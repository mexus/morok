//! Unit tests for the profiler data model, table rendering (GPU-free), and
//! [`RunProfile::merge`] accumulation semantics.

use std::time::Duration;

use svod_device::PmcCounter;

use crate::profiler::{PmcSelection, ProfileOptions, RunProfile, StageProfile, parse_pmc};

#[test]
fn pmc_counter_token_roundtrip() {
    for c in [PmcCounter::SqBusyCycles, PmcCounter::SqWaves, PmcCounter::SqInstsValu] {
        assert_eq!(PmcCounter::from_token(c.token()), Some(c), "token roundtrip for {c:?}");
    }
    assert_eq!(PmcCounter::from_token("nope"), None);
    assert_eq!(PmcCounter::from_token("BUSY"), Some(PmcCounter::SqBusyCycles), "case-insensitive alias");
}

#[test]
fn pmc_selection_resolution() {
    assert!(!PmcSelection::None.is_enabled());
    assert!(PmcSelection::None.counters().is_empty());
    assert!(PmcSelection::Default.is_enabled());
    assert_eq!(PmcSelection::Default.counters().len(), 3);
    let custom = PmcSelection::Custom(vec![PmcCounter::SqInstsValu]);
    assert_eq!(custom.counters(), vec![PmcCounter::SqInstsValu]);
}

#[test]
fn parse_pmc_values() {
    assert_eq!(parse_pmc(""), PmcSelection::None);
    assert_eq!(parse_pmc("0"), PmcSelection::None);
    assert_eq!(parse_pmc("1"), PmcSelection::Default);
    assert_eq!(parse_pmc("valu,waves"), PmcSelection::Custom(vec![PmcCounter::SqInstsValu, PmcCounter::SqWaves]));
    // All-unknown tokens fall back to the default set rather than an empty selection.
    assert_eq!(parse_pmc("bogus"), PmcSelection::Default);
}

#[test]
fn profile_options_default() {
    let o = ProfileOptions::default();
    assert_eq!(o.iters, 1);
    assert!(o.static_analysis);
    assert_eq!(o.counters, PmcSelection::None);
}

#[test]
fn render_table_empty_and_host_only() {
    // Empty report renders nothing.
    assert_eq!(RunProfile::default().render_table(), "");

    // A host-only stage (no kernels) renders a single wall line, no metric table.
    let mut rp = RunProfile::default();
    rp.push(StageProfile::host("mel", Duration::from_millis(3)));
    let out = rp.render_table();
    assert!(out.contains("mel"), "host stage name present: {out:?}");
    assert!(out.contains("host"), "host stage tagged host: {out:?}");
    assert!(!out.contains("GFLOP/s"), "no metric columns for host-only: {out:?}");
}

#[test]
fn merge_accumulates_same_named_stages_and_appends_new() {
    let mut a = RunProfile::default();
    a.push(StageProfile::host("mel", Duration::from_millis(2)));
    let mut enc = StageProfile::host("encoder", Duration::from_millis(10));
    enc.meta.insert("rtf".into(), "0.02".into());
    a.push(enc);

    let mut b = RunProfile::default();
    let mut enc2 = StageProfile::host("encoder", Duration::from_millis(5)); // same name → sum wall + meta
    enc2.meta.insert("chunks".into(), "4".into());
    b.push(enc2);
    b.push(StageProfile::host("decode", Duration::from_millis(3))); // new name → appended

    a.merge(b);

    let names: Vec<&str> = a.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["mel", "encoder", "decode"], "matched stays in place, new appends");
    assert_eq!(a.stage("mel").unwrap().wall, Duration::from_millis(2), "untouched");

    let enc = a.stage("encoder").unwrap();
    assert_eq!(enc.wall, Duration::from_millis(15), "10 + 5 summed");
    assert_eq!(enc.meta.get("rtf").map(String::as_str), Some("0.02"), "kept");
    assert_eq!(enc.meta.get("chunks").map(String::as_str), Some("4"), "folded in");
}
