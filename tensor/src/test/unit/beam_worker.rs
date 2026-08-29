use super::*;

fn response(index: usize) -> WorkerResponse {
    WorkerResponse { index, result: None, error: None }
}

fn overdue(timeout: Duration) -> BusyTask {
    BusyTask { index: 0, started: Instant::now() - timeout }
}

/// TA6: a response written exactly on the deadline is already in the channel,
/// so it must be delivered instead of dropping the candidate and SIGKILLing a
/// healthy helper.
#[test]
fn a_response_on_the_deadline_is_delivered_not_timed_out() {
    let timeout = Duration::from_millis(50);
    let (send, responses) = mpsc::channel();
    send.send(Ok(response(7))).unwrap();

    let outcome = poll_slot(&responses, Some(&overdue(timeout)), timeout);
    assert!(matches!(outcome, SlotOutcome::Response(response) if response.index == 7), "on-deadline response dropped");
}

#[test]
fn an_empty_channel_past_the_deadline_times_out() {
    let timeout = Duration::from_millis(50);
    let (_send, responses) = mpsc::channel::<std::io::Result<WorkerResponse>>();
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), timeout), SlotOutcome::TimedOut));
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), Duration::ZERO), SlotOutcome::Idle));
    assert!(matches!(poll_slot(&responses, None, timeout), SlotOutcome::Idle), "an idle worker never times out");
}

#[test]
fn a_closed_helper_fails_the_slot_regardless_of_the_deadline() {
    let timeout = Duration::from_millis(50);
    let (send, responses) = mpsc::channel();
    send.send(Err(std::io::Error::other("stdout closed"))).unwrap();
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), timeout), SlotOutcome::Failed(Some(_))));

    let (send, responses) = mpsc::channel::<std::io::Result<WorkerResponse>>();
    drop(send);
    assert!(matches!(poll_slot(&responses, Some(&overdue(timeout)), timeout), SlotOutcome::Failed(None)));
}

/// TA5: a resolution failure must not be latched — the next attempt has to run
/// the resolver again, and only a success is remembered.
#[test]
fn helper_resolution_failures_are_not_cached() {
    let cache = std::sync::Mutex::new(None);
    let attempts = std::cell::Cell::new(0usize);
    let resolve = || {
        attempts.set(attempts.get() + 1);
        match attempts.get() {
            1 => Err(BeamWorker::HelperUnavailable { reason: "not built yet".into() }),
            _ => Ok(std::path::PathBuf::from("/helper/svod-beam-worker")),
        }
    };

    assert!(cached_helper_path(&cache, resolve).is_err(), "first resolution must fail");
    let resolved = cached_helper_path(&cache, resolve).expect("a failure must not be latched");
    assert_eq!(resolved, std::path::PathBuf::from("/helper/svod-beam-worker"));
    assert_eq!(cached_helper_path(&cache, resolve).unwrap(), resolved);
    assert_eq!(attempts.get(), 2, "a cached success must not re-resolve");
}

/// A `SVOD_BEAM_WORKER` that is not a file is reported, not silently ignored.
#[test]
fn a_non_file_helper_override_is_rejected() {
    let helper = tempfile::NamedTempFile::new().expect("helper stand-in");
    let missing = helper.path().with_extension("absent");
    unsafe { std::env::set_var("SVOD_BEAM_WORKER", &missing) };
    let failure = resolve_helper_path().expect_err("a missing helper must not resolve");
    unsafe { std::env::remove_var("SVOD_BEAM_WORKER") };
    assert!(
        matches!(&failure, BeamWorker::HelperUnavailable { reason } if reason.contains("is not a file")),
        "{failure}"
    );
}

/// The helper path comes from cargo's own artifact report, not a guessed
/// `target/<profile>/` layout.
#[test]
fn last_executable_takes_cargos_final_artifact() {
    let messages = concat!(
        r#"{"reason":"compiler-artifact","target":{"name":"svod-tensor"},"executable":null}"#,
        "\n",
        r#"{"reason":"compiler-artifact","executable":"/custom/target/dir/debug/svod-beam-worker"}"#,
        "\n",
        r#"{"reason":"build-finished","success":true}"#,
        "\n",
    );
    assert_eq!(
        last_executable(messages.as_bytes()),
        Some(std::path::PathBuf::from("/custom/target/dir/debug/svod-beam-worker"))
    );
    assert_eq!(last_executable(b"not json\n"), None);
}

/// CV3: pool failures are typed, so a caller can tell an out-of-order helper
/// from an unavailable one instead of matching on a rendered string.
#[test]
fn worker_misorder_is_a_distinguishable_variant() {
    let misorder = BeamWorker::WorkerMisorder { got: 3, expected: Some(1) };
    assert!(matches!(misorder, BeamWorker::WorkerMisorder { got: 3, expected: Some(1) }));
    assert!(misorder.to_string().contains("expected Some(1)"), "{misorder}");
}
