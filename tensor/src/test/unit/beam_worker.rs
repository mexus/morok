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

/// TA5: a bad `SVOD_BEAM_WORKER` must not be latched — the next resolution with
/// a valid helper has to succeed.
#[test]
fn helper_resolution_failures_are_not_cached() {
    let helper = tempfile::NamedTempFile::new().expect("helper stand-in");
    let missing = helper.path().with_extension("absent");

    unsafe { std::env::set_var("SVOD_BEAM_WORKER", &missing) };
    let failure = helper_path().expect_err("a missing helper must not resolve");
    assert!(failure.contains("is not a file"), "{failure}");

    unsafe { std::env::set_var("SVOD_BEAM_WORKER", helper.path()) };
    let resolved = helper_path().expect("a valid helper must resolve after a failure");
    unsafe { std::env::remove_var("SVOD_BEAM_WORKER") };
    assert_eq!(resolved, helper.path());
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
