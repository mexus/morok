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
