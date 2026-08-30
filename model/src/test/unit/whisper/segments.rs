use crate::whisper::{WhisperTokenizer, split_into_segments};

fn tokenizer() -> WhisperTokenizer {
    WhisperTokenizer::from_hub(true, 99).unwrap()
}

#[test]
fn unfinished_timestamp_tail_is_not_emitted() {
    let tokenizer = tokenizer();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = vec![timestamp];
    tokens.extend(tokenizer.encode(" hello"));
    tokens.extend([timestamp + 50, timestamp + 50]);
    tokens.extend(tokenizer.encode(" unfinished"));

    let segments = split_into_segments(&tokens, &tokenizer, 30.0);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start, 0.0);
    assert_eq!(segments[0].end, 1.0);
}

#[test]
fn segment_without_timestamps_spans_real_window() {
    let tokenizer = tokenizer();
    let tokens = tokenizer.encode(" hello");
    let segments = split_into_segments(&tokens, &tokenizer, 2.5);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 0.0);
    assert_eq!(segments[0].end, 2.5);
}

#[test]
fn last_timestamp_limits_unpaired_segment() {
    let tokenizer = tokenizer();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = vec![timestamp];
    tokens.extend(tokenizer.encode(" hello"));
    tokens.push(timestamp + 75);
    tokens.extend(tokenizer.encode(" tail"));

    let segments = split_into_segments(&tokens, &tokenizer, 4.0);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].end, 1.5);
}

#[test]
fn timestamp_segments_are_clipped_to_real_audio_extent() {
    let tokenizer = tokenizer();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = vec![timestamp + 50];
    tokens.extend(tokenizer.encode(" hello"));
    tokens.extend([timestamp + 500, timestamp + 500]);
    tokens.extend(tokenizer.encode(" beyond"));
    tokens.push(timestamp + 600);

    let segments = split_into_segments(&tokens, &tokenizer, 2.5);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 1.0);
    assert_eq!(segments[0].end, 2.5);
}
