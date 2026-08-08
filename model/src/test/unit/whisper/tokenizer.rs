use crate::whisper::WhisperTokenizer;

#[test]
fn word_split_preserves_text_and_tokens() {
    let tokenizer = WhisperTokenizer::from_hub(true, 99).unwrap();
    let tokens = tokenizer.encode(" hello world!");
    let (words, word_tokens) = tokenizer.split_to_word_tokens_for_language(&tokens, Some("en"));
    assert_eq!(words.concat(), " hello world!");
    assert_eq!(word_tokens.concat(), tokens);
    assert_eq!(words, [" hello", " world", "!"]);
}

#[test]
fn no_space_languages_split_on_unicode_boundaries() {
    let tokenizer = WhisperTokenizer::from_hub(true, 99).unwrap();
    let tokens = tokenizer.encode("你好世界");
    let (words, word_tokens) = tokenizer.split_to_word_tokens_for_language(&tokens, Some("zh"));
    assert_eq!(words.concat(), "你好世界");
    assert_eq!(word_tokens.concat(), tokens);
    assert!(words.len() > 1);
}

#[test]
fn timestamps_are_word_boundaries() {
    let tokenizer = WhisperTokenizer::from_hub(true, 99).unwrap();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = tokenizer.encode(" hello");
    tokens.push(timestamp + 10);
    tokens.extend(tokenizer.encode("world"));

    let (words, word_tokens) = tokenizer.split_to_word_tokens_for_language(&tokens, Some("en"));
    assert_eq!(word_tokens.concat(), tokens);
    assert_eq!(word_tokens[1], [timestamp + 10]);
    assert_eq!(words[2], "world");
}

#[test]
fn punctuation_is_kept_separate_for_timing_attachment() {
    let tokenizer = WhisperTokenizer::from_hub(true, 99).unwrap();
    let tokens = tokenizer.encode(" hello!");
    let (words, _) = tokenizer.split_to_word_tokens_for_language(&tokens, Some("en"));
    assert_eq!(words, [" hello", "!"]);
}
