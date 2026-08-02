//! Whisper BPE tokenizer — delegates to `tiktoken-rs` CoreBPE.
//!
//! The tokenizer loads a `.tiktoken` rank file (base64-encoded BPE merges),
//! appends Whisper's special tokens (SOT, language, task, timestamp, …), and
//! provides Whisper-specific helpers on top of the battle-tested CoreBPE.

use std::collections::HashMap;

use tiktoken_rs::CoreBPE;

use super::error::{Error, Result};

// ─── Language codes ─────────────────────────────────────────────────────────

pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

/// GPT-2 BPE regex pattern used by tiktoken (same as whisper/tokenizer.py).
const GPT2_PAT: &str = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// ─── Tokenizer ──────────────────────────────────────────────────────────────

pub struct WhisperTokenizer {
    /// The CoreBPE engine (tiktoken-rs) — handles encode/decode.
    bpe: CoreBPE,
    /// Special token strings → ids.
    special_tokens: HashMap<String, u32>,
    /// Base vocabulary size (before special tokens).
    n_vocab_base: usize,
    /// Whether this is a multilingual model.
    pub multilingual: bool,
    /// Number of languages supported.
    pub num_languages: usize,
    /// Configured language code (e.g. "en").
    pub language: Option<String>,
    /// Configured task: "transcribe" or "translate".
    pub task: Option<String>,
}

impl WhisperTokenizer {
    /// Build a tokenizer from a `.tiktoken` rank file (base64-encoded BPE
    /// ranks, one per line: `<base64_token> <rank>`).
    pub fn new(
        tiktoken_data: &str,
        multilingual: bool,
        num_languages: usize,
        language: Option<&str>,
        task: Option<&str>,
    ) -> Result<Self> {
        // Parse the .tiktoken file into a rank map.
        let encoder: rustc_hash::FxHashMap<Vec<u8>, u32> = parse_tiktoken_ranks(tiktoken_data)?;

        let n_vocab_base = encoder.len();

        // Build Whisper special tokens with sequential IDs starting at n_vocab_base.
        let specials = build_special_tokens(multilingual, num_languages);
        let mut special_tokens: rustc_hash::FxHashMap<String, u32> = rustc_hash::FxHashMap::default();
        for (i, s) in specials.iter().enumerate() {
            special_tokens.insert(s.clone(), (n_vocab_base + i) as u32);
        }

        // Build the CoreBPE — uses FxHashMap internally.
        let bpe = CoreBPE::new(encoder, special_tokens.clone(), GPT2_PAT)
            .map_err(|e| Error::Tokenizer { msg: format!("CoreBPE::new: {e}") })?;

        // Convert special_tokens to std HashMap for our lookups
        let special_tokens: HashMap<String, u32> = special_tokens.into_iter().collect();

        Ok(Self {
            bpe,
            special_tokens,
            n_vocab_base,
            multilingual,
            num_languages,
            language: language.map(|s| s.to_string()),
            task: task.map(|s| s.to_string()),
        })
    }

    // ─── Tokenizer loading helpers ────────────────────────────────────────────

    /// Load tokenizer for a Whisper model. Uses embedded tiktoken data
    /// (from the `openai/whisper` submodule) — no runtime download.
    pub fn from_hub(multilingual: bool, num_languages: usize) -> Result<Self> {
        let data = if multilingual {
            include_str!("assets/multilingual.tiktoken")
        } else {
            include_str!("assets/gpt2.tiktoken")
        };
        Self::new(data, multilingual, num_languages, Some("en"), Some("transcribe"))
    }

    /// Load the tiktoken data from a local file.
    pub fn from_file(path: &std::path::Path, multilingual: bool, num_languages: usize) -> Result<Self> {
        let data =
            std::fs::read_to_string(path).map_err(|e| Error::Tokenizer { msg: format!("read tiktoken file: {e}") })?;
        Self::new(&data, multilingual, num_languages, Some("en"), Some("transcribe"))
    }

    // ─── Special token accessors ────────────────────────────────────────────

    pub fn eot(&self) -> u32 {
        self.special_tokens["<|endoftext|>"]
    }

    pub fn sot(&self) -> u32 {
        self.special_tokens["<|startoftranscript|>"]
    }

    pub fn transcribe(&self) -> u32 {
        self.special_tokens["<|transcribe|>"]
    }

    pub fn translate(&self) -> u32 {
        self.special_tokens["<|translate|>"]
    }

    pub fn sot_prev(&self) -> u32 {
        self.special_tokens["<|startofprev|>"]
    }

    pub fn sot_lm(&self) -> u32 {
        self.special_tokens["<|startoflm|>"]
    }

    pub fn no_speech(&self) -> Option<u32> {
        self.special_tokens.get("<|nospeech|>").copied()
    }

    pub fn no_timestamps(&self) -> u32 {
        self.special_tokens["<|notimestamps|>"]
    }

    pub fn timestamp_begin(&self) -> u32 {
        self.special_tokens["<|0.00|>"]
    }

    /// SOT sequence: [sot, [language], [task]]
    pub fn sot_sequence(&self) -> Vec<u32> {
        let mut seq = vec![self.sot()];
        if let Some(lang) = &self.language
            && let Some(&tok) = self.special_tokens.get(&format!("<|{lang}|>"))
        {
            seq.push(tok);
        }
        if let Some(task) = &self.task {
            let tok = if task == "transcribe" { self.transcribe() } else { self.translate() };
            seq.push(tok);
        }
        seq
    }

    /// SOT sequence including `<|notimestamps|>`.
    pub fn sot_sequence_including_notimestamps(&self) -> Vec<u32> {
        let mut seq = self.sot_sequence();
        seq.push(self.no_timestamps());
        seq
    }

    /// All language token IDs.
    pub fn all_language_tokens(&self) -> Vec<u32> {
        LANGUAGES
            .iter()
            .take(self.num_languages)
            .filter_map(|(code, _)| self.special_tokens.get(&format!("<|{code}|>")).copied())
            .collect()
    }

    /// Language codes matching [`all_language_tokens`](Self::all_language_tokens).
    pub fn all_language_codes(&self) -> Vec<String> {
        LANGUAGES
            .iter()
            .take(self.num_languages)
            .filter_map(|(code, _)| self.special_tokens.get(&format!("<|{code}|>")).map(|_| code.to_string()))
            .collect()
    }

    /// Look up the language token for a code string.
    pub fn language_token_for(&self, code: &str) -> Option<u32> {
        self.special_tokens.get(&format!("<|{code}|>")).copied()
    }

    /// Look up the language code for a token id.
    pub fn code_for_token(&self, token: u32) -> Option<String> {
        LANGUAGES
            .iter()
            .take(self.num_languages)
            .find(|(code, _)| self.special_tokens.get(&format!("<|{code}|>")).map(|t| *t == token).unwrap_or(false))
            .map(|(code, _)| code.to_string())
    }

    // ─── Encode / decode (delegated to CoreBPE) ──────────────────────────────

    /// Encode text using BPE (no special tokens).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_ordinary(text)
    }

    /// Decode token IDs to text, filtering out special/timestamp tokens.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let filtered: Vec<u32> = token_ids.iter().filter(|&&t| t < self.timestamp_begin()).copied().collect();
        self.bpe.decode(&filtered).unwrap_or_default()
    }

    /// Decode including timestamp tokens (annotated).
    pub fn decode_with_timestamps(&self, token_ids: &[u32]) -> String {
        let ts_begin = self.timestamp_begin();
        let mut result = String::new();
        for &id in token_ids {
            if id >= ts_begin {
                let secs = (id - ts_begin) as f32 / super::config::TOKENS_PER_SECOND;
                result.push_str(&format!("<|{secs:.2}|>"));
            } else {
                // Decode single token via CoreBPE
                if let Ok(s) = self.bpe.decode(&[id]) {
                    result.push_str(&s);
                }
            }
        }
        result
    }

    /// Split tokens into words and their constituent token lists.
    /// Uses CoreBPE's `decode_bytes` for accurate per-token byte mapping.
    pub fn split_to_word_tokens(&self, tokens: &[u32]) -> (Vec<String>, Vec<Vec<u32>>) {
        let ts_begin = self.timestamp_begin();
        let eot = self.eot();

        let mut words: Vec<String> = Vec::new();
        let mut word_tokens: Vec<Vec<u32>> = Vec::new();
        let mut current_chars = String::new();
        let mut current_tokens = Vec::new();

        for &tok in tokens {
            current_tokens.push(tok);

            // Timestamp / EOT token: commit current word, push special as its own
            if tok >= ts_begin || tok == eot {
                if !current_chars.is_empty() {
                    words.push(std::mem::take(&mut current_chars));
                    word_tokens.push(current_tokens[..current_tokens.len() - 1].to_vec());
                }
                if tok < eot {
                    words.push(format!("<|{:.2}|>", (tok - ts_begin) as f32 / super::config::TOKENS_PER_SECOND));
                } else {
                    words.push("<|endoftext|>".to_string());
                }
                word_tokens.push(vec![tok]);
                current_tokens.clear();
                continue;
            }

            // Decode this single token to check for word boundaries
            if tok < self.n_vocab_base as u32
                && let Ok(decoded) = self.bpe.decode(&[tok])
            {
                let starts_new = decoded.starts_with(' ') && !current_chars.is_empty();
                if starts_new {
                    words.push(std::mem::take(&mut current_chars));
                    word_tokens.push(current_tokens[..current_tokens.len() - 1].to_vec());
                    current_tokens.clear();
                    current_tokens.push(tok);
                    current_chars.push_str(&decoded[1..]);
                } else {
                    current_chars.push_str(&decoded);
                }
            }
        }
        if !current_chars.is_empty() {
            words.push(std::mem::take(&mut current_chars));
            word_tokens.push(std::mem::take(&mut current_tokens));
        }

        (words, word_tokens)
    }

    /// Non-speech tokens to suppress (matching whisper/tokenizer.py).
    pub fn non_speech_tokens(&self) -> Vec<u32> {
        let symbols = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』";
        let extras = [
            "<<",
            ">>",
            "<<<",
            ">>>",
            "--",
            "---",
            "-(",
            "-[",
            "('",
            "(\"",
            "((",
            "))",
            "(((",
            ")))",
            "[[",
            "]]",
            "{{",
            "}}",
            "♪♪",
            "♪♪♪",
        ];
        let misc = "♩♪♫♬♭♮♯";

        let mut result: Vec<u32> = Vec::new();

        for s in symbols.chars() {
            let ids = self.encode(&s.to_string());
            if ids.len() == 1 {
                result.push(ids[0]);
            }
            let ids = self.encode(&format!(" {s}"));
            if ids.len() == 1 {
                result.push(ids[0]);
            }
        }
        for e in &extras {
            let ids = self.encode(e);
            if ids.len() == 1 {
                result.push(ids[0]);
            }
            let ids = self.encode(&format!(" {e}"));
            if ids.len() == 1 {
                result.push(ids[0]);
            }
        }
        for c in misc.chars() {
            let ids = self.encode(&c.to_string());
            if !ids.is_empty() {
                result.push(ids[0]);
            }
        }
        let dash = self.encode(" -");
        if !dash.is_empty() {
            result.push(dash[0]);
        }
        let quote = self.encode(" '");
        if !quote.is_empty() {
            result.push(quote[0]);
        }

        result.sort();
        result.dedup();
        result
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Parse a `.tiktoken` file (base64-encoded token → rank per line).
fn parse_tiktoken_ranks(data: &str) -> Result<rustc_hash::FxHashMap<Vec<u8>, u32>> {
    use base64::Engine;
    let mut ranks = rustc_hash::FxHashMap::default();
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let token_b64 = parts.next().ok_or_else(|| Error::Tokenizer { msg: "malformed tiktoken line".into() })?;
        let rank_str = parts.next().ok_or_else(|| Error::Tokenizer { msg: "malformed tiktoken line".into() })?;
        let token_bytes = if token_b64 == "=" {
            // Python's base64.b64decode("=") returns empty bytes
            Vec::new()
        } else {
            base64::engine::general_purpose::STANDARD
                .decode(token_b64)
                .map_err(|e| Error::Tokenizer { msg: format!("base64 decode: {e}") })?
        };
        let rank: u32 = rank_str.parse().map_err(|e| Error::Tokenizer { msg: format!("rank parse: {e}") })?;
        ranks.insert(token_bytes, rank);
    }
    Ok(ranks)
}

fn build_special_tokens(_multilingual: bool, num_languages: usize) -> Vec<String> {
    let mut specials = vec!["<|endoftext|>".to_string(), "<|startoftranscript|>".to_string()];
    for (code, _) in LANGUAGES.iter().take(num_languages) {
        specials.push(format!("<|{code}|>"));
    }
    specials.push("<|translate|>".into());
    specials.push("<|transcribe|>".into());
    specials.push("<|startoflm|>".into());
    specials.push("<|startofprev|>".into());
    specials.push("<|nospeech|>".into());
    specials.push("<|notimestamps|>".into());
    for i in 0..=1500 {
        specials.push(format!("<|{:.2}|>", i as f32 * 0.02));
    }
    specials
}
