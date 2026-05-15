//! SentencePiece `.model` (protobuf) loader. Pieces retain their `▁`
//! (U+2581) prefix on word-initial tokens; consumers replace `▁` with a
//! space for natural detokenization.

use std::path::{Path, PathBuf};

use snafu::{ResultExt, Snafu};

/// Partial decode of `ModelProto` — only the `pieces` array. prost skips
/// unknown tags. Tag source of truth: upstream `google/sentencepiece`
/// repo's `src/sentencepiece_model.proto`.
#[derive(prost::Message)]
struct SpModelProto {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<SpPiece>,
}

#[derive(prost::Message)]
struct SpPiece {
    /// The piece string, e.g. `"▁hello"` (`U+2581` = SP space marker) or
    /// `"<unk>"` for control tokens.
    #[prost(string, optional, tag = "1")]
    piece: Option<String>,
    /// `enum Type { NORMAL = 1; UNKNOWN = 2; CONTROL = 3; USER_DEFINED = 4; BYTE = 6; UNUSED = 5 }`.
    #[prost(int32, optional, tag = "3")]
    r#type: Option<i32>,
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("reading SentencePiece model from {}: {source}", path.display()))]
    Io { path: PathBuf, source: std::io::Error },
    #[snafu(display("parsing SentencePiece model at {}: {source}", path.display()))]
    Decode { path: PathBuf, source: prost::DecodeError },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Read a SentencePiece `.model` file and return per-id raw pieces.
///
/// Special tokens (UNKNOWN=2, CONTROL=3, BYTE=6, UNUSED=5) are mapped to the
/// empty string so they elide from the transcript on the (rare) chance the
/// model emits one.
pub fn load_vocab(path: &Path) -> Result<Vec<String>> {
    use prost::Message;
    let bytes = std::fs::read(path).context(IoSnafu { path: path.to_path_buf() })?;
    let proto = SpModelProto::decode(&*bytes).context(DecodeSnafu { path: path.to_path_buf() })?;
    let mut pieces = Vec::with_capacity(proto.pieces.len());
    for p in proto.pieces {
        let kind = p.r#type.unwrap_or(1);
        // Type 1 = NORMAL, 4 = USER_DEFINED. Everything else (UNKNOWN,
        // CONTROL, BYTE, UNUSED) is non-emittable: store empty so the
        // transcript stays clean if the predictor accidentally lands there.
        let s = if kind == 1 || kind == 4 { p.piece.unwrap_or_default() } else { String::new() };
        pieces.push(s);
    }
    Ok(pieces)
}
