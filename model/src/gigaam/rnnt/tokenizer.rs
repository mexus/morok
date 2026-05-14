//! SentencePiece tokenizer loader. Reads a `.model` (protobuf) file and
//! extracts the per-id pieces. Pieces retain their `▁` (U+2581) prefix on
//! word-initial tokens; the call site concatenates and replaces `▁` with a
//! space for natural detokenization.

use std::path::Path;

use snafu::ResultExt;

use crate::gigaam::Result;
use crate::gigaam::error::{ConfigIoSnafu, Error};

/// Minimal subset of the SentencePiece `ModelProto` schema needed to read out
/// the `pieces` array. We don't need the trainer/normalizer specs, the score
/// field, or any other top-level fields — prost silently skips unknown tags
/// during decode, so this partial schema suffices.
///
/// Source of truth for tags: `submodules/GigaAM/.../sentencepiece_model.proto`
/// (or upstream `google/sentencepiece` repo `src/sentencepiece_model.proto`).
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

/// Read a SentencePiece `.model` file and return per-id raw pieces.
///
/// Special tokens (UNKNOWN=2, CONTROL=3, BYTE=6, UNUSED=5) are mapped to the
/// empty string so they elide from the transcript on the (rare) chance the
/// model emits one.
pub(crate) fn load_sentencepiece_vocab(path: &Path) -> Result<Vec<String>> {
    use prost::Message;
    let bytes = std::fs::read(path).context(ConfigIoSnafu)?;
    let proto = SpModelProto::decode(&*bytes).map_err(|e| Error::DecoderConfig {
        message: format!("failed to parse SentencePiece model at {}: {e}", path.display()),
    })?;
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
