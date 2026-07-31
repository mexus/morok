//! BGE-M3 — multilingual embedding model and reranker (`BAAI/bge-m3`).
//!
//! Built on the [`crate::xlm_roberta`] backbone. Three retrieval heads:
//!
//! - **Dense**: CLS pooling + L2 normalize → `(B, D)`
//! - **Sparse**: Linear → ReLU → scatter-to-vocab → `(B, vocab_size)`
//! - **ColBERT**: per-token Linear (skip CLS) + L2 normalize → `(B, L-1, Dc)`
//!
//! Also provides [`BgeRerankerV2M3`] (`BAAI/bge-reranker-v2-m3`), a cross-encoder
//! reranker sharing the same XLM-RoBERTa backbone with a classification head.

mod colbert_head;
mod embedder;
mod jit;
mod reranker;
mod scoring;
mod sparse_head;

pub use colbert_head::ColbertHead;
pub use embedder::{BgeM3, BgeM3Output, EncodeOpts};
pub use jit::{BgeM3ColbertJit, BgeM3DenseJit, BgeRerankerJit};
pub use reranker::BgeRerankerV2M3;
pub use scoring::{colbert_score, dense_score, hybrid_score, sparse_score};
pub use sparse_head::SparseHead;
