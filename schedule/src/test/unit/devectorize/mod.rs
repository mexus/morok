//! Devectorizer test suite (devectorize.rs).
//!
//! Focused tests for Tinygrad's devectorizer2 and Rust's shaped STACK mapping.
//!
//! # Test Organization
//!
//! - `helpers`: Test builders and assertion helpers
//! - `bool_storage`: bool->uint8 conversion tests
//! - `alu_devectorization`: no_vectorized_alu tests
//! - `pipeline`: End-to-end devectorize() tests
//! - `edge_cases`: Corner cases and regression tests

pub mod alu_devectorization;
pub mod bool_storage;
pub mod edge_cases;
pub mod fp8_decomp;
pub mod gep_movement;
pub mod helpers;
pub mod late_gater;
pub mod new_patterns;
pub mod pipeline;
pub mod reduce_to_acc;
