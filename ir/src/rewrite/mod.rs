//! Graph rewrite engine with fixed-point iteration.
//!
//! This module implements the core graph rewriting algorithm that applies
//! pattern-based transformations to UOp graphs.

pub mod engine;

pub use engine::{NoMatcher, graph_rewrite, graph_rewrite_bottom_up, graph_rewrite_with_bpm};
