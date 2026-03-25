//! ONNX model frontend for Morok.
//!
//! This crate provides ONNX model parsing and conversion to Morok Tensors.
//!
//! # Example
//!
//! ```ignore
//! use morok_onnx::OnnxImporter;
//!
//! // Import model — returns inputs, outputs, and variables as normal Morok types
//! let model = OnnxImporter::new().import("model.onnx", &[("batch", 4)])?;
//!
//! // Outputs are lazy Tensors — realize to execute
//! let result = model.outputs["output"].realize()?;
//!
//! // Dynamic dimensions are auto-extracted as Variables
//! for (name, var) in &model.variables {
//!     println!("{name}: bounds {:?}", var.bounds());
//! }
//! ```

pub mod error;
pub mod importer;
pub mod parser;
pub mod registry;

pub use error::{Error, Result};
pub use importer::{OnnxImporter, OnnxModel};

#[cfg(test)]
mod test;
