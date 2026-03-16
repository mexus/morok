#![allow(non_snake_case)]

use super::helpers::run_onnx_node_test;
include!(concat!(env!("OUT_DIR"), "/onnx_node_tests.rs"));
