#![allow(non_snake_case)]

use super::helpers::run_onnx_light_test;
include!(concat!(env!("OUT_DIR"), "/onnx_light_tests.rs"));
