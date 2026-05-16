//! Smoke tests for `Tensor::rand`: realizes, values are in `[0, 1)`, and the
//! sequence is deterministic for a fixed seed across `manual_seed` resets.
//!
//! These tests share the global RNG state (`manual_seed` is a process-wide
//! operation), so they serialize against each other via `RAND_TEST_LOCK`.

use crate::Tensor;
use crate::rand::manual_seed;

use super::RAND_TEST_LOCK;

fn realize_f32(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<f32> {
    t.realize_with(config).expect("realize");
    t.as_vec::<f32>().expect("read")
}

crate::codegen_tests! {
    fn rand_produces_finite_values_in_unit_interval(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(1337);
        let mut t = Tensor::rand(&[16]).expect("build rand graph");
        let v = realize_f32(&mut t, &config);
        assert_eq!(v.len(), 16);
        for (i, &x) in v.iter().enumerate() {
            assert!(x.is_finite(), "rand[{i}] = {x} is non-finite");
            assert!((0.0..1.0).contains(&x), "rand[{i}] = {x} is outside [0, 1)");
        }
    }

    fn rand_is_deterministic_after_manual_seed_reset(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(42);
        let mut a = Tensor::rand(&[8]).unwrap();
        let v_a = realize_f32(&mut a, &config);

        manual_seed(42);
        let mut b = Tensor::rand(&[8]).unwrap();
        let v_b = realize_f32(&mut b, &config);

        assert_eq!(v_a, v_b, "same seed must yield same output");
    }

    fn rand_advances_counter_between_calls(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(7);
        let mut first = Tensor::rand(&[8]).unwrap();
        let v_first = realize_f32(&mut first, &config);
        let mut second = Tensor::rand(&[8]).unwrap();
        let v_second = realize_f32(&mut second, &config);
        assert_ne!(v_first, v_second, "consecutive draws must differ (counter must advance)");
    }

    fn rand_differs_across_seeds(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(1);
        let mut a = Tensor::rand(&[8]).unwrap();
        let v_a = realize_f32(&mut a, &config);
        manual_seed(2);
        let mut b = Tensor::rand(&[8]).unwrap();
        let v_b = realize_f32(&mut b, &config);
        assert_ne!(v_a, v_b, "different seeds must yield different output");
    }
}
