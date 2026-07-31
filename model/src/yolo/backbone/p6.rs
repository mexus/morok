//! [`YoloBackboneP6`] — YOLO v26-P6 backbone (layers 0–12).
//!
//! Deeper than the standard backbone: adds a P6/64 stage (Conv→C3k2) after
//! P5/32, and P5 is narrowed to 768 channels. SPPF uses `shortcut=false`.

use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use crate::yolo::blocks::attention::C2PSA;
use crate::yolo::blocks::conv::YoloConv;
use crate::yolo::blocks::csp::C3k2;
use crate::yolo::blocks::sppf::Sppf;
use crate::yolo::config::{YoloScale, make_depth, scale_channels};
use crate::yolo::error::Result;

/// P6 backbone channels after scaling: `[c0, c1, c2, c3, c5, c6]`.
/// `c5 = scale_channels(768)` (P5), `c6 = scale_channels(1024)` (P6).
pub fn p6_scaled_channels(scale: YoloScale) -> [usize; 6] {
    let sc = |yaml_c| scale_channels(yaml_c, scale);
    [sc(64), sc(128), sc(256), sc(512), sc(768), sc(1024)]
}

/// Full YOLO v26-P6 backbone (layers 0–12).
///
/// Forward returns four skip-connection outputs: `(l4, l6, l8, l12)`
/// at strides 8, 16, 32, 64.
#[derive(Clone)]
pub struct YoloBackboneP6 {
    pub conv0: YoloConv,
    pub conv1: YoloConv,
    pub c3k2_2: C3k2,
    pub conv3: YoloConv,
    pub c3k2_4: C3k2,
    pub conv5: YoloConv,
    pub c3k2_6: C3k2,
    pub conv7: YoloConv,
    pub c3k2_8: C3k2,
    pub conv9: YoloConv,
    pub c3k2_10: C3k2,
    pub sppf11: Sppf,
    pub c2psa12: C2PSA,
}

impl YoloBackboneP6 {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [c0, c1, c2, c3, c5, c6] = p6_scaled_channels(scale);
        Self {
            conv0: YoloConv::empty(3, c0, 3, 2, true),
            conv1: YoloConv::empty(c0, c1, 3, 2, true),
            c3k2_2: C3k2::empty(c1, c2, d(2), true, 0.25, false, false),
            conv3: YoloConv::empty(c2, c2, 3, 2, true),
            c3k2_4: C3k2::empty(c2, c3, d(2), true, 0.25, false, false),
            conv5: YoloConv::empty(c3, c3, 3, 2, true),
            c3k2_6: C3k2::empty(c3, c3, d(2), true, 0.5, true, false),
            conv7: YoloConv::empty(c3, c5, 3, 2, true),
            c3k2_8: C3k2::empty(c5, c5, d(2), true, 0.5, true, false),
            conv9: YoloConv::empty(c5, c6, 3, 2, true),
            c3k2_10: C3k2::empty(c6, c6, d(2), true, 0.5, true, false),
            sppf11: Sppf::empty(c6, c6, 5, 3, false),
            c2psa12: C2PSA::empty(c6, c6, d(2), 0.5),
        }
    }

    /// Run backbone layers 0–12, returning `(l4, l6, l8, l12)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let x = self.conv0.forward(x)?;
        let x = self.conv1.forward(&x)?;
        let x = self.c3k2_2.forward(&x)?;
        let x = self.conv3.forward(&x)?;
        let l4 = self.c3k2_4.forward(&x)?;
        let x = self.conv5.forward(&l4)?;
        let l6 = self.c3k2_6.forward(&x)?;
        let x = self.conv7.forward(&l6)?;
        let l8 = self.c3k2_8.forward(&x)?;
        let x = self.conv9.forward(&l8)?;
        let x = self.c3k2_10.forward(&x)?;
        let x = self.sppf11.forward(&x)?;
        let l12 = self.c2psa12.forward(&x)?;
        Ok((l4, l6, l8, l12))
    }
}

impl HasStateDict for YoloBackboneP6 {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        let mut sd = self.conv0.state_dict(&p(0));
        sd.extend(self.conv1.state_dict(&p(1)));
        sd.extend(self.c3k2_2.state_dict(&p(2)));
        sd.extend(self.conv3.state_dict(&p(3)));
        sd.extend(self.c3k2_4.state_dict(&p(4)));
        sd.extend(self.conv5.state_dict(&p(5)));
        sd.extend(self.c3k2_6.state_dict(&p(6)));
        sd.extend(self.conv7.state_dict(&p(7)));
        sd.extend(self.c3k2_8.state_dict(&p(8)));
        sd.extend(self.conv9.state_dict(&p(9)));
        sd.extend(self.c3k2_10.state_dict(&p(10)));
        sd.extend(self.sppf11.state_dict(&p(11)));
        sd.extend(self.c2psa12.state_dict(&p(12)));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        self.conv0.load_state_dict(sd, &p(0))?;
        self.conv1.load_state_dict(sd, &p(1))?;
        self.c3k2_2.load_state_dict(sd, &p(2))?;
        self.conv3.load_state_dict(sd, &p(3))?;
        self.c3k2_4.load_state_dict(sd, &p(4))?;
        self.conv5.load_state_dict(sd, &p(5))?;
        self.c3k2_6.load_state_dict(sd, &p(6))?;
        self.conv7.load_state_dict(sd, &p(7))?;
        self.c3k2_8.load_state_dict(sd, &p(8))?;
        self.conv9.load_state_dict(sd, &p(9))?;
        self.c3k2_10.load_state_dict(sd, &p(10))?;
        self.sppf11.load_state_dict(sd, &p(11))?;
        self.c2psa12.load_state_dict(sd, &p(12))?;
        Ok(())
    }
}
