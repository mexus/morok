//! ResNet building blocks: weight-holding wrappers around the tensor
//! primitives that implement [`HasStateDict`](crate::state::HasStateDict) for
//! safetensors round-trips.
//!
//! All ResNet convs are bias-less; biases live only in the BN affine
//! parameters. BatchNorm is stored with the running variance already folded
//! into `invstd = 1 / sqrt(var + eps)` so the forward path is a pure
//! affine-and-shift — the fold happens once in
//! [`crate::resnet::remap::fold_batchnorm`].

use morok_dtype::DType;
use morok_tensor::Tensor;
use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::config::BlockKind;
use super::error::{Result, TensorSnafu};

fn zeros(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape, DType::Float32).expect("zeros for resnet placeholder must succeed")
}

fn ones(shape: &[usize]) -> Tensor {
    Tensor::ones(shape, DType::Float32).expect("ones for resnet placeholder must succeed")
}

// ---------------------------------------------------------------------------
// Conv2dWeights
// ---------------------------------------------------------------------------

/// Bias-less 2D convolution wrapper. `weight` has layout
/// `[out_ch, in_ch / groups, kH, kW]` (same as PyTorch / timm).
#[derive(Clone)]
pub(crate) struct Conv2dWeights {
    pub weight: Tensor,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}

impl Conv2dWeights {
    pub fn empty(out_ch: usize, in_ch: usize, kernel: usize, stride: usize, padding: usize) -> Self {
        Self { weight: zeros(&[out_ch, in_ch, kernel, kernel]), stride, padding, groups: 1 }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.padding as isize;
        x.conv2d()
            .weight(&self.weight)
            .groups(self.groups)
            .stride(&[self.stride, self.stride])
            .padding(&[(p, p), (p, p)])
            .call()
            .context(TensorSnafu)
    }
}

impl HasStateDict for Conv2dWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BatchNormWeights
// ---------------------------------------------------------------------------

/// BN with the running variance pre-folded into `invstd`. State-dict round-trip
/// uses the canonical timm/PyTorch keys `weight` (→ `scale`), `bias`,
/// `running_mean` (→ `mean`), `running_var` (→ `invstd` after fold).
#[derive(Clone)]
pub(crate) struct BatchNormWeights {
    pub scale: Tensor,
    pub bias: Tensor,
    pub mean: Tensor,
    pub invstd: Tensor,
}

impl BatchNormWeights {
    pub fn empty(channels: usize) -> Self {
        Self { scale: ones(&[channels]), bias: zeros(&[channels]), mean: zeros(&[channels]), invstd: ones(&[channels]) }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.batchnorm()
            .scale(&self.scale)
            .bias(&self.bias)
            .mean(&self.mean)
            .invstd(&self.invstd)
            .call()
            .context(TensorSnafu)
    }
}

impl HasStateDict for BatchNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.scale.clone());
        sd.insert(prefixed(prefix, "bias"), self.bias.clone());
        sd.insert(prefixed(prefix, "running_mean"), self.mean.clone());
        sd.insert(prefixed(prefix, "running_var"), self.invstd.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.scale = get_tensor(sd, &prefixed(prefix, "weight"))?;
        self.bias = get_tensor(sd, &prefixed(prefix, "bias"))?;
        self.mean = get_tensor(sd, &prefixed(prefix, "running_mean"))?;
        self.invstd = get_tensor(sd, &prefixed(prefix, "running_var"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BasicBlock — used by ResNet-18 / ResNet-34
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct BasicBlock {
    pub conv1: Conv2dWeights,
    pub bn1: BatchNormWeights,
    pub conv2: Conv2dWeights,
    pub bn2: BatchNormWeights,
    pub downsample: Option<(Conv2dWeights, BatchNormWeights)>,
}

impl BasicBlock {
    pub fn empty(in_planes: usize, planes: usize, stride: usize) -> Self {
        let expansion = BlockKind::Basic.expansion();
        let downsample = if stride != 1 || in_planes != planes * expansion {
            Some((
                Conv2dWeights::empty(planes * expansion, in_planes, 1, stride, 0),
                BatchNormWeights::empty(planes * expansion),
            ))
        } else {
            None
        };
        Self {
            conv1: Conv2dWeights::empty(planes, in_planes, 3, stride, 1),
            bn1: BatchNormWeights::empty(planes),
            conv2: Conv2dWeights::empty(planes, planes, 3, 1, 1),
            bn2: BatchNormWeights::empty(planes),
            downsample,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(x)?)?;
        let out = out.relu().context(TensorSnafu)?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?;
        let shortcut = match &self.downsample {
            Some((c, b)) => b.forward(&c.forward(x)?)?,
            None => x.clone(),
        };
        out.try_add(&shortcut).context(TensorSnafu)?.relu().context(TensorSnafu)
    }
}

impl HasStateDict for BasicBlock {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.conv1.state_dict(&prefixed(prefix, "conv1"));
        sd.extend(self.bn1.state_dict(&prefixed(prefix, "bn1")));
        sd.extend(self.conv2.state_dict(&prefixed(prefix, "conv2")));
        sd.extend(self.bn2.state_dict(&prefixed(prefix, "bn2")));
        if let Some((c, b)) = &self.downsample {
            sd.extend(c.state_dict(&prefixed(prefix, "downsample.0")));
            sd.extend(b.state_dict(&prefixed(prefix, "downsample.1")));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.conv1.load_state_dict(sd, &prefixed(prefix, "conv1"))?;
        self.bn1.load_state_dict(sd, &prefixed(prefix, "bn1"))?;
        self.conv2.load_state_dict(sd, &prefixed(prefix, "conv2"))?;
        self.bn2.load_state_dict(sd, &prefixed(prefix, "bn2"))?;
        if let Some((c, b)) = &mut self.downsample {
            c.load_state_dict(sd, &prefixed(prefix, "downsample.0"))?;
            b.load_state_dict(sd, &prefixed(prefix, "downsample.1"))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bottleneck — used by ResNet-50 / 101 / 152 (v1.5 stride placement)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct Bottleneck {
    pub conv1: Conv2dWeights,
    pub bn1: BatchNormWeights,
    pub conv2: Conv2dWeights,
    pub bn2: BatchNormWeights,
    pub conv3: Conv2dWeights,
    pub bn3: BatchNormWeights,
    pub downsample: Option<(Conv2dWeights, BatchNormWeights)>,
}

impl Bottleneck {
    pub fn empty(in_planes: usize, planes: usize, stride: usize) -> Self {
        let expansion = BlockKind::Bottleneck.expansion();
        let out_ch = planes * expansion;
        let downsample = if stride != 1 || in_planes != out_ch {
            Some((Conv2dWeights::empty(out_ch, in_planes, 1, stride, 0), BatchNormWeights::empty(out_ch)))
        } else {
            None
        };
        Self {
            conv1: Conv2dWeights::empty(planes, in_planes, 1, 1, 0),
            bn1: BatchNormWeights::empty(planes),
            conv2: Conv2dWeights::empty(planes, planes, 3, stride, 1),
            bn2: BatchNormWeights::empty(planes),
            conv3: Conv2dWeights::empty(out_ch, planes, 1, 1, 0),
            bn3: BatchNormWeights::empty(out_ch),
            downsample,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(x)?)?.relu().context(TensorSnafu)?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?.relu().context(TensorSnafu)?;
        let out = self.bn3.forward(&self.conv3.forward(&out)?)?;
        let shortcut = match &self.downsample {
            Some((c, b)) => b.forward(&c.forward(x)?)?,
            None => x.clone(),
        };
        out.try_add(&shortcut).context(TensorSnafu)?.relu().context(TensorSnafu)
    }
}

impl HasStateDict for Bottleneck {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.conv1.state_dict(&prefixed(prefix, "conv1"));
        sd.extend(self.bn1.state_dict(&prefixed(prefix, "bn1")));
        sd.extend(self.conv2.state_dict(&prefixed(prefix, "conv2")));
        sd.extend(self.bn2.state_dict(&prefixed(prefix, "bn2")));
        sd.extend(self.conv3.state_dict(&prefixed(prefix, "conv3")));
        sd.extend(self.bn3.state_dict(&prefixed(prefix, "bn3")));
        if let Some((c, b)) = &self.downsample {
            sd.extend(c.state_dict(&prefixed(prefix, "downsample.0")));
            sd.extend(b.state_dict(&prefixed(prefix, "downsample.1")));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.conv1.load_state_dict(sd, &prefixed(prefix, "conv1"))?;
        self.bn1.load_state_dict(sd, &prefixed(prefix, "bn1"))?;
        self.conv2.load_state_dict(sd, &prefixed(prefix, "conv2"))?;
        self.bn2.load_state_dict(sd, &prefixed(prefix, "bn2"))?;
        self.conv3.load_state_dict(sd, &prefixed(prefix, "conv3"))?;
        self.bn3.load_state_dict(sd, &prefixed(prefix, "bn3"))?;
        if let Some((c, b)) = &mut self.downsample {
            c.load_state_dict(sd, &prefixed(prefix, "downsample.0"))?;
            b.load_state_dict(sd, &prefixed(prefix, "downsample.1"))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Block (enum dispatch) + ResidualStage
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum Block {
    Basic(BasicBlock),
    Bottleneck(Bottleneck),
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Block::Basic(b) => b.forward(x),
            Block::Bottleneck(b) => b.forward(x),
        }
    }
}

impl HasStateDict for Block {
    fn state_dict(&self, prefix: &str) -> StateDict {
        match self {
            Block::Basic(b) => b.state_dict(prefix),
            Block::Bottleneck(b) => b.state_dict(prefix),
        }
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        match self {
            Block::Basic(b) => b.load_state_dict(sd, prefix),
            Block::Bottleneck(b) => b.load_state_dict(sd, prefix),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ResidualStage {
    pub blocks: Vec<Block>,
}

impl ResidualStage {
    /// Construct a fresh stage. The first block may downsample (`stride`);
    /// remaining blocks always have stride 1. Channel width follows the
    /// canonical schedule: every block in the stage emits `planes * expansion`
    /// channels, and the next block sees that as its `in_planes`.
    pub fn empty(kind: BlockKind, in_planes: usize, planes: usize, num_blocks: usize, stride: usize) -> Self {
        let expansion = kind.expansion();
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut current_in = in_planes;
        for i in 0..num_blocks {
            let s = if i == 0 { stride } else { 1 };
            let block = match kind {
                BlockKind::Basic => Block::Basic(BasicBlock::empty(current_in, planes, s)),
                BlockKind::Bottleneck => Block::Bottleneck(Bottleneck::empty(current_in, planes, s)),
            };
            blocks.push(block);
            current_in = planes * expansion;
        }
        Self { blocks }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        Ok(x)
    }
}

impl HasStateDict for ResidualStage {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (i, b) in self.blocks.iter().enumerate() {
            sd.extend(b.state_dict(&prefixed(prefix, &i.to_string())));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        for (i, b) in self.blocks.iter_mut().enumerate() {
            b.load_state_dict(sd, &prefixed(prefix, &i.to_string()))?;
        }
        Ok(())
    }
}
