use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::basic_block::{BasicBlock, BlockKind};
use super::bottleneck::Bottleneck;
use super::error::Result;

#[derive(Clone)]
pub enum Block {
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
pub struct ResidualStage {
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
