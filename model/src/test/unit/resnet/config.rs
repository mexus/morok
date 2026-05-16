use crate::resnet::{BlockKind, ResNetDepth};
use test_case::test_case;

#[test_case(ResNetDepth::R18, [2, 2, 2, 2], BlockKind::Basic, 1; "r18")]
#[test_case(ResNetDepth::R34, [3, 4, 6, 3], BlockKind::Basic, 1; "r34")]
#[test_case(ResNetDepth::R50, [3, 4, 6, 3], BlockKind::Bottleneck, 4; "r50")]
#[test_case(ResNetDepth::R101, [3, 4, 23, 3], BlockKind::Bottleneck, 4; "r101")]
#[test_case(ResNetDepth::R152, [3, 8, 36, 3], BlockKind::Bottleneck, 4; "r152")]
fn depth_schedule(depth: ResNetDepth, layers: [usize; 4], block: BlockKind, expansion: usize) {
    assert_eq!(depth.layers(), layers);
    assert_eq!(depth.block(), block);
    assert_eq!(depth.expansion(), expansion);
}
