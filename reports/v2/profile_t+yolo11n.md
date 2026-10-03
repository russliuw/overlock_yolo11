# Profile: t+yolo11n

- parameters (unique): **34,583,034**
- trainable: 34,583,018
- buffers: 166,999
- partial MACs (conv/linear/BN only): 49842812928
- na2d_av MACs: 491929600
- complete FLOPs: NOT AVAILABLE (no complete FLOPs profiler covers the custom operators)

## buckets

| bucket | unique tensors | numel | trainable | buffers |
|---|---|---|---|---|
| backbone_overlock | 1761 | 33,127,096 | 33,127,096 | 158,917 |
| adapters | 9 | 230,400 | 230,400 | 1,027 |
| neck_head_native | 136 | 1,225,538 | 1,225,522 | 7,049 |

## uncounted operators

- natten na2d_av (custom op)
- einops.einsum dynamic-kernel weight generation
- torch.matmul/softmax inside DynamicConvBlock
- F.interpolate / adaptive_avg_pool2d
- GRN (norm/mean reductions)
- LayerScale (groupwise F.conv2d)
- DFL softmax projection

Parameter counts do not change between 224 and 640; only spatial cost does.
