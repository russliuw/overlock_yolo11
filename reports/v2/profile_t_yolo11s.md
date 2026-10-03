# Profile: t+yolo11s

- parameters (unique): **37,576,026**
- trainable: 37,576,010
- buffers: 174,263
- partial MACs (conv/linear/BN only): 53145699328
- na2d_av MACs: 491929600
- complete FLOPs: NOT AVAILABLE (no complete FLOPs profiler covers the custom operators)

## buckets

| bucket | unique tensors | numel | trainable | buffers |
|---|---|---|---|---|
| backbone_overlock | 1761 | 33,127,096 | 33,127,096 | 158,917 |
| adapters | 9 | 460,800 | 460,800 | 2,051 |
| neck_head_native | 136 | 3,988,130 | 3,988,114 | 13,289 |

## uncounted operators

- natten na2d_av (custom op)
- einops.einsum dynamic-kernel weight generation
- torch.matmul/softmax inside DynamicConvBlock
- F.interpolate / adaptive_avg_pool2d
- GRN (norm/mean reductions)
- LayerScale (groupwise F.conv2d)
- DFL softmax projection

Parameter counts do not change between 224 and 640; only spatial cost does.
