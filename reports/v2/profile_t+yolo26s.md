# Profile: t+yolo26s

- parameters (unique): **38,098,420**
- trainable: 38,098,420
- buffers: 182,296
- partial MACs (conv/linear/BN only): 53626185728
- na2d_av MACs: 491929600
- complete FLOPs: NOT AVAILABLE (no complete FLOPs profiler covers the custom operators)

## buckets

| bucket | unique tensors | numel | trainable | buffers |
|---|---|---|---|---|
| backbone_overlock | 1761 | 33,127,096 | 33,127,096 | 158,917 |
| adapters | 9 | 460,800 | 460,800 | 2,051 |
| neck_head_native | 246 | 4,510,524 | 4,510,524 | 21,322 |

## uncounted operators

- natten na2d_av (custom op)
- einops.einsum dynamic-kernel weight generation
- torch.matmul/softmax inside DynamicConvBlock
- F.interpolate / adaptive_avg_pool2d
- GRN (norm/mean reductions)
- LayerScale (groupwise F.conv2d)
- DFL softmax projection

Parameter counts do not change between 224 and 640; only spatial cost does.
