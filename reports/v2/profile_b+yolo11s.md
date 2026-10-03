# Profile: b+yolo11s

- parameters (unique): **100,626,442**
- trainable: 100,626,426
- buffers: 397,163
- partial MACs (conv/linear/BN only): 149658742528
- na2d_av MACs: 1226035200
- complete FLOPs: NOT AVAILABLE (no complete FLOPs profiler covers the custom operators)

## buckets

| bucket | unique tensors | numel | trainable | buffers |
|---|---|---|---|---|
| backbone_overlock | 3085 | 96,091,496 | 96,091,496 | 381,817 |
| adapters | 9 | 546,816 | 546,816 | 2,051 |
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
