"""ISOLATED COPY of the official OverLoCK *detection* backbone (V1 provenance header).

Origin
------
``/Users/lw/Documents/CNN-Mamba/OverLoCK-main/detection/models/overlock.py``
SHA256 (source read for this copy): see ``reports/source_manifest.json``.
Upstream: OverLoCK, https://arxiv.org/abs/2502.20087 (LMMMEng/OverLoCK), license of the
original repository applies.  Author credits: original OverLoCK authors; the
``apply_rpb`` routine is itself borrowed upstream from
https://tinyurl.com/mrbub4t3 ; ``DilatedReparamBlock`` follows UniRepLKNet
(https://github.com/AILab-CVC/UniRepLKNet).

Why a copy exists
-----------------
The official detection file imports ``mmdet`` (registry), ``mmcv/mmengine`` (checkpoint
IO + logger) and ``natten.functional.na2d_av``; none of ``natten``/``mmcv``/``mmengine``/
``mmdet`` exists in the target environment.  Per DESIGN.md 5.1 this single source file is
vendored as an isolated, provenance-annotated adapter.  Nothing in
``OverLoCK-main`` is modified.

Exact modification list (nothing else changed)
----------------------------------------------
1. Module docstring replaced by this provenance header.
2. Imports: dropped ``torch.distributed`` (only ``is_available/is_initialized`` are now
   touched), ``from natten.functional import na2d_av``, ``from timm.models.registry import
   register_model``, ``from mmdet.models.builder import MODELS``,
   ``from mmdet.utils import get_root_logger``, the ``mmcv/mmengine load_checkpoint``
   try-block, and ``from torch.utils.checkpoint import checkpoint`` (replaced by a local
   ``_checkpoint`` helper bound to ``torch.utils.checkpoint.checkpoint`` as
   ``use_reentrant`` is version dependent).  ``timm`` is kept for ``DropPath``; ``to_2tuple``
   was inlined because ``nn.Conv2d`` accepts int kernel sizes directly.
3. ``get_conv2d``: the iGEMM import attempt is kept but it is only reached when
   ``attempt_use_lk_impl=True``.  V1 always passes ``use_gemm=False`` (DESIGN.md 4.1), so
   the real ``nn.Conv2d`` large-kernel path runs and no ``depthwise_conv2d_implicit_gemm``
   import is attempted unless explicitly requested.
4. NATTEN: ``na2d_av(attn, value, kernel_size=K)`` calls are routed through the explicit
   backend interface (``self._na2d_av``), which is either the real NATTEN kernel or the
   exact differentiable CPU reference.  No substitution by pooling/conv/identity.
5. ``OverLoCK.__init__``: ``use_ds`` / ``projection`` / ``num_classes`` are accepted and
   recorded but the classification ``head``/``aux_head`` are *never constructed*.  The
   official detection code builds them and then does ``del self.head`` unconditionally,
   which raises when ``use_ds=False``.  Removing the dead classification-head
   construction changes no remaining parameter name, no shape and no forward
   computation; it only removes parameters that were created and immediately deleted
   (DESIGN.md 4.1).
6. Registry decorators and the ``pretrained=... -> download URL`` bodies are removed;
   model construction is an explicit factory (:func:`build_overlock_b`) and weights are
   loaded explicitly and audited by :mod:`overlock_yolo.checkpoint`.  ``_convert_sync_batchnorm``
   is kept as a no-op without a process group.
7. Added ``OverLoCK.forward_multiscale`` which is exactly ``forward_features`` with an
   assertion on the detection output signature; ``forward_features``/``forward`` are
   byte-for-byte the same computation as upstream.

Everything else -- depth, kernels, overview/focus branches, ``high_level_proj``,
``patch_embedx``, ``h_proj``, relative position bias, the two softmaxes, the
interpolate-and-restore of the dynamic blocks for small feature maps, ``extra_norm``
placement -- is the upstream code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn

try:  # timm is a hard dependency of the official implementation
    from timm.layers import DropPath
except Exception:  # pragma: no cover - timm layout differences
    try:
        from timm.models.layers import DropPath
    except Exception:  # pragma: no cover
        DropPath = None

from .attention_backend import BackendCache, backend_cache_report, na2d_av

try:  # pragma: no cover - torch version dependent signature
    from torch.utils.checkpoint import checkpoint as _torch_checkpoint
except Exception:  # pragma: no cover
    _torch_checkpoint = None

__all__ = [
    "LayerNorm2d",
    "OverLoCK",
    "build_overlock",
    "build_overlock_b",
    "OVERLOCK_B_DETECTION_CONFIG",
    "OVERLOCK_SOURCE",
    "VARIANT_FACTORIES",
    "stem",
    "downsample",
]

#: Provenance of the vendored implementation (duplicated in reports/v2/compatibility.json).
OVERLOCK_SOURCE = {
    "upstream_file": "OverLoCK-main/detection/models/overlock.py",
    "upstream_repo": "https://github.com/LMMMEng/OverLoCK",
    "paper": "https://arxiv.org/abs/2502.20087",
    "variants": "xt/t/s/b (detection / mmdet factory bodies, lines 861-946)",
    "vendored_as": "overlock_yolo/backbone.py (isolated copy; nothing in OverLoCK-main is modified)",
}


def _checkpoint(fn, *args, **kwargs):
    """``torch.utils.checkpoint.checkpoint`` with ``use_reentrant=False`` when supported."""
    if _torch_checkpoint is None:  # pragma: no cover
        return fn(*args, **kwargs)
    try:
        return _torch_checkpoint(fn, *args, use_reentrant=False, **kwargs)
    except TypeError:  # pragma: no cover - very old torch
        return _torch_checkpoint(fn, *args, **kwargs)


def get_conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
    attempt_use_lk_impl=True,
):
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    need_large_impl = (
        kernel_size[0] == kernel_size[1]
        and kernel_size[0] > 5
        and padding == (kernel_size[0] // 2, kernel_size[1] // 2)
    )

    if attempt_use_lk_impl and need_large_impl:
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        except Exception:
            DepthWiseConv2dImplicitGEMM = None
        if (
            DepthWiseConv2dImplicitGEMM is not None
            and need_large_impl
            and in_channels == out_channels
            and out_channels == groups
            and stride == 1
            and dilation == 1
        ):
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        return F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    slices = []
    for i in range(kernel.size(1)):
        slices.append(F.conv_transpose2d(kernel[:, i : i + 1, :, :], identity_kernel, stride=dilate_rate))
    return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
    )


def downsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
    )


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        return x * self.proj(x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x.contiguous()


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer, as in ConvNeXt V2 / upstream OverLoCK."""

    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        return (self.gamma * Nx + 1) * x


class DilatedReparamBlock(nn.Module):
    """Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)."""

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=channels,
            bias=deploy,
            attempt_use_lk_impl=attempt_use_lk_impl,
        )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError("Dilated Reparam Block requires kernel_size >= 5")

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(
                    "dil_conv_k{}_{}".format(k, r),
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=k,
                        stride=1,
                        padding=(r * (k - 1) + 1) // 2,
                        dilation=r,
                        groups=channels,
                        bias=False,
                    ),
                )
                self.__setattr__("dil_bn_k{}_{}".format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, "origin_bn"):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__("dil_conv_k{}_{}".format(k, r))
            bn = self.__getattr__("dil_bn_k{}_{}".format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, "origin_bn"):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__("dil_conv_k{}_{}".format(k, r))
                bn = self.__getattr__("dil_bn_k{}_{}".format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(
                origin_k.size(0),
                origin_k.size(0),
                origin_k.size(2),
                stride=1,
                padding=origin_k.size(2) // 2,
                dilation=1,
                groups=origin_k.size(0),
                bias=True,
                attempt_use_lk_impl=self.attempt_use_lk_impl,
            )
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__("origin_bn")
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__("dil_conv_k{}_{}".format(k, r))
                self.__delattr__("dil_bn_k{}_{}".format(k, r))


class CTXDownsample(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
        )
        self.h_proj = nn.Sequential(
            nn.Conv2d(h_dim // 4, h_dim // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim // 4),
        )

    def forward(self, x, ctx):
        x = self.x_proj(x)
        ctx = self.h_proj(ctx)
        return (x, ctx)


class ResDWConv(nn.Conv2d):
    """Depthwise convolution with residual connection."""

    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + super().forward(x)


class RepConvBlock(nn.Module):
    def __init__(
        self,
        dim=64,
        kernel_size=7,
        mlp_ratio=4,
        ls_init_value=None,
        res_scale=False,
        drop_path=0,
        norm_layer=LayerNorm2d,
        use_gemm=False,
        deploy=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint
        mlp_dim = int(dim * mlp_ratio)
        self.dwconv = ResDWConv(dim, kernel_size=3)
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)
        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(
                dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm
            ),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            nn.Identity(),
        )
        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward_features(self, x):
        x = self.dwconv(x)
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            x = x + self.drop_path(self.ls(self.proj[:-1](x)))
        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            x = _checkpoint(self.forward_features, x)
        else:
            x = self.forward_features(x)
        return x


class DynamicConvBlock(nn.Module):
    """Context-Mixing Dynamic Kernel block with the overview/focus (SMK) branch."""

    def __init__(
        self,
        dim=64,
        ctx_dim=32,
        kernel_size=7,
        smk_size=5,
        num_heads=2,
        mlp_ratio=4,
        ls_init_value=None,
        res_scale=False,
        drop_path=0,
        norm_layer=LayerNorm2d,
        is_first=False,
        is_last=False,
        use_gemm=False,
        deploy=False,
        use_checkpoint=False,
        na2d_av_fn=None,
        **kwargs,
    ):
        super().__init__()

        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim * mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint
        # backend hook: real NATTEN kernel or the exact differentiable reference op
        self._na2d_av = na2d_av_fn if na2d_av_fn is not None else na2d_av

        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1)
            self.h_scale = LayerScale(ctx_dim, init_value=1)

        self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
        self.norm1 = norm_layer(out_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim),
        )

        self.weight_query = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1)

        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.lepe = nn.Sequential(
            DilatedReparamBlock(
                dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm
            ),
            nn.BatchNorm2d(dim),
        )

        self.se_layer = SEModule(dim)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_dim, kernel_size=1),
        )

        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, out_dim, kernel_size=1),
        )

        self.ls1 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)

        self.get_rpb()

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        """RPB implementation directly borrowed from https://tinyurl.com/mrbub4t3 (unchanged)."""
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size // 2] = height - (kernel_size - 1)
        num_repeat_w[kernel_size // 2] = width - (kernel_size - 1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2 * kernel_size - 1)) + idx_w.repeat_interleave(
            num_repeat_w
        )
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size**2))
        return attn + rpb

    def _forward_inner(self, x, h_x, h_r):
        input_resoltion = x.shape[2:]
        B, C, H, W = x.shape
        B, C_h, H_h, W_h = h_x.shape

        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_f = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f
        x_f = self.norm1(x_f)
        x = self.fusion(x_f)
        gate = self.gate(x)
        lepe = self.lepe(x)

        is_pad = False
        if min(H, W) < self.kernel_size:
            # official internal interpolation for feature maps smaller than the kernel,
            # restored to the input resolution afterwards. Kept as upstream.
            is_pad = True
            if H < W:
                size = (self.kernel_size, int(self.kernel_size / H * W))
            else:
                size = (int(self.kernel_size / W * H), self.kernel_size)

            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            x_f = F.interpolate(x_f, size=size, mode="bilinear", align_corners=False)
            H, W = size

        query, key = torch.split(x_f, split_size_or_sections=[C, C_h], dim=1)
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)
        query = rearrange(query, "b (g c) h w -> b g c (h w)", g=self.num_heads)
        key = rearrange(key, "b (g c) h w -> b g c (h w)", g=self.num_heads)
        weight = einsum(query, key, "b g c n, b g c l -> b g n l")
        weight = rearrange(weight, "b g n l -> b l g n").contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, "b l g (h w) -> b g h w l", h=H, w=W)

        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size**2, self.kernel_size**2], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(x, "b (m g c) h w -> m b g h w c", m=2, g=self.num_heads)

        x1 = self._na2d_av(attn1, value[0], kernel_size=self.smk_size)
        x2 = self._na2d_av(attn2, value[1], kernel_size=self.kernel_size)

        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, "b g h w c -> b (g c) h w", h=H, w=W)

        if is_pad:
            x = F.adaptive_avg_pool2d(x, input_resoltion)

        x = self.dyconv_proj(x)

        x = x + lepe
        x = self.se_layer(x)

        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))

        x = self.dwconv2(x)

        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if self.is_last:
            return (x, None)
        l_x, h_x = torch.split(x, split_size_or_sections=[C, C_h], dim=1)
        return (l_x, h_x)

    def forward(self, x, h_x, h_r):
        if self.use_checkpoint and x.requires_grad:
            x = _checkpoint(self._forward_inner, x, h_x, h_r)
        else:
            x = self._forward_inner(x, h_x, h_r)
        return x


class OverLoCK(nn.Module):
    """An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels.

    https://arxiv.org/abs/2502.20087 -- detection variant, returning four feature maps.
    """

    def __init__(
        self,
        depth=(2, 2, 2, 2),
        sub_depth=(4, 2),
        in_chans=3,
        embed_dim=(96, 192, 384, 768),
        kernel_size=(7, 7, 7, 7),
        mlp_ratio=(4, 4, 4, 4),
        sub_mlp_ratio=(4, 4),
        sub_num_heads=(4, 8),
        ls_init_value=(None, None, 1, 1),
        res_scale=True,
        smk_size=5,
        deploy=False,
        use_gemm=True,
        use_ds=True,
        drop_rate=0,
        drop_path_rate=0,
        norm_layer=LayerNorm2d,
        projection=1024,
        num_classes=1000,
        use_checkpoint=(0, 0, 0, 0),
        attention_backend: str = "auto",
        device=None,
        row_chunk: int = 0,
    ):
        super().__init__()

        fusion_dim = embed_dim[-1] + embed_dim[-1] // 4
        self.num_features = self.embed_dim = list(embed_dim)
        #: set by :func:`build_overlock`; kept as an attribute so the detection signature can be
        #: asserted per variant (P3/P4/P5 channels differ across xt/t/s/b).
        self.variant = "unknown"
        self.overlock_config = {}

        # --- lazy, per-device attention backend selection (DESIGN_V2.md 7.2) ----------
        # Resolution happens on the device of the tensor actually entering na2d_av, so a model
        # constructed on CPU and later moved with .to("cuda") re-resolves instead of reusing a
        # stale CPU answer, and an explicit 'natten' request on CPU fails loudly.
        self.attention_backend_requested = attention_backend
        self._backend_cache = BackendCache(attention_backend)
        self._row_chunk = row_chunk
        self._backend_reports = {}

        def _na2d(attn, value, kernel_size):
            resolved = self._backend_cache.resolve(value.device)
            self._backend_reports[str(value.device)] = resolved.as_dict()
            return na2d_av(attn, value, kernel_size, resolved=resolved, row_chunk=self._row_chunk)

        # --- untouched upstream computation -----------------------------------------
        self.patch_embed1 = stem(in_chans, embed_dim[0])
        self.patch_embed2 = downsample(embed_dim[0], embed_dim[1])
        self.patch_embed3 = downsample(embed_dim[1], embed_dim[2])
        self.patch_embed4 = downsample(embed_dim[2], embed_dim[3])
        self.high_level_proj = nn.Conv2d(embed_dim[-1], embed_dim[-1] // 4, kernel_size=1)
        self.patch_embedx = CTXDownsample(embed_dim[2], embed_dim[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth) + sum(sub_depth))]

        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.blocks3 = nn.ModuleList()
        self.blocks4 = nn.ModuleList()
        self.sub_blocks3 = nn.ModuleList()
        self.sub_blocks4 = nn.ModuleList()

        for i in range(depth[0]):
            self.blocks1.append(
                RepConvBlock(
                    dim=embed_dim[0],
                    kernel_size=kernel_size[0],
                    mlp_ratio=mlp_ratio[0],
                    ls_init_value=ls_init_value[0],
                    res_scale=res_scale,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[0]),
                )
            )

        for i in range(depth[1]):
            self.blocks2.append(
                RepConvBlock(
                    dim=embed_dim[1],
                    kernel_size=kernel_size[1],
                    mlp_ratio=mlp_ratio[1],
                    ls_init_value=ls_init_value[1],
                    res_scale=res_scale,
                    drop_path=dpr[i + depth[0]],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[1]),
                )
            )

        for i in range(depth[2]):
            self.blocks3.append(
                RepConvBlock(
                    dim=embed_dim[2],
                    kernel_size=kernel_size[2],
                    mlp_ratio=mlp_ratio[2],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depth[:2])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[2]),
                )
            )

        for i in range(depth[3]):
            self.blocks4.append(
                RepConvBlock(
                    dim=embed_dim[3],
                    kernel_size=kernel_size[3],
                    mlp_ratio=mlp_ratio[3],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depth[:3])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[3]),
                )
            )

        for i in range(sub_depth[0]):
            self.sub_blocks3.append(
                DynamicConvBlock(
                    dim=embed_dim[2],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[2],
                    num_heads=sub_num_heads[0],
                    pool_size=7,
                    mlp_ratio=sub_mlp_ratio[0],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depth)],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    is_first=(i == 0),
                    use_checkpoint=(i < use_checkpoint[2]),
                    na2d_av_fn=_na2d,
                )
            )

        for i in range(sub_depth[1]):
            self.sub_blocks4.append(
                DynamicConvBlock(
                    dim=embed_dim[3],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[-1],
                    num_heads=sub_num_heads[1],
                    pool_size=7,
                    mlp_ratio=sub_mlp_ratio[1],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depth) + sub_depth[0]],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    is_first=False,
                    is_last=(i == sub_depth[1] - 1),
                    use_checkpoint=(i < use_checkpoint[3]),
                    na2d_av_fn=_na2d,
                )
            )

        self.h_proj = nn.Sequential(
            nn.Conv2d(embed_dim[-1], fusion_dim, kernel_size=1),
            LayerScale(fusion_dim, init_value=1e-5),
        )

        # --- DIFFERENCE #5: upstream builds and then unconditionally deletes the
        # classification head/aux_head (which breaks for use_ds=False). V1 accepts the
        # same kwargs for signature compatibility but never constructs them.
        self.use_ds = use_ds
        self.projection = projection
        self.num_classes = num_classes
        self.detection_head_constructed = False

        self.extra_norm = nn.ModuleList()
        for idx in range(4):
            dim = embed_dim[idx]
            if idx >= 2:
                dim = dim + embed_dim[-1] // 4
            self.extra_norm.append(norm_layer(dim))
        self.extra_norm.append(norm_layer(embed_dim[-1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _convert_sync_batchnorm(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    # ----------------------------------------------------------------------------- forward
    def forward_pre_features(self, x):
        outs = []
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
        outs.append(self.extra_norm[0](x))

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        outs.append(self.extra_norm[1](x))
        return outs

    def forward_base_features(self, x):
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)

        ctx = self.patch_embed4(x)
        for blk in self.blocks4:
            ctx = blk(ctx)
        return (x, ctx)

    def forward_sub_features(self, x, ctx):
        outs = []
        ctx_cls = ctx
        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, size=x.shape[2:], mode="bilinear", align_corners=False)

        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)

        outs.append(self.extra_norm[2](torch.cat([x, ctx], dim=1)))

        x, ctx = self.patch_embedx(x, ctx)
        for idx, blk in enumerate(self.sub_blocks4):
            x, ctx = blk(x, ctx, ctx_ori)

        ctx = self.extra_norm[-1](ctx_cls)
        x = self.extra_norm[3](x) + self.h_proj(ctx)

        outs.append(x)
        return outs

    def forward_features(self, x):
        x0, x1 = self.forward_pre_features(x)
        x, ctx = self.forward_base_features(x1)
        x2, x3 = self.forward_sub_features(x, ctx)
        return (x0, x1, x2, x3)

    def forward(self, x):
        return self.forward_features(x)

    def forward_multiscale(self, x, strict: bool = True):
        """``forward_features`` plus an assertion on this variant's detection output signature."""
        from .variants import FEATURE_INFO, P3P4P5_CHANNELS

        assert x.dim() == 4 and x.shape[1] == 3, f"backbone expects BCHW RGB input, got {tuple(x.shape)}"
        h, w = x.shape[-2:]
        assert h % 32 == 0 and w % 32 == 0, (
            f"the detection backbone requires input padded to a multiple of 32 (letterbox), got {h}x{w}"
        )
        outs = self.forward_features(x)
        if strict:
            signature = FEATURE_INFO[self.variant]
            for i, (feat, (stride, ch)) in enumerate(zip(outs, signature)):
                exp_hw = (h // stride, w // stride)
                if tuple(feat.shape[1:]) != (ch, *exp_hw):
                    raise AssertionError(
                        f"{self.variant} detection output x{i}: expected (C,H,W)=({ch},{exp_hw[0]},{exp_hw[1]}), "
                        f"got {tuple(feat.shape[1:])}; contract P3/P4/P5={P3P4P5_CHANNELS[self.variant]}"
                    )
        return outs

    # ------------------------------------------------------------------ backend reporting
    def attention_backend_report(self) -> dict:
        """Per-device resolution report (also lists devices never used yet)."""
        cache = backend_cache_report([self._backend_cache])
        cache["resolutions"] = dict(self._backend_reports)
        cache["row_chunk"] = self._row_chunk
        if not self._backend_reports:
            cache["reason"] = "no na2d_av call has run yet; the backend is resolved lazily at forward time"
        return cache

    def na2d_av_blocks(self) -> list:
        """The dynamic (context-mixing) blocks that call ``na2d_av`` -- one per sub-block."""
        return list(self.sub_blocks3) + list(self.sub_blocks4)


#: Official detection-variant (``overlock_b``) construction arguments -- DESIGN_V2.md 4.2.
#: Kept as the explicit, backwards-compatible wrapper target for :func:`build_overlock_b`.
OVERLOCK_B_DETECTION_CONFIG = dict(
    depth=[8, 8, 10, 4],
    sub_depth=[20, 4],
    embed_dim=[80, 160, 384, 576],
    kernel_size=[17, 15, 13, 7],
    mlp_ratio=[4, 4, 4, 4],
    sub_mlp_ratio=[3, 3],
    sub_num_heads=[6, 9],
    smk_size=5,
    deploy=False,
    use_gemm=False,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_checkpoint=[0, 0, 0, 0],
)

#: Architecture-defining kwargs that a caller may never override: they come from the official
#: per-variant factory body only, so no "mostly Base" configuration can be smuggled in.
_PROTECTED_VARIANT_KEYS = frozenset(
    {
        "depth",
        "sub_depth",
        "embed_dim",
        "kernel_size",
        "mlp_ratio",
        "sub_mlp_ratio",
        "sub_num_heads",
        "smk_size",
        "ls_init_value",
        "res_scale",
        "num_classes",
        "projection",
        "use_ds",
    }
)

#: Variant id -> official factory body (transcribed, see overlock_yolo/variants.py).
VARIANT_FACTORIES = {
    "xt": dict(depth=[2, 2, 3, 2], sub_depth=[6, 2], embed_dim=[56, 112, 256, 336], sub_num_heads=[4, 6]),
    "t": dict(depth=[4, 4, 6, 2], sub_depth=[12, 2], embed_dim=[64, 128, 256, 512], sub_num_heads=[4, 8]),
    "s": dict(depth=[6, 6, 8, 3], sub_depth=[16, 3], embed_dim=[64, 128, 320, 512], sub_num_heads=[8, 16]),
    "b": dict(depth=[8, 8, 10, 4], sub_depth=[20, 4], embed_dim=[80, 160, 384, 576], sub_num_heads=[6, 9]),
}


def build_overlock(
    variant: str = "t",
    attention_backend: str = "auto",
    device=None,
    row_chunk: int = 0,
    deploy: bool = False,
    use_gemm: bool = False,
    **kwargs,
) -> OverLoCK:
    """Explicit factory for the official OverLoCK *detection* backbone of any variant.

    Replaces the upstream ``@MODELS.register_module() def overlock_<v>(pretrained=...)`` entry
    points, whose ``pretrained`` argument is rewritten into a GitHub download URL.  Weights are
    loaded separately and auditably by :mod:`overlock_yolo.checkpoint`.

    ``depth``/``sub_depth``/``embed_dim``/``sub_num_heads`` (and the other architecture keys)
    are fixed by the official factory body for the requested variant and cannot be overridden;
    ``deploy=True`` (reparameterised checkpoints) is refused so a fused structure can never be
    mistaken for the training structure.
    """
    from .variants import VARIANTS_ORDER, variant_config

    if variant not in VARIANTS_ORDER:
        raise KeyError(f"unknown OverLoCK variant {variant!r}; expected one of {VARIANTS_ORDER}")
    if deploy:
        raise ValueError(
            "deploy=True builds the reparameterised (merged large-kernel) structure and is not "
            "supported: it cannot represent the training-time detectors and must not be used to "
            "load a training checkpoint. Use deploy=False."
        )
    overlap = _PROTECTED_VARIANT_KEYS & set(kwargs)
    if overlap:
        raise TypeError(
            f"build_overlock: the architecture keys {sorted(overlap)} are fixed by the official "
            f"'{variant}' factory body and may not be overridden"
        )

    cfg = variant_config(variant)
    cfg.update(kernel_size=[17, 15, 13, 7], mlp_ratio=[4, 4, 4, 4], sub_mlp_ratio=[3, 3], smk_size=5)
    cfg.update(
        deploy=False,
        use_gemm=use_gemm,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=[0, 0, 0, 0],
        in_chans=3,
    )
    cfg.update(kwargs)
    model = OverLoCK(attention_backend=attention_backend, device=device, row_chunk=row_chunk, **cfg)
    model.variant = variant
    model.overlock_config = {k: (list(v) if isinstance(v, list) else v) for k, v in cfg.items()}
    model._convert_sync_batchnorm()
    return model


def build_overlock_b(attention_backend: str = "auto", device=None, row_chunk: int = 0, **kwargs) -> OverLoCK:
    """Backwards-compatible wrapper for the OverLoCK-Base detection backbone."""
    overlap = set(OVERLOCK_B_DETECTION_CONFIG) & set(kwargs) - {"use_gemm"}
    if overlap:
        raise TypeError(f"build_overlock_b: overriding official config keys is not allowed: {sorted(overlap)}")
    return build_overlock(
        "b",
        attention_backend=attention_backend,
        device=device,
        row_chunk=row_chunk,
        use_gemm=kwargs.pop("use_gemm", OVERLOCK_B_DETECTION_CONFIG["use_gemm"]),
        **kwargs,
    )
