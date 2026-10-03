"""V03: NATTEN backend resolution + differentiable CPU ``na2d_av`` reference.

Independent naive-loop oracle, corner/edge/interior coverage, H==K, rectangular and
minimal sizes, K=3/5, gradients, chunk equivalence, and fail-fast behaviour.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from overlock_yolo.attention_backend import (  # noqa: E402
    AttentionBackendError,
    env_report,
    na2d_av,
    na2d_av_reference,
    natten_available,
    resolve_backend,
)


def naive_na2d_av(attn: torch.Tensor, value: torch.Tensor, K: int) -> torch.Tensor:
    """Independent triple-loop oracle (no unfold/gather/matmul shortcuts)."""
    B, Hh, H, W, KK = attn.shape
    D = value.shape[-1]
    assert KK == K * K
    out = torch.zeros(B, Hh, H, W, D, dtype=value.dtype)
    for b in range(B):
        for hh in range(Hh):
            for y in range(H):
                sy = max(0, min(y - K // 2, H - K))
                for x in range(W):
                    sx = max(0, min(x - K // 2, W - K))
                    for i in range(K):
                        for j in range(K):
                            out[b, hh, y, x] += attn[b, hh, y, x, i * K + j] * value[b, hh, sy + i, sx + j]
    return out


class TestBackendResolution(unittest.TestCase):
    def test_auto_on_cpu_resolves_to_reference(self):
        r = resolve_backend("auto", device=torch.device("cpu"))
        self.assertEqual(r.resolved, "torch_reference")
        self.assertFalse(r.is_native)
        self.assertIn("NATTEN", r.reason)
        self.assertIn("resolved", r.as_dict())

    def test_natten_request_fails_loudly_when_absent(self):
        if natten_available():
            self.skipTest("NATTEN present in this environment; absence path not applicable")
        for dev in (torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
            with self.assertRaises(AttentionBackendError) as ctx:
                resolve_backend("natten", device=dev)
            self.assertIn("natten", str(ctx.exception).lower())

    def test_torch_reference_explicit(self):
        r = resolve_backend("torch_reference", device=torch.device("cpu"))
        self.assertEqual(r.resolved, "torch_reference")

    def test_unknown_backend_rejected(self):
        with self.assertRaises(AttentionBackendError):
            resolve_backend("magic")

    def test_env_report_has_no_fake_flag(self):
        rep = env_report()
        self.assertIn("natten_importable", rep)
        if not rep["natten_importable"]:
            self.assertIsNotNone(rep["natten_import_error"])


class TestReferenceAgainstOracle(unittest.TestCase):
    CASES = [
        (2, 2, 3, 3, 2, 3),  # corner-heavy tiny, K=2
        (2, 2, 5, 7, 3, 6),  # rectangular, K=3
        (1, 3, 4, 4, 4, 2),  # H==W==K
        (2, 1, 6, 6, 5, 4),  # K=5
        (1, 1, 1, 1, 1, 1),  # minimal, K=1
        (1, 2, 7, 3, 3, 5),  # tall
        (1, 2, 2, 5, 2, 3),  # wide
        (1, 2, 9, 9, 5, 8),
        (1, 2, 8, 13, 3, 4),
        (1, 2, 5, 7, 5, 6),  # H==K, W>K
        (1, 1, 11, 11, 7, 2),  # K=7 (overview/focus region sizes)
    ]

    def test_matches_oracle(self):
        torch.manual_seed(0)
        for (b, heads, h, w, k, d) in self.CASES:
            with self.subTest(shape=(b, heads, h, w, k, d)):
                a = torch.rand(b, heads, h, w, k * k)
                v = torch.randn(b, heads, h, w, d)
                got = na2d_av_reference(a, v, k)
                exp = naive_na2d_av(a, v, k)
                self.assertEqual(tuple(got.shape), (b, heads, h, w, d))
                self.assertTrue(torch.allclose(got, exp, atol=1e-6), f"max diff {(got - exp).abs().max()}")

    def test_corner_edge_interior_explicitly(self):
        """Corners and edges must use the FULL shifted window, not zero padding or per-point clamp."""
        torch.manual_seed(3)
        h = w = 5
        k = 3
        v = torch.arange(h * w, dtype=torch.float).reshape(1, 1, h, w, 1)
        # one-hot attention on the single neighbour (i=0, j=0) -> must pick start_y/start_x
        a = torch.zeros(1, 1, h, w, k * k)
        a[..., 0] = 1.0
        out = na2d_av_reference(a, v, k)[..., 0]
        for y in range(h):
            for x in range(w):
                sy = max(0, min(y - k // 2, h - k))
                sx = max(0, min(x - k // 2, w - k))
                self.assertEqual(float(out[0, 0, y, x]), float(v[0, 0, sy, sx, 0]), f"corner/edge mismatch at {(y, x)}")
        # a zero-padding based formulation would return 0 at this clamped position; the
        # correct full-shift semantics return the real clamped neighbour value
        self.assertEqual(float(out[0, 0, h - 1, w - 1]), float(v[0, 0, h - k, w - k, 0]))
        self.assertNotEqual(float(out[0, 0, h - 1, w - 1]), 0.0)

    def test_softmax_not_reapplied(self):
        """The op is linear in attn; if it re-softmaxed, scaling attn would not scale the output."""
        torch.manual_seed(4)
        a = torch.rand(1, 2, 4, 4, 9)
        v = torch.randn(1, 2, 4, 4, 3)
        o1 = na2d_av_reference(a, v, 3)
        o2 = na2d_av_reference(2.0 * a, v, 3)
        self.assertTrue(torch.allclose(o2, 2.0 * o1, atol=1e-6))

    def test_semantics_match_official_constant_pad_interior(self):
        """Interior positions agree with a constant-pad unfold conv (independent formulation)."""
        torch.manual_seed(5)
        B, Hh, H, W, K, D = 1, 2, 6, 6, 3, 4
        a = torch.rand(B, Hh, H, W, K * K)
        v = torch.randn(B, Hh, H, W, D)
        pad = K // 2
        vp = v.permute(0, 1, 4, 2, 3).reshape(B * Hh * D, H, W)
        vp = torch.nn.functional.pad(vp, (pad, pad, pad, pad))
        pat = vp.unfold(1, K, 1).unfold(2, K, 1)  # [B*Hh*D,H,W,K,K]
        pat = pat.reshape(B, Hh, D, H, W, K * K)
        conv_like = (a.unsqueeze(2) * pat).sum(-1).permute(0, 1, 3, 4, 2)  # [B,heads,H,W,D]
        got = na2d_av_reference(a, v, K)
        i = pad
        self.assertTrue(torch.allclose(got[:, :, i : H - i, i : W - i], conv_like[:, :, i : H - i, i : W - i], atol=1e-6))

    def test_row_chunk_equivalence(self):
        torch.manual_seed(6)
        a = torch.rand(2, 3, 5, 7, 9)
        v = torch.randn(2, 3, 5, 7, 4)
        full = na2d_av_reference(a, v, 3)
        for chunk in (1, 4, 7, 100):
            self.assertTrue(torch.equal(full, na2d_av_reference(a, v, 3, row_chunk=chunk)))

    def test_gradients_finite_and_nonzero(self):
        torch.manual_seed(7)
        a = torch.rand(1, 2, 4, 5, 9, requires_grad=True)
        v = torch.randn(1, 2, 4, 5, 3, requires_grad=True)
        out = na2d_av_reference(a, v, 3)
        out.sum().backward()
        for name, g in (("attn", a.grad), ("value", v.grad)):
            self.assertIsNotNone(g)
            self.assertTrue(torch.isfinite(g).all(), f"{name} grad not finite")
            self.assertGreater(float(g.abs().sum()), 0.0, f"{name} grad is all zeros")

    def test_gradcheck_value(self):
        torch.manual_seed(8)
        a = (torch.rand(1, 1, 3, 4, 9, dtype=torch.double) + 0.1)
        v = torch.randn(1, 1, 3, 4, 2, dtype=torch.double, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(lambda vv: na2d_av_reference(a, vv, 3), (v,), eps=1e-6, atol=1e-5)
        )

    def test_gradcheck_attn(self):
        torch.manual_seed(9)
        a = torch.rand(1, 1, 3, 3, 4, dtype=torch.double, requires_grad=True)
        v = torch.randn(1, 1, 3, 3, 2, dtype=torch.double)
        self.assertTrue(
            torch.autograd.gradcheck(lambda aa: na2d_av_reference(aa, v, 2), (a,), eps=1e-6, atol=1e-5)
        )

    def test_dispatch_through_public_op(self):
        torch.manual_seed(10)
        a = torch.rand(1, 2, 4, 4, 9)
        v = torch.randn(1, 2, 4, 4, 3)
        r = resolve_backend("auto", device=torch.device("cpu"))
        self.assertTrue(torch.equal(na2d_av(a, v, 3, resolved=r), na2d_av_reference(a, v, 3)))
        self.assertTrue(torch.equal(na2d_av(a, v, 3, backend="torch_reference"), na2d_av_reference(a, v, 3)))

    def test_dtype_and_accumulation(self):
        torch.manual_seed(11)
        a = torch.rand(1, 1, 3, 3, 4)
        v = torch.randn(1, 1, 3, 3, 2)
        self.assertEqual(na2d_av_reference(a, v, 2).dtype, torch.float32)
        a16, v16 = a.to(torch.bfloat16), v.to(torch.bfloat16)
        out16 = na2d_av_reference(a16, v16, 2)
        self.assertEqual(out16.dtype, torch.bfloat16)
        ref32 = na2d_av_reference(a16.float(), v16.float(), 2)
        self.assertTrue(torch.allclose(out16.float(), ref32, atol=2e-2))


class TestFailFast(unittest.TestCase):
    def test_h_smaller_than_k(self):
        with self.assertRaises(AttentionBackendError):
            na2d_av_reference(torch.rand(1, 1, 2, 5, 9), torch.randn(1, 1, 2, 5, 3), 3)

    def test_w_smaller_than_k(self):
        with self.assertRaises(AttentionBackendError):
            na2d_av_reference(torch.rand(1, 1, 5, 2, 25), torch.randn(1, 1, 5, 2, 3), 5)

    def test_wrong_last_dim(self):
        with self.assertRaises(AttentionBackendError):
            na2d_av_reference(torch.rand(1, 1, 4, 4, 8), torch.randn(1, 1, 4, 4, 3), 3)

    def test_mismatched_spatial(self):
        with self.assertRaises(AttentionBackendError):
            na2d_av_reference(torch.rand(1, 1, 4, 5, 9), torch.randn(1, 1, 4, 4, 3), 3)

    def test_dtype_mismatch(self):
        with self.assertRaises(AttentionBackendError):
            na2d_av_reference(torch.rand(1, 1, 4, 4, 9), torch.randn(1, 1, 4, 4, 3, dtype=torch.double), 3)

    def test_rank_mismatch(self):
        with self.assertRaises(AttentionBackendError):
            na2d_av_reference(torch.rand(1, 4, 4, 9), torch.randn(1, 1, 4, 4, 3), 3)

    def test_bad_kernel_size(self):
        with self.assertRaises(AttentionBackendError):
            na2d_av_reference(torch.rand(1, 1, 4, 4, 9), torch.randn(1, 1, 4, 4, 3), 0)
        with self.assertRaises(AttentionBackendError):
            na2d_av_reference(torch.rand(1, 1, 4, 4, 9), torch.randn(1, 1, 4, 4, 3), 3.0)

    def test_unsupported_dilation_and_causal(self):
        a = torch.rand(1, 1, 4, 4, 9)
        v = torch.randn(1, 1, 4, 4, 3)
        with self.assertRaises(AttentionBackendError):
            na2d_av(a, v, 3, dilation=2)
        with self.assertRaises(AttentionBackendError):
            na2d_av(a, v, 3, is_causal=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
