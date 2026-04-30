import math
import numpy as np
import pytest

import aargrad as ag
from aargrad import nn
from aargrad.optim import SGD


def _has_cuda():
    try:
        ag.Tensor.zeros([1], ag.DType.Fp32, ag.Device.CUDA)
    except Exception:
        return False
    return True


def mha_ref(x, q_w, k_w, v_w, o_w, q_b, k_b, v_b, o_b, n_heads, causal=False):
    B, T, D = x.shape
    d_head = D // n_heads
    Q = x @ q_w.T + q_b
    K = x @ k_w.T + k_b
    V = x @ v_w.T + v_b
    Q = Q.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
    scores = np.einsum("bhid,bhjd->bhij", Q, K) / math.sqrt(d_head)
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    attn = e / e.sum(axis=-1, keepdims=True)
    out = np.einsum("bhij,bhjd->bhid", attn, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ o_w.T + o_b


def _params_to_numpy(mha):
    return {
        "q_w": ag.to_numpy(mha.q_proj.weight.tensor),
        "k_w": ag.to_numpy(mha.k_proj.weight.tensor),
        "v_w": ag.to_numpy(mha.v_proj.weight.tensor),
        "o_w": ag.to_numpy(mha.out_proj.weight.tensor),
        "q_b": ag.to_numpy(mha.q_proj.bias.tensor),
        "k_b": ag.to_numpy(mha.k_proj.bias.tensor),
        "v_b": ag.to_numpy(mha.v_proj.bias.tensor),
        "o_b": ag.to_numpy(mha.out_proj.bias.tensor),
    }


def test_mha_forward_matches_reference():
    rng = np.random.default_rng(0)
    B, T, D, H = 2, 5, 8, 2
    x_np = rng.standard_normal((B, T, D)).astype(np.float32)
    mha = nn.MultiHeadAttention(D, H, bias=True, rng=np.random.default_rng(1))
    out = ag.to_numpy(mha(ag.from_numpy(x_np)))
    ref = mha_ref(x_np, **_params_to_numpy(mha), n_heads=H)
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_mha_causal_matches_reference():
    rng = np.random.default_rng(2)
    B, T, D, H = 2, 5, 8, 2
    x_np = rng.standard_normal((B, T, D)).astype(np.float32)
    mha = nn.MultiHeadAttention(D, H, causal=True, bias=True, rng=np.random.default_rng(3))
    out = ag.to_numpy(mha(ag.from_numpy(x_np)))
    ref = mha_ref(x_np, **_params_to_numpy(mha), n_heads=H, causal=True)
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_mha_backward_finite_diff():
    rng = np.random.default_rng(4)
    B, T, D, H = 1, 3, 4, 2
    x_np = rng.standard_normal((B, T, D)).astype(np.float32)
    mha = nn.MultiHeadAttention(D, H, causal=True, bias=True, rng=np.random.default_rng(5))
    x = ag.from_numpy(x_np, requires_grad=True)
    ag.sum_all(mha(x)).backward()
    dx = ag.to_numpy(x.grad)

    params = _params_to_numpy(mha)
    eps = 1e-3
    dx_num = np.zeros_like(dx)
    for idx in np.ndindex(x_np.shape):
        xp = x_np.copy(); xp[idx] += eps
        xm = x_np.copy(); xm[idx] -= eps
        lp = float(np.sum(mha_ref(xp, **params, n_heads=H, causal=True)))
        lm = float(np.sum(mha_ref(xm, **params, n_heads=H, causal=True)))
        dx_num[idx] = (lp - lm) / (2 * eps)
    np.testing.assert_allclose(dx, dx_num, atol=2e-2)


def test_mha_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    B, T, D, H = 2, 6, 8, 2
    x_np = np.random.default_rng(6).standard_normal((B, T, D)).astype(np.float32)
    cpu_mha = nn.MultiHeadAttention(D, H, causal=True, bias=True, rng=np.random.default_rng(7))
    gpu_mha = nn.MultiHeadAttention(D, H, causal=True, bias=True, rng=np.random.default_rng(7))
    gpu_mha.to(ag.Device.CUDA)
    cpu_out = ag.to_numpy(cpu_mha(ag.from_numpy(x_np)))
    gpu_out = ag.to_numpy(gpu_mha(ag.from_numpy(x_np).to(ag.Device.CUDA)).to(ag.Device.CPU))
    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-4)


def test_transformer_block_loss_decreases():
    rng = np.random.default_rng(8)
    B, T, D, H = 2, 4, 8, 2
    x_np = rng.standard_normal((B, T, D)).astype(np.float32)
    y_np = rng.standard_normal((B, T, D)).astype(np.float32)

    block = nn.Sequential(
        nn.LayerNorm(D),
        nn.MultiHeadAttention(D, H, causal=True, bias=True, rng=np.random.default_rng(9)),
        nn.LayerNorm(D),
        nn.Linear(D, 4 * D, bias=True, rng=np.random.default_rng(10)),
        nn.GELU(),
        nn.Linear(4 * D, D, bias=True, rng=np.random.default_rng(11)),
    )
    optim = SGD(list(block.parameters()), lr=0.005)

    losses = []
    for _ in range(150):
        x = ag.from_numpy(x_np)
        y_neg = ag.from_numpy(-y_np)
        pred = block(x)
        diff = ag.add(pred, y_neg)
        loss = ag.sum_all(ag.mul(diff, diff))
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(ag.to_numpy(loss).item())

    assert losses[-1] < losses[0] * 0.7, f"loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
