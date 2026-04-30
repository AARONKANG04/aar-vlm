import math
import numpy as np
import pytest

import aargrad as ag
from aargrad import nn


def _has_cuda():
    try:
        ag.Tensor.zeros([1], ag.DType.Fp32, ag.Device.CUDA)
    except Exception:
        return False
    return True


def attention_ref(q, k, v, causal=False):
    d = q.shape[-1]
    scores = (q @ k.T) / math.sqrt(d)
    if causal:
        T = scores.shape[-1]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    attn = e / e.sum(axis=-1, keepdims=True)
    return attn @ v


def test_attention_forward_matches_reference():
    rng = np.random.default_rng(0)
    q_np = rng.standard_normal((4, 8)).astype(np.float32)
    k_np = rng.standard_normal((4, 8)).astype(np.float32)
    v_np = rng.standard_normal((4, 8)).astype(np.float32)

    q, k, v = (ag.from_numpy(t) for t in (q_np, k_np, v_np))
    out = ag.to_numpy(nn.scaled_dot_product_attention(q, k, v))
    np.testing.assert_allclose(out, attention_ref(q_np, k_np, v_np), atol=1e-5)


def test_causal_attention_matches_reference():
    rng = np.random.default_rng(1)
    q_np = rng.standard_normal((4, 8)).astype(np.float32)
    k_np = rng.standard_normal((4, 8)).astype(np.float32)
    v_np = rng.standard_normal((4, 8)).astype(np.float32)

    q, k, v = (ag.from_numpy(t) for t in (q_np, k_np, v_np))
    out = ag.to_numpy(nn.scaled_dot_product_attention(q, k, v, causal=True))
    np.testing.assert_allclose(out, attention_ref(q_np, k_np, v_np, causal=True), atol=1e-5)


def test_apply_causal_mask_zeros_upper_triangle_after_softmax():
    rng = np.random.default_rng(2)
    scores_np = rng.standard_normal((4, 4)).astype(np.float32)
    masked = ag.apply_causal_mask(ag.from_numpy(scores_np))
    probs = ag.to_numpy(ag.softmax(masked))
    for i in range(4):
        for j in range(4):
            if j > i:
                assert probs[i, j] == 0.0


def test_scale_backward_passes_alpha():
    a = ag.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
    ag.sum_all(ag.scale(a, 2.5)).backward()
    np.testing.assert_array_equal(ag.to_numpy(a.grad), np.full(3, 2.5, dtype=np.float32))


def test_attention_backward_finite_diff():
    rng = np.random.default_rng(3)
    q_np = rng.standard_normal((3, 4)).astype(np.float32)
    k_np = rng.standard_normal((3, 4)).astype(np.float32)
    v_np = rng.standard_normal((3, 4)).astype(np.float32)

    q = ag.from_numpy(q_np, requires_grad=True)
    k = ag.from_numpy(k_np)
    v = ag.from_numpy(v_np)
    loss = ag.sum_all(nn.scaled_dot_product_attention(q, k, v, causal=True))
    loss.backward()
    dq = ag.to_numpy(q.grad)

    def loss_at(q_arr):
        return float(np.sum(attention_ref(q_arr, k_np, v_np, causal=True)))

    eps = 1e-3
    dq_num = np.zeros_like(dq)
    for i in range(q_np.shape[0]):
        for j in range(q_np.shape[1]):
            qp = q_np.copy(); qp[i, j] += eps
            qm = q_np.copy(); qm[i, j] -= eps
            dq_num[i, j] = (loss_at(qp) - loss_at(qm)) / (2 * eps)

    np.testing.assert_allclose(dq, dq_num, atol=1e-2)


def test_attention_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(4)
    q_np = rng.standard_normal((6, 8)).astype(np.float32)
    k_np = rng.standard_normal((6, 8)).astype(np.float32)
    v_np = rng.standard_normal((6, 8)).astype(np.float32)

    cpu = nn.scaled_dot_product_attention(
        *(ag.from_numpy(t) for t in (q_np, k_np, v_np)), causal=True
    )
    gpu = nn.scaled_dot_product_attention(
        *(ag.from_numpy(t).to(ag.Device.CUDA) for t in (q_np, k_np, v_np)),
        causal=True,
    )
    np.testing.assert_allclose(
        ag.to_numpy(cpu), ag.to_numpy(gpu.to(ag.Device.CPU)), atol=1e-5
    )
