import math
import numpy as np
import pytest

import aargrad as ag


def _has_cuda():
    try:
        ag.Tensor.zeros([1], ag.DType.Fp32, ag.Device.CUDA)
    except Exception:
        return False
    return True


def attention_ref(q, k, v, causal=False):
    d = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(d)
    if causal:
        T = scores.shape[-1]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    attn = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(attn, v)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("T", [8, 32, 33, 64])
def test_flash_attention_forward(device, causal, T):
    if device == "cuda" and not _has_cuda():
        pytest.skip("CUDA not available")
    dev = ag.Device.CUDA if device == "cuda" else ag.Device.CPU
    B, H, d = 2, 4, 16
    rng = np.random.default_rng(hash((device, causal, T)) & 0xFFFFFFFF)
    q_np = rng.standard_normal((B, H, T, d)).astype(np.float32)
    k_np = rng.standard_normal((B, H, T, d)).astype(np.float32)
    v_np = rng.standard_normal((B, H, T, d)).astype(np.float32)

    q, k, v = (ag.from_numpy(t).to(dev) for t in (q_np, k_np, v_np))
    out = ag.to_numpy(ag.flash_attention(q, k, v, causal).to(ag.Device.CPU))
    np.testing.assert_allclose(out, attention_ref(q_np, k_np, v_np, causal=causal), atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_flash_attention_backward_finite_diff(device):
    if device == "cuda" and not _has_cuda():
        pytest.skip("CUDA not available")
    dev = ag.Device.CUDA if device == "cuda" else ag.Device.CPU
    B, H, T, d = 1, 2, 4, 8
    rng = np.random.default_rng(7)
    q_np = rng.standard_normal((B, H, T, d)).astype(np.float32)
    k_np = rng.standard_normal((B, H, T, d)).astype(np.float32)
    v_np = rng.standard_normal((B, H, T, d)).astype(np.float32)

    q = ag.from_numpy(q_np, requires_grad=True).to(dev)
    q.requires_grad = True
    k = ag.from_numpy(k_np).to(dev)
    v = ag.from_numpy(v_np).to(dev)
    loss = ag.sum_all(ag.flash_attention(q, k, v, True))
    loss.backward()
    dq = ag.to_numpy(q.grad.to(ag.Device.CPU))

    def loss_at(q_arr):
        return float(np.sum(attention_ref(q_arr, k_np, v_np, causal=True)))

    eps = 1e-3
    dq_num = np.zeros_like(dq)
    it = np.ndindex(*q_np.shape)
    for idx in it:
        qp = q_np.copy(); qp[idx] += eps
        qm = q_np.copy(); qm[idx] -= eps
        dq_num[idx] = (loss_at(qp) - loss_at(qm)) / (2 * eps)

    np.testing.assert_allclose(dq, dq_num, atol=2e-2)


def test_flash_attention_cpu_cuda_parity():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    B, H, T, d = 2, 4, 64, 16
    rng = np.random.default_rng(11)
    q_np = rng.standard_normal((B, H, T, d)).astype(np.float32)
    k_np = rng.standard_normal((B, H, T, d)).astype(np.float32)
    v_np = rng.standard_normal((B, H, T, d)).astype(np.float32)

    def run(dev, requires_grad):
        q = ag.from_numpy(q_np).to(dev)
        k = ag.from_numpy(k_np).to(dev)
        v = ag.from_numpy(v_np).to(dev)
        if requires_grad:
            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True
        out = ag.flash_attention(q, k, v, True)
        loss = ag.sum_all(out)
        loss.backward()
        return (
            ag.to_numpy(out.to(ag.Device.CPU)),
            ag.to_numpy(q.grad.to(ag.Device.CPU)),
            ag.to_numpy(k.grad.to(ag.Device.CPU)),
            ag.to_numpy(v.grad.to(ag.Device.CPU)),
        )

    cpu_out, cpu_dq, cpu_dk, cpu_dv = run(ag.Device.CPU, True)
    gpu_out, gpu_dq, gpu_dk, gpu_dv = run(ag.Device.CUDA, True)
    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-4)
    np.testing.assert_allclose(cpu_dq, gpu_dq, atol=1e-3)
    np.testing.assert_allclose(cpu_dk, gpu_dk, atol=1e-3)
    np.testing.assert_allclose(cpu_dv, gpu_dv, atol=1e-3)
