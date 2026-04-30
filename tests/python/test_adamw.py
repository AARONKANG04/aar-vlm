import numpy as np
import pytest

import aargrad as ag
from aargrad import nn
from aargrad.optim import AdamW


def _has_cuda():
    try:
        ag.Tensor.zeros([1], ag.DType.Fp32, ag.Device.CUDA)
    except Exception:
        return False
    return True


def _adamw_numpy_step(param, grad, m, v, lr, b1, b2, eps, wd, t):
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad * grad
    m_hat = m / bc1
    v_hat = v / bc2
    param = param * (1 - lr * wd) - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adamw_matches_numpy(device):
    if device == "cuda" and not _has_cuda():
        pytest.skip("CUDA not available")
    dev = ag.Device.CUDA if device == "cuda" else ag.Device.CPU

    rng = np.random.default_rng(0)
    p_np = rng.standard_normal((4, 5)).astype(np.float32)
    g_np = rng.standard_normal((4, 5)).astype(np.float32)
    m_np = np.zeros_like(p_np)
    v_np = np.zeros_like(p_np)

    lr, b1, b2, eps, wd = 1e-2, 0.9, 0.999, 1e-8, 0.01

    p_t = ag.from_numpy(p_np.copy()).to(dev)
    g_t = ag.from_numpy(g_np.copy()).to(dev)
    m_t = ag.Tensor.zeros(list(p_np.shape), ag.DType.Fp32, dev)
    v_t = ag.Tensor.zeros(list(p_np.shape), ag.DType.Fp32, dev)

    for t in range(1, 6):
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        ag.adamw_step(p_t, g_t, m_t, v_t, lr, b1, b2, eps, wd, bc1, bc2)
        p_np, m_np, v_np = _adamw_numpy_step(p_np, g_np, m_np, v_np, lr, b1, b2, eps, wd, t)

    np.testing.assert_allclose(ag.to_numpy(p_t.to(ag.Device.CPU)), p_np, atol=1e-6)
    np.testing.assert_allclose(ag.to_numpy(m_t.to(ag.Device.CPU)), m_np, atol=1e-6)
    np.testing.assert_allclose(ag.to_numpy(v_t.to(ag.Device.CPU)), v_np, atol=1e-6)


def test_adamw_cpu_cuda_parity():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(1)
    p_init = rng.standard_normal((6, 7)).astype(np.float32)
    g_np = rng.standard_normal((6, 7)).astype(np.float32)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.05

    def run(dev):
        p = ag.from_numpy(p_init.copy()).to(dev)
        g = ag.from_numpy(g_np.copy()).to(dev)
        m = ag.Tensor.zeros(list(p_init.shape), ag.DType.Fp32, dev)
        v = ag.Tensor.zeros(list(p_init.shape), ag.DType.Fp32, dev)
        for t in range(1, 11):
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
            ag.adamw_step(p, g, m, v, lr, b1, b2, eps, wd, bc1, bc2)
        return ag.to_numpy(p.to(ag.Device.CPU))

    np.testing.assert_allclose(run(ag.Device.CUDA), run(ag.Device.CPU), atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adamw_mlp_loss_decreases(device):
    if device == "cuda" and not _has_cuda():
        pytest.skip("CUDA not available")
    dev = ag.Device.CUDA if device == "cuda" else ag.Device.CPU

    rng = np.random.default_rng(0)
    A_true = rng.standard_normal((4, 8)).astype(np.float32)
    B_true = rng.standard_normal((8, 1)).astype(np.float32)
    X = rng.standard_normal((32, 4)).astype(np.float32)
    Y = np.maximum(X @ A_true, 0) @ B_true

    model = nn.Sequential(
        nn.Linear(4, 8, bias=True, rng=np.random.default_rng(1)),
        nn.GELU(),
        nn.Linear(8, 1, bias=True, rng=np.random.default_rng(2)),
    )
    model.to(dev)
    optim = AdamW(list(model.parameters()), lr=1e-2, weight_decay=0.01)

    losses = []
    for _ in range(200):
        x = ag.from_numpy(X).to(dev)
        y = ag.from_numpy(Y).to(dev)
        pred = model(x)
        diff = ag.sub(pred, y)
        loss = ag.sum_all(ag.mul(diff, diff))
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(ag.to_numpy(loss.to(ag.Device.CPU)).item())

    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
