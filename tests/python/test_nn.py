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


def test_parameter_marks_requires_grad():
    t = ag.from_numpy(np.zeros((2, 2), dtype=np.float32))
    assert t.requires_grad is False
    p = nn.Parameter(t)
    assert p.tensor.requires_grad is True


def test_module_registers_parameters_and_submodules():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(ag.from_numpy(np.zeros((2, 2), dtype=np.float32)))

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(ag.from_numpy(np.zeros((3,), dtype=np.float32)))
            self.inner = Inner()
            self.b = nn.Parameter(ag.from_numpy(np.zeros((4,), dtype=np.float32)))

    m = Outer()
    params = list(m.parameters())
    assert len(params) == 3
    assert params[0].tensor.shape == [3]
    assert params[1].tensor.shape == [4]
    assert params[2].tensor.shape == [2, 2]


def test_linear_forward_shape():
    rng = np.random.default_rng(0)
    layer = nn.Linear(8, 4, rng=rng)
    x = ag.from_numpy(rng.standard_normal((2, 8)).astype(np.float32))
    y = layer(x)
    assert y.shape == [2, 4]


def test_linear_backward_finite_diff():
    rng = np.random.default_rng(1)
    layer = nn.Linear(3, 2, rng=rng)
    x_np = rng.standard_normal((4, 3)).astype(np.float32)

    x = ag.from_numpy(x_np, requires_grad=True)
    y = layer(x)
    loss = ag.sum_all(ag.mul(y, y))
    loss.backward()
    w_grad = ag.to_numpy(layer.weight.tensor.grad).copy()
    w_val = ag.to_numpy(layer.weight.tensor).copy()

    def loss_at(w_modified):
        w_t = ag.from_numpy(w_modified)
        x_p = ag.from_numpy(x_np)
        y_p = ag.matmul_a_bt(x_p, w_t)
        return ag.to_numpy(ag.sum_all(ag.mul(y_p, y_p))).item()

    eps = 1e-3
    numerical = np.zeros_like(w_grad)
    for i in range(w_val.shape[0]):
        for j in range(w_val.shape[1]):
            wp = w_val.copy(); wp[i, j] += eps
            wm = w_val.copy(); wm[i, j] -= eps
            numerical[i, j] = (loss_at(wp) - loss_at(wm)) / (2 * eps)

    np.testing.assert_allclose(w_grad, numerical, atol=1e-2)


def test_module_to_cuda_preserves_requires_grad():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(0)
    model = nn.Sequential(nn.Linear(4, 8, rng=rng), nn.ReLU(), nn.Linear(8, 1, rng=rng))
    model.to(ag.Device.CUDA)
    for p in model.parameters():
        assert p.tensor.device == ag.Device.CUDA
        assert p.tensor.requires_grad is True


def test_sgd_step_updates_in_place():
    p = nn.Parameter(ag.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32)))
    loss = ag.sum_all(p.tensor)
    loss.backward()

    SGD([p], lr=0.1).step()

    np.testing.assert_allclose(
        ag.to_numpy(p.tensor),
        np.array([0.9, 1.9, 2.9], dtype=np.float32),
    )
