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


_erf_vec = np.vectorize(math.erf)


def gelu_ref(x):
    return 0.5 * x * (1.0 + _erf_vec(x / math.sqrt(2.0)))


def test_sub_forward_and_backward():
    a_np = np.array([3.0, 5.0, 7.0], dtype=np.float32)
    b_np = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    a = ag.from_numpy(a_np, requires_grad=True)
    b = ag.from_numpy(b_np, requires_grad=True)
    c = ag.sub(a, b)
    np.testing.assert_array_equal(ag.to_numpy(c), a_np - b_np)
    ag.sum_all(c).backward()
    np.testing.assert_array_equal(ag.to_numpy(a.grad), np.ones_like(a_np))
    np.testing.assert_array_equal(ag.to_numpy(b.grad), -np.ones_like(b_np))


def test_gelu_forward_matches_erf_reference():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((4, 5)).astype(np.float32)
    x = ag.from_numpy(x_np)
    y = ag.to_numpy(ag.gelu(x))
    np.testing.assert_allclose(y, gelu_ref(x_np), atol=1e-6)


def test_gelu_backward_finite_diff():
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((3, 4)).astype(np.float32) * 1.5
    x = ag.from_numpy(x_np, requires_grad=True)
    loss = ag.sum_all(ag.gelu(x))
    loss.backward()
    g = ag.to_numpy(x.grad)

    eps = 1e-3
    numerical = np.zeros_like(g)
    for i in range(x_np.shape[0]):
        for j in range(x_np.shape[1]):
            xp = x_np.copy(); xp[i, j] += eps
            xm = x_np.copy(); xm[i, j] -= eps
            lp = ag.to_numpy(ag.sum_all(ag.gelu(ag.from_numpy(xp)))).item()
            lm = ag.to_numpy(ag.sum_all(ag.gelu(ag.from_numpy(xm)))).item()
            numerical[i, j] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(g, numerical, atol=1e-3)


def test_add_bias_forward():
    x_np = np.arange(12, dtype=np.float32).reshape(4, 3)
    b_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y = ag.add_bias(ag.from_numpy(x_np), ag.from_numpy(b_np))
    np.testing.assert_array_equal(ag.to_numpy(y), x_np + b_np)


def test_add_bias_backward_finite_diff():
    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((5, 3)).astype(np.float32)
    b_np = rng.standard_normal((3,)).astype(np.float32)
    x = ag.from_numpy(x_np, requires_grad=True)
    b = ag.from_numpy(b_np, requires_grad=True)
    loss = ag.sum_all(ag.mul(ag.add_bias(x, b), ag.add_bias(x, b)))
    loss.backward()
    dx = ag.to_numpy(x.grad)
    db = ag.to_numpy(b.grad)

    def fwd(x_arr, b_arr):
        y = x_arr + b_arr
        return float(np.sum(y * y))

    eps = 1e-3
    dx_num = np.zeros_like(dx)
    for i in range(x_np.shape[0]):
        for j in range(x_np.shape[1]):
            xp = x_np.copy(); xp[i, j] += eps
            xm = x_np.copy(); xm[i, j] -= eps
            dx_num[i, j] = (fwd(xp, b_np) - fwd(xm, b_np)) / (2 * eps)
    db_num = np.zeros_like(db)
    for j in range(b_np.shape[0]):
        bp = b_np.copy(); bp[j] += eps
        bm = b_np.copy(); bm[j] -= eps
        db_num[j] = (fwd(x_np, bp) - fwd(x_np, bm)) / (2 * eps)

    np.testing.assert_allclose(dx, dx_num, atol=1e-2)
    np.testing.assert_allclose(db, db_num, atol=1e-2)


def test_linear_with_bias_forward_shape():
    layer = nn.Linear(4, 6, bias=True, rng=np.random.default_rng(0))
    x = ag.from_numpy(np.zeros((2, 4), dtype=np.float32))
    y = layer(x)
    assert y.shape == [2, 6]
    params = list(layer.parameters())
    assert len(params) == 2  # weight + bias


def test_linear_without_bias_has_one_param():
    layer = nn.Linear(4, 6, bias=False, rng=np.random.default_rng(0))
    assert len(list(layer.parameters())) == 1


def test_layernorm_module_forward():
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    ln = nn.LayerNorm(8)
    y = ag.to_numpy(ln(ag.from_numpy(x_np)))
    mu = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    expected = (x_np - mu) / np.sqrt(var + 1e-5)
    np.testing.assert_allclose(y, expected, atol=1e-5)


def test_gelu_module_matches_op():
    x = ag.from_numpy(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(ag.to_numpy(nn.GELU()(x)), ag.to_numpy(ag.gelu(x)))


def test_cuda_ops_match_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((6, 4)).astype(np.float32)
    b_np = rng.standard_normal((4,)).astype(np.float32)
    y_np = rng.standard_normal((6, 4)).astype(np.float32)

    x_cpu = ag.from_numpy(x_np); y_cpu = ag.from_numpy(y_np); b_cpu = ag.from_numpy(b_np)
    x_gpu = ag.from_numpy(x_np).to(ag.Device.CUDA)
    y_gpu = ag.from_numpy(y_np).to(ag.Device.CUDA)
    b_gpu = ag.from_numpy(b_np).to(ag.Device.CUDA)

    np.testing.assert_allclose(
        ag.to_numpy(ag.sub(x_cpu, y_cpu)),
        ag.to_numpy(ag.sub(x_gpu, y_gpu).to(ag.Device.CPU)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ag.to_numpy(ag.gelu(x_cpu)),
        ag.to_numpy(ag.gelu(x_gpu).to(ag.Device.CPU)),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        ag.to_numpy(ag.add_bias(x_cpu, b_cpu)),
        ag.to_numpy(ag.add_bias(x_gpu, b_gpu).to(ag.Device.CPU)),
        atol=1e-6,
    )
