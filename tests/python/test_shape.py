import numpy as np
import pytest

import aargrad as ag


def _has_cuda():
    try:
        ag.Tensor.zeros([1], ag.DType.Fp32, ag.Device.CUDA)
    except Exception:
        return False
    return True


def test_reshape_round_trip():
    x_np = np.arange(24, dtype=np.float32)
    t = ag.from_numpy(x_np)
    r = ag.reshape(t, [2, 3, 4])
    assert r.shape == [2, 3, 4]
    np.testing.assert_array_equal(ag.to_numpy(r), x_np.reshape(2, 3, 4))


def test_reshape_inferred_dim():
    x = ag.from_numpy(np.arange(12, dtype=np.float32))
    r = ag.reshape(x, [3, -1])
    assert r.shape == [3, 4]


def test_reshape_backward():
    x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = ag.from_numpy(x_np, requires_grad=True)
    r = ag.reshape(x, [4, 3])
    ag.sum_all(ag.mul(r, r)).backward()
    np.testing.assert_allclose(ag.to_numpy(x.grad), 2 * x_np)


def test_transpose_matches_numpy():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    x = ag.from_numpy(x_np)
    y = ag.to_numpy(ag.transpose(x, 1, 2))
    np.testing.assert_array_equal(y, np.swapaxes(x_np, 1, 2))


def test_transpose_is_involutive():
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    x = ag.from_numpy(x_np)
    y = ag.transpose(ag.transpose(x, 0, 2), 0, 2)
    np.testing.assert_array_equal(ag.to_numpy(y), x_np)


def test_transpose_backward():
    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    x = ag.from_numpy(x_np, requires_grad=True)
    y = ag.transpose(x, 0, 2)
    ag.sum_all(ag.mul(y, y)).backward()
    np.testing.assert_allclose(ag.to_numpy(x.grad), 2 * x_np, atol=1e-5)


def test_shape_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    x_cpu = ag.from_numpy(x_np)
    x_gpu = ag.from_numpy(x_np).to(ag.Device.CUDA)
    np.testing.assert_array_equal(
        ag.to_numpy(ag.transpose(x_cpu, 1, 3)),
        ag.to_numpy(ag.transpose(x_gpu, 1, 3).to(ag.Device.CPU)),
    )
