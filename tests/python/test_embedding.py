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


def test_embedding_forward_matches_numpy():
    rng = np.random.default_rng(0)
    V, D = 7, 4
    w_np = rng.standard_normal((V, D)).astype(np.float32)
    ids_np = np.array([[0, 1, 5], [3, 0, 6]], dtype=np.int64)
    w = ag.from_numpy(w_np)
    ids = ag.from_numpy(ids_np)
    out = ag.to_numpy(ag.embedding(w, ids))
    assert out.shape == (2, 3, D)
    np.testing.assert_array_equal(out, w_np[ids_np])


def test_embedding_module_forward_shape():
    rng = np.random.default_rng(1)
    emb = nn.Embedding(10, 6, rng=rng)
    ids = ag.from_numpy(np.zeros((2, 4), dtype=np.int64))
    out = emb(ids)
    assert out.shape == [2, 4, 6]


def test_embedding_backward_scatter_add_repeated_ids():
    rng = np.random.default_rng(2)
    V, D = 5, 3
    w_np = rng.standard_normal((V, D)).astype(np.float32)
    ids_np = np.array([0, 2, 0, 1, 2, 0], dtype=np.int64)
    w = ag.from_numpy(w_np, requires_grad=True)
    ids = ag.from_numpy(ids_np)
    out = ag.embedding(w, ids)
    ag.sum_all(ag.mul(out, out)).backward()
    grad = ag.to_numpy(w.grad)
    expected = np.zeros_like(w_np)
    for i, idx in enumerate(ids_np):
        expected[idx] += 2 * w_np[idx]
    np.testing.assert_allclose(grad, expected, atol=1e-5)


def test_embedding_backward_finite_diff():
    rng = np.random.default_rng(3)
    V, D = 4, 3
    w_np = rng.standard_normal((V, D)).astype(np.float32)
    ids_np = np.array([0, 1, 2, 1, 3], dtype=np.int64)
    w = ag.from_numpy(w_np, requires_grad=True)
    ids = ag.from_numpy(ids_np)
    out = ag.embedding(w, ids)
    ag.sum_all(ag.mul(out, out)).backward()
    dw = ag.to_numpy(w.grad)

    eps = 1e-3
    dw_num = np.zeros_like(dw)
    for idx in np.ndindex(w_np.shape):
        wp = w_np.copy(); wp[idx] += eps
        wm = w_np.copy(); wm[idx] -= eps
        lp = float(np.sum(wp[ids_np] ** 2))
        lm = float(np.sum(wm[ids_np] ** 2))
        dw_num[idx] = (lp - lm) / (2 * eps)
    np.testing.assert_allclose(dw, dw_num, atol=1e-3)


def test_embedding_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(4)
    V, D = 8, 5
    w_np = rng.standard_normal((V, D)).astype(np.float32)
    ids_np = np.array([[0, 7, 3], [2, 2, 5]], dtype=np.int64)
    w_cpu = ag.from_numpy(w_np)
    w_gpu = ag.from_numpy(w_np).to(ag.Device.CUDA)
    ids_cpu = ag.from_numpy(ids_np)
    ids_gpu = ag.from_numpy(ids_np).to(ag.Device.CUDA)
    cpu_out = ag.to_numpy(ag.embedding(w_cpu, ids_cpu))
    gpu_out = ag.to_numpy(ag.embedding(w_gpu, ids_gpu).to(ag.Device.CPU))
    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-6)
