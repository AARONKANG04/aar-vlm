import numpy as np
import pytest

import aargrad as ag


def _has_cuda():
    try:
        ag.Tensor.zeros([1], ag.DType.Fp32, ag.Device.CUDA)
    except Exception:
        return False
    return True


def test_bmm_matches_numpy():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    b_np = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
    out = ag.to_numpy(ag.bmm(ag.from_numpy(a_np), ag.from_numpy(b_np)))
    np.testing.assert_allclose(out, a_np @ b_np, atol=1e-4)


def test_bmm_a_bt_matches_numpy():
    rng = np.random.default_rng(1)
    a_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    b_np = rng.standard_normal((2, 3, 6, 5)).astype(np.float32)
    out = ag.to_numpy(ag.bmm_a_bt(ag.from_numpy(a_np), ag.from_numpy(b_np)))
    np.testing.assert_allclose(out, np.einsum("...ik,...jk->...ij", a_np, b_np), atol=1e-4)


def test_bmm_at_b_matches_numpy():
    rng = np.random.default_rng(2)
    a_np = rng.standard_normal((2, 3, 5, 4)).astype(np.float32)
    b_np = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
    out = ag.to_numpy(ag.bmm_at_b(ag.from_numpy(a_np), ag.from_numpy(b_np)))
    np.testing.assert_allclose(out, np.einsum("...ki,...kj->...ij", a_np, b_np), atol=1e-4)


def test_bmm_backward_finite_diff():
    rng = np.random.default_rng(3)
    a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b_np = rng.standard_normal((2, 4, 5)).astype(np.float32)
    a = ag.from_numpy(a_np, requires_grad=True)
    b = ag.from_numpy(b_np, requires_grad=True)
    ag.sum_all(ag.bmm(a, b)).backward()
    da = ag.to_numpy(a.grad)
    db = ag.to_numpy(b.grad)

    eps = 1e-3
    da_num = np.zeros_like(da)
    for idx in np.ndindex(a_np.shape):
        ap = a_np.copy(); ap[idx] += eps
        am = a_np.copy(); am[idx] -= eps
        da_num[idx] = (float(np.sum(ap @ b_np)) - float(np.sum(am @ b_np))) / (2 * eps)
    np.testing.assert_allclose(da, da_num, atol=1e-2)

    db_num = np.zeros_like(db)
    for idx in np.ndindex(b_np.shape):
        bp = b_np.copy(); bp[idx] += eps
        bm = b_np.copy(); bm[idx] -= eps
        db_num[idx] = (float(np.sum(a_np @ bp)) - float(np.sum(a_np @ bm))) / (2 * eps)
    np.testing.assert_allclose(db, db_num, atol=1e-2)


def test_bmm_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(4)
    a_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    b_np = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
    cpu_out = ag.to_numpy(ag.bmm(ag.from_numpy(a_np), ag.from_numpy(b_np)))
    gpu_out = ag.to_numpy(
        ag.bmm(ag.from_numpy(a_np).to(ag.Device.CUDA),
               ag.from_numpy(b_np).to(ag.Device.CUDA)).to(ag.Device.CPU)
    )
    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-4)
