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
        ag.to_numpy(ag.contiguous(ag.transpose(x_gpu, 1, 3)).to(ag.Device.CPU)),
    )


def test_transpose_is_metadata_only():
    rng = np.random.default_rng(10)
    x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    x = ag.from_numpy(x_np)
    y = ag.transpose(x, 0, 2)
    assert y.shape == [4, 3, 2]
    assert y.strides == [x.strides[2], x.strides[1], x.strides[0]]
    assert y.is_contiguous() is False
    assert x.is_contiguous() is True


def test_slice_matches_numpy():
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((4, 6, 8)).astype(np.float32)
    x = ag.from_numpy(x_np)
    s = ag.slice(x, 1, 2, 5)
    assert s.shape == [4, 3, 8]
    np.testing.assert_array_equal(ag.to_numpy(s), x_np[:, 2:5, :])
    s2 = ag.slice(x, 2, 1, 7)
    np.testing.assert_array_equal(ag.to_numpy(s2), x_np[:, :, 1:7])


def test_slice_squeeze_chain():
    rng = np.random.default_rng(12)
    x_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    x = ag.from_numpy(x_np)
    sliced = ag.slice(x, 1, 1, 2)
    sq = ag.squeeze(sliced, 1)
    assert sq.shape == [2, 4, 5]
    np.testing.assert_array_equal(ag.to_numpy(sq), x_np[:, 1, :, :])


def test_slice_backward_accumulates():
    rng = np.random.default_rng(13)
    x_np = rng.standard_normal((3, 4, 5)).astype(np.float32)
    x = ag.from_numpy(x_np, requires_grad=True)
    a = ag.slice(x, 1, 0, 2)
    b = ag.slice(x, 1, 2, 4)
    out = ag.add(ag.sum_all(ag.mul(a, a)), ag.sum_all(ag.mul(b, b)))
    out.backward()
    np.testing.assert_allclose(ag.to_numpy(x.grad), 2 * x_np, atol=1e-5)


def test_unsqueeze_squeeze_round_trip():
    rng = np.random.default_rng(14)
    x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    x = ag.from_numpy(x_np)
    y = ag.unsqueeze(x, 1)
    assert y.shape == [2, 1, 3, 4]
    np.testing.assert_array_equal(ag.to_numpy(y), x_np[:, None, :, :])
    z = ag.squeeze(y, 1)
    assert z.shape == [2, 3, 4]
    np.testing.assert_array_equal(ag.to_numpy(z), x_np)


def test_contiguous_identity_when_contiguous():
    rng = np.random.default_rng(15)
    x_np = rng.standard_normal((3, 4)).astype(np.float32)
    x = ag.from_numpy(x_np)
    c = ag.contiguous(x)
    assert c.is_contiguous()
    np.testing.assert_array_equal(ag.to_numpy(c), x_np)


def test_contiguous_materializes_strided():
    rng = np.random.default_rng(16)
    x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    x = ag.from_numpy(x_np)
    t = ag.transpose(x, 0, 2)
    assert not t.is_contiguous()
    c = ag.contiguous(t)
    assert c.is_contiguous()
    np.testing.assert_array_equal(ag.to_numpy(c), np.swapaxes(x_np, 0, 2))


def test_bmm_a_bt_with_strided_inputs():
    rng = np.random.default_rng(17)
    B, H, T, dh = 2, 3, 5, 4
    qkv_np = rng.standard_normal((B, T, 3, H, dh)).astype(np.float32)
    qkv = ag.from_numpy(qkv_np)
    q = ag.transpose(ag.squeeze(ag.slice(qkv, 2, 0, 1), 2), 1, 2)
    k = ag.transpose(ag.squeeze(ag.slice(qkv, 2, 1, 2), 2), 1, 2)
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    out = ag.to_numpy(ag.bmm_a_bt(q, k))
    q_np = qkv_np[:, :, 0, :, :].transpose(0, 2, 1, 3)
    k_np = qkv_np[:, :, 1, :, :].transpose(0, 2, 1, 3)
    ref = np.einsum("bhid,bhjd->bhij", q_np, k_np)
    np.testing.assert_allclose(out, ref, atol=1e-5)
