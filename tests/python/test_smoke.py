import numpy as np

import aargrad as ag


def test_enums_present():
    assert ag.Device.CPU is not None
    assert ag.DType.Fp32 is not None


def test_factories():
    z = ag.Tensor.zeros([2, 3], ag.DType.Fp32)
    o = ag.Tensor.ones([2, 3], ag.DType.Fp32)
    assert z.numel == 6
    np.testing.assert_array_equal(ag.to_numpy(z), np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(ag.to_numpy(o), np.ones((2, 3), dtype=np.float32))


def test_from_to_numpy_round_trip():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = ag.from_numpy(x)
    assert t.shape == [2, 3]
    np.testing.assert_array_equal(ag.to_numpy(t), x)


def test_fp16_round_trip():
    x = np.arange(6, dtype=np.float16).reshape(2, 3)
    t = ag.from_numpy(x)
    assert t.dtype == ag.DType.Fp16
    np.testing.assert_array_equal(ag.to_numpy(t), x)


def test_scalar_to_numpy():
    a = ag.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    s = ag.sum_all(a)
    assert ag.to_numpy(s).item() == 6.0


def test_add_backward_grads_are_ones():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((4, 4)).astype(np.float32)
    b_np = rng.standard_normal((4, 4)).astype(np.float32)
    a = ag.from_numpy(a_np, requires_grad=True)
    b = ag.from_numpy(b_np, requires_grad=True)
    c = ag.add(a, b)
    s = ag.sum_all(c)
    s.backward()
    np.testing.assert_allclose(ag.to_numpy(a.grad), np.ones_like(a_np))
    np.testing.assert_allclose(ag.to_numpy(b.grad), np.ones_like(b_np))


def test_explicit_grad_output_backward():
    a_np = np.arange(4, dtype=np.float32).reshape(2, 2)
    b_np = np.arange(4, 8, dtype=np.float32).reshape(2, 2)
    a = ag.from_numpy(a_np, requires_grad=True)
    b = ag.from_numpy(b_np, requires_grad=True)
    c = ag.add(a, b)
    grad = ag.from_numpy(np.full((2, 2), 2.0, dtype=np.float32))
    c.backward(grad)
    expected = np.full((2, 2), 2.0, dtype=np.float32)
    np.testing.assert_allclose(ag.to_numpy(a.grad), expected)
    np.testing.assert_allclose(ag.to_numpy(b.grad), expected)


def test_matmul_runs():
    eye = np.eye(3, dtype=np.float32)
    rhs = np.arange(9, dtype=np.float32).reshape(3, 3)
    a = ag.from_numpy(eye)
    b = ag.from_numpy(rhs)
    c = ag.matmul(a, b)
    np.testing.assert_allclose(ag.to_numpy(c), rhs)
