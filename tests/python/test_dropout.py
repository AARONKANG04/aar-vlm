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


def test_dropout_drop_rate_close_to_p():
    ag.manual_seed(0)
    n = 20000
    x_np = np.ones((n,), dtype=np.float32)
    x = ag.from_numpy(x_np)
    out = ag.to_numpy(ag.dropout(x, 0.3))
    drop_frac = float((out == 0.0).mean())
    assert abs(drop_frac - 0.3) < 0.02, f"expected ~0.3, got {drop_frac:.4f}"


def test_dropout_survivors_are_rescaled():
    ag.manual_seed(1)
    p = 0.4
    x_np = np.full((1000,), 2.0, dtype=np.float32)
    x = ag.from_numpy(x_np)
    out = ag.to_numpy(ag.dropout(x, p))
    survivors = out[out != 0.0]
    expected = 2.0 / (1.0 - p)
    np.testing.assert_allclose(survivors, expected, atol=1e-5)


def test_dropout_reproducible_with_manual_seed():
    x = ag.from_numpy(np.random.default_rng(0).standard_normal((50,)).astype(np.float32))
    ag.manual_seed(123)
    a = ag.to_numpy(ag.dropout(x, 0.5))
    ag.manual_seed(123)
    b = ag.to_numpy(ag.dropout(x, 0.5))
    np.testing.assert_array_equal(a, b)


def test_dropout_consecutive_calls_differ():
    ag.manual_seed(7)
    x = ag.from_numpy(np.ones((200,), dtype=np.float32))
    a = ag.to_numpy(ag.dropout(x, 0.5))
    b = ag.to_numpy(ag.dropout(x, 0.5))
    assert not np.array_equal(a, b)


def test_dropout_backward_matches_mask():
    ag.manual_seed(42)
    x_np = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    x = ag.from_numpy(x_np, requires_grad=True)
    ag.manual_seed(42)
    out = ag.dropout(x, 0.4)
    out_np = ag.to_numpy(out)
    ag.sum_all(out).backward()
    grad = ag.to_numpy(x.grad)
    expected_mask = np.where(out_np != 0.0, 1.0 / (1.0 - 0.4), 0.0).astype(np.float32)
    np.testing.assert_allclose(grad, expected_mask, atol=1e-5)


def test_dropout_zero_p_is_identity():
    x_np = np.random.default_rng(0).standard_normal((3, 5)).astype(np.float32)
    x = ag.from_numpy(x_np, requires_grad=True)
    out = ag.dropout(x, 0.0)
    np.testing.assert_array_equal(ag.to_numpy(out), x_np)
    ag.sum_all(out).backward()
    np.testing.assert_array_equal(ag.to_numpy(x.grad), np.ones_like(x_np))


def test_module_dropout_eval_passthrough():
    drop = nn.Dropout(0.5)
    x_np = np.random.default_rng(0).standard_normal((100,)).astype(np.float32)
    x = ag.from_numpy(x_np)
    drop.eval()
    out = ag.to_numpy(drop(x))
    np.testing.assert_array_equal(out, x_np)


def test_module_train_propagates_to_children():
    block = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 4))
    block.eval()
    for m in block._modules.values():
        assert m.training is False
    block.train()
    for m in block._modules.values():
        assert m.training is True


def test_module_dropout_train_actually_drops():
    ag.manual_seed(99)
    drop = nn.Dropout(0.5)
    drop.train()
    x = ag.from_numpy(np.ones((1000,), dtype=np.float32))
    out = ag.to_numpy(drop(x))
    drop_frac = float((out == 0.0).mean())
    assert 0.4 < drop_frac < 0.6


def test_dropout_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    x_np = np.random.default_rng(0).standard_normal((128,)).astype(np.float32)
    ag.manual_seed(2026)
    cpu_out = ag.to_numpy(ag.dropout(ag.from_numpy(x_np), 0.3))
    ag.manual_seed(2026)
    gpu_out = ag.to_numpy(
        ag.dropout(ag.from_numpy(x_np).to(ag.Device.CUDA), 0.3).to(ag.Device.CPU)
    )
    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-6)
