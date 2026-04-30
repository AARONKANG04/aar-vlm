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


def _ce_ref(logits, targets, ignore_index=-100):
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    m = logits.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))).squeeze(-1)
    correct = logits[np.arange(logits.shape[0]), targets.clip(min=0)]
    per_row = lse - correct
    return float(per_row[mask].mean())


def test_cross_entropy_forward_matches_reference():
    rng = np.random.default_rng(0)
    N, V = 6, 5
    logits_np = rng.standard_normal((N, V)).astype(np.float32)
    targets_np = np.array([0, 4, 2, 1, 3, 0], dtype=np.int64)
    logits = ag.from_numpy(logits_np)
    targets = ag.from_numpy(targets_np)
    loss = ag.to_numpy(ag.cross_entropy(logits, targets)).item()
    np.testing.assert_allclose(loss, _ce_ref(logits_np, targets_np), atol=1e-5)


def test_cross_entropy_ignore_index_masks():
    rng = np.random.default_rng(1)
    N, V = 4, 6
    logits_np = rng.standard_normal((N, V)).astype(np.float32)
    targets_np = np.array([2, -100, 1, -100], dtype=np.int64)
    logits = ag.from_numpy(logits_np)
    targets = ag.from_numpy(targets_np)
    loss = ag.to_numpy(ag.cross_entropy(logits, targets, ignore_index=-100)).item()
    np.testing.assert_allclose(loss, _ce_ref(logits_np, targets_np, -100), atol=1e-5)


def test_cross_entropy_backward_finite_diff():
    rng = np.random.default_rng(2)
    N, V = 4, 5
    logits_np = rng.standard_normal((N, V)).astype(np.float32)
    targets_np = np.array([1, 4, 0, 2], dtype=np.int64)
    logits = ag.from_numpy(logits_np, requires_grad=True)
    targets = ag.from_numpy(targets_np)
    ag.cross_entropy(logits, targets).backward()
    dl = ag.to_numpy(logits.grad)

    eps = 1e-3
    dl_num = np.zeros_like(dl)
    for idx in np.ndindex(logits_np.shape):
        lp = logits_np.copy(); lp[idx] += eps
        lm = logits_np.copy(); lm[idx] -= eps
        dl_num[idx] = (_ce_ref(lp, targets_np) - _ce_ref(lm, targets_np)) / (2 * eps)
    np.testing.assert_allclose(dl, dl_num, atol=1e-3)


def test_cross_entropy_module_flattens_BT():
    rng = np.random.default_rng(3)
    B, T, V = 2, 3, 4
    logits_np = rng.standard_normal((B, T, V)).astype(np.float32)
    targets_np = rng.integers(0, V, size=(B, T)).astype(np.int64)
    logits = ag.from_numpy(logits_np)
    targets = ag.from_numpy(targets_np)
    ce = nn.CrossEntropyLoss()
    loss = ag.to_numpy(ce(logits, targets)).item()
    np.testing.assert_allclose(
        loss,
        _ce_ref(logits_np.reshape(B * T, V), targets_np.reshape(B * T)),
        atol=1e-5,
    )


def test_toy_lm_loss_decreases():
    rng = np.random.default_rng(4)
    V, D, T, B = 12, 8, 6, 3
    ids_np = rng.integers(0, V, size=(B, T + 1)).astype(np.int64)
    inputs_np = ids_np[:, :-1]
    targets_np = ids_np[:, 1:]

    model = nn.Sequential(
        nn.Embedding(V, D, rng=np.random.default_rng(5)),
        nn.Linear(D, V, bias=True, rng=np.random.default_rng(6)),
    )
    ce = nn.CrossEntropyLoss()
    optim = SGD(list(model.parameters()), lr=0.1)

    losses = []
    for _ in range(80):
        x = ag.from_numpy(inputs_np)
        y = ag.from_numpy(targets_np)
        logits = model(x)
        loss = ce(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(ag.to_numpy(loss).item())
    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_cross_entropy_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(7)
    N, V = 8, 7
    logits_np = rng.standard_normal((N, V)).astype(np.float32)
    targets_np = np.array([0, 1, -100, 3, 4, -100, 2, 6], dtype=np.int64)
    logits_cpu = ag.from_numpy(logits_np)
    logits_gpu = ag.from_numpy(logits_np).to(ag.Device.CUDA)
    targets_cpu = ag.from_numpy(targets_np)
    targets_gpu = ag.from_numpy(targets_np).to(ag.Device.CUDA)
    cpu_loss = ag.to_numpy(ag.cross_entropy(logits_cpu, targets_cpu)).item()
    gpu_loss = ag.to_numpy(
        ag.cross_entropy(logits_gpu, targets_gpu).to(ag.Device.CPU)
    ).item()
    np.testing.assert_allclose(cpu_loss, gpu_loss, atol=1e-5)
