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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mlp_loss_decreases(device):
    if device == "cuda" and not _has_cuda():
        pytest.skip("CUDA not available")
    dev = ag.Device.CUDA if device == "cuda" else ag.Device.CPU

    rng = np.random.default_rng(0)
    A_true = rng.standard_normal((4, 8)).astype(np.float32)
    B_true = rng.standard_normal((8, 1)).astype(np.float32)
    X = rng.standard_normal((32, 4)).astype(np.float32)
    Y = np.maximum(X @ A_true, 0) @ B_true

    model = nn.Sequential(
        nn.Linear(4, 8, rng=np.random.default_rng(1)),
        nn.ReLU(),
        nn.Linear(8, 1, rng=np.random.default_rng(2)),
    )
    model.to(dev)
    optim = SGD(list(model.parameters()), lr=0.01)

    losses = []
    for _ in range(200):
        x = ag.from_numpy(X).to(dev)
        y_neg = ag.from_numpy(-Y).to(dev)
        pred = model(x)
        diff = ag.add(pred, y_neg)
        loss = ag.sum_all(ag.mul(diff, diff))
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(ag.to_numpy(loss.to(ag.Device.CPU)).item())

    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
