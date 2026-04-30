# aar-vlm

A from-scratch deep learning stack: tensor library, autograd, neural-net layer, and (eventually) a vision-language model — all built up from `cudaMalloc` and `<cuda_runtime.h>`. No PyTorch, no JAX, no cuBLAS.

The repo is split into two Python packages:

- **`aargrad/`** — the framework. Tensor, ops, autograd, `nn`, `optim`. Mirrors what `torch` is to a model.
- **`vlm/`** — the model code that will be built on `aargrad`: pretraining loop, SFT, DPO, model definitions. Currently a placeholder.

The C++/CUDA core lives in `cppsrc/` and is exposed to Python via pybind11 as `aargrad._core`.

---

## Status

| Component                | CPU | CUDA | Autograd |
|--------------------------|:---:|:----:|:--------:|
| `add`, `mul`, `relu`     |  ✓  |   ✓  |     ✓    |
| `sum_all`                |  ✓  |   ✓  |     ✓    |
| `matmul`                 |  ✓  |   ✓  |     ✓    |
| `matmul_a_bt`, `matmul_at_b` | ✓ |   ✓  |    ✓    |
| `softmax` (last-dim)     |  ✓  |   ✓  |     ✓    |
| `layernorm` (last-dim, with γ/β) | ✓ | ✓ |   ✓    |
| `scaled_add_inplace`     |  ✓  |   ✓  | (no — optimizer primitive) |
| `nn.Module`, `Parameter`, `Sequential`, `Linear`, `ReLU` | — | — | — |
| `optim.SGD`              |  —  |   —  |     —    |

End-to-end: a 2-layer MLP trains on CUDA — forward, backward, optimizer step, all on GPU.

---

## Quick start

### Prerequisites

- CMake ≥ 3.24, a C++20 compiler.
- (Optional) CUDA Toolkit ≥ 12.0 for GPU support. CMake autodetects via `check_language(CUDA)`; without it the build silently falls back to a CPU-only build (CUDA ops throw `"CUDA support not built; rebuild with CUDA"` if called).
- Python ≥ 3.10 with `numpy` for the Python bindings.

### Build the C++ core + run C++ tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For an explicit CUDA build (e.g. on a T4):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
ctest --test-dir build --output-on-failure -R Cuda    # CUDA-only tests
```

### Build the Python package + run Python tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . --config-settings cmake.define.BUILD_PYTHON=ON
pytest tests/python/ -v
```

`pip install -e .` triggers `scikit-build-core`, which runs CMake under the hood. Editable installs rebuild on re-import (per `tool.scikit-build.editable.rebuild = true`).

---

## A small tour

### Tensor + ops

```python
import numpy as np
import aargrad as ag

x = ag.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
y = ag.from_numpy(np.ones((2, 3), dtype=np.float32))
z = ag.add(x, y)
print(ag.to_numpy(z))           # [[1,2,3],[4,5,6]]
print(ag.to_numpy(ag.sum_all(z)))  # 21.0

# Move to GPU
x_gpu = x.to(ag.Device.CUDA)
```

### Autograd

```python
a = ag.from_numpy(np.array([1.,2.,3.], dtype=np.float32), requires_grad=True)
b = ag.from_numpy(np.array([4.,5.,6.], dtype=np.float32), requires_grad=True)
loss = ag.sum_all(ag.mul(a, b))
loss.backward()
print(ag.to_numpy(a.grad))  # [4,5,6]  (= b)
print(ag.to_numpy(b.grad))  # [1,2,3]  (= a)
```

The autograd tape lives on `Tensor.grad_fn`; backward is iterative (no recursion → no stack-overflow risk on deep chains, see [test_autograd.cpp](tests/cpp/test_autograd.cpp)).

### Modules and training

```python
import numpy as np
import aargrad as ag
from aargrad import nn
from aargrad.optim import SGD

# Synthetic regression task
rng = np.random.default_rng(0)
X = rng.standard_normal((32, 4)).astype(np.float32)
Y = (np.maximum(X @ rng.standard_normal((4, 8)), 0)
     @ rng.standard_normal((8, 1))).astype(np.float32)

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
model.to(ag.Device.CUDA)             # move all params to GPU in one call
optim = SGD(model.parameters(), lr=0.01)

for step in range(200):
    x = ag.from_numpy(X).to(ag.Device.CUDA)
    y_neg = ag.from_numpy(-Y).to(ag.Device.CUDA)
    pred = model(x)
    loss = ag.sum_all(ag.mul(ag.add(pred, y_neg), ag.add(pred, y_neg)))
    optim.zero_grad()
    loss.backward()
    optim.step()
```

The full version of this training loop is exercised on both CPU and CUDA in [tests/python/test_train_mlp.py](tests/python/test_train_mlp.py).

---

## Architecture

### Tensor and storage

`Tensor` is a value type holding a `shared_ptr<Storage>` plus shape, dtype, device, and three autograd fields (`requires_grad`, `grad_fn`, `grad_slot`). Copying a Tensor shares storage and grad slot; this is what lets `Parameter` wrappers and Module attributes refer to the same logical tensor without manual reference juggling.

Allocation goes through a `Device`-tagged allocator (`cpu_allocator()` / `cuda_allocator()`); CUDA buffers come from `cudaMalloc`.

### Autograd

Tape-based. Each op that returns a Tensor with `requires_grad=true` constructs a `Function` subclass, records its input edges (leaves or upstream `grad_fn`s), saves any tensors needed for backward, and stamps `grad_fn` onto the output tensor.

`Tensor::backward()` runs an **iterative** topological sort over the recorded graph (`run_backward` in [cppsrc/autograd/backward.cpp](cppsrc/autograd/backward.cpp)) — no recursion, so a 10000-op chain doesn't blow the stack. There's a regression test for that exact scenario.

Gradient accumulation across fan-out edges and across multiple `backward()` calls goes through `add_grads`, which routes to a CUDA kernel when both operands live on the GPU.

### CPU/CUDA dispatch

Each op has the same shape: a public function in `cppsrc/ops/<op>.{cpp,hpp}` that does shape checking, autograd graph construction, and dispatches to `*_cpu` (defined in the same .cpp) or `*_cuda` (defined in `<op>.cu`, forward-declared at the top of the .cpp). For non-CUDA builds, `<op>_cuda_stub.cpp` provides linker-satisfying symbols that throw at runtime.

The `if(HAS_CUDA)` block in [CMakeLists.txt](CMakeLists.txt) picks `.cu` vs `_cuda_stub.cpp` per source file.

### `nn.Module` (Python)

Module/Parameter intentionally live in **Python**, not C++ — same as PyTorch. The reason is ergonomics: `__setattr__`-driven auto-registration of parameters and submodules is dramatically cleaner in Python, and the framework's hot path (forward, backward, kernel launches) is already in C++.

`Module.to(device)` rebuilds each `Parameter.tensor` on the new device because `Tensor::to()` returns a fresh tensor with autograd state stripped — `requires_grad` has to be re-set to give it a fresh `grad_slot`. The optimizer holds the `Parameter` *wrapper* (a stable handle), so swapping `parameter.tensor` is transparent to it. There's no "construct optimizer after `.to()`" footgun.

### Optimizer primitives

`SGD.step()` calls `aargrad.scaled_add_inplace(p, grad, -lr)` — a tiny in-place fused-update kernel. No autograd; it writes through `Tensor::data()` and never touches `grad_fn` / `grad_slot`. The next forward builds a fresh autograd graph against the updated parameter values.

---

## Project layout

```
aar-vlm/
├── aargrad/                # Framework package (Python entry point)
│   ├── __init__.py         # Re-exports of _core symbols + nn + optim
│   ├── nn.py               # Module, Parameter, Linear, ReLU, Sequential
│   ├── optim.py            # SGD
│   └── _core.*.so          # pybind11 extension (built into here)
├── vlm/                    # Model-code package (currently empty placeholder)
│   └── __init__.py
├── cppsrc/                 # C++/CUDA source
│   ├── core/               # Tensor, storage, allocators, dtype
│   ├── autograd/           # Function ABI, iterative backward
│   ├── ops/                # add/mul/relu/sum_all + matmul + softmax + layernorm
│   │                       # Each op: <op>.{cpp,cu,_cuda_stub.cpp,hpp}
│   └── python/binding.cpp  # pybind11 module definition
├── tests/
│   ├── cpp/                # gtest binaries (test_tensor, test_autograd_ops, ...)
│   └── python/             # pytest (test_smoke, test_nn, test_train_mlp)
├── CMakeLists.txt
└── pyproject.toml          # scikit-build-core + pybind11
```

---

## Roadmap

Implemented:
- [x] Tensor (CPU + CUDA) with shape, dtype, device-tagged storage
- [x] 6 forward ops with autograd: add, mul, relu, sum_all, matmul (incl. transpose variants), softmax, layernorm
- [x] Iterative tape-based autograd; `loss.backward()` works on CUDA-resident graphs
- [x] Python bindings via pybind11
- [x] `nn.Module`, `Parameter`, `Linear`, `ReLU`, `Sequential`
- [x] `optim.SGD`
- [x] End-to-end MLP training on CPU and CUDA

Tier 2 (next):
- [ ] `add_bias` op → `nn.Linear(bias=True)`
- [ ] `gelu` op + `nn.GELU`
- [ ] `nn.LayerNorm` Module wrapping the existing `layernorm` op (owns γ, β)
- [ ] `optim.AdamW`
- [ ] `nn.Embedding` (gather kernel)
- [ ] `nn.CrossEntropyLoss` (log-softmax + nll)
- [ ] `state_dict` / `load_state_dict`
- [ ] `aargrad.sub` (so the MLP loop drops the `add(-y)` workaround)

Toward a VLM:
- [ ] Causal mask + scaled-dot-product attention (composes existing matmul + softmax)
- [ ] Multi-head attention, transformer block
- [ ] Tokenizer + dataset loading
- [ ] Image patch embedding + vision encoder
- [ ] Cross-modal fusion
- [ ] Pretraining loop in `vlm/`
- [ ] SFT, DPO

---

## Development

### Running tests

C++ (gtest, via ctest):

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure
```

Python (pytest):

```bash
pytest tests/python/ -v
```

Specific test:

```bash
pytest tests/python/test_train_mlp.py::test_mlp_loss_decreases -v
./build/test_autograd_ops --gtest_filter='AutogradOps.*'
```

### CUDA-only tests

CUDA gtest cases use `GTEST_SKIP()` under `#ifndef HAS_CUDA`; they show as `[Skipped]` on CPU-only builds. To run only those on a GPU host:

```bash
ctest --test-dir build --output-on-failure -R Cuda
```

CUDA Python tests use `pytest.skip(...)` after a probe (`Tensor.zeros([1], Fp32, CUDA)` raising → no CUDA).

### Adding a new op

Mirror the existing ops:

1. `cppsrc/ops/<op>.hpp` — public signature.
2. `cppsrc/ops/<op>.cpp` — CPU impl, forward-decl of `<op>_cuda`, autograd `Function` subclass with `backward()`, public function that builds the autograd node.
3. `cppsrc/ops/<op>.cu` — CUDA kernel + host wrapper using the `BLOCK=256` / `grid_for(n)` convention.
4. `cppsrc/ops/<op>_cuda_stub.cpp` — throwing stub for non-CUDA builds.
5. List all four files in [CMakeLists.txt](CMakeLists.txt) under the `if(HAS_CUDA)` / `else` branches.
6. Bind in [cppsrc/python/binding.cpp](cppsrc/python/binding.cpp) with `gil_release()`.
7. Re-export from [aargrad/__init__.py](aargrad/__init__.py).
8. Tests in `tests/cpp/test_<op>.cpp` (CPU, finite-diff backward, CUDA-matches-CPU) and `tests/python/test_<op>.py` if user-facing.

---

## Why build it

I'm writing this to understand the stack from `cudaMalloc` up. The autograd tape, the kernel launch, how storage and gradients connect through a backward pass — none of it really clicks until you've built it yourself. End goal: a VLM with no black boxes left.
