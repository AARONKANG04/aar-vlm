import math

import numpy as np

from . import _core


class Parameter:
    def __init__(self, tensor):
        tensor.requires_grad = True
        self.tensor = tensor


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.tensor.zero_grad()

    def to(self, device):
        for p in self._parameters.values():
            new_t = p.tensor.to(device)
            new_t.requires_grad = True
            p.tensor = new_t
        for m in self._modules.values():
            m.to(device)
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features, *, bias=False, rng=None):
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        bound = math.sqrt(6.0 / in_features)
        w = rng.uniform(-bound, bound, size=(out_features, in_features)).astype(np.float32)
        self.weight = Parameter(_core.from_numpy(w))
        if bias:
            self.bias = Parameter(_core.from_numpy(np.zeros((out_features,), dtype=np.float32)))
        else:
            self.bias = None

    def forward(self, x):
        y = _core.matmul_a_bt(x, self.weight.tensor)
        if self.bias is not None:
            y = _core.add_bias(y, self.bias.tensor)
        return y


class ReLU(Module):
    def forward(self, x):
        return _core.relu(x)


class GELU(Module):
    def forward(self, x):
        return _core.gelu(x)


class LayerNorm(Module):
    def __init__(self, normalized_features, eps=1e-5):
        super().__init__()
        self.weight = Parameter(_core.from_numpy(np.ones((normalized_features,), dtype=np.float32)))
        self.bias = Parameter(_core.from_numpy(np.zeros((normalized_features,), dtype=np.float32)))
        self.eps = eps

    def forward(self, x):
        return _core.layernorm(x, self.weight.tensor, self.bias.tensor, self.eps)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, f"layer_{i}", layer)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


def scaled_dot_product_attention(q, k, v, *, causal=False):
    d_head = q.shape[-1]
    scores = _core.matmul_a_bt(q, k)
    scores = _core.scale(scores, 1.0 / math.sqrt(d_head))
    if causal:
        scores = _core.apply_causal_mask(scores)
    attn = _core.softmax(scores)
    return _core.matmul(attn, v)
