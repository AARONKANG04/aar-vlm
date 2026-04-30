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


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, *, rng=None):
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        w = rng.standard_normal((num_embeddings, embedding_dim)).astype(np.float32)
        self.weight = Parameter(_core.from_numpy(w))

    def forward(self, ids):
        return _core.embedding(self.weight.tensor, ids)


class CrossEntropyLoss(Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if len(logits.shape) > 2:
            *lead, V = logits.shape
            N = 1
            for d in lead:
                N *= d
            logits = _core.reshape(logits, [N, V])
            targets = _core.reshape(targets, [N])
        return _core.cross_entropy(logits, targets, self.ignore_index)


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


class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads, *, causal=False, bias=True, rng=None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        rng = rng if rng is not None else np.random.default_rng()
        self.qkv_proj = Linear(d_model, 3 * d_model, bias=bias, rng=rng)
        self.out_proj = Linear(d_model, d_model, bias=bias, rng=rng)

    def _take(self, qkv, idx):
        s = _core.slice(qkv, 2, idx, idx + 1)
        s = _core.squeeze(s, 2)
        return _core.transpose(s, 1, 2)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = _core.reshape(qkv, [B, T, 3, self.n_heads, self.d_head])
        q = self._take(qkv, 0)
        k = self._take(qkv, 1)
        v = self._take(qkv, 2)
        scores = _core.scale(_core.bmm_a_bt(q, k), 1.0 / math.sqrt(self.d_head))
        if self.causal:
            scores = _core.apply_causal_mask(scores)
        attn = _core.softmax(scores)
        out = _core.transpose(_core.bmm(attn, v), 1, 2)
        out = _core.reshape(out, [B, T, self.d_model])
        return self.out_proj(out)
