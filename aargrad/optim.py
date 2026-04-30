from . import _core


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.tensor.zero_grad()

    def step(self):
        for p in self.params:
            g = p.tensor.grad
            if g is None:
                continue
            _core.scaled_add_inplace(p.tensor, g, -self.lr)


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._state = {}

    def zero_grad(self):
        for p in self.params:
            p.tensor.zero_grad()

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for p in self.params:
            g = p.tensor.grad
            if g is None:
                continue
            key = id(p)
            st = self._state.get(key)
            if st is None:
                m = _core.Tensor.zeros(p.tensor.shape, p.tensor.dtype, p.tensor.device)
                v = _core.Tensor.zeros(p.tensor.shape, p.tensor.dtype, p.tensor.device)
                st = (m, v)
                self._state[key] = st
            m, v = st
            _core.adamw_step(p.tensor, g, m, v,
                             self.lr, self.beta1, self.beta2,
                             self.eps, self.weight_decay, bc1, bc2)
