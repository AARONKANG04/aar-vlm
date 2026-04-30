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
