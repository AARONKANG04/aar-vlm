#include "autograd/function.hpp"

namespace vlm {
    void Function::record_input(const Tensor& t) {
        Input edge;
        edge.needs_grad = t.requires_grad;
        if (t.requires_grad) {
            if (t.grad_fn) {
                edge.grad_fn = t.grad_fn;
            } else {
                edge.leaf = t;
            }
        }
        inputs.push_back(std::move(edge));
    }

    bool any_requires_grad(std::initializer_list<const Tensor*> tensors) {
        for (const Tensor* t : tensors) {
            if (t->requires_grad) return true;
        }
        return false;
    }
}
