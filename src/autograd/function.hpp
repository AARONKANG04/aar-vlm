#pragma once

#include <memory>
#include <vector>

#include "core/tensor.hpp"

namespace vlm {
    class Function {
    public:
        struct Input {
            std::shared_ptr<Function> grad_fn;
            Tensor leaf;
            bool needs_grad = false;
        };

        std::vector<Input> inputs;

        virtual ~Function() = default;
        virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;
        virtual const char* name() const { return "Function"; }

        void record_input(const Tensor& t);
    };

    bool any_requires_grad(std::initializer_list<const Tensor*> tensors);
}
