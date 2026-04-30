#include "autograd/backward.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "autograd/function.hpp"
#include "ops/shape.hpp"

namespace vlm {
    Tensor add_cuda(const Tensor& a, const Tensor& b);

    namespace {
        Tensor add_grads(const Tensor& a, const Tensor& b) {
            if (a.shape != b.shape) {
                throw std::runtime_error("autograd: grad shape mismatch");
            }
            if (a.device != b.device) {
                throw std::runtime_error("autograd: grad device mismatch");
            }
            Tensor ac = a.is_contiguous() ? a : contiguous(a);
            Tensor bc = b.is_contiguous() ? b : contiguous(b);
            if (ac.device == Device::CUDA) {
                return add_cuda(ac, bc);
            }
            Tensor out = Tensor::empty(ac.shape, ac.dtype, ac.device);
            const float* ap = static_cast<const float*>(ac.data());
            const float* bp = static_cast<const float*>(bc.data());
            float* op = static_cast<float*>(out.data());
            const size_t n = ac.numel();
            for (size_t i = 0; i < n; ++i) op[i] = ap[i] + bp[i];
            return out;
        }

        void accumulate_into_leaf(Tensor& leaf, const Tensor& grad) {
            if (!leaf.grad_slot) return;
            auto& slot = *leaf.grad_slot;
            if (slot) {
                slot = std::make_shared<Tensor>(add_grads(*slot, grad));
            } else {
                slot = std::make_shared<Tensor>(
                    grad.is_contiguous() ? grad : contiguous(grad));
            }
        }
    }

    void run_backward(const Tensor& root, const Tensor& grad_output) {
        if (!root.grad_fn) return;

        struct Frame {
            std::shared_ptr<Function> fn;
            bool expanded;
        };
        std::vector<Frame> stack;
        std::vector<std::shared_ptr<Function>> order;
        std::unordered_set<Function*> visited;

        stack.push_back({root.grad_fn, false});
        while (!stack.empty()) {
            Frame& top = stack.back();
            if (!top.expanded) {
                if (visited.count(top.fn.get())) {
                    stack.pop_back();
                    continue;
                }
                visited.insert(top.fn.get());
                top.expanded = true;
                for (const auto& edge : top.fn->inputs) {
                    if (edge.grad_fn && !visited.count(edge.grad_fn.get())) {
                        stack.push_back({edge.grad_fn, false});
                    }
                }
            } else {
                order.push_back(std::move(top.fn));
                stack.pop_back();
            }
        }
        std::reverse(order.begin(), order.end());

        std::unordered_map<Function*, Tensor> grads;
        grads[root.grad_fn.get()] = grad_output;

        for (auto& fn : order) {
            auto it = grads.find(fn.get());
            if (it == grads.end()) continue;
            Tensor grad_at = std::move(it->second);
            grads.erase(it);

            std::vector<Tensor> input_grads = fn->backward(grad_at);
            if (input_grads.size() != fn->inputs.size()) {
                throw std::runtime_error("autograd: backward returned wrong number of grads");
            }

            for (size_t i = 0; i < fn->inputs.size(); ++i) {
                auto& edge = fn->inputs[i];
                if (!edge.needs_grad) continue;
                const Tensor& g = input_grads[i];
                if (edge.grad_fn) {
                    auto it2 = grads.find(edge.grad_fn.get());
                    if (it2 == grads.end()) {
                        grads[edge.grad_fn.get()] = g;
                    } else {
                        it2->second = add_grads(it2->second, g);
                    }
                } else {
                    accumulate_into_leaf(edge.leaf, g);
                }
            }
        }
    }
}
