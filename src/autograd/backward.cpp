#include "autograd/backward.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "autograd/function.hpp"

namespace vlm {
    namespace {
        Tensor add_grads(const Tensor& a, const Tensor& b) {
            if (a.shape != b.shape) {
                throw std::runtime_error("autograd: grad shape mismatch");
            }
            if (a.device != b.device) {
                throw std::runtime_error("autograd: grad device mismatch");
            }
            Tensor out = Tensor::empty(a.shape, a.dtype, a.device);
            if (a.device == Device::CPU) {
                const float* ap = static_cast<const float*>(a.data());
                const float* bp = static_cast<const float*>(b.data());
                float* op = static_cast<float*>(out.data());
                const size_t n = a.numel();
                for (size_t i = 0; i < n; ++i) op[i] = ap[i] + bp[i];
            } else {
                throw std::runtime_error("autograd: CUDA grad accumulation not yet supported");
            }
            return out;
        }

        void accumulate_into_leaf(Tensor& leaf, const Tensor& grad) {
            if (!leaf.grad_slot) return;
            auto& slot = *leaf.grad_slot;
            if (slot) {
                slot = std::make_shared<Tensor>(add_grads(*slot, grad));
            } else {
                slot = std::make_shared<Tensor>(grad);
            }
        }
    }

    void run_backward(const Tensor& root, const Tensor& grad_output) {
        if (!root.grad_fn) return;

        std::vector<std::shared_ptr<Function>> order;
        std::unordered_set<Function*> visited;
        std::function<void(const std::shared_ptr<Function>&)> dfs =
            [&](const std::shared_ptr<Function>& fn) {
                if (visited.count(fn.get())) return;
                visited.insert(fn.get());
                for (const auto& edge : fn->inputs) {
                    if (edge.grad_fn) dfs(edge.grad_fn);
                }
                order.push_back(fn);
            };
        dfs(root.grad_fn);
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
