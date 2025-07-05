//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#pragma once
#include "tensor.h"
#include <cmath>
#include "nn_interfaces.h"
#include "nn_loss.h"

namespace utec {
namespace neural_network {

template <typename T>
class ReLU : public LayerBase<T> {
    utec::algebra::Tensor<T,2> last_input;
public:
    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
        last_input = x;
        auto y = x;
        for (auto& v : y)
            v = std::max(static_cast<T>(0), v);
        return y;
    }
    utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& grad_out) override {
        Tensor<T, 2> grad_input(grad_out.shape());
        for (size_t i = 0; i < grad_out.size(); ++i) {
            grad_input.begin()[i] = (last_input.begin()[i] > 0) ? grad_out.begin()[i] : 0;
        }
        return grad_input;
    }
};

template <typename T>
class Sigmoid : public LayerBase<T> {
    utec::algebra::Tensor<T,2> last_output;
public:
    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
        auto y = x;
        for (auto& v : y)
            v = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-v));
        last_output = y;
        return y;
    }
    utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& grad) override {
        auto g = grad;
        for (size_t i = 0; i < g.size(); ++i)
            g.begin()[i] = grad.begin()[i] * last_output.begin()[i] * (1 - last_output.begin()[i]);
        return g;
    }
};

} // namespace neural_network
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H