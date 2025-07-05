//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#pragma once
#include "tensor.h"
#include <vector>
#include <cmath>

namespace utec::neural_network {

template<typename T = float>
class SGD {
    T lr;
public:
    SGD(T lr) : lr(lr) {}
    template<typename Tensor>
    void update(Tensor& W, const Tensor& dW) {
        for (size_t i = 0; i < W.size(); ++i)
            W.begin()[i] -= lr * dW.begin()[i];
    }
};

template<typename T = float>
class Adam {
    T lr, beta1, beta2;
    std::vector<T> m, v;
    int t = 0;
public:
    Adam(T lr, T beta1 = 0.9, T beta2 = 0.999)
        : lr(lr), beta1(beta1), beta2(beta2) {}
    template<typename Tensor>
    void update(Tensor& W, const Tensor& dW) {
        if (m.size() != W.size()) {
            m.assign(W.size(), T{});
            v.assign(W.size(), T{});
            t = 0;
        }
        ++t;
        for (size_t i = 0; i < W.size(); ++i) {
            m[i] = beta1 * m[i] + (1 - beta1) * dW.begin()[i];
            v[i] = beta2 * v[i] + (1 - beta2) * dW.begin()[i] * dW.begin()[i];
            T m_hat = m[i] / (1 - std::pow(beta1, t));
            T v_hat = v[i] / (1 - std::pow(beta2, t));
            W.begin()[i] -= lr * m_hat / (std::sqrt(v_hat) + 1e-8);
        }
    }
};

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
