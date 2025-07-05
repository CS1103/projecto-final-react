//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#pragma once
#include "tensor.h"

namespace utec {
namespace neural_network {

template<typename T>
class LayerBase {
public:
    LayerBase() = default;
    virtual ~LayerBase() = default;
    virtual utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>&) = 0;
    virtual utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>&) = 0;
    virtual void update_params(T /*lr*/) {}
};

} // namespace neural_network
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
