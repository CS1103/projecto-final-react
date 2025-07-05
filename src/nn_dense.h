#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#pragma once
#include "tensor.h"
#include "nn_interfaces.h"
#include <vector>
#include <array>
#include <functional>
#include "nn_loss.h"

namespace utec::neural_network {

    template <typename T>
    class Dense : public LayerBase<T> {
    private:
        utec::algebra::Tensor<T, 2> W;
        utec::algebra::Tensor<T, 1> b;
        utec::algebra::Tensor<T, 2> last_input;
        utec::algebra::Tensor<T, 2> dW;
        utec::algebra::Tensor<T, 1> db;

    public:
        Dense() = default;

        template<typename WInit, typename BInit>
        Dense(size_t in_features, size_t out_features, WInit weight_init, BInit bias_init)
                : W(std::array<size_t,2>{in_features, out_features}),
                  b(std::array<size_t,1>{out_features}),
                  dW(std::array<size_t,2>{in_features, out_features}),
                  db(std::array<size_t,1>{out_features})
        {
            weight_init(W);

            utec::algebra::Tensor<T,2> b2d(std::array<size_t,2>{b.shape()[0], 1});
            for (size_t i = 0; i < b.shape()[0]; ++i)
                b2d(i,0) = b(i);

            bias_init(b2d);

            for (size_t i = 0; i < b.shape()[0]; ++i)
                b(i) = b2d(i,0);
        }

        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& X) override {
            last_input = X;
            auto x_shape = X.shape();
            auto w_shape = W.shape();
            utec::algebra::Tensor<T, 2> result(std::array<size_t,2>{x_shape[0], w_shape[1]});
            for (size_t i = 0; i < x_shape[0]; ++i)
                for (size_t j = 0; j < w_shape[1]; ++j) {
                    T sum = 0;
                    for (size_t k = 0; k < w_shape[0]; ++k)
                        sum += X(i, k) * W(k, j);
                    result(i, j) = sum + b(j);
                }
            return result;
        }

        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad_out) override {
            // dW = last_input^T . grad_out
            auto last_input_T = last_input.transpose(); // (in_features, batch)
            dW = last_input_T.dot(grad_out);           // (in_features, out_features)

            // db = suma de grad_out sobre el batch
            db = utec::algebra::Tensor<T, 1>(std::array<size_t,1>{grad_out.shape()[1]});
            for (size_t j = 0; j < grad_out.shape()[1]; ++j) {
                T sum = 0;
                for (size_t i = 0; i < grad_out.shape()[0]; ++i)
                    sum += grad_out(i, j);
                db(j) = sum;
            }

            // grad_input = grad_out . W^T
            auto W_T = W.transpose(); // (out_features, in_features)
            auto grad_input = grad_out.dot(W_T); // (batch, in_features)
            return grad_input;
        }

        void update_params(T lr) override {
            for (size_t i = 0; i < W.size(); ++i)
                W.begin()[i] -= lr * dW.begin()[i];
            for (size_t i = 0; i < b.size(); ++i)
                b.begin()[i] -= lr * db.begin()[i];
        }
    };

} // namespace utec::neural_network

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H