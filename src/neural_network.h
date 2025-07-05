#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "tensor.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include <vector>
#include <memory>
#include <type_traits>  // Para std::is_same_v

namespace utec::neural_network {

    template <typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<LayerBase<T>>> layers;

    public:
        template <typename Layer>
        void add_layer(std::unique_ptr<Layer> layer) {
            layers.push_back(std::move(layer));
        }

        utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
            utec::algebra::Tensor<T, 2> out = X;
            for (auto& layer : layers)
                out = layer->forward(out);
            return out;
        }

        template <template <typename> class Loss>
        void train(const utec::algebra::Tensor<T, 2>& X,
                   const utec::algebra::Tensor<T, 2>& Y,
                   size_t epochs,
                   size_t batch_size,
                   T lr) {
            // Instancia del tipo de pérdida con T como tipo interno
            using LossType = Loss<T>;

            // Validación en tiempo de compilación
            static_assert(std::is_same_v<LossType, MSELoss<T>> ||
                          std::is_same_v<LossType, BCELoss<T>>,
                          "Loss debe ser MSELoss<T> o BCELoss<T>");

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                utec::algebra::Tensor<T, 2> out = X;
                for (auto& layer : layers) {
                    out = layer->forward(out);
                }

                LossType loss_fn(out, Y);
                T loss = loss_fn.loss();
                auto grad = loss_fn.loss_gradient();

                for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    grad = (*it)->backward(grad);
                }

                for (auto& layer : layers) {
                    layer->update_params(lr);
                }
            }
        }
    };

} // namespace utec::neural_network

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
