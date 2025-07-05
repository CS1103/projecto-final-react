//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <functional>
#include <iomanip>

namespace utec {
    namespace algebra {

        

        template <typename T, size_t Rank>
        class Tensor {
        public:
            using Shape = std::array<size_t, Rank>;

            auto begin() { return data_.begin(); }
            auto end() { return data_.end(); }
            auto begin() const { return data_.begin(); }
            auto end() const { return data_.end(); }
            auto cbegin() const { return data_.cbegin(); }
            auto cend() const { return data_.cend(); }



        private:
            Shape shape_;
            Shape strides_;  // Nuevo: strides precalculados
            std::vector<T> data_;

            // Calcula strides a partir de la forma
            static Shape compute_strides(const Shape& shape) {
                Shape strides;
                strides[Rank - 1] = 1;
                // Cambiar int por size_t
                for (size_t i = Rank - 2; i < Rank; --i) {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                return strides;
            }


            size_t total_size() const {
        size_t sz = 1;
        for (auto d : shape_) sz *= d;
        return sz;
    }

        public:
            Tensor() = default;
            // ...dentro de Tensor<T, Rank>...
            template <typename... Dims>
            explicit Tensor(Dims... dims) {
                static_assert(sizeof...(Dims) == Rank, "Incorrect number of dimensions");
                shape_ = make_shape(dims...);
                size_t total_size = std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
                data_.resize(total_size);
                strides_ = compute_strides(shape_);
            }

            Tensor& operator=(std::initializer_list<T> values) {
                // Reutilizar inicialización interna
                if (values.size() > data_.size()) {
                    throw std::invalid_argument("Number of elements in initializer_list exceeds tensor capacity.");
                }

                data_.assign(values.begin(), values.end());
                return *this;
            }

            // Constructor por forma
            Tensor(const Shape& shape) : shape_(shape), strides_(compute_strides(shape)) {
                size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
                data_.resize(total_size);
            }

            size_t size() const { return data_.size(); }


            void fill(const T& value) {
                std::fill(data_.begin(), data_.end(), value);
            }

        
            void reshape(const Shape& new_shape) {
                size_t new_total_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
                data_.resize(new_total_size); // Rellena con ceros si es mayor, descarta si es menor
                shape_ = new_shape;
                strides_ = compute_strides(new_shape);
            }

            template <typename... Dims>
            void reshape(Dims... dims) {
                if (sizeof...(Dims) != Rank) {
                    throw std::invalid_argument("Number of dimensions do not match with 2");
                }
                Shape new_shape = make_shape(dims...);
                size_t new_total_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
                data_.resize(new_total_size); // Igual que arriba
                shape_ = new_shape;
                strides_ = compute_strides(shape_);
            }
            template <typename... Dims>
            static std::array<size_t, Rank> make_shape(Dims... dims) {
                static_assert(sizeof...(Dims) == Rank, "Incorrect number of dimensions");
                return std::array<size_t, Rank>{static_cast<size_t>(dims)...};
}
// ...existing code...

            



            // Copia/movimiento
            Tensor(const Tensor& other) = default;
            Tensor(Tensor&& other) = default;
            Tensor& operator=(const Tensor& other) = default;
            Tensor& operator=(Tensor&& other) = default;

            const Shape& shape() const { return shape_; }

            size_t flatten_index(const std::array<size_t, Rank>& idxs) const {
                size_t index = 0;
                for (size_t i = 0; i < Rank; ++i) {
                    index += idxs[i] * strides_[i];
                }
                return index;
            }

            // Acceso constante
            template <typename... Idxs>
            const T& operator()(Idxs... idxs) const {
                static_assert(sizeof...(Idxs) == Rank, "Incorrect number of indices");
                std::array<size_t, Rank> idx_array{static_cast<size_t>(idxs)...};
                return data_[flatten_index(idx_array)];
            }

            // Acceso no constante
            template <typename... Idxs>
            T& operator()(Idxs... idxs) {
                static_assert(sizeof...(Idxs) == Rank, "Incorrect number of indices");
                std::array<size_t, Rank> idx_array{static_cast<size_t>(idxs)...};
                return data_[flatten_index(idx_array)];
            }

            // Acceso con array directamente
            T& operator()(const std::array<size_t, Rank>& idxs) {
                return data_[flatten_index(idxs)];
            }

            const T& operator()(const std::array<size_t, Rank>& idxs) const {
                return data_[flatten_index(idxs)];
            }

            std::vector<T>& data() { return data_; }
            const std::vector<T>& data() const { return data_; }


            Tensor<T, 2> transpose() const {
                const auto& s = this->shape();
                Tensor<T, 2> result(std::array<size_t, 2>{s[1], s[0]});
                for (size_t i = 0; i < s[0]; ++i)
                    for (size_t j = 0; j < s[1]; ++j)
                        result(j, i) = (*this)(i, j);
                return result;
            }

            Tensor<T, 2> dot(const Tensor<T, 2>& other) const {
                auto s1 = this->shape();
                auto s2 = other.shape();
                if (s1[1] != s2[0])
                    throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
                Tensor<T, 2> result(std::array<size_t, 2>{s1[0], s2[1]});
                for (size_t i = 0; i < s1[0]; ++i) {
                    for (size_t j = 0; j < s2[1]; ++j) {
                        T sum = 0;
                        for (size_t k = 0; k < s1[1]; ++k) {
                            sum += (*this)(i, k) * other(k, j);
                        }
                        result(i, j) = sum;
                    }
                }
                return result;
            }
        };


        template <typename T, size_t Rank>
        Tensor<T, Rank> operator*(const Tensor<T, Rank>& lhs, const Tensor<T, Rank>& rhs) {
            auto shape1 = lhs.shape();
            auto shape2 = rhs.shape();

            // Calcula la shape resultante con broadcasting
            std::array<size_t, Rank> result_shape;
            for (size_t i = 0; i < Rank; ++i) {
                if (shape1[i] == shape2[i]) result_shape[i] = shape1[i];
                else if (shape1[i] == 1) result_shape[i] = shape2[i];
                else if (shape2[i] == 1) result_shape[i] = shape1[i];
                else throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }

            Tensor<T, Rank> result(result_shape);

            std::array<size_t, Rank> idx{};
            std::function<void(size_t)> recur = [&](size_t dim) {
                if (dim == Rank) {
                    std::array<size_t, Rank> idx1, idx2;
                    for (size_t d = 0; d < Rank; ++d) {
                        idx1[d] = (shape1[d] == 1) ? 0 : idx[d];
                        idx2[d] = (shape2[d] == 1) ? 0 : idx[d];
                    }
                    result(idx) = lhs(idx1) * rhs(idx2);
                } else {
                    for (size_t i = 0; i < result_shape[dim]; ++i) {
                        idx[dim] = i;
                        recur(dim + 1);
                    }
                }
            };
            recur(0);
            return result;
        }


// Operaciones tensor-escalar y escalar-tensor
        template <typename T, size_t Rank>
        Tensor<T, Rank> operator+(const Tensor<T, Rank>& t, const T& scalar) {
            Tensor<T, Rank> result = t;
            for (auto& v : result.data()) v += scalar;
            return result;
        }
        template <typename T, size_t Rank>
        Tensor<T, Rank> operator+(const T& scalar, const Tensor<T, Rank>& t) {
            return t + scalar;
        }
        template <typename T, size_t Rank>
        Tensor<T, Rank> operator-(const Tensor<T, Rank>& t, const T& scalar) {
            Tensor<T, Rank> result = t;
            for (auto& v : result.data()) v -= scalar;
            return result;
        }
        template <typename T, size_t Rank>
        Tensor<T, Rank> operator-(const T& scalar, const Tensor<T, Rank>& t) {
            Tensor<T, Rank> result = t;
            for (auto& v : result.data()) v = scalar - v;
            return result;
        }
        template <typename T, size_t Rank>
        Tensor<T, Rank> operator*(const Tensor<T, Rank>& t, const T& scalar) {
            Tensor<T, Rank> result = t;
            for (auto& v : result.data()) v *= scalar;
            return result;
        }
        template <typename T, size_t Rank>
        Tensor<T, Rank> operator*(const T& scalar, const Tensor<T, Rank>& t) {
            return t * scalar;
        }
        template <typename T, size_t Rank>
        Tensor<T, Rank> operator/(const Tensor<T, Rank>& t, const T& scalar) {
            Tensor<T, Rank> result = t;
            for (auto& v : result.data()) v /= scalar;
            return result;
        }
        template <typename T, size_t Rank>
        Tensor<T, Rank> operator/(const T& scalar, const Tensor<T, Rank>& t) {
            Tensor<T, Rank> result = t;
            for (auto& v : result.data()) v = scalar / v;
            return result;
        }
        // Comparación
        template <typename T, size_t Rank>
        bool operator==(const Tensor<T, Rank>& lhs, const Tensor<T, Rank>& rhs) {
            return lhs.shape() == rhs.shape() && lhs.data() == rhs.data();
        }

        // Operador suma
        template <typename T, size_t Rank>
        Tensor<T, Rank> operator+(const Tensor<T, Rank>& lhs, const Tensor<T, Rank>& rhs) {
            auto shape1 = lhs.shape();
            auto shape2 = rhs.shape();

            // Calcula la shape resultante con broadcasting
            std::array<size_t, Rank> result_shape;
            for (size_t i = 0; i < Rank; ++i) {
                if (shape1[i] == shape2[i]) result_shape[i] = shape1[i];
                else if (shape1[i] == 1) result_shape[i] = shape2[i];
                else if (shape2[i] == 1) result_shape[i] = shape1[i];
                else throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }

            Tensor<T, Rank> result(result_shape);

            std::array<size_t, Rank> idx{};
            std::function<void(size_t)> recur = [&](size_t dim) {
                if (dim == Rank) {
                    std::array<size_t, Rank> idx1, idx2;
                    for (size_t d = 0; d < Rank; ++d) {
                        idx1[d] = (shape1[d] == 1) ? 0 : idx[d];
                        idx2[d] = (shape2[d] == 1) ? 0 : idx[d];
                    }
                    result(idx) = lhs(idx1) + rhs(idx2);
                } else {
                    for (size_t i = 0; i < result_shape[dim]; ++i) {
                        idx[dim] = i;
                        recur(dim + 1);
                    }
                }
            };
            recur(0);
            return result;
        }

        template <typename T, size_t Rank>
        Tensor<T, Rank> operator-(const Tensor<T, Rank>& lhs, const Tensor<T, Rank>& rhs) {
            auto shape1 = lhs.shape();
            auto shape2 = rhs.shape();

            // Calcula la shape resultante con broadcasting
            std::array<size_t, Rank> result_shape;
            for (size_t i = 0; i < Rank; ++i) {
                if (shape1[i] == shape2[i]) result_shape[i] = shape1[i];
                else if (shape1[i] == 1) result_shape[i] = shape2[i];
                else if (shape2[i] == 1) result_shape[i] = shape1[i];
                else throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }

            Tensor<T, Rank> result(result_shape);

            std::array<size_t, Rank> idx{};
            std::function<void(size_t)> recur = [&](size_t dim) {
                if (dim == Rank) {
                    std::array<size_t, Rank> idx1, idx2;
                    for (size_t d = 0; d < Rank; ++d) {
                        idx1[d] = (shape1[d] == 1) ? 0 : idx[d];
                        idx2[d] = (shape2[d] == 1) ? 0 : idx[d];
                    }
                    result(idx) = lhs(idx1) - rhs(idx2);
                } else {
                    for (size_t i = 0; i < result_shape[dim]; ++i) {
                        idx[dim] = i;
                        recur(dim + 1);
                    }
                }
            };
            recur(0);
            return result;
        }

        // Impresión recursiva
        namespace detail {
            template <typename T, size_t Rank>
            void print_recursive(std::ostream& os, const Tensor<T, Rank>& tensor,
                                 const typename Tensor<T, Rank>::Shape& shape,
                                 size_t dim, std::array<size_t, Rank>& index, bool is_top = false) {
                if (dim == Rank - 1) {
                    for (size_t i = 0; i < shape[dim]; ++i) {
                        index[dim] = i;
                        os << tensor(index);
                        if (i + 1 != shape[dim]) os << " ";
                    }
                } else {
                    os << "{\n";
                    for (size_t i = 0; i < shape[dim]; ++i) {
                        index[dim] = i;
                        print_recursive(os, tensor, shape, dim + 1, index, false);
                        os << "\n";
                    }
                    os << "}";
                    if (is_top) os << "\n";
                }
            }
        }


        template <typename T, size_t Rank>
        std::ostream& operator<<(std::ostream& os, const Tensor<T, Rank>& tensor) {
            typename Tensor<T, Rank>::Shape shape = tensor.shape();
            std::array<size_t, Rank> index{};
            detail::print_recursive(os, tensor, shape, 0, index, true);
            return os;
        }

        // Transposición 2D
        template <typename T, size_t Rank, typename = std::enable_if_t<Rank == 2>>
        Tensor<T, 2> transpose_2d(const Tensor<T, Rank>& tensor) {
            auto shape = tensor.shape();
            Tensor<T, 2> result(std::array<size_t, 2>{shape[1], shape[0]});
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    result(j, i) = tensor(i, j);
                }
            }
            return result;
        }

        template <typename T>
        Tensor<T, 2> transpose_2d(const Tensor<T, 1>&) {
            throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
        }

        // Transposición 2D para Rank > 2
        template <typename T, size_t Rank>
        std::enable_if_t<(Rank > 2), Tensor<T, Rank>>
        transpose_2d(const Tensor<T, Rank>& tensor) {
            auto shape = tensor.shape();
            auto new_shape = shape;
            std::swap(new_shape[Rank - 2], new_shape[Rank - 1]);
            Tensor<T, Rank> result(new_shape);

            std::array<size_t, Rank> idx_src{}, idx_dst{};
            std::function<void(size_t)> recur = [&](size_t dim) {
                if (dim == Rank - 2) {
                    for (size_t i = 0; i < shape[Rank - 2]; ++i) {
                        for (size_t j = 0; j < shape[Rank - 1]; ++j) {
                            idx_src[Rank - 2] = i;
                            idx_src[Rank - 1] = j;
                            idx_dst[Rank - 2] = j;
                            idx_dst[Rank - 1] = i;
                            result(idx_dst) = tensor(idx_src);
                        }
                    }
                } else {
                    for (size_t k = 0; k < shape[dim]; ++k) {
                        idx_src[dim] = idx_dst[dim] = k;
                        recur(dim + 1);
                    }
                }
            };
            recur(0);
            return result;
        }



// Producto matricial Rank == 2
        template <typename T>
        Tensor<T, 2> matrix_product(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
            auto shape1 = A.shape();
            auto shape2 = B.shape();

            if (shape1[1] != shape2[0]) {
                throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
            }

            Tensor<T, 2> result(std::array<size_t, 2>{shape1[0], shape2[1]});

            for (size_t i = 0; i < shape1[0]; ++i) {
                for (size_t j = 0; j < shape2[1]; ++j) {
                    T sum = T();
                    for (size_t k = 0; k < shape1[1]; ++k) {
                        sum += A(i, k) * B(k, j);
                    }
                    result(i, j) = sum;
                }
            }

            return result;
        }

        // Producto matricial generalizado para Rank >= 3
        template <typename T, size_t Rank>
        Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B) {
            auto shape1 = A.shape();
            auto shape2 = B.shape();

            // Verifica batch dimensions
            for (size_t i = 0; i + 2 < Rank; ++i) {
                if (shape1[i] != shape2[i]) {
                    // Si las dimensiones de matrices son compatibles, lanza el mensaje especial
                    if (shape1[Rank - 1] == shape2[Rank - 2]) {
                        throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
                    } else {
                        throw std::invalid_argument("Shapes do not match");
                    }
                }
            }

            // Verifica dimensiones de multiplicación de matrices
            if (shape1[Rank - 1] != shape2[Rank - 2]) {
                throw std::invalid_argument("Shapes do not match");
            }

            auto result_shape = shape1;
            result_shape[Rank - 1] = shape2[Rank - 1];

            Tensor<T, Rank> result(result_shape);

            std::array<size_t, Rank> index_a{}, index_b{}, index_r{};

            std::function<void(size_t)> compute = [&](size_t dim) {
                if (dim == Rank - 2) {
                    for (size_t i = 0; i < shape1[Rank - 2]; ++i) {
                        index_a[Rank - 2] = index_r[Rank - 2] = i;
                        for (size_t j = 0; j < shape2[Rank - 1]; ++j) {
                            index_b[Rank - 1] = index_r[Rank - 1] = j;
                            T sum = T();
                            for (size_t k = 0; k < shape1[Rank - 1]; ++k) {
                                index_a[Rank - 1] = k;
                                index_b[Rank - 2] = k;
                                sum += A(index_a) * B(index_b);
                            }
                            result(index_r) = sum;
                        }
                    }
                } else {
                    for (size_t i = 0; i < shape1[dim]; ++i) {
                        index_a[dim] = index_b[dim] = index_r[dim] = i;
                        compute(dim + 1);
                    }
                }
            };

            compute(0);
            return result;
        }

    }  // namespace algebra
}  // namespace utec

// ...existing code...

template<typename T, size_t DIMS, typename Func>
utec::algebra::Tensor<T, DIMS> apply(const utec::algebra::Tensor<T, DIMS>& input, Func&& func) {
    auto result = input;
    for (auto& v : result) v = func(v);
    return result;
}

template <typename T, size_t N>
using Tensor = utec::algebra::Tensor<T, N>;

// ...existing code...

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

