#include "CudaOps.h"

#ifndef CNNV_HAS_CUDA

namespace cnnv::nn::ops::cuda_backend {

bool available() { return false; }

const char* status() {
    return "CUDA tensor kernels were not compiled";
}

bool try_matmul(const float*, const float*, float*,
                std::size_t, std::size_t, std::size_t) {
    return false;
}

bool try_matmul_bias(const float*, const float*, const float*, float*,
                     std::size_t, std::size_t, std::size_t) {
    return false;
}

bool try_conv2d(const float*, const float*, const float*, float*,
                std::size_t, std::size_t, std::size_t,
                std::size_t, std::size_t, std::size_t,
                std::size_t, std::size_t,
                std::size_t, std::size_t) {
    return false;
}

bool try_scaled_dot_product_attention(const float*, const float*,
                                      const float*, float*, float*,
                                      std::size_t, std::size_t,
                                      std::size_t) {
    return false;
}

}  // namespace cnnv::nn::ops::cuda_backend

#endif
