#include "MatMul.h"

#include "CudaOps.h"

namespace cnnv::nn::ops {

void matmul(const float* A, const float* B, float* C,
            std::size_t M, std::size_t K, std::size_t N) {
    if (cuda_backend::try_matmul(A, B, C, M, K, N)) return;

    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

void matmul_bias(const float* A, const float* B, const float* bias, float* C,
                 std::size_t M, std::size_t K, std::size_t N) {
    if (cuda_backend::try_matmul_bias(A, B, bias, C, M, K, N)) return;

    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float acc = bias ? bias[n] : 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

}  // namespace cnnv::nn::ops
