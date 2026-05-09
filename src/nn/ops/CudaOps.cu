#include "nn/ops/CudaOps.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace cnnv::nn::ops::cuda_backend {

namespace {

constexpr std::size_t kMinGpuWork = 32768;

bool runtimeAvailable() {
    static int cached = -1;
    if (cached >= 0) return cached == 1;

    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        (void)cudaGetLastError();
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}

bool ok(cudaError_t err) {
    if (err == cudaSuccess) return true;
    (void)cudaGetLastError();
    return false;
}

void freeAll(float* a, float* b, float* c, float* d) {
    if (a) (void)cudaFree(a);
    if (b) (void)cudaFree(b);
    if (c) (void)cudaFree(c);
    if (d) (void)cudaFree(d);
}

bool copyToDevice(const float* src, std::size_t count, float** dst) {
    *dst = nullptr;
    if (count == 0) return true;
    const std::size_t bytes = count * sizeof(float);
    if (!ok(cudaMalloc(reinterpret_cast<void**>(dst), bytes))) return false;
    if (!ok(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice))) {
        (void)cudaFree(*dst);
        *dst = nullptr;
        return false;
    }
    return true;
}

__global__ void matmulKernel(const float* A, const float* B,
                             const float* bias, float* C,
                             std::size_t M, std::size_t K, std::size_t N) {
    const std::size_t n =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t m =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    float acc = bias ? bias[n] : 0.0f;
    for (std::size_t k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = acc;
}

__global__ void conv2dKernel(const float* input, const float* weight,
                             const float* bias, float* output,
                             std::size_t Ci, std::size_t Hin, std::size_t Win,
                             std::size_t Co, std::size_t Kh, std::size_t Kw,
                             std::size_t strideH, std::size_t strideW,
                             std::size_t padH, std::size_t padW,
                             std::size_t Hout, std::size_t Wout) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = Co * Hout * Wout;
    if (idx >= total) return;

    const std::size_t ox = idx % Wout;
    const std::size_t oy = (idx / Wout) % Hout;
    const std::size_t co = idx / (Hout * Wout);

    float acc = bias ? bias[co] : 0.0f;
    for (std::size_t ci = 0; ci < Ci; ++ci) {
        for (std::size_t ky = 0; ky < Kh; ++ky) {
            const long iy = static_cast<long>(oy * strideH + ky) -
                            static_cast<long>(padH);
            if (iy < 0 || iy >= static_cast<long>(Hin)) continue;
            for (std::size_t kx = 0; kx < Kw; ++kx) {
                const long ix = static_cast<long>(ox * strideW + kx) -
                                static_cast<long>(padW);
                if (ix < 0 || ix >= static_cast<long>(Win)) continue;
                const std::size_t inputIdx =
                    ci * Hin * Win +
                    static_cast<std::size_t>(iy) * Win +
                    static_cast<std::size_t>(ix);
                const std::size_t weightIdx =
                    co * Ci * Kh * Kw + ci * Kh * Kw + ky * Kw + kx;
                acc += input[inputIdx] * weight[weightIdx];
            }
        }
    }
    output[idx] = acc;
}

__device__ std::size_t qkvIndex(std::size_t t, std::size_t h, std::size_t d,
                                std::size_t heads, std::size_t headDim) {
    return t * heads * headDim + h * headDim + d;
}

__global__ void attentionKernel(const float* Q, const float* K,
                                const float* V, float* out, float* weights,
                                std::size_t tokens, std::size_t heads,
                                std::size_t headDim) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t totalRows = heads * tokens;
    if (row >= totalRows) return;

    const std::size_t h = row / tokens;
    const std::size_t i = row % tokens;
    const float scale = rsqrtf(static_cast<float>(headDim));

    float maxLogit = -3.402823466e+38f;
    for (std::size_t j = 0; j < tokens; ++j) {
        float dot = 0.0f;
        for (std::size_t d = 0; d < headDim; ++d) {
            dot += Q[qkvIndex(i, h, d, heads, headDim)] *
                   K[qkvIndex(j, h, d, heads, headDim)];
        }
        maxLogit = fmaxf(maxLogit, dot * scale);
    }

    float sumExp = 0.0f;
    for (std::size_t j = 0; j < tokens; ++j) {
        float dot = 0.0f;
        for (std::size_t d = 0; d < headDim; ++d) {
            dot += Q[qkvIndex(i, h, d, heads, headDim)] *
                   K[qkvIndex(j, h, d, heads, headDim)];
        }
        const float e = expf(dot * scale - maxLogit);
        if (weights) weights[(h * tokens + i) * tokens + j] = e;
        sumExp += e;
    }
    const float invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;

    for (std::size_t d = 0; d < headDim; ++d) {
        float acc = 0.0f;
        for (std::size_t j = 0; j < tokens; ++j) {
            float w = 0.0f;
            if (weights) {
                const std::size_t wIdx = (h * tokens + i) * tokens + j;
                w = weights[wIdx] * invSum;
                weights[wIdx] = w;
            } else {
                float dot = 0.0f;
                for (std::size_t kd = 0; kd < headDim; ++kd) {
                    dot += Q[qkvIndex(i, h, kd, heads, headDim)] *
                           K[qkvIndex(j, h, kd, heads, headDim)];
                }
                w = expf(dot * scale - maxLogit) * invSum;
            }
            acc += w * V[qkvIndex(j, h, d, heads, headDim)];
        }
        out[qkvIndex(i, h, d, heads, headDim)] = acc;
    }
}

bool runMatmul(const float* A, const float* B, const float* bias, float* C,
               std::size_t M, std::size_t K, std::size_t N) {
    if (!runtimeAvailable()) return false;
    if (M == 0 || K == 0 || N == 0) return true;
    if (M * K * N < kMinGpuWork) return false;

    float* dA = nullptr;
    float* dB = nullptr;
    float* dBias = nullptr;
    float* dC = nullptr;
    if (!copyToDevice(A, M * K, &dA)) return false;
    if (!copyToDevice(B, K * N, &dB)) {
        freeAll(dA, nullptr, nullptr, nullptr);
        return false;
    }
    if (bias && !copyToDevice(bias, N, &dBias)) {
        freeAll(dA, dB, nullptr, nullptr);
        return false;
    }
    if (!ok(cudaMalloc(reinterpret_cast<void**>(&dC), M * N * sizeof(float)))) {
        freeAll(dA, dB, dBias, nullptr);
        return false;
    }

    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((N + block.x - 1) / block.x),
                    static_cast<unsigned int>((M + block.y - 1) / block.y));
    matmulKernel<<<grid, block>>>(dA, dB, dBias, dC, M, K, N);
    if (!ok(cudaPeekAtLastError()) ||
        !ok(cudaMemcpy(C, dC, M * N * sizeof(float),
                       cudaMemcpyDeviceToHost))) {
        freeAll(dA, dB, dBias, dC);
        return false;
    }

    freeAll(dA, dB, dBias, dC);
    return true;
}

}  // namespace

bool available() {
    return runtimeAvailable();
}

const char* status() {
    return runtimeAvailable() ? "CUDA tensor kernels available"
                              : "CUDA compiled, but no usable CUDA device/driver";
}

bool try_matmul(const float* A, const float* B, float* C,
                std::size_t M, std::size_t K, std::size_t N) {
    return runMatmul(A, B, nullptr, C, M, K, N);
}

bool try_matmul_bias(const float* A, const float* B, const float* bias,
                     float* C, std::size_t M, std::size_t K, std::size_t N) {
    return runMatmul(A, B, bias, C, M, K, N);
}

bool try_conv2d(const float* input, const float* weight, const float* bias,
                float* output,
                std::size_t Ci, std::size_t Hin, std::size_t Win,
                std::size_t Co, std::size_t Kh, std::size_t Kw,
                std::size_t strideH, std::size_t strideW,
                std::size_t padH, std::size_t padW) {
    if (!runtimeAvailable()) return false;
    const std::size_t Hout = (Hin + 2 * padH - Kh) / strideH + 1;
    const std::size_t Wout = (Win + 2 * padW - Kw) / strideW + 1;
    const std::size_t outputCount = Co * Hout * Wout;
    if (outputCount == 0) return true;
    if (outputCount * Ci * Kh * Kw < kMinGpuWork) return false;

    float* dInput = nullptr;
    float* dWeight = nullptr;
    float* dBias = nullptr;
    float* dOutput = nullptr;
    if (!copyToDevice(input, Ci * Hin * Win, &dInput)) return false;
    if (!copyToDevice(weight, Co * Ci * Kh * Kw, &dWeight)) {
        freeAll(dInput, nullptr, nullptr, nullptr);
        return false;
    }
    if (bias && !copyToDevice(bias, Co, &dBias)) {
        freeAll(dInput, dWeight, nullptr, nullptr);
        return false;
    }
    if (!ok(cudaMalloc(reinterpret_cast<void**>(&dOutput),
                       outputCount * sizeof(float)))) {
        freeAll(dInput, dWeight, dBias, nullptr);
        return false;
    }

    constexpr unsigned int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((outputCount + block - 1) / block);
    conv2dKernel<<<grid, block>>>(dInput, dWeight, dBias, dOutput,
                                  Ci, Hin, Win, Co, Kh, Kw,
                                  strideH, strideW, padH, padW,
                                  Hout, Wout);
    if (!ok(cudaPeekAtLastError()) ||
        !ok(cudaMemcpy(output, dOutput, outputCount * sizeof(float),
                       cudaMemcpyDeviceToHost))) {
        freeAll(dInput, dWeight, dBias, dOutput);
        return false;
    }

    freeAll(dInput, dWeight, dBias, dOutput);
    return true;
}

bool try_scaled_dot_product_attention(const float* Q, const float* K,
                                      const float* V, float* out,
                                      float* weights,
                                      std::size_t tokens, std::size_t heads,
                                      std::size_t headDim) {
    if (!runtimeAvailable()) return false;
    if (tokens == 0 || heads == 0 || headDim == 0) return true;
    if (tokens * heads * tokens * headDim < kMinGpuWork) return false;

    const std::size_t qkvCount = tokens * heads * headDim;
    const std::size_t weightCount = heads * tokens * tokens;
    float* dQ = nullptr;
    float* dK = nullptr;
    float* dV = nullptr;
    float* dOut = nullptr;
    float* dWeights = nullptr;
    if (!copyToDevice(Q, qkvCount, &dQ)) return false;
    if (!copyToDevice(K, qkvCount, &dK)) {
        freeAll(dQ, nullptr, nullptr, nullptr);
        return false;
    }
    if (!copyToDevice(V, qkvCount, &dV)) {
        freeAll(dQ, dK, nullptr, nullptr);
        return false;
    }
    if (!ok(cudaMalloc(reinterpret_cast<void**>(&dOut),
                       qkvCount * sizeof(float)))) {
        freeAll(dQ, dK, dV, nullptr);
        return false;
    }
    if (weights && !ok(cudaMalloc(reinterpret_cast<void**>(&dWeights),
                                  weightCount * sizeof(float)))) {
        freeAll(dQ, dK, dV, dOut);
        return false;
    }

    constexpr unsigned int block = 128;
    const unsigned int grid =
        static_cast<unsigned int>((heads * tokens + block - 1) / block);
    attentionKernel<<<grid, block>>>(dQ, dK, dV, dOut, dWeights,
                                     tokens, heads, headDim);
    if (!ok(cudaPeekAtLastError()) ||
        !ok(cudaMemcpy(out, dOut, qkvCount * sizeof(float),
                       cudaMemcpyDeviceToHost))) {
        freeAll(dQ, dK, dV, dOut);
        if (dWeights) (void)cudaFree(dWeights);
        return false;
    }
    if (weights &&
        !ok(cudaMemcpy(weights, dWeights, weightCount * sizeof(float),
                       cudaMemcpyDeviceToHost))) {
        freeAll(dQ, dK, dV, dOut);
        if (dWeights) (void)cudaFree(dWeights);
        return false;
    }

    freeAll(dQ, dK, dV, dOut);
    if (dWeights) (void)cudaFree(dWeights);
    return true;
}

}  // namespace cnnv::nn::ops::cuda_backend
