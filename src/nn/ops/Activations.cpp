#include "Activations.h"

#include <algorithm>
#include <cmath>

namespace cnnv::nn::ops {

void relu(float* x, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

void relu(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

void clipped_relu_i16_to_i8(const std::int16_t* in, std::int8_t* out,
                            std::size_t n, int shift) {
    for (std::size_t i = 0; i < n; ++i) {
        int v = in[i] >> shift;
        if (v < 0) v = 0;
        if (v > 127) v = 127;
        out[i] = static_cast<std::int8_t>(v);
    }
}

void clipped_relu(const float* in, float* out, std::size_t n, float max_val) {
    for (std::size_t i = 0; i < n; ++i) {
        float v = in[i];
        if (v < 0.0f) v = 0.0f;
        if (v > max_val) v = max_val;
        out[i] = v;
    }
}

void sigmoid(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
}

void tanh_act(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::tanh(in[i]);
    }
}

void gelu(const float* in, float* out, std::size_t n) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoef = 0.044715f;
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float inner = kSqrt2OverPi * (x + kCoef * x * x * x);
        out[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void softmax(float* x, std::size_t n) {
    if (n == 0) return;
    float max_v = x[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (x[i] > max_v) max_v = x[i];
    }
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_v);
        sum += x[i];
    }
    if (sum > 0.0f) {
        const float inv = 1.0f / sum;
        for (std::size_t i = 0; i < n; ++i) x[i] *= inv;
    }
}

void softmax(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = in[i];
    softmax(out, n);
}

}  // namespace cnnv::nn::ops
