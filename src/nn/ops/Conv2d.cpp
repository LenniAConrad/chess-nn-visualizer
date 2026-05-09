#include "Conv2d.h"

#include "CudaOps.h"

namespace cnnv::nn::ops {

void conv2d(const float* input, const float* weight, const float* bias,
            float* output,
            std::size_t Ci, std::size_t Hin, std::size_t Win,
            std::size_t Co, std::size_t Kh, std::size_t Kw,
            std::size_t strideH, std::size_t strideW,
            std::size_t padH, std::size_t padW) {
    if (cuda_backend::try_conv2d(input, weight, bias, output,
                                 Ci, Hin, Win, Co, Kh, Kw,
                                 strideH, strideW, padH, padW)) {
        return;
    }

    const std::size_t Hout = conv2d_out_dim(Hin, Kh, strideH, padH);
    const std::size_t Wout = conv2d_out_dim(Win, Kw, strideW, padW);

    for (std::size_t co = 0; co < Co; ++co) {
        const float b = bias ? bias[co] : 0.0f;
        for (std::size_t oy = 0; oy < Hout; ++oy) {
            for (std::size_t ox = 0; ox < Wout; ++ox) {
                float acc = b;
                for (std::size_t ci = 0; ci < Ci; ++ci) {
                    for (std::size_t ky = 0; ky < Kh; ++ky) {
                        const long iy = static_cast<long>(oy * strideH + ky) -
                                        static_cast<long>(padH);
                        if (iy < 0 || iy >= static_cast<long>(Hin)) continue;
                        for (std::size_t kx = 0; kx < Kw; ++kx) {
                            const long ix =
                                static_cast<long>(ox * strideW + kx) -
                                static_cast<long>(padW);
                            if (ix < 0 || ix >= static_cast<long>(Win)) continue;
                            const std::size_t in_idx =
                                ci * Hin * Win +
                                static_cast<std::size_t>(iy) * Win +
                                static_cast<std::size_t>(ix);
                            const std::size_t w_idx =
                                co * Ci * Kh * Kw + ci * Kh * Kw + ky * Kw + kx;
                            acc += input[in_idx] * weight[w_idx];
                        }
                    }
                }
                output[co * Hout * Wout + oy * Wout + ox] = acc;
            }
        }
    }
}

}  // namespace cnnv::nn::ops
