#include "nn/nnue/NnueLoader.h"

#include "io/WeightFileReader.h"
#include "nn/nnue/NnueFeatureEncoder.h"

#include <stdexcept>
#include <string>

namespace cnnv::nn::nnue {

namespace {

std::vector<float> readFloatArray(cnnv::io::WeightFileReader& r,
                                  const std::string& label,
                                  std::size_t expected) {
    std::uint32_t length = r.read_u32();
    if (length != expected) {
        throw std::runtime_error(
            "NnueLoader: " + label + " length mismatch (got " +
            std::to_string(length) + ", expected " + std::to_string(expected) +
            ")");
    }
    std::vector<float> out(length);
    r.read_f32_array(out.data(), length);
    return out;
}

}  // namespace

Weights Loader::load(const std::string& path) {
    cnnv::io::WeightFileReader r;
    r.open(path, "NNUE");

    const std::uint32_t version = r.read_u32();
    if (version != kVersion) {
        throw std::runtime_error(
            "NnueLoader: unsupported version " + std::to_string(version));
    }

    const std::uint32_t featureCount = r.read_u32();
    if (featureCount != static_cast<std::uint32_t>(FeatureEncoder::kFeatureCount)) {
        throw std::runtime_error(
            "NnueLoader: feature count mismatch (got " +
            std::to_string(featureCount) + ", expected " +
            std::to_string(FeatureEncoder::kFeatureCount) + ")");
    }

    const std::uint32_t hiddenSize = r.read_u32();
    if (hiddenSize == 0) {
        throw std::runtime_error("NnueLoader: hiddenSize must be positive");
    }

    Weights w;
    w.hiddenSize = static_cast<int>(hiddenSize);
    w.outputScale = r.read_f32();
    w.featureBias = readFloatArray(r, "featureBias", hiddenSize);
    w.featureWeights = readFloatArray(
        r, "featureWeights",
        static_cast<std::size_t>(featureCount) * hiddenSize);
    w.outputWeights = readFloatArray(r, "outputWeights",
                                     static_cast<std::size_t>(hiddenSize) * 2);
    w.outputBias = r.read_f32();

    if (r.tell() != r.size()) {
        throw std::runtime_error(
            "NnueLoader: trailing bytes after parsed payload (" +
            std::to_string(r.size() - r.tell()) + " left)");
    }
    return w;
}

Weights Loader::makeFallback(int hiddenSize) {
    Weights w;
    w.hiddenSize = hiddenSize;
    w.featureBias.assign(static_cast<std::size_t>(hiddenSize), 0.0f);
    w.featureWeights.assign(
        static_cast<std::size_t>(FeatureEncoder::kFeatureCount) * hiddenSize,
        0.0f);
    w.outputWeights.assign(static_cast<std::size_t>(hiddenSize) * 2, 0.0f);
    w.outputBias = 0.0f;
    w.outputScale = 1.0f;
    return w;
}

}  // namespace cnnv::nn::nnue
