#include "nn/lc0_cnn/Lc0CnnLoader.h"

#include "io/WeightFileReader.h"

#include <stdexcept>
#include <string>

namespace cnnv::nn::lc0_cnn {

namespace {

std::vector<float> readFloatArray(cnnv::io::WeightFileReader& r,
                                  std::size_t expected,
                                  const char* label) {
    const std::uint32_t length = r.read_u32();
    if (length != expected) {
        throw std::runtime_error(
            std::string("Lc0CnnLoader: ") + label + " length mismatch (got " +
            std::to_string(length) + ", expected " + std::to_string(expected) +
            ")");
    }
    std::vector<float> out(length);
    if (length > 0) r.read_f32_array(out.data(), length);
    return out;
}

ConvLayer readConv(cnnv::io::WeightFileReader& r) {
    ConvLayer c;
    c.outChannels = static_cast<int>(r.read_u32());
    c.inChannels  = static_cast<int>(r.read_u32());
    c.kernel      = static_cast<int>(r.read_u32());
    const std::size_t kk = static_cast<std::size_t>(c.kernel) * c.kernel;
    c.weights = readFloatArray(
        r, static_cast<std::size_t>(c.outChannels) * c.inChannels * kk,
        "conv weights");
    c.bias = readFloatArray(r, static_cast<std::size_t>(c.outChannels),
                            "conv bias");
    return c;
}

DenseLayer readDense(cnnv::io::WeightFileReader& r, int expectedOut) {
    DenseLayer d;
    d.outDim = static_cast<int>(r.read_u32());
    d.inDim  = static_cast<int>(r.read_u32());
    if (d.outDim != expectedOut) {
        throw std::runtime_error(
            "Lc0CnnLoader: dense outDim mismatch (got " +
            std::to_string(d.outDim) + ", expected " +
            std::to_string(expectedOut) + ")");
    }
    d.weights = readFloatArray(
        r, static_cast<std::size_t>(d.outDim) * d.inDim, "dense weights");
    d.bias = readFloatArray(r, static_cast<std::size_t>(d.outDim),
                            "dense bias");
    return d;
}

SeUnit readSeUnit(cnnv::io::WeightFileReader& r, int expectedChannels) {
    SeUnit s;
    s.present = (r.read_u8() != 0);
    if (!s.present) return s;
    s.hidden   = static_cast<int>(r.read_u32());
    s.channels = static_cast<int>(r.read_u32());
    if (s.channels != expectedChannels) {
        throw std::runtime_error(
            "Lc0CnnLoader: SE unit channel mismatch (got " +
            std::to_string(s.channels) + ", expected " +
            std::to_string(expectedChannels) + ")");
    }
    s.w1 = readFloatArray(
        r, static_cast<std::size_t>(s.hidden) * s.channels, "se.w1");
    s.b1 = readFloatArray(r, static_cast<std::size_t>(s.hidden), "se.b1");
    s.w2 = readFloatArray(
        r, static_cast<std::size_t>(s.channels) * 2 * s.hidden, "se.w2");
    s.b2 = readFloatArray(r, static_cast<std::size_t>(s.channels) * 2,
                          "se.b2");
    return s;
}

std::size_t convParams(const ConvLayer& c) {
    return c.weights.size() + c.bias.size();
}
std::size_t denseParams(const DenseLayer& d) {
    return d.weights.size() + d.bias.size();
}
std::size_t seParams(const SeUnit& s) {
    if (!s.present) return 0;
    return s.w1.size() + s.b1.size() + s.w2.size() + s.b2.size();
}

}  // namespace

Weights Loader::load(const std::string& path) {
    cnnv::io::WeightFileReader r;
    r.open(path, "LC0J");

    if (r.read_u32() != kVersion) {
        throw std::runtime_error("Lc0CnnLoader: unsupported version");
    }

    Weights w;
    w.inputChannels  = static_cast<int>(r.read_u32());
    w.trunkChannels  = static_cast<int>(r.read_u32());
    const int residualBlocks = static_cast<int>(r.read_u32());
    w.policyChannels = static_cast<int>(r.read_u32());
    w.valueChannels  = static_cast<int>(r.read_u32());
    w.valueHidden    = static_cast<int>(r.read_u32());
    const int policyMapLength = static_cast<int>(r.read_u32());
    w.wdlOutputs     = static_cast<int>(r.read_u32());
    if (w.wdlOutputs != 3) {
        throw std::runtime_error(
            "Lc0CnnLoader: expected wdlOutputs=3 but got " +
            std::to_string(w.wdlOutputs));
    }

    w.inputLayer = readConv(r);
    std::size_t params = convParams(w.inputLayer);

    w.blocks.reserve(static_cast<std::size_t>(residualBlocks));
    for (int i = 0; i < residualBlocks; ++i) {
        ResidualBlock rb;
        rb.conv1 = readConv(r);
        rb.conv2 = readConv(r);
        rb.se    = readSeUnit(r, rb.conv2.outChannels);
        params += convParams(rb.conv1) + convParams(rb.conv2) + seParams(rb.se);
        w.blocks.push_back(std::move(rb));
    }

    w.policyStem   = readConv(r);
    w.policyOutput = readConv(r);
    w.valueConv    = readConv(r);
    w.valueFc1     = readDense(r, w.valueHidden);
    w.valueFc2     = readDense(r, w.wdlOutputs);

    params += convParams(w.policyStem) + convParams(w.policyOutput);
    params += convParams(w.valueConv) + denseParams(w.valueFc1) +
              denseParams(w.valueFc2);

    const std::uint32_t mapEntries = r.read_u32();
    if (static_cast<int>(mapEntries) != policyMapLength) {
        throw std::runtime_error(
            "Lc0CnnLoader: policy map length mismatch");
    }
    w.policyMap.resize(mapEntries);
    for (std::size_t i = 0; i < mapEntries; ++i) {
        w.policyMap[i] = static_cast<int>(r.read_i32());
    }

    if (r.tell() != r.size()) {
        throw std::runtime_error("Lc0CnnLoader: trailing bytes after payload");
    }

    w.parameterCount = params;
    return w;
}

}  // namespace cnnv::nn::lc0_cnn
