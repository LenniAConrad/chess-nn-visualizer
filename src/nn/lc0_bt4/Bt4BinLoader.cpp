#include "nn/lc0_bt4/Bt4BinLoader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cnnv::nn::lc0_bt4 {

namespace {

// BT4J magic ("BT4J", little-endian int) and supported format version.
constexpr std::uint32_t kMagic = 0x4A345442u;
constexpr std::uint32_t kVersion = 2u;
constexpr std::uint32_t kMaxStringLength = 1'000'000u;
constexpr std::int32_t kMaxEncoderBlocks = 256;

/**
 * @brief Kleiner Little-Endian-Cursor ueber den geladenen BT4J-Blob.
 *
 * Der Cursor ist absichtlich stumpf und eng gehalten: erst Grenzen pruefen,
 * dann lesen. Kaputte Modelldateien sollen hier rausfliegen und nicht spaeter
 * irgendwo im Forward-Pass fuer Chaos sorgen.
 */
class Cursor {
   public:
    explicit Cursor(std::vector<std::uint8_t> bytes) : m_bytes(std::move(bytes)) {}

    std::int32_t readI32() {
        require(4);
        std::uint32_t v = static_cast<std::uint32_t>(m_bytes[m_pos]) |
                          (static_cast<std::uint32_t>(m_bytes[m_pos + 1]) << 8) |
                          (static_cast<std::uint32_t>(m_bytes[m_pos + 2]) << 16) |
                          (static_cast<std::uint32_t>(m_bytes[m_pos + 3]) << 24);
        m_pos += 4;
        return static_cast<std::int32_t>(v);
    }

    std::uint32_t readU32() { return static_cast<std::uint32_t>(readI32()); }

    float readF32() {
        std::int32_t bits = readI32();
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    std::uint8_t readU8() {
        require(1);
        return m_bytes[m_pos++];
    }

    bool readBool() { return readU8() != 0; }

    std::string readString() {
        std::int32_t length = readNonNegativeCount("string length");
        if (static_cast<std::uint32_t>(length) > kMaxStringLength) {
            throw std::runtime_error("BT4: invalid string length");
        }
        const std::size_t n = static_cast<std::size_t>(length);
        require(n);
        std::string s(reinterpret_cast<const char*>(&m_bytes[m_pos]), n);
        m_pos += n;
        return s;
    }

    /**
     * @brief Liest ein laengenpraefixiertes float32-Array.
     *
     * Der Count kommt aus der Datei, also ist er nicht vertrauenswuerdig. Erst
     * wird gegen die restliche Payload gecheckt; dann wird allokiert. Kein RAM-
     * Drama nur wegen einem kaputten Test- oder Model-Blob.
     */
    std::vector<float> readFloatArray() {
        std::int32_t count = readNonNegativeCount("float array count");
        const std::size_t n = static_cast<std::size_t>(count);
        if (n > remaining() / sizeof(float)) {
            throw std::runtime_error("BT4: float array count exceeds file payload");
        }
        std::vector<float> out(n);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = readF32();
        return out;
    }

    bool hasRemaining() const { return m_pos < m_bytes.size(); }

    /**
     * @brief Liest einen Count, der nicht negativ sein darf.
     *
     * Viele BT4J-Felder starten mit genau so einem signed Laengenwert. Der
     * kleine Helper haelt die langweilige Negativ-Pruefung an einer Stelle,
     * damit die eigentlichen Reader nur noch ihre Spezialgrenzen checken.
     */
    std::int32_t readNonNegativeCount(const char* label) {
        const std::int32_t count = readI32();
        if (count < 0) {
            throw std::runtime_error("BT4: negative " + std::string(label));
        }
        return count;
    }

   private:
    std::size_t remaining() const { return m_bytes.size() - m_pos; }

    void require(std::size_t n) const {
        if (n > remaining()) {
            throw std::runtime_error("BT4: unexpected end of file");
        }
    }

    std::vector<std::uint8_t> m_bytes;
    std::size_t m_pos = 0;
};

Bt4Activation parseActivation(const std::string& name) {
    if (name == "NONE") return Bt4Activation::None;
    if (name == "RELU") return Bt4Activation::Relu;
    if (name == "MISH") return Bt4Activation::Mish;
    if (name == "SWISH") return Bt4Activation::Swish;
    if (name == "TANH") return Bt4Activation::Tanh;
    throw std::runtime_error("BT4: unknown activation enum '" + name + "'");
}

Dense readDense(Cursor& in) {
    Dense d;
    d.inDim = in.readI32();
    d.outDim = in.readI32();
    d.weights = in.readFloatArray();
    d.bias = in.readFloatArray();
    return d;
}

Arch readArchitecture(Cursor& in) {
    Arch a;
    a.name = in.readString();
    a.inputFormat = in.readString();
    a.inputEmbedding = in.readString();
    a.inputChannels = in.readI32();
    a.tokens = in.readI32();
    a.embeddingSize = in.readI32();
    a.encoderLayers = in.readI32();
    a.attentionHeads = in.readI32();
    a.policySize = in.readI32();
    a.layerNormEpsilon = in.readF32();
    a.ffnHiddenSize = in.readI32();
    a.smolgenHiddenChannels = in.readI32();
    a.smolgenHiddenSize = in.readI32();
    a.smolgenPerHeadDim = in.readI32();
    a.smolgenGlobalSize = in.readI32();
    a.defaultActivation = parseActivation(in.readString());
    a.smolgenActivation = parseActivation(in.readString());
    a.ffnActivation = parseActivation(in.readString());
    a.hasInputPreproc = in.readBool();
    a.hasInputEmbFfn = in.readBool();
    a.hasInputGates = in.readBool();
    a.hasSmolgen = in.readBool();
    return a;
}

InputStack readInputStack(Cursor& in, const Arch& arch) {
    InputStack s;
    if (arch.hasInputPreproc) {
        s.hasPreproc = true;
        s.preproc = readDense(in);
    }
    s.embedding = readDense(in);
    if (arch.inputEmbedding == "PE_DENSE") {
        s.hasEmbLn = true;
        s.embLnGamma = in.readFloatArray();
        s.embLnBeta = in.readFloatArray();
    }
    if (arch.hasInputGates) {
        s.hasGates = true;
        s.multGate = in.readFloatArray();
        s.addGate = in.readFloatArray();
    }
    if (arch.hasInputEmbFfn) {
        s.hasEmbFfn = true;
        s.embFfn.dense1 = readDense(in);
        s.embFfn.dense2 = readDense(in);
        s.embFfnLnGamma = in.readFloatArray();
        s.embFfnLnBeta = in.readFloatArray();
    }
    return s;
}

Smolgen readSmolgen(Cursor& in) {
    Smolgen sm;
    sm.compress = readDense(in);
    sm.dense1 = readDense(in);
    sm.ln1Gamma = in.readFloatArray();
    sm.ln1Beta = in.readFloatArray();
    sm.dense2 = readDense(in);
    sm.ln2Gamma = in.readFloatArray();
    sm.ln2Beta = in.readFloatArray();
    return sm;
}

Attention readAttention(Cursor& in, bool hasSmolgen) {
    Attention a;
    a.heads = in.readI32();
    a.query = readDense(in);
    a.key = readDense(in);
    a.value = readDense(in);
    a.out = readDense(in);
    if (hasSmolgen) {
        a.hasSmolgen = true;
        a.smolgen = readSmolgen(in);
    }
    return a;
}

EncoderBlock readEncoderBlock(Cursor& in, bool hasSmolgen) {
    EncoderBlock b;
    b.attention = readAttention(in, hasSmolgen);
    b.ffnIn = readDense(in);
    b.ffnOut = readDense(in);
    b.ln1Gamma = in.readFloatArray();
    b.ln1Beta = in.readFloatArray();
    b.ln2Gamma = in.readFloatArray();
    b.ln2Beta = in.readFloatArray();
    b.activation = parseActivation(in.readString());
    b.alpha = in.readF32();
    return b;
}

/**
 * @brief Liest die Encoder-Block-Liste mit einer harten Plausibilitaetsgrenze.
 *
 * Echte kleine BT4-Modelle kommen weit unter diesem Limit raus. Wenn eine Datei
 * Millionen Blocks behauptet, ist das kein Modell mehr, sondern Quatsch, also
 * direkt abbrechen bevor `reserve()` gross einkauft.
 */
std::vector<EncoderBlock> readEncoderBlocks(Cursor& in, bool hasSmolgen) {
    std::int32_t count = in.readNonNegativeCount("encoder block count");
    if (count > kMaxEncoderBlocks) {
        throw std::runtime_error("BT4: encoder block count too large");
    }
    std::vector<EncoderBlock> blocks;
    blocks.reserve(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) {
        blocks.push_back(readEncoderBlock(in, hasSmolgen));
    }
    return blocks;
}

PolicyHead readPolicyHead(Cursor& in) {
    PolicyHead h;
    h.embedding = readDense(in);
    h.encoders = readEncoderBlocks(in, /*hasSmolgen=*/false);
    h.query = readDense(in);
    h.key = readDense(in);
    h.promotionWeights = in.readFloatArray();
    h.activation = parseActivation(in.readString());
    return h;
}

ValueHead readValueHead(Cursor& in) {
    ValueHead h;
    h.embedding = readDense(in);
    h.fc1 = readDense(in);
    h.fc2 = readDense(in);
    h.activation = parseActivation(in.readString());
    return h;
}

std::vector<std::uint8_t> readWholeFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("BT4: cannot open '" + path + "'");
    const std::streamsize size = file.tellg();
    if (size < 0) throw std::runtime_error("BT4: cannot size '" + path + "'");
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (size > 0 &&
        !file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("BT4: short read on '" + path + "'");
    }
    return bytes;
}

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::size_t checkedProduct(int a, int b, const std::string& label) {
    require(a >= 0 && b >= 0, "BT4: negative dimension in " + label);
    return static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
}

void validateArray(const std::vector<float>& values,
                   std::size_t expected,
                   const std::string& label) {
    require(values.size() == expected,
            "BT4: " + label + " length mismatch (got " +
                std::to_string(values.size()) + ", expected " +
                std::to_string(expected) + ")");
}

void validateDense(const Dense& d, const std::string& label) {
    require(d.inDim > 0, "BT4: " + label + " has non-positive input dim");
    require(d.outDim > 0, "BT4: " + label + " has non-positive output dim");
    validateArray(d.weights, checkedProduct(d.outDim, d.inDim, label),
                  label + ".weights");
    validateArray(d.bias, static_cast<std::size_t>(d.outDim),
                  label + ".bias");
}

void validateDenseShape(const Dense& d,
                        int expectedIn,
                        int expectedOut,
                        const std::string& label) {
    validateDense(d, label);
    require(d.inDim == expectedIn,
            "BT4: " + label + " input dim mismatch (got " +
                std::to_string(d.inDim) + ", expected " +
                std::to_string(expectedIn) + ")");
    require(d.outDim == expectedOut,
            "BT4: " + label + " output dim mismatch (got " +
                std::to_string(d.outDim) + ", expected " +
                std::to_string(expectedOut) + ")");
}

void validateAttention(const Attention& a,
                       int inputDim,
                       int outputDim,
                       int tokens,
                       bool sharedSmolgen,
                       const std::string& label) {
    require(a.heads > 0, "BT4: " + label + " has non-positive head count");
    validateDense(a.query, label + ".query");
    validateDense(a.key, label + ".key");
    validateDense(a.value, label + ".value");
    validateDense(a.out, label + ".out");
    require(a.query.inDim == inputDim && a.key.inDim == inputDim &&
                a.value.inDim == inputDim,
            "BT4: " + label + " qkv input dim mismatch");
    require(a.query.outDim == a.key.outDim &&
                a.query.outDim == a.value.outDim,
            "BT4: " + label + " qkv output dims differ");
    require(a.query.outDim % a.heads == 0,
            "BT4: " + label + " qkv dim is not divisible by heads");
    require(a.out.inDim == a.query.outDim && a.out.outDim == outputDim,
            "BT4: " + label + " output projection shape mismatch");

    if (!a.hasSmolgen) return;
    require(sharedSmolgen, "BT4: " + label + " has smolgen without shared weights");
    validateDenseShape(a.smolgen.compress, inputDim,
                       a.smolgen.compress.outDim, label + ".smolgen.compress");
    validateDense(a.smolgen.dense1, label + ".smolgen.dense1");
    require(a.smolgen.dense1.inDim == a.smolgen.compress.outDim,
            "BT4: " + label + " smolgen dense1 input dim mismatch");
    validateArray(a.smolgen.ln1Gamma,
                  static_cast<std::size_t>(a.smolgen.dense1.outDim),
                  label + ".smolgen.ln1Gamma");
    validateArray(a.smolgen.ln1Beta,
                  static_cast<std::size_t>(a.smolgen.dense1.outDim),
                  label + ".smolgen.ln1Beta");
    validateDense(a.smolgen.dense2, label + ".smolgen.dense2");
    require(a.smolgen.dense2.inDim == a.smolgen.dense1.outDim,
            "BT4: " + label + " smolgen dense2 input dim mismatch");
    require(a.smolgen.dense2.outDim % a.heads == 0,
            "BT4: " + label + " smolgen output dim is not divisible by heads");
    validateArray(a.smolgen.ln2Gamma,
                  static_cast<std::size_t>(a.smolgen.dense2.outDim),
                  label + ".smolgen.ln2Gamma");
    validateArray(a.smolgen.ln2Beta,
                  static_cast<std::size_t>(a.smolgen.dense2.outDim),
                  label + ".smolgen.ln2Beta");
}

void validateEncoderBlock(const EncoderBlock& block,
                          int tokens,
                          int dim,
                          bool sharedSmolgen,
                          const std::string& label) {
    validateAttention(block.attention, dim, dim, tokens, sharedSmolgen,
                      label + ".attention");
    validateDense(block.ffnIn, label + ".ffnIn");
    require(block.ffnIn.inDim == dim,
            "BT4: " + label + " ffnIn input dim mismatch");
    validateDenseShape(block.ffnOut, block.ffnIn.outDim, dim,
                       label + ".ffnOut");
    validateArray(block.ln1Gamma, static_cast<std::size_t>(dim),
                  label + ".ln1Gamma");
    validateArray(block.ln1Beta, static_cast<std::size_t>(dim),
                  label + ".ln1Beta");
    validateArray(block.ln2Gamma, static_cast<std::size_t>(dim),
                  label + ".ln2Gamma");
    validateArray(block.ln2Beta, static_cast<std::size_t>(dim),
                  label + ".ln2Beta");
}

void validateWeights(const Bt4RealWeights& w) {
    const Arch& a = w.arch;
    require(a.inputChannels == 112,
            "BT4: expected 112 input channels but got " +
                std::to_string(a.inputChannels));
    require(a.tokens == 64,
            "BT4: expected 64 board tokens but got " +
                std::to_string(a.tokens));
    require(a.embeddingSize > 0, "BT4: non-positive embedding size");
    require(a.policySize > 0, "BT4: non-positive policy size");
    require(a.layerNormEpsilon > 0.0f, "BT4: non-positive layer norm epsilon");
    require(static_cast<int>(w.encoders.size()) == a.encoderLayers,
            "BT4: encoder block count does not match architecture metadata");

    int inputWidth = a.inputChannels;
    if (a.inputEmbedding == "PE_MAP") {
        inputWidth += a.tokens;
    } else if (a.inputEmbedding == "PE_DENSE") {
        require(w.input.hasPreproc,
                "BT4: PE_DENSE input embedding requires preproc weights");
        validateDenseShape(w.input.preproc, a.tokens * 12,
                           w.input.preproc.outDim, "input.preproc");
        require(w.input.preproc.outDim % a.tokens == 0,
                "BT4: input.preproc output dim must divide into tokens");
        inputWidth += w.input.preproc.outDim / a.tokens;
        validateArray(w.input.embLnGamma, static_cast<std::size_t>(a.embeddingSize),
                      "input.embLnGamma");
        validateArray(w.input.embLnBeta, static_cast<std::size_t>(a.embeddingSize),
                      "input.embLnBeta");
    }
    validateDenseShape(w.input.embedding, inputWidth, a.embeddingSize,
                       "input.embedding");

    if (w.input.hasGates) {
        const std::size_t flowSize =
            checkedProduct(a.tokens, a.embeddingSize, "input gates");
        validateArray(w.input.multGate, flowSize, "input.multGate");
        validateArray(w.input.addGate, flowSize, "input.addGate");
    }
    if (w.input.hasEmbFfn) {
        validateDenseShape(w.input.embFfn.dense1, a.embeddingSize,
                           w.input.embFfn.dense1.outDim, "input.embFfn.dense1");
        validateDenseShape(w.input.embFfn.dense2,
                           w.input.embFfn.dense1.outDim, a.embeddingSize,
                           "input.embFfn.dense2");
        validateArray(w.input.embFfnLnGamma,
                      static_cast<std::size_t>(a.embeddingSize),
                      "input.embFfnLnGamma");
        validateArray(w.input.embFfnLnBeta,
                      static_cast<std::size_t>(a.embeddingSize),
                      "input.embFfnLnBeta");
    }

    for (std::size_t i = 0; i < w.encoders.size(); ++i) {
        validateEncoderBlock(w.encoders[i], a.tokens, a.embeddingSize,
                             a.hasSmolgen, "encoder" + std::to_string(i));
    }
    if (a.hasSmolgen) {
        require(!w.smolgenW.empty(), "BT4: missing shared smolgen weights");
        const int perHead =
            w.encoders.empty()
                ? 0
                : w.encoders.front().attention.smolgen.dense2.outDim /
                      w.encoders.front().attention.heads;
        require(perHead > 0, "BT4: invalid smolgen per-head width");
        validateArray(w.smolgenW,
                      static_cast<std::size_t>(a.tokens) *
                          static_cast<std::size_t>(a.tokens) *
                          static_cast<std::size_t>(perHead),
                      "smolgenW");
    }

    validateDenseShape(w.policyHead.embedding, a.embeddingSize,
                       w.policyHead.embedding.outDim, "policy.embedding");
    int policyDim = w.policyHead.embedding.outDim;
    for (std::size_t i = 0; i < w.policyHead.encoders.size(); ++i) {
        validateEncoderBlock(w.policyHead.encoders[i], a.tokens, policyDim,
                             false, "policy.encoder" + std::to_string(i));
    }
    validateDenseShape(w.policyHead.query, policyDim, w.policyHead.query.outDim,
                       "policy.query");
    validateDenseShape(w.policyHead.key, policyDim, w.policyHead.query.outDim,
                       "policy.key");
    validateArray(w.policyHead.promotionWeights,
                  static_cast<std::size_t>(4) *
                      static_cast<std::size_t>(w.policyHead.query.outDim),
                  "policy.promotionWeights");

    validateDenseShape(w.valueHead.embedding, a.embeddingSize,
                       w.valueHead.embedding.outDim, "value.embedding");
    validateDenseShape(w.valueHead.fc1,
                       a.tokens * w.valueHead.embedding.outDim,
                       w.valueHead.fc1.outDim, "value.fc1");
    validateDenseShape(w.valueHead.fc2, w.valueHead.fc1.outDim, 3,
                       "value.fc2");
}

}  // namespace

bool loadBt4Real(const std::string& path, Bt4RealWeights& out,
                 std::string& error) {
    try {
        Cursor in(readWholeFile(path));

        const std::uint32_t magic = in.readU32();
        if (magic != kMagic) {
            error = "BT4: invalid magic (not a BT4J file)";
            return false;
        }
        const std::uint32_t version = in.readU32();
        if (version != kVersion) {
            error = "BT4: unsupported version " + std::to_string(version) +
                    " (expected 2)";
            return false;
        }

        Bt4RealWeights w;
        w.arch = readArchitecture(in);
        w.input = readInputStack(in, w.arch);
        w.encoders = readEncoderBlocks(in, w.arch.hasSmolgen);
        if (w.arch.hasSmolgen) {
            w.smolgenW = in.readFloatArray();
        }
        w.policyHead = readPolicyHead(in);
        w.valueHead = readValueHead(in);

        if (in.hasRemaining()) {
            error = "BT4: unexpected bytes at end of weights file";
            return false;
        }
        validateWeights(w);

        out = std::move(w);
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

}  // namespace cnnv::nn::lc0_bt4
