/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "alice_native_v2_network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <new>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../movegen.h"
#include "../../position.h"
#include "../alice_native/alice_native_features.h"
#include "alice_native_v2_session.h"
#include "manifest.h"

namespace Stockfish::Eval::NNUE::AliceNativeV2 {

struct Network::Parameters {
    struct DenseStack {
        std::array<i32, Fc0BiasElementsPerStack>  fc0Bias{};
        std::array<i8, Fc0WeightElementsPerStack> fc0Weight{};
        std::array<i8, Fc0WeightElementsPerStack> fc0InterleavedWeight{};
        std::array<i32, Fc1BiasElementsPerStack>  fc1Bias{};
        std::array<i8, Fc1WeightElementsPerStack> fc1Weight{};
        std::array<i32, Fc2BiasElementsPerStack>  fc2Bias{};
        std::array<i8, Fc2WeightElementsPerStack> fc2Weight{};
    };

    bool allocate_features() {
        pieceSquareWeight.reset(new (std::nothrow) i16[PieceSquareWeightElements]);
        pieceSquarePsqt.reset(new (std::nothrow) i32[PieceSquarePsqtElements]);
        return pieceSquareWeight && pieceSquarePsqt;
    }

    std::array<i16, FtBiasElements>     ftBias{};
    std::unique_ptr<i16[]>              pieceSquareWeight;
    std::unique_ptr<i32[]>              pieceSquarePsqt;
    std::array<DenseStack, LayerStacks> dense{};
    WireMetadata                        wire;
    u64                                 generation = 0;
    std::array<std::array<u8, 32>, TensorCount> tensorDigests{};
};

namespace {

constexpr u64 MaximumManifestBytes = 65536;

enum TensorIndex : usize {
    FtBias,
    PieceSquareWeight,
    PieceSquarePsqt,
    Fc0Bias,
    Fc0Weight,
    Fc1Bias,
    Fc1Weight,
    Fc2Bias,
    Fc2Weight,
};

constexpr std::array<std::string_view, TensorCount> TensorNames = {
  "ft.bias",          "pieceSquare.weight", "pieceSquare.psqt",
  "stack.fc0.bias",   "stack.fc0.weight",   "stack.fc1.bias",
  "stack.fc1.weight", "stack.fc2.bias",     "stack.fc2.weight",
};

constexpr std::array<u64, TensorCount> TensorBytes = {
  FtBiasElements * 2,
  PieceSquareWeightElements * 2,
  PieceSquarePsqtElements * 4,
  u64(LayerStacks) * Fc0BiasElementsPerStack * 4,
  u64(LayerStacks) * Fc0WeightElementsPerStack,
  u64(LayerStacks) * Fc1BiasElementsPerStack * 4,
  u64(LayerStacks) * Fc1WeightElementsPerStack,
  u64(LayerStacks) * Fc2BiasElementsPerStack * 4,
  u64(LayerStacks) * Fc2WeightElementsPerStack,
};

u32 rotate_right(u32 value, unsigned shift) { return (value >> shift) | (value << (32 - shift)); }

class Sha256 {
   public:
    void update(const u8* input, usize size) {
        totalBytes += size;
        while (size)
        {
            const usize take = std::min(size, block.size() - buffered);
            std::memcpy(block.data() + buffered, input, take);
            buffered += take;
            input += take;
            size -= take;
            if (buffered == block.size())
            {
                process_block(block.data());
                buffered = 0;
            }
        }
    }

    std::array<u8, 32> finish() {
        const u64 bitLength = totalBytes * 8;
        block[buffered++]   = 0x80;
        if (buffered > 56)
        {
            std::fill(block.begin() + buffered, block.end(), 0);
            process_block(block.data());
            buffered = 0;
        }
        std::fill(block.begin() + buffered, block.begin() + 56, 0);
        for (usize i = 0; i < 8; ++i)
            block[63 - i] = u8(bitLength >> (8 * i));
        process_block(block.data());

        std::array<u8, 32> digest{};
        for (usize i = 0; i < state.size(); ++i)
            for (usize j = 0; j < 4; ++j)
                digest[4 * i + j] = u8(state[i] >> (24 - 8 * j));
        return digest;
    }

   private:
    void process_block(const u8* source) {
        static constexpr std::array<u32, 64> Constants = {
          0x428A2F98u, 0x71374491u, 0xB5C0FBCFu, 0xE9B5DBA5u, 0x3956C25Bu, 0x59F111F1u,
          0x923F82A4u, 0xAB1C5ED5u, 0xD807AA98u, 0x12835B01u, 0x243185BEu, 0x550C7DC3u,
          0x72BE5D74u, 0x80DEB1FEu, 0x9BDC06A7u, 0xC19BF174u, 0xE49B69C1u, 0xEFBE4786u,
          0x0FC19DC6u, 0x240CA1CCu, 0x2DE92C6Fu, 0x4A7484AAu, 0x5CB0A9DCu, 0x76F988DAu,
          0x983E5152u, 0xA831C66Du, 0xB00327C8u, 0xBF597FC7u, 0xC6E00BF3u, 0xD5A79147u,
          0x06CA6351u, 0x14292967u, 0x27B70A85u, 0x2E1B2138u, 0x4D2C6DFCu, 0x53380D13u,
          0x650A7354u, 0x766A0ABBu, 0x81C2C92Eu, 0x92722C85u, 0xA2BFE8A1u, 0xA81A664Bu,
          0xC24B8B70u, 0xC76C51A3u, 0xD192E819u, 0xD6990624u, 0xF40E3585u, 0x106AA070u,
          0x19A4C116u, 0x1E376C08u, 0x2748774Cu, 0x34B0BCB5u, 0x391C0CB3u, 0x4ED8AA4Au,
          0x5B9CCA4Fu, 0x682E6FF3u, 0x748F82EEu, 0x78A5636Fu, 0x84C87814u, 0x8CC70208u,
          0x90BEFFFAu, 0xA4506CEBu, 0xBEF9A3F7u, 0xC67178F2u,
        };
        std::array<u32, 64> words{};
        for (usize i = 0; i < 16; ++i)
            words[i] = (u32(source[4 * i]) << 24) | (u32(source[4 * i + 1]) << 16)
                     | (u32(source[4 * i + 2]) << 8) | u32(source[4 * i + 3]);
        for (usize i = 16; i < words.size(); ++i)
        {
            const u32 sigma0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18)
                             ^ (words[i - 15] >> 3);
            const u32 sigma1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19)
                             ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + sigma0 + words[i - 7] + sigma1;
        }
        u32 a = state[0], b = state[1], c = state[2], d = state[3];
        u32 e = state[4], f = state[5], g = state[6], h = state[7];
        for (usize i = 0; i < words.size(); ++i)
        {
            const u32 choice     = (e & f) ^ (~e & g);
            const u32 majority   = (a & b) ^ (a & c) ^ (b & c);
            const u32 sigma0     = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const u32 sigma1     = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const u32 temporary1 = h + sigma1 + choice + Constants[i] + words[i];
            const u32 temporary2 = sigma0 + majority;
            h = g; g = f; f = e; e = d + temporary1;
            d = c; c = b; b = a; a = temporary1 + temporary2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    std::array<u32, 8> state = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    std::array<u8, 64> block{};
    usize              buffered   = 0;
    u64                totalBytes = 0;
};

std::string digest_string(const std::array<u8, 32>& digest) {
    std::ostringstream out;
    out << std::hex << std::uppercase << std::setfill('0');
    for (u8 byte : digest)
        out << std::setw(2) << unsigned(byte);
    return out.str();
}

u32 crc32(const u8* bytes, usize size) {
    u32 crc = 0xFFFFFFFFu;
    for (usize index = 0; index < size; ++index)
    {
        crc ^= bytes[index];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc ^ 0xFFFFFFFFu;
}

u32 load_u32(const u8* bytes) {
    return u32(bytes[0]) | (u32(bytes[1]) << 8) | (u32(bytes[2]) << 16)
         | (u32(bytes[3]) << 24);
}

u64 load_u64(const u8* bytes) {
    u64 value = 0;
    for (usize index = 0; index < 8; ++index)
        value |= u64(bytes[index]) << (8 * index);
    return value;
}

class AuthenticatingReader {
   public:
    explicit AuthenticatingReader(std::istream& source) : input(source) {}

    bool read(void* destination, usize bytes, Sha256* tensorHash = nullptr) {
        input.read(static_cast<char*>(destination), std::streamsize(bytes));
        if (input.gcount() != std::streamsize(bytes))
            return false;
        const auto* data = static_cast<const u8*>(destination);
        wholeHash.update(data, bytes);
        if (tensorHash)
            tensorHash->update(data, bytes);
        consumed += bytes;
        return true;
    }

    bool read_u32(u32& value) {
        std::array<u8, 4> bytes{};
        if (!read(bytes.data(), bytes.size()))
            return false;
        value = load_u32(bytes.data());
        return true;
    }

    u64 bytes_consumed() const { return consumed; }
    std::array<u8, 32> finish() { return wholeHash.finish(); }

   private:
    std::istream& input;
    Sha256        wholeHash;
    u64           consumed = 0;
};

std::string path_string(const std::filesystem::path& path) {
#if defined(_WIN32)
    return utf8_from_wstring(path.wstring());
#else
    return path.string();
#endif
}

std::string normalized_path(const std::filesystem::path& path) {
    std::error_code       error;
    std::filesystem::path normalized = std::filesystem::weakly_canonical(path, error);
    if (error)
    {
        error.clear();
        normalized = std::filesystem::absolute(path, error);
    }
    return path_string(error ? path : normalized);
}

std::optional<std::string> normalize_required_sha(std::string_view expected,
                                                  std::string&     normalized) {
    if (expected.size() != 64)
        return "expected SHA-256 must contain exactly 64 hexadecimal characters";
    normalized.assign(expected);
    for (char& character : normalized)
    {
        const unsigned char value = static_cast<unsigned char>(character);
        if (!std::isxdigit(value))
            return "expected SHA-256 contains a non-hexadecimal character";
        character = char(std::toupper(value));
    }
    return std::nullopt;
}

bool read_i8_tensor(AuthenticatingReader& reader,
                    i8*                   destination,
                    u64                   elements,
                    Sha256&               tensorHash,
                    std::string_view      name,
                    u64                   flatBase,
                    std::string&          error) {
    constexpr usize ChunkElements = 65536;
    u64             completed     = 0;
    while (completed < elements)
    {
        const usize take = usize(std::min<u64>(ChunkElements, elements - completed));
        if (!reader.read(destination + completed, take, &tensorHash))
        {
            error = std::string(name) + " is truncated at flat index "
                  + std::to_string(flatBase + completed);
            return false;
        }
        for (usize index = 0; index < take; ++index)
            if (destination[completed + index] == std::numeric_limits<i8>::min())
            {
                error = std::string(name) + " contains forbidden -128 at flat index "
                      + std::to_string(flatBase + completed + index);
                return false;
            }
        completed += take;
    }
    return true;
}

bool read_i16_tensor(AuthenticatingReader& reader,
                     i16*                  destination,
                     u64                   elements,
                     Sha256&               tensorHash,
                     std::string_view      name,
                     u64                   flatBase,
                     std::string&          error) {
    std::array<u8, 65536> buffer{};
    u64                   completed = 0;
    while (completed < elements)
    {
        const usize take = usize(std::min<u64>(buffer.size() / 2, elements - completed));
        if (!reader.read(buffer.data(), take * 2, &tensorHash))
        {
            error = std::string(name) + " is truncated at flat index "
                  + std::to_string(flatBase + completed);
            return false;
        }
        for (usize index = 0; index < take; ++index)
        {
            const u16 raw = u16(buffer[2 * index]) | (u16(buffer[2 * index + 1]) << 8);
            i16       value;
            std::memcpy(&value, &raw, sizeof(value));
            if (value == std::numeric_limits<i16>::min())
            {
                error = std::string(name) + " contains forbidden -32768 at flat index "
                      + std::to_string(flatBase + completed + index);
                return false;
            }
            destination[completed + index] = value;
        }
        completed += take;
    }
    return true;
}

bool read_i32_tensor(AuthenticatingReader& reader,
                     i32*                  destination,
                     u64                   elements,
                     Sha256&               tensorHash,
                     std::string_view      name,
                     u64                   flatBase,
                     std::string&          error) {
    std::array<u8, 65536> buffer{};
    u64                   completed = 0;
    while (completed < elements)
    {
        const usize take = usize(std::min<u64>(buffer.size() / 4, elements - completed));
        if (!reader.read(buffer.data(), take * 4, &tensorHash))
        {
            error = std::string(name) + " is truncated at flat index "
                  + std::to_string(flatBase + completed);
            return false;
        }
        for (usize index = 0; index < take; ++index)
        {
            const u32 raw = load_u32(buffer.data() + 4 * index);
            i32       value;
            std::memcpy(&value, &raw, sizeof(value));
            if (value == std::numeric_limits<i32>::min())
            {
                error = std::string(name) + " contains forbidden INT32_MIN at flat index "
                      + std::to_string(flatBase + completed + index);
                return false;
            }
            destination[completed + index] = value;
        }
        completed += take;
    }
    return true;
}

template<typename Parameters>
ParameterView make_parameter_view(const Parameters& parameters) {
    ParameterView view;
    view.ftBias            = parameters.ftBias.data();
    view.pieceSquareWeight = parameters.pieceSquareWeight.get();
    view.pieceSquarePsqt   = parameters.pieceSquarePsqt.get();
    for (usize stack = 0; stack < LayerStacks; ++stack)
    {
        const auto& source      = parameters.dense[stack];
        auto&       destination = view.dense[stack];
        destination.fc0Bias     = source.fc0Bias.data();
        destination.fc0Weight   = source.fc0Weight.data();
        destination.fc0InterleavedWeight = source.fc0InterleavedWeight.data();
        destination.fc1Bias     = source.fc1Bias.data();
        destination.fc1Weight   = source.fc1Weight.data();
        destination.fc2Bias     = source.fc2Bias.data();
        destination.fc2Weight   = source.fc2Weight.data();
    }
    return view;
}

template<typename Range>
void write_array(std::ostream& out, const Range& values) {
    out << '[';
    bool first = true;
    for (const auto value : values)
    {
        if (!first)
            out << ',';
        first = false;
        out << i64(value);
    }
    out << ']';
}

void write_piece_indices(std::ostream& out,
                         const AliceNative::PerspectivePieceSnapshot& snapshot) {
    out << '[';
    for (usize index = 0; index < snapshot.pieces.size(); ++index)
    {
        if (index)
            out << ',';
        out << snapshot.pieces[index];
    }
    out << ']';
}

bool same_accumulators(const IntegerAccumulatorSet& left, const IntegerAccumulatorSet& right) {
    for (Color perspective : {WHITE, BLACK})
        if (left[perspective].values != right[perspective].values
            || left[perspective].psqt != right[perspective].psqt)
            return false;
    return true;
}

template<typename Parameters>
std::optional<std::string> validate_numeric_envelopes(const Parameters& parameters) {
    std::array<i16, TransformerWidth> minimumWeight{};
    std::array<i16, TransformerWidth> maximumWeight{};
    minimumWeight.fill(std::numeric_limits<i16>::max());
    maximumWeight.fill(std::numeric_limits<i16>::min() + 1);
    std::array<i32, PsqtBuckets> minimumPsqt{};
    std::array<i32, PsqtBuckets> maximumPsqt{};
    minimumPsqt.fill(std::numeric_limits<i32>::max());
    maximumPsqt.fill(std::numeric_limits<i32>::min() + 1);
    for (u64 row = 0; row < PieceSquareDimensions; ++row)
    {
        const i16* weights = parameters.pieceSquareWeight.get() + row * TransformerWidth;
        for (usize lane = 0; lane < TransformerWidth; ++lane)
        {
            minimumWeight[lane] = std::min(minimumWeight[lane], weights[lane]);
            maximumWeight[lane] = std::max(maximumWeight[lane], weights[lane]);
        }
        const i32* psqt = parameters.pieceSquarePsqt.get() + row * PsqtBuckets;
        for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
        {
            minimumPsqt[bucket] = std::min(minimumPsqt[bucket], psqt[bucket]);
            maximumPsqt[bucket] = std::max(maximumPsqt[bucket], psqt[bucket]);
        }
    }
    for (usize lane = 0; lane < TransformerWidth; ++lane)
    {
        const i64 lower = parameters.ftBias[lane]
                        + 32 * i64(std::min<i16>(0, minimumWeight[lane]));
        const i64 upper = parameters.ftBias[lane]
                        + 32 * i64(std::max<i16>(0, maximumWeight[lane]));
        if (lower < -32767 || upper > 32767)
            return "AliceNative-v2 any-32 feature envelope exceeds signed i16 at lane "
                 + std::to_string(lane) + ": [" + std::to_string(lower) + ", "
                 + std::to_string(upper) + "]";
    }
    for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
    {
        const i64 lower = 32 * i64(std::min<i32>(0, minimumPsqt[bucket]));
        const i64 upper = 32 * i64(std::max<i32>(0, maximumPsqt[bucket]));
        if (lower < -2147483647LL || upper > 2147483647LL)
            return "AliceNative-v2 any-32 PSQT envelope exceeds signed i32 at bucket "
                 + std::to_string(bucket) + ": [" + std::to_string(lower) + ", "
                 + std::to_string(upper) + "]";
    }

    const auto validate_affine = [](std::string_view label, const i8* weights,
                                    const i32* biases, usize outputs, usize stride,
                                    usize inputs) -> std::optional<std::string> {
        for (usize output = 0; output < outputs; ++output)
        {
            i64 lower = biases[output], upper = biases[output];
            for (usize input = 0; input < inputs; ++input)
            {
                const i8 weight = weights[output * stride + input];
                if (weight < 0) lower += i64(weight) * 127;
                else upper += i64(weight) * 127;
            }
            if (lower < std::numeric_limits<i32>::min()
                || upper > std::numeric_limits<i32>::max())
                return std::string(label) + " envelope exceeds signed i32 at row "
                     + std::to_string(output);
        }
        return std::nullopt;
    };
    for (usize stack = 0; stack < LayerStacks; ++stack)
    {
        const auto& dense = parameters.dense[stack];
        if (auto error = validate_affine("AliceNative-v2 fc0", dense.fc0Weight.data(),
                                         dense.fc0Bias.data(), Hidden1, DenseInput, DenseInput))
            return error;
        if (auto error = validate_affine("AliceNative-v2 fc1", dense.fc1Weight.data(),
                                         dense.fc1Bias.data(), Hidden2, Hidden2, Hidden1))
            return error;
        if (auto error = validate_affine("AliceNative-v2 fc2", dense.fc2Weight.data(),
                                         dense.fc2Bias.data(), 1, Hidden2, Hidden2))
            return error;
        // The AVX2 FC0 kernel uses saturating i16 additions for two adjacent
        // four-byte chunks. Seal every possible nonnegative u8 activation so
        // this optimization is bit-exact rather than data-dependent.
        for (usize output = 0; output < Hidden1; ++output)
            for (usize block = 0; block < DenseInput; block += 8)
                for (usize pair = 0; pair < 2; ++pair)
                {
                    i64 lower = 0;
                    i64 upper = 0;
                    for (usize offset : {2 * pair, 2 * pair + 1, 2 * pair + 4,
                                         2 * pair + 5})
                    {
                        const i8 weight = dense.fc0Weight[output * DenseInput + block + offset];
                        if (weight < 0)
                            lower += i64(weight) * FeatureScale;
                        else
                            upper += i64(weight) * FeatureScale;
                    }
                    if (lower < std::numeric_limits<i16>::min()
                        || upper > std::numeric_limits<i16>::max())
                        return "AliceNative-v2 fc0 AVX2 saturation envelope is unsafe at stack "
                             + std::to_string(stack) + " row " + std::to_string(output)
                             + " input block " + std::to_string(block);
                }
    }
    return std::nullopt;
}

template<typename DenseStack>
void prepare_fc0_interleaved(DenseStack& dense) {
    for (usize index = 0; index < dense.fc0Weight.size(); ++index)
    {
        const usize interleaved =
          (index / 4) % (DenseInput / 4) * Hidden1 * 4 + index / DenseInput * 4 + index % 4;
        dense.fc0InterleavedWeight[interleaved] = dense.fc0Weight[index];
    }
}

}  // namespace

Network::Network() = default;
Network::~Network() { assert(activeLeases.load(std::memory_order_acquire) == 0); }

Network::Lease::Lease(const Network* owner, const Parameters* parameters) noexcept :
    network(owner), pinned(parameters) {}
Network::Lease::~Lease() { reset(); }
Network::Lease::Lease(Lease&& other) noexcept : network(other.network), pinned(other.pinned) {
    other.network = nullptr;
    other.pinned  = nullptr;
}
Network::Lease& Network::Lease::operator=(Lease&& other) noexcept {
    if (this != &other)
    {
        reset();
        network = other.network;
        pinned  = other.pinned;
        other.network = nullptr;
        other.pinned  = nullptr;
    }
    return *this;
}
Network::Lease::operator bool() const noexcept { return network && pinned; }
ParameterView Network::Lease::parameter_view() const noexcept {
    return pinned ? make_parameter_view(*pinned) : ParameterView{};
}
u64 Network::Lease::generation() const noexcept { return pinned ? pinned->generation : 0; }
std::string_view Network::Lease::normalized_path() const noexcept {
    return pinned ? std::string_view(pinned->wire.normalizedPath) : std::string_view{};
}
std::string_view Network::Lease::sha256() const noexcept {
    return pinned ? std::string_view(pinned->wire.sha256) : std::string_view{};
}
void Network::Lease::reset() noexcept {
    if (network)
        network->release_lease();
    network = nullptr;
    pinned  = nullptr;
}

std::optional<Network::Lease> Network::acquire_lease(std::string& error) const noexcept {
    error.clear();
    if (replacementInProgress.load(std::memory_order_acquire))
    {
        error = "AliceNative-v2 parameter replacement is in progress.";
        return std::nullopt;
    }
    activeLeases.fetch_add(1, std::memory_order_acq_rel);
    if (replacementInProgress.load(std::memory_order_acquire))
    {
        release_lease();
        error = "AliceNative-v2 parameter replacement is in progress.";
        return std::nullopt;
    }
    const Parameters* parameters = active.get();
    if (!parameters)
    {
        release_lease();
        error = "AliceNative-v2 parameters are not loaded.";
        return std::nullopt;
    }
    return std::optional<Lease>(Lease(this, parameters));
}

void Network::release_lease() const noexcept {
    const u64 previous = activeLeases.fetch_sub(1, std::memory_order_acq_rel);
    assert(previous > 0);
    (void) previous;
}

std::optional<std::string> Network::load(const std::filesystem::path& file,
                                         std::string_view             expectedSha256) {
    const auto reject = [&](std::string reason) -> std::optional<std::string> {
        lastError = std::move(reason);
        return lastError;
    };
    if (activeLeases.load(std::memory_order_acquire))
        return reject("AliceNative-v2 replacement is rejected while a search lease is active");
    bool expectedReplacement = false;
    if (!replacementInProgress.compare_exchange_strong(expectedReplacement, true,
                                                        std::memory_order_acq_rel))
        return reject("another AliceNative-v2 replacement is already in progress");
    struct Guard {
        std::atomic_bool& flag;
        ~Guard() { flag.store(false, std::memory_order_release); }
    } guard{replacementInProgress};
    if (activeLeases.load(std::memory_order_acquire))
        return reject("AliceNative-v2 replacement is rejected while a search lease is active");

    std::string expected;
    if (auto error = normalize_required_sha(expectedSha256, expected))
        return reject(*error);
    if (active && active->generation == std::numeric_limits<u64>::max())
        return reject("AliceNative-v2 parameter generation is exhausted");

    std::ifstream input(file, std::ios::binary);
    if (!input)
        return reject("AliceNative-v2 parameter file could not be opened: " + normalized_path(file));
    input.seekg(0, std::ios::end);
    const std::streampos end = input.tellg();
    if (end == std::streampos(-1))
        return reject("AliceNative-v2 parameter size could not be derived");
    const u64 fileSize = u64(end);
    input.seekg(0, std::ios::beg);

    AuthenticatingReader reader(input);
    std::array<u8, HeaderBytes> header{};
    if (!reader.read(header.data(), header.size()))
        return reject("AliceNative-v2 header is truncated");
    if (!std::equal(WireMagic.begin(), WireMagic.end(), header.begin()))
        return reject("AliceNative-v2 wire magic mismatch");
    const u32 version       = load_u32(header.data() + 8);
    const u32 manifestBytes = load_u32(header.data() + 12);
    const u64 payloadBytes  = load_u64(header.data() + 16);
    const u32 manifestCrc   = load_u32(header.data() + 24);
    const u32 headerCrc     = load_u32(header.data() + 28);
    if (crc32(header.data(), 28) != headerCrc)
        return reject("AliceNative-v2 header CRC32 mismatch");
    if (version != WireVersion)
        return reject("AliceNative-v2 wire version mismatch");
    if (manifestBytes == 0 || manifestBytes > MaximumManifestBytes)
        return reject("AliceNative-v2 manifest length is outside 1..65536");
    if (manifestBytes != CanonicalManifestBytes)
        return reject("AliceNative-v2 canonical manifest length mismatch");
    if (payloadBytes != PayloadBytes)
        return reject("AliceNative-v2 payload length mismatch");
    const u64 expectedFileBytes = HeaderBytes + u64(manifestBytes) + payloadBytes;
    if (fileSize < expectedFileBytes)
        return reject("AliceNative-v2 parameter file is truncated");
    if (fileSize > expectedFileBytes)
        return reject("AliceNative-v2 parameter file has trailing data");

    std::string manifest(manifestBytes, '\0');
    if (!reader.read(manifest.data(), manifest.size()))
        return reject("AliceNative-v2 manifest is truncated");
    if (crc32(reinterpret_cast<const u8*>(manifest.data()), manifest.size()) != manifestCrc)
        return reject("AliceNative-v2 manifest CRC32 mismatch");
    Sha256 manifestHasher;
    manifestHasher.update(reinterpret_cast<const u8*>(manifest.data()), manifest.size());
    const std::string manifestDigest = digest_string(manifestHasher.finish());
    if (manifestDigest != ManifestSha256)
        return reject("AliceNative-v2 canonical manifest SHA-256 mismatch");
    const auto require_manifest = [&](std::string_view token, std::string_view label)
      -> std::optional<std::string> {
        if (manifest.find(token) == std::string::npos)
            return "AliceNative-v2 manifest is missing " + std::string(label);
        return std::nullopt;
    };
    if (auto error = require_manifest("\"architecture\":\"AliceNative-v2-M512\"", "architecture"))
        return reject(*error);
    if (auto error = require_manifest("\"quantization\":\"alice-native-piece-i16-psqt-i32-dense-i8-v1\"", "quantization"))
        return reject(*error);
    if (auto error = require_manifest(std::string("\"quantizationContractSha256\":\"")
                                        + std::string(QuantizationContractSha256) + "\"",
                                      "quantization contract identity"))
        return reject(*error);
    if (auto error = require_manifest(std::string("\"checkpointAdapter\":{\"format\":\"")
                                        + std::string(CheckpointAdapterId) + "\",\"sha256\":\""
                                        + std::string(CheckpointAdapterSha256) + "\"}",
                                      "checkpoint adapter identity"))
        return reject(*error);

    std::unique_ptr<Parameters> candidate(new (std::nothrow) Parameters());
    if (!candidate || !candidate->allocate_features())
        return reject("AliceNative-v2 parameter allocation failed");
    std::array<Sha256, TensorCount> tensorHashes;
    std::string parseError;
    u32 featureHash = 0;
    if (!reader.read_u32(featureHash) || featureHash != FeatureComponentHash)
        return reject("AliceNative-v2 feature component hash mismatch");
    if (!read_i16_tensor(reader, candidate->ftBias.data(), FtBiasElements, tensorHashes[FtBias],
                         TensorNames[FtBias], 0, parseError)
        || !read_i16_tensor(reader, candidate->pieceSquareWeight.get(),
                            PieceSquareWeightElements, tensorHashes[PieceSquareWeight],
                            TensorNames[PieceSquareWeight], 0, parseError)
        || !read_i32_tensor(reader, candidate->pieceSquarePsqt.get(), PieceSquarePsqtElements,
                            tensorHashes[PieceSquarePsqt], TensorNames[PieceSquarePsqt], 0,
                            parseError))
        return reject(parseError);

    for (usize stack = 0; stack < LayerStacks; ++stack)
    {
        u32 denseHash = 0;
        if (!reader.read_u32(denseHash) || denseHash != DenseComponentHash)
            return reject("AliceNative-v2 dense component hash mismatch at stack "
                          + std::to_string(stack));
        auto& dense = candidate->dense[stack];
        if (!read_i32_tensor(reader, dense.fc0Bias.data(), Fc0BiasElementsPerStack,
                             tensorHashes[Fc0Bias], TensorNames[Fc0Bias],
                             stack * Fc0BiasElementsPerStack, parseError)
            || !read_i8_tensor(reader, dense.fc0Weight.data(), Fc0WeightElementsPerStack,
                               tensorHashes[Fc0Weight], TensorNames[Fc0Weight],
                               stack * Fc0WeightElementsPerStack, parseError)
            || !read_i32_tensor(reader, dense.fc1Bias.data(), Fc1BiasElementsPerStack,
                                tensorHashes[Fc1Bias], TensorNames[Fc1Bias],
                                stack * Fc1BiasElementsPerStack, parseError)
            || !read_i8_tensor(reader, dense.fc1Weight.data(), Fc1WeightElementsPerStack,
                               tensorHashes[Fc1Weight], TensorNames[Fc1Weight],
                               stack * Fc1WeightElementsPerStack, parseError)
            || !read_i32_tensor(reader, dense.fc2Bias.data(), Fc2BiasElementsPerStack,
                                tensorHashes[Fc2Bias], TensorNames[Fc2Bias],
                                stack * Fc2BiasElementsPerStack, parseError)
            || !read_i8_tensor(reader, dense.fc2Weight.data(), Fc2WeightElementsPerStack,
                               tensorHashes[Fc2Weight], TensorNames[Fc2Weight],
                               stack * Fc2WeightElementsPerStack, parseError))
            return reject(parseError);
        for (usize row = 0; row < Hidden2; ++row)
            for (usize column = Hidden1; column < Hidden2; ++column)
                if (dense.fc1Weight[row * Hidden2 + column] != 0)
                    return reject("AliceNative-v2 FC1 SIMD padding is nonzero at stack "
                                  + std::to_string(stack) + " row " + std::to_string(row)
                                  + " column " + std::to_string(column));
        prepare_fc0_interleaved(dense);
    }

    if (auto error = validate_numeric_envelopes(*candidate))
        return reject(*error);

    if (reader.bytes_consumed() != fileSize || reader.bytes_consumed() != expectedFileBytes)
        return reject("AliceNative-v2 parser did not consume the exact file size");
    if (input.peek() != std::char_traits<char>::eof())
        return reject("AliceNative-v2 file contains trailing data");
    const std::string fileDigest = digest_string(reader.finish());
    if (fileDigest != expected)
        return reject("AliceNative-v2 whole-file SHA-256 mismatch: expected " + expected
                      + ", got " + fileDigest);

    for (usize tensor = 0; tensor < TensorCount; ++tensor)
        candidate->tensorDigests[tensor] = tensorHashes[tensor].finish();
    candidate->wire.normalizedPath = normalized_path(file);
    candidate->wire.bytes          = fileSize;
    candidate->wire.payloadBytes   = payloadBytes;
    candidate->wire.manifestBytes  = manifestBytes;
    candidate->wire.sha256         = fileDigest;
    candidate->wire.manifestSha256 = manifestDigest;
    candidate->wire.version        = version;
    candidate->generation          = active ? active->generation + 1 : 1;

    active = std::move(candidate);
    lastError.clear();
    return std::nullopt;
}

bool Network::loaded() const { return bool(active); }
u64 Network::generation() const { return active ? active->generation : 0; }
const std::string& Network::last_error() const { return lastError; }

std::string Network::status_line() const {
    if (!active)
        return "AliceNative-v2 M512 parameters are not loaded"
             + (lastError.empty() ? std::string(".") : ": " + lastError);
    std::ostringstream out;
    out << "AliceNative-v2 M512 parameters loaded generation=" << active->generation
        << " path=\"" << active->wire.normalizedPath << "\" bytes=" << active->wire.bytes
        << " payload_bytes=" << active->wire.payloadBytes << " sha256=" << active->wire.sha256
        << " manifest_sha256=" << active->wire.manifestSha256 << " version=0x" << std::hex
        << std::uppercase << active->wire.version << " search=available";
    return out.str();
}

std::string Network::tensor_status_line() const {
    if (!active)
        return "AliceNative-v2 tensor identities are unavailable.";
    std::ostringstream out;
    out << "AliceNative-v2 tensors generation=" << active->generation;
    for (usize tensor = 0; tensor < TensorCount; ++tensor)
        out << ' ' << TensorNames[tensor] << "_bytes=" << TensorBytes[tensor] << ' '
            << TensorNames[tensor] << "_sha256=" << digest_string(active->tensorDigests[tensor]);
    return out.str();
}

std::optional<std::string>
Network::probe(std::string_view tensor, u64 index, std::string& report) const {
    if (!active)
        return "AliceNative-v2 parameters are not loaded.";
    i64 value = 0;
    if (tensor == "ft.bias")
    {
        if (index >= FtBiasElements) return "ft.bias index is out of range";
        value = active->ftBias[index];
    }
    else if (tensor == "pieceSquare.weight")
    {
        if (index >= PieceSquareWeightElements) return "pieceSquare.weight index is out of range";
        value = active->pieceSquareWeight[index];
    }
    else if (tensor == "pieceSquare.psqt")
    {
        if (index >= PieceSquarePsqtElements) return "pieceSquare.psqt index is out of range";
        value = active->pieceSquarePsqt[index];
    }
    else
    {
        usize tensorIndex = TensorCount;
        for (usize candidate = Fc0Bias; candidate < TensorCount; ++candidate)
            if (tensor == TensorNames[candidate]) tensorIndex = candidate;
        if (tensorIndex == TensorCount)
            return "unknown AliceNative-v2 tensor: " + std::string(tensor);
        const std::array<u64, 6> perStack = {
          Fc0BiasElementsPerStack, Fc0WeightElementsPerStack, Fc1BiasElementsPerStack,
          Fc1WeightElementsPerStack, Fc2BiasElementsPerStack, Fc2WeightElementsPerStack};
        const u64 count = perStack[tensorIndex - Fc0Bias];
        if (index >= u64(LayerStacks) * count) return "dense tensor index is out of range";
        const usize stack = usize(index / count), local = usize(index % count);
        const auto& dense = active->dense[stack];
        switch (tensorIndex)
        {
        case Fc0Bias: value = dense.fc0Bias[local]; break;
        case Fc0Weight: value = dense.fc0Weight[local]; break;
        case Fc1Bias: value = dense.fc1Bias[local]; break;
        case Fc1Weight: value = dense.fc1Weight[local]; break;
        case Fc2Bias: value = dense.fc2Bias[local]; break;
        case Fc2Weight: value = dense.fc2Weight[local]; break;
        default: return "internal AliceNative-v2 probe mapping failure";
        }
    }
    std::ostringstream out;
    out << "alice_native_v2_parameter generation " << active->generation << " tensor " << tensor
        << " index " << index << " value " << value;
    report = out.str();
    return std::nullopt;
}

std::optional<std::string> Network::integer_trace(const Position& position,
                                                  std::string&    report) const {
    if (!active)
        return "AliceNative-v2 integer trace requires loaded parameters.";
    AliceNative::PieceSnapshot snapshot;
    if (auto error = AliceNative::build_piece_snapshot(position, snapshot)) return error;
    IntegerAccumulatorSet accumulators;
    const ParameterView view = make_parameter_view(*active);
    for (Color perspective : {WHITE, BLACK})
        if (auto error = refresh_accumulator(view, snapshot[perspective],
                                             accumulators[perspective]))
            return error;
    IntegerStages stages;
    if (auto error = evaluate(view, position, accumulators, stages)) return error;
    i32 fastRaw = 0, fastAdjusted = 0;
    if (auto error = evaluate_value(view, position, accumulators, false, fastRaw)) return error;
    if (auto error = evaluate_value(view, position, accumulators, true, fastAdjusted)) return error;
    if (fastRaw != stages.rawValue || fastAdjusted != stages.adjustedValue)
        return "AliceNative-v2 optimized inference differs from the scalar integer trace.";

    std::ostringstream out;
    out << "{\"schema\":\"alice-native-v2-index-trace-v1\",\"architecture\":\"AliceNative-v2-M512\",\"generation\":"
        << active->generation << ",\"networkSha256\":\"" << active->wire.sha256
        << "\",\"fen\":\"" << position.fen() << "\",\"sideToMove\":"
        << int(stages.sideToMove) << ",\"pieceCount\":"
        << stages.pieceCount << ",\"phase\":" << stages.phase << ",\"pieceFeatures\":[";
    for (Color perspective : {WHITE, BLACK})
    {
        if (perspective != WHITE) out << ',';
        write_piece_indices(out, snapshot[perspective]);
    }
    out << "],\"featureAccumulator\":[";
    for (Color perspective : {WHITE, BLACK})
    {
        if (perspective != WHITE) out << ',';
        write_array(out, accumulators[perspective].values);
    }
    out << "],\"psqtAccumulator\":[";
    for (Color perspective : {WHITE, BLACK})
    {
        if (perspective != WHITE) out << ',';
        write_array(out, accumulators[perspective].psqt);
    }
    out << "],\"transformedByPerspective\":[";
    for (Color perspective : {WHITE, BLACK})
    {
        if (perspective != WHITE) out << ',';
        write_array(out, stages.transformed[perspective]);
    }
    out << "],\"denseInput\":"; write_array(out, stages.denseInput);
    out << ",\"fc0Raw\":"; write_array(out, stages.fc0Raw);
    out << ",\"fc0Linear\":"; write_array(out, stages.fc0);
    std::array<i32, Hidden2> fc1Input{};
    std::copy(stages.fc0.begin(), stages.fc0.end(), fc1Input.begin());
    out << ",\"fc1Input\":"; write_array(out, fc1Input);
    out << ",\"fc1Raw\":"; write_array(out, stages.fc1Raw);
    out << ",\"fc1Linear\":"; write_array(out, stages.fc1);
    const i64 psqtDifference =
      i64(accumulators[stages.sideToMove].psqt[stages.phase])
      - i64(accumulators[~stages.sideToMove].psqt[stages.phase]);
    out << ",\"fc2Raw\":" << stages.fc2Raw << ",\"psqtDifference\":"
        << psqtDifference << ",\"materialRaw\":" << stages.materialRaw
        << ",\"mixedRaw\":" << stages.combinedRaw
        << ",\"nativeNnueValue\":" << stages.rawValue
        << ",\"adjustedEntertainmentWeightingApplied\":"
        << (stages.entertainment != 0 ? "true" : "false")
        << ",\"entertainment\":" << stages.entertainment << ",\"adjustedRaw\":"
        << stages.adjustedRaw << ",\"rawValue\":" << stages.rawValue
        << ",\"adjustedValue\":" << stages.adjustedValue << '}';
    report = out.str();
    return std::nullopt;
}

std::optional<std::string>
Network::verify_incremental(Position& position, Depth depth, IncrementalVerificationStats& stats) const {
    stats = {};
    if (!active) return "AliceNative-v2 incremental verification requires loaded parameters.";
    if (depth < 0 || depth > 2) return "AliceNative-v2 incremental depth must be between 0 and 2.";
    const ParameterView view = make_parameter_view(*active);
    const std::string rootFen = position.fen();
    const Key rootKey = position.key();
    AliceNative::PieceSnapshot rootSnapshot;
    if (auto error = AliceNative::build_piece_snapshot(position, rootSnapshot)) return error;
    IntegerAccumulatorSet rootAccumulators;
    for (Color perspective : {WHITE, BLACK})
        if (auto error = refresh_accumulator(view, rootSnapshot[perspective],
                                             rootAccumulators[perspective])) return error;

    std::function<std::optional<std::string>(Depth, const AliceNative::PieceSnapshot&,
                                             const IntegerAccumulatorSet&)> visit;
    visit = [&](Depth remaining, const AliceNative::PieceSnapshot& snapshot,
                const IntegerAccumulatorSet& incremental) -> std::optional<std::string> {
        ++stats.positions;
        AliceNative::PieceSnapshot fullSnapshot;
        if (auto error = AliceNative::build_piece_snapshot(position, fullSnapshot)) return error;
        IntegerAccumulatorSet full;
        for (Color perspective : {WHITE, BLACK})
            if (auto error = refresh_accumulator(view, fullSnapshot[perspective], full[perspective]))
                return error;
        ++stats.accumulatorComparisons;
        if (!same_accumulators(incremental, full))
            return "AliceNative-v2 incremental accumulator differs from full refresh.";
        IntegerStages incrementalStages, fullStages;
        if (auto error = evaluate(view, position, incremental, incrementalStages)) return error;
        if (auto error = evaluate(view, position, full, fullStages)) return error;
        ++stats.stageComparisons;
        if (!same_stages(incrementalStages, fullStages))
            return "AliceNative-v2 incremental stages differ from full refresh.";
        i32 fastRaw = 0, fastAdjusted = 0;
        if (auto error = evaluate_value(view, position, incremental, false, fastRaw)) return error;
        if (auto error = evaluate_value(view, position, incremental, true, fastAdjusted))
            return error;
        ++stats.fastPathComparisons;
        if (fastRaw != incrementalStages.rawValue
            || fastAdjusted != incrementalStages.adjustedValue)
            return "AliceNative-v2 optimized inference differs from scalar stages.";
        if (remaining == 0) return std::nullopt;

        const std::string parentFen = position.fen();
        const Key parentKey = position.key();
        std::vector<Move> moves;
        for (Move move : MoveList<LEGAL>(position)) moves.push_back(move);
        for (Move move : moves)
        {
            const Piece moved = position.moved_piece(move);
            stats.captures += position.capture(move);
            stats.promotions += move.type_of() == PROMOTION;
            stats.castlings += move.type_of() == CASTLING;
            stats.kingMoves += type_of(moved) == KING;
            StateInfo state;
            Dirties dirties;
            position.do_move(move, state, position.gives_check(move), dirties, nullptr, nullptr);
            AliceNative::PieceSnapshot childSnapshot;
            auto error = AliceNative::build_piece_snapshot(position, childSnapshot);
            IntegerAccumulatorSet child = incremental;
            if (!error)
                for (Color perspective : {WHITE, BLACK})
                {
                    const auto& before = snapshot[perspective];
                    const auto& after  = childSnapshot[perspective];
                    if (before.kingSquare != after.kingSquare || before.kingBoard != after.kingBoard)
                    {
                        error = refresh_accumulator(view, after, child[perspective]);
                        ++stats.fullRefreshes[perspective];
                    }
                    else
                    {
                        AccumulatorDeltaStats delta;
                        error = update_accumulator(view, before, after, child[perspective], delta);
                        stats.pieceAdds += delta.pieceAdds;
                        stats.pieceRemoves += delta.pieceRemoves;
                    }
                    if (error) break;
                }
            ++stats.transitions;
            if (!error) error = visit(remaining - 1, childSnapshot, child);
            position.undo_move(move);
            ++stats.undoChecks;
            if (position.fen() != parentFen || position.key() != parentKey)
                return "AliceNative-v2 incremental verification did not restore its parent.";
            if (error) return error;
        }
        return std::nullopt;
    };

    auto error = visit(depth, rootSnapshot, rootAccumulators);
    if (position.fen() != rootFen || position.key() != rootKey)
        return "AliceNative-v2 incremental verification did not restore the root.";
    return error;
}

std::optional<std::string>
Network::verify_session(Position& position, Depth depth, std::string& report) const {
    report.clear();
    if (depth < 0 || depth > 2)
        return "AliceNative-v2 search-session verification depth must be between 0 and 2.";

    std::string leaseError;
    auto        lease = acquire_lease(leaseError);
    if (!lease)
        return "AliceNative-v2 search-session verification requires loaded parameters: "
             + leaseError;

    const Parameters*   parameters         = lease->pinned;
    const ParameterView view               = lease->parameter_view();
    const u64           verifiedGeneration = lease->generation();
    const std::string   verifiedSha256(lease->sha256());
    const std::string   rootFen        = position.fen();
    const Key           rootKey        = position.key();
    const Bitboard      rootBoardB     = position.state()->boardB;
    const Color         rootSideToMove = position.side_to_move();
    const int           rootPieceCount = position.count<ALL_PIECES>();

    std::unique_ptr<SearchSession> session(new (std::nothrow) SearchSession(
      view, verifiedGeneration, verifiedSha256, position));
    if (!session)
        return "AliceNative-v2 search-session allocation failed.";

    const auto describe_failure = [](std::string_view                action,
                                     const AliceSearch::EvalFailure& failure) {
        std::ostringstream out;
        out << "AliceNative-v2 search session " << action
            << " failed: code=" << AliceSearch::failure_code_name(failure.code)
            << " stage=" << AliceSearch::failure_stage_name(failure.stage)
            << " generation=" << failure.generation << " ply=" << failure.ply
            << " perspective=" << failure.perspective << ".";
        return out.str();
    };

    if (!session->ready())
    {
        Value                    ignored = VALUE_ZERO;
        AliceSearch::EvalFailure failure;
        session->evaluate(position, ignored, failure);
        return describe_failure("initialization", failure);
    }

    u64 positions         = 0;
    u64 transitions       = 0;
    u64 captures          = 0;
    u64 promotions        = 0;
    u64 castlings         = 0;
    u64 kingMoves         = 0;
    u64 accumulatorChecks = 0;
    u64 stageChecks       = 0;
    u64 valueChecks       = 0;
    u64 undoChecks        = 0;
    u64 nullTransitions   = 0;
    u64 nullAccumulatorChecks = 0;
    u64 nullValueChecks   = 0;
    u64 nullUndoChecks    = 0;

    std::function<std::optional<std::string>(Depth)> visit;
    visit = [&](Depth remaining) -> std::optional<std::string> {
        ++positions;
        if (!session->matches_current(position))
            return "AliceNative-v2 search session did not match the current position.";

        Value                    sessionValue = VALUE_ZERO;
        AliceSearch::EvalFailure evaluationFailure;
        if (!session->evaluate(position, sessionValue, evaluationFailure))
            return describe_failure("evaluation", evaluationFailure);

        AliceNative::PieceSnapshot fullSnapshot;
        if (auto error = AliceNative::build_piece_snapshot(position, fullSnapshot))
            return error;
        IntegerAccumulatorSet full;
        for (Color perspective : {WHITE, BLACK})
            if (auto error = refresh_accumulator(view, fullSnapshot[perspective],
                                                 full[perspective]))
                return error;

        const auto& current = session->current_accumulators();
        if (!same_accumulators(current, full))
            return "AliceNative-v2 search-session accumulator mismatch at " + position.fen()
                 + ".";
        ++accumulatorChecks;

        IntegerStages currentStages, fullStages;
        if (auto error = evaluate(view, position, current, currentStages))
            return error;
        if (auto error = evaluate(view, position, full, fullStages))
            return error;
        if (!same_stages(currentStages, fullStages))
            return "AliceNative-v2 search-session integer-stage mismatch at " + position.fen()
                 + ".";
        ++stageChecks;
        if (sessionValue != fullStages.adjustedValue)
            return "AliceNative-v2 search-session value mismatch at " + position.fen() + ".";
        ++valueChecks;

        // Stockfish's null-move search changes identity and side to move without changing
        // any piece feature. Exercise that exact transition explicitly: the session must
        // copy its accumulators, evaluate from the opposite perspective, and restore the
        // parent transactionally.
        if (!position.checkers())
        {
            const std::string nullParentFen    = position.fen();
            const Key         nullParentKey    = position.key();
            const Bitboard    nullParentBoardB = position.state()->boardB;
            const Color       nullParentSide   = position.side_to_move();
            const int         nullParentPieces = position.count<ALL_PIECES>();

            StateInfo nullState;
            position.do_null_move(nullState);
            ++nullTransitions;

            std::optional<std::string> nullError;
            bool                       nullPushed = false;
            AliceSearch::EvalFailure   nullPushFailure;
            if (!session->push_null(position, nullPushFailure))
                nullError = describe_failure("null push", nullPushFailure);
            else
            {
                nullPushed = true;
                Value                    nullSessionValue = VALUE_ZERO;
                AliceSearch::EvalFailure nullEvaluationFailure;
                if (!session->evaluate(position, nullSessionValue, nullEvaluationFailure))
                    nullError = describe_failure("null evaluation", nullEvaluationFailure);
                else
                {
                    AliceNative::PieceSnapshot nullSnapshot;
                    IntegerAccumulatorSet      nullFull;
                    if (auto error = AliceNative::build_piece_snapshot(position, nullSnapshot))
                        nullError = error;
                    for (Color perspective : {WHITE, BLACK})
                        if (!nullError)
                            if (auto error = refresh_accumulator(
                                  view, nullSnapshot[perspective], nullFull[perspective]))
                                nullError = error;
                    if (!nullError && !same_accumulators(session->current_accumulators(), nullFull))
                        nullError = "AliceNative-v2 null-move accumulator mismatch.";
                    if (!nullError)
                    {
                        ++nullAccumulatorChecks;
                        i32 nullFullValue = 0;
                        if (auto error = evaluate_value(view, position, nullFull, true,
                                                        nullFullValue))
                            nullError = error;
                        else if (nullSessionValue != nullFullValue)
                            nullError = "AliceNative-v2 null-move value mismatch.";
                        else
                            ++nullValueChecks;
                    }
                }
            }

            position.undo_null_move();
            if (nullPushed)
            {
                AliceSearch::EvalFailure nullPopFailure;
                if (!session->pop(position, nullPopFailure) && !nullError)
                    nullError = describe_failure("null pop", nullPopFailure);
                else if (!nullError)
                    ++nullUndoChecks;
            }

            if (!nullError
                && (position.fen() != nullParentFen || position.key() != nullParentKey
                    || position.state()->boardB != nullParentBoardB
                    || position.side_to_move() != nullParentSide
                    || position.count<ALL_PIECES>() != nullParentPieces
                    || !session->matches_current(position)))
                nullError = "AliceNative-v2 null move did not restore its parent position.";
            if (nullError)
                return nullError;
        }

        if (remaining == 0)
            return std::nullopt;

        const std::string nodeFen    = position.fen();
        const Key         nodeKey    = position.key();
        const Bitboard    nodeBoardB = position.state()->boardB;
        const Color       nodeSide   = position.side_to_move();
        const int         nodePieces = position.count<ALL_PIECES>();
        std::vector<Move> legalMoves;
        for (Move move : MoveList<LEGAL>(position))
            legalMoves.push_back(move);

        for (Move move : legalMoves)
        {
            const Piece moved = position.moved_piece(move);
            captures += position.capture(move);
            promotions += move.type_of() == PROMOTION;
            castlings += move.type_of() == CASTLING;
            kingMoves += type_of(moved) == KING;

            StateInfo state;
            Dirties   dirties;
            position.do_move(move, state, position.gives_check(move), dirties, nullptr, nullptr);
            ++transitions;

            AliceSearch::EvalFailure pushFailure;
            if (!session->push(position, dirties, pushFailure))
            {
                position.undo_move(move);
                return describe_failure("push", pushFailure);
            }

            auto error = visit(remaining - 1);
            position.undo_move(move);

            AliceSearch::EvalFailure popFailure;
            if (!session->pop(position, popFailure))
                return describe_failure("pop", popFailure);
            ++undoChecks;

            if (position.fen() != nodeFen || position.key() != nodeKey
                || position.state()->boardB != nodeBoardB || position.side_to_move() != nodeSide
                || position.count<ALL_PIECES>() != nodePieces
                || !session->matches_current(position))
                return "AliceNative-v2 search session did not restore a parent position.";
            if (error)
                return error;
        }
        return std::nullopt;
    };

    if (auto error = visit(depth))
        return error;
    if (position.fen() != rootFen || position.key() != rootKey
        || position.state()->boardB != rootBoardB || position.side_to_move() != rootSideToMove
        || position.count<ALL_PIECES>() != rootPieceCount || session->ply() != 0
        || !session->matches_current(position))
        return "AliceNative-v2 search session did not restore its root position.";
    if (active.get() != parameters || active->generation != verifiedGeneration
        || active->wire.sha256 != verifiedSha256)
        return "AliceNative-v2 parameters changed during search-session verification.";

    const RuntimeSessionStats& runtime = session->stats();
    if (runtime.evaluations != positions + nullValueChecks
        || runtime.pushes != transitions + nullTransitions
        || runtime.pops != transitions + nullTransitions
        || runtime.nullPushes != nullTransitions || runtime.nullPops != nullTransitions
        || accumulatorChecks != positions
        || stageChecks != positions || valueChecks != positions || undoChecks != transitions
        || nullAccumulatorChecks != nullTransitions || nullValueChecks != nullTransitions
        || nullUndoChecks != nullTransitions)
        return "AliceNative-v2 search-session counters violated their traversal invariants.";

    std::ostringstream out;
    out << "alice_native_v2 session verified generation " << verifiedGeneration << " positions "
        << positions << " transitions " << transitions << " captures " << captures
        << " promotions " << promotions << " castlings " << castlings << " king_moves "
        << kingMoves << " evaluations " << runtime.evaluations << " pushes " << runtime.pushes
        << " pops " << runtime.pops << " refreshes " << runtime.fullRefreshes[WHITE] << ','
        << runtime.fullRefreshes[BLACK] << " piece_adds " << runtime.pieceAdds
        << " piece_removes " << runtime.pieceRemoves << " max_piece_events "
        << runtime.maxPieceEvents << " accumulator_checks " << accumulatorChecks
        << " integer_stage_checks " << stageChecks << " value_checks " << valueChecks
        << " undo_checks " << undoChecks << " null_transitions " << nullTransitions
        << " null_accumulator_checks " << nullAccumulatorChecks << " null_value_checks "
        << nullValueChecks << " null_undo_checks " << nullUndoChecks << " depth " << depth
        << " search available";
    report = out.str();
    return std::nullopt;
}

}  // namespace Stockfish::Eval::NNUE::AliceNativeV2
