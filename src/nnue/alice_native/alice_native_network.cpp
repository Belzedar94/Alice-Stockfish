/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "alice_native_network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../movegen.h"
#include "../../position.h"
#include "../simd.h"
#include "alice_native_features.h"
#include "alice_native_inference.h"
#include "alice_native_session.h"

namespace Stockfish::Eval::NNUE::AliceNative {

struct QualificationNetwork::Parameters {
    struct DenseStack {
        std::array<i32, Fc0BiasElementsPerStack>  fc0Bias{};
        std::array<i8, Fc0WeightElementsPerStack> fc0Weight{};
        std::array<i32, Fc1BiasElementsPerStack>  fc1Bias{};
        std::array<i8, Fc1WeightElementsPerStack> fc1Weight{};
        std::array<i32, Fc2BiasElementsPerStack>  fc2Bias{};
        std::array<i8, Fc2WeightElementsPerStack> fc2Weight{};
    };

    bool allocate_features() {
        threatWeight.reset(new (std::nothrow) i8[ThreatWeightElements]);
        threatPsqt.reset(new (std::nothrow) i32[ThreatPsqtElements]);
        pieceSquareWeight.reset(new (std::nothrow) i16[PieceSquareWeightElements]);
        pieceSquarePsqt.reset(new (std::nothrow) i32[PieceSquarePsqtElements]);
        return threatWeight && threatPsqt && pieceSquareWeight && pieceSquarePsqt;
    }

    std::array<i16, FtBiasElements>     ftBias{};
    std::unique_ptr<i8[]>               threatWeight;
    std::unique_ptr<i32[]>              threatPsqt;
    std::unique_ptr<i16[]>              pieceSquareWeight;
    std::unique_ptr<i32[]>              pieceSquarePsqt;
    std::array<DenseStack, LayerStacks> dense{};

    WireMetadata                                wire;
    u64                                         generation = 0;
    std::array<std::array<u8, 32>, TensorCount> tensorDigests{};
};

namespace {

constexpr u32 LegacyWireVersion    = 0x7AF32F20u;
constexpr u64 MaximumManifestBytes = 65536;

enum TensorIndex : usize {
    FtBias,
    ThreatWeight,
    ThreatPsqt,
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
  "ft.bias",          "threat.weight",  "threat.psqt",      "pieceSquare.weight",
  "pieceSquare.psqt", "stack.fc0.bias", "stack.fc0.weight", "stack.fc1.bias",
  "stack.fc1.weight", "stack.fc2.bias", "stack.fc2.weight",
};

constexpr std::array<u64, TensorCount> TensorBytes = {
  FtBiasElements * 2,
  ThreatWeightElements,
  ThreatPsqtElements * 4,
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
          0x428A2F98u, 0x71374491u, 0xB5C0FBCFu, 0xE9B5DBA5u, 0x3956C25Bu, 0x59F111F1u, 0x923F82A4u,
          0xAB1C5ED5u, 0xD807AA98u, 0x12835B01u, 0x243185BEu, 0x550C7DC3u, 0x72BE5D74u, 0x80DEB1FEu,
          0x9BDC06A7u, 0xC19BF174u, 0xE49B69C1u, 0xEFBE4786u, 0x0FC19DC6u, 0x240CA1CCu, 0x2DE92C6Fu,
          0x4A7484AAu, 0x5CB0A9DCu, 0x76F988DAu, 0x983E5152u, 0xA831C66Du, 0xB00327C8u, 0xBF597FC7u,
          0xC6E00BF3u, 0xD5A79147u, 0x06CA6351u, 0x14292967u, 0x27B70A85u, 0x2E1B2138u, 0x4D2C6DFCu,
          0x53380D13u, 0x650A7354u, 0x766A0ABBu, 0x81C2C92Eu, 0x92722C85u, 0xA2BFE8A1u, 0xA81A664Bu,
          0xC24B8B70u, 0xC76C51A3u, 0xD192E819u, 0xD6990624u, 0xF40E3585u, 0x106AA070u, 0x19A4C116u,
          0x1E376C08u, 0x2748774Cu, 0x34B0BCB5u, 0x391C0CB3u, 0x4ED8AA4Au, 0x5B9CCA4Fu, 0x682E6FF3u,
          0x748F82EEu, 0x78A5636Fu, 0x84C87814u, 0x8CC70208u, 0x90BEFFFAu, 0xA4506CEBu, 0xBEF9A3F7u,
          0xC67178F2u,
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

        u32 a = state[0];
        u32 b = state[1];
        u32 c = state[2];
        u32 d = state[3];
        u32 e = state[4];
        u32 f = state[5];
        u32 g = state[6];
        u32 h = state[7];

        for (usize i = 0; i < words.size(); ++i)
        {
            const u32 choice     = (e & f) ^ (~e & g);
            const u32 majority   = (a & b) ^ (a & c) ^ (b & c);
            const u32 sigma0     = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const u32 sigma1     = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const u32 temporary1 = h + sigma1 + choice + Constants[i] + words[i];
            const u32 temporary2 = sigma0 + majority;
            h                    = g;
            g                    = f;
            f                    = e;
            e                    = d + temporary1;
            d                    = c;
            c                    = b;
            b                    = a;
            a                    = temporary1 + temporary2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    std::array<u32, 8> state = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    std::array<u8, 64> block{};
    usize              buffered   = 0;
    u64                totalBytes = 0;
};

class AuthenticatingReader {
   public:
    explicit AuthenticatingReader(std::istream& source) :
        input(source) {}

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
        value =
          u32(bytes[0]) | (u32(bytes[1]) << 8) | (u32(bytes[2]) << 16) | (u32(bytes[3]) << 24);
        return true;
    }

    u64                bytes_consumed() const { return consumed; }
    std::array<u8, 32> finish() { return wholeHash.finish(); }

   private:
    std::istream& input;
    Sha256        wholeHash;
    u64           consumed = 0;
};

std::string digest_string(const std::array<u8, 32>& digest) {
    std::ostringstream out;
    out << std::hex << std::uppercase << std::setfill('0');
    for (u8 byte : digest)
        out << std::setw(2) << unsigned(byte);
    return out.str();
}

std::string sha256(std::string_view bytes) {
    Sha256 hash;
    hash.update(reinterpret_cast<const u8*>(bytes.data()), bytes.size());
    return digest_string(hash.finish());
}

std::optional<std::string> sha256_file(const std::filesystem::path& path, std::string& result) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return "could not reopen the file for SHA-256";

    Sha256                  hash;
    std::array<char, 65536> buffer{};
    while (input)
    {
        input.read(buffer.data(), buffer.size());
        const std::streamsize count = input.gcount();
        if (count > 0)
            hash.update(reinterpret_cast<const u8*>(buffer.data()), usize(count));
    }
    if (!input.eof())
        return "failed while reading the file for SHA-256";

    result = digest_string(hash.finish());
    return std::nullopt;
}

bool read_u32(std::istream& input, u32& value) {
    std::array<u8, 4> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!input)
        return false;
    value = u32(bytes[0]) | (u32(bytes[1]) << 8) | (u32(bytes[2]) << 16) | (u32(bytes[3]) << 24);
    return true;
}

std::string hex32(u32 value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << value;
    return out.str();
}

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

std::optional<std::string> normalized_expected_sha(const std::optional<std::string>& expected,
                                                   std::string&                      normalized) {
    normalized.clear();
    if (!expected || expected->empty())
        return std::nullopt;
    if (expected->size() != 64)
        return "expected SHA-256 must contain exactly 64 hexadecimal characters";
    normalized = *expected;
    for (char& character : normalized)
    {
        const unsigned char value = static_cast<unsigned char>(character);
        if (!std::isxdigit(value))
            return "expected SHA-256 contains a non-hexadecimal character";
        character = char(std::toupper(value));
    }
    return std::nullopt;
}

std::optional<std::string> normalized_required_sha(std::string_view expected,
                                                   std::string&     normalized) {
    if (expected.empty())
        return "expected SHA-256 is mandatory for native parameter loading";
    return normalized_expected_sha(std::string(expected), normalized);
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
            const u32 raw = u32(buffer[4 * index]) | (u32(buffer[4 * index + 1]) << 8)
                          | (u32(buffer[4 * index + 2]) << 16) | (u32(buffer[4 * index + 3]) << 24);
            i32 value;
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

void update_i8_digest(Sha256& hash, const i8* source, u64 elements) {
    constexpr u64 ChunkElements = 1 << 20;
    u64           completed     = 0;
    while (completed < elements)
    {
        const usize take = usize(std::min<u64>(ChunkElements, elements - completed));
        hash.update(reinterpret_cast<const u8*>(source + completed), take);
        completed += take;
    }
}

void update_i16_digest(Sha256& hash, const i16* source, u64 elements) {
    std::array<u8, 65536> buffer{};
    u64                   completed = 0;
    while (completed < elements)
    {
        const usize take = usize(std::min<u64>(buffer.size() / 2, elements - completed));
        for (usize index = 0; index < take; ++index)
        {
            u16 raw;
            std::memcpy(&raw, source + completed + index, sizeof(raw));
            buffer[2 * index]     = u8(raw);
            buffer[2 * index + 1] = u8(raw >> 8);
        }
        hash.update(buffer.data(), take * 2);
        completed += take;
    }
}

void update_i32_digest(Sha256& hash, const i32* source, u64 elements) {
    std::array<u8, 65536> buffer{};
    u64                   completed = 0;
    while (completed < elements)
    {
        const usize take = usize(std::min<u64>(buffer.size() / 4, elements - completed));
        for (usize index = 0; index < take; ++index)
        {
            u32 raw;
            std::memcpy(&raw, source + completed + index, sizeof(raw));
            buffer[4 * index]     = u8(raw);
            buffer[4 * index + 1] = u8(raw >> 8);
            buffer[4 * index + 2] = u8(raw >> 16);
            buffer[4 * index + 3] = u8(raw >> 24);
        }
        hash.update(buffer.data(), take * 4);
        completed += take;
    }
}

void affine_bounds(const i8*            weights,
                   const i32*           biases,
                   usize                outputs,
                   usize                inputs,
                   std::array<i64, L2>& lower,
                   std::array<i64, L2>& upper) {
    for (usize output = 0; output < outputs; ++output)
    {
        i64 minimum = biases[output];
        i64 maximum = biases[output];
        for (usize input = 0; input < inputs; ++input)
        {
            const i8 weight = weights[output * inputs + input];
            if (weight < 0)
                minimum += i64(weight) * 127;
            else
                maximum += i64(weight) * 127;
        }
        lower[output] = minimum;
        upper[output] = maximum;
    }
}

bool require_bounds(std::string_view           label,
                    const std::array<i64, L2>& lower,
                    const std::array<i64, L2>& upper,
                    usize                      outputs,
                    i64                        minimum,
                    i64                        maximum,
                    std::string_view           domain,
                    std::string&               error) {
    for (usize output = 0; output < outputs; ++output)
        if (lower[output] < minimum || upper[output] > maximum)
        {
            error = std::string(label) + " affine envelope exceeds " + std::string(domain)
                  + " at row " + std::to_string(output) + ": [" + std::to_string(lower[output])
                  + ", " + std::to_string(upper[output]) + "]";
            return false;
        }
    return true;
}

template<typename Range>
void write_integer_array(std::ostream& out, const Range& values) {
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

template<typename Feature>
void write_feature_indices(std::ostream& out, const std::vector<Feature>& features) {
    out << '[';
    for (usize index = 0; index < features.size(); ++index)
    {
        if (index)
            out << ',';
        out << features[index].index;
    }
    out << ']';
}

using LoadedAccumulator    = IntegerAccumulator;
using LoadedAccumulatorSet = IntegerAccumulatorSet;

template<typename Parameters>
ParameterView make_parameter_view(const Parameters& parameters) {
    ParameterView view;
    view.ftBias            = parameters.ftBias.data();
    view.threatWeight      = parameters.threatWeight.get();
    view.threatPsqt        = parameters.threatPsqt.get();
    view.pieceSquareWeight = parameters.pieceSquareWeight.get();
    view.pieceSquarePsqt   = parameters.pieceSquarePsqt.get();
    for (usize stack = 0; stack < LayerStacks; ++stack)
    {
        const auto& source      = parameters.dense[stack];
        auto&       destination = view.dense[stack];
        destination.fc0Bias     = source.fc0Bias.data();
        destination.fc0Weight   = source.fc0Weight.data();
        destination.fc1Bias     = source.fc1Bias.data();
        destination.fc1Weight   = source.fc1Weight.data();
        destination.fc2Bias     = source.fc2Bias.data();
        destination.fc2Weight   = source.fc2Weight.data();
    }
    return view;
}

struct alignas(CacheLineSize) SimdLoadedAccumulator {
    alignas(CacheLineSize) std::array<i16, L1> values{};
    alignas(CacheLineSize) std::array<i32, PsqtBuckets> psqt{};
};

using SimdLoadedAccumulatorSet = std::array<SimdLoadedAccumulator, COLOR_NB>;

template<typename Parameters>
std::optional<std::string> refresh_loaded_accumulator(const Parameters&       parameters,
                                                      const PerspectiveTrace& trace,
                                                      LoadedAccumulator&      accumulator) {
    return refresh_integer_accumulator(make_parameter_view(parameters), trace, accumulator);
}

template<typename Feature, typename Update>
void apply_loaded_feature_delta(const std::vector<Feature>& before,
                                const std::vector<Feature>& after,
                                Update&&                    update,
                                u64&                        adds,
                                u64&                        removes) {
    usize beforeIndex = 0;
    usize afterIndex  = 0;
    while (beforeIndex < before.size() || afterIndex < after.size())
    {
        if (afterIndex == after.size()
            || (beforeIndex < before.size() && before[beforeIndex].index < after[afterIndex].index))
        {
            update(before[beforeIndex++].index, -1);
            ++removes;
        }
        else if (beforeIndex == before.size()
                 || after[afterIndex].index < before[beforeIndex].index)
        {
            update(after[afterIndex++].index, 1);
            ++adds;
        }
        else
        {
            ++beforeIndex;
            ++afterIndex;
        }
    }
}

constexpr bool loaded_simd_available() {
#if defined(USE_AVX2) || defined(USE_SSSE3)
    return true;
#else
    return false;
#endif
}

template<typename Parameters>
void update_loaded_simd_piece(const Parameters&      parameters,
                              IndexType              index,
                              i32                    sign,
                              SimdLoadedAccumulator& accumulator) {
    const i16* row = parameters.pieceSquareWeight.get() + u64(index) * L1;
#if defined(USE_AVX2)
    for (usize lane = 0; lane < L1; lane += 16)
    {
        const __m256i current =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator.values.data() + lane));
        const __m256i weight = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + lane));
        const __m256i next =
          sign > 0 ? _mm256_add_epi16(current, weight) : _mm256_sub_epi16(current, weight);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator.values.data() + lane), next);
    }
#elif defined(USE_SSE2)
    for (usize lane = 0; lane < L1; lane += 8)
    {
        const __m128i current =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(accumulator.values.data() + lane));
        const __m128i weight = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + lane));
        const __m128i next =
          sign > 0 ? _mm_add_epi16(current, weight) : _mm_sub_epi16(current, weight);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(accumulator.values.data() + lane), next);
    }
#else
    for (usize lane = 0; lane < L1; ++lane)
        accumulator.values[lane] = i16(accumulator.values[lane] + sign * row[lane]);
#endif

    const i32* psqtRow = parameters.pieceSquarePsqt.get() + u64(index) * PsqtBuckets;
#if defined(USE_AVX2)
    const __m256i current =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator.psqt.data()));
    const __m256i weight = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(psqtRow));
    const __m256i next =
      sign > 0 ? _mm256_add_epi32(current, weight) : _mm256_sub_epi32(current, weight);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator.psqt.data()), next);
#elif defined(USE_SSE2)
    for (usize bucket = 0; bucket < PsqtBuckets; bucket += 4)
    {
        const __m128i current =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(accumulator.psqt.data() + bucket));
        const __m128i weight = _mm_loadu_si128(reinterpret_cast<const __m128i*>(psqtRow + bucket));
        const __m128i next =
          sign > 0 ? _mm_add_epi32(current, weight) : _mm_sub_epi32(current, weight);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(accumulator.psqt.data() + bucket), next);
    }
#else
    for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
        accumulator.psqt[bucket] += sign * psqtRow[bucket];
#endif
}

template<typename Parameters>
void update_loaded_simd_threat(const Parameters&      parameters,
                               IndexType              index,
                               i32                    sign,
                               SimdLoadedAccumulator& accumulator) {
    const i8* row = parameters.threatWeight.get() + u64(index) * L1;
#if defined(USE_AVX2)
    for (usize lane = 0; lane < L1; lane += 16)
    {
        const __m256i current =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator.values.data() + lane));
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + lane));
        const __m256i weight = _mm256_cvtepi8_epi16(packed);
        const __m256i next =
          sign > 0 ? _mm256_add_epi16(current, weight) : _mm256_sub_epi16(current, weight);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator.values.data() + lane), next);
    }
#elif defined(USE_SSE2)
    using namespace SIMD;
    for (usize lane = 0; lane < L1; lane += 8)
    {
        const __m128i current =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(accumulator.values.data() + lane));
        u64 packed = 0;
        std::memcpy(&packed, row + lane, sizeof(packed));
        const __m128i weight = vec_convert_8_16(packed);
        const __m128i next =
          sign > 0 ? _mm_add_epi16(current, weight) : _mm_sub_epi16(current, weight);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(accumulator.values.data() + lane), next);
    }
#else
    for (usize lane = 0; lane < L1; ++lane)
        accumulator.values[lane] = i16(accumulator.values[lane] + sign * row[lane]);
#endif

    const i32* psqtRow = parameters.threatPsqt.get() + u64(index) * PsqtBuckets;
#if defined(USE_AVX2)
    const __m256i current =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator.psqt.data()));
    const __m256i weight = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(psqtRow));
    const __m256i next =
      sign > 0 ? _mm256_add_epi32(current, weight) : _mm256_sub_epi32(current, weight);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator.psqt.data()), next);
#elif defined(USE_SSE2)
    for (usize bucket = 0; bucket < PsqtBuckets; bucket += 4)
    {
        const __m128i current =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(accumulator.psqt.data() + bucket));
        const __m128i weight = _mm_loadu_si128(reinterpret_cast<const __m128i*>(psqtRow + bucket));
        const __m128i next =
          sign > 0 ? _mm_add_epi32(current, weight) : _mm_sub_epi32(current, weight);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(accumulator.psqt.data() + bucket), next);
    }
#else
    for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
        accumulator.psqt[bucket] += sign * psqtRow[bucket];
#endif
}

template<typename Parameters>
void refresh_loaded_simd_accumulator(const Parameters&       parameters,
                                     const PerspectiveTrace& trace,
                                     SimdLoadedAccumulator&  accumulator) {
    accumulator = {};
    std::copy(parameters.ftBias.begin(), parameters.ftBias.end(), accumulator.values.begin());
    for (const auto& feature : trace.pieces)
        update_loaded_simd_piece(parameters, feature.index, 1, accumulator);
    for (const auto& feature : trace.threats)
        update_loaded_simd_threat(parameters, feature.index, 1, accumulator);
}

template<typename Parameters>
void update_loaded_simd_accumulator(const Parameters&       parameters,
                                    const PerspectiveTrace& before,
                                    const PerspectiveTrace& after,
                                    SimdLoadedAccumulator&  accumulator) {
    u64 ignoredAdds    = 0;
    u64 ignoredRemoves = 0;
    apply_loaded_feature_delta(
      before.pieces, after.pieces,
      [&](IndexType index, i32 sign) {
          update_loaded_simd_piece(parameters, index, sign, accumulator);
      },
      ignoredAdds, ignoredRemoves);
    ignoredAdds = ignoredRemoves = 0;
    apply_loaded_feature_delta(
      before.threats, after.threats,
      [&](IndexType index, i32 sign) {
          update_loaded_simd_threat(parameters, index, sign, accumulator);
      },
      ignoredAdds, ignoredRemoves);
}

std::optional<std::string> compare_loaded_simd(const SimdLoadedAccumulator& simd,
                                               const LoadedAccumulator&     scalar,
                                               Color                        perspective) {
    for (usize lane = 0; lane < L1; ++lane)
        if (simd.values[lane] != scalar.values[lane])
            return "Alice native loaded SIMD feature accumulator differs from scalar at perspective "
                 + std::to_string(int(perspective)) + " lane " + std::to_string(lane) + ".";
    for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
        if (simd.psqt[bucket] != scalar.psqt[bucket])
            return "Alice native loaded SIMD PSQT accumulator differs from scalar at perspective "
                 + std::to_string(int(perspective)) + " bucket " + std::to_string(bucket) + ".";
    return std::nullopt;
}

template<typename Parameters>
std::optional<std::string> update_loaded_accumulator(const Parameters&       parameters,
                                                     const PerspectiveTrace& before,
                                                     const PerspectiveTrace& after,
                                                     LoadedAccumulator&      accumulator,
                                                     LoadedIncrementalVerificationStats& stats) {
    AccumulatorDeltaStats delta;
    auto error = update_integer_accumulator(make_parameter_view(parameters), before, after,
                                            accumulator, delta);
    stats.pieceAdds += delta.pieceAdds;
    stats.pieceRemoves += delta.pieceRemoves;
    stats.threatAdds += delta.threatAdds;
    stats.threatRemoves += delta.threatRemoves;
    stats.maxPieceEvents  = std::max(stats.maxPieceEvents, delta.pieceAdds + delta.pieceRemoves);
    stats.maxThreatEvents = std::max(stats.maxThreatEvents, delta.threatAdds + delta.threatRemoves);
    return error;
}

template<typename Parameters>
std::optional<std::string> evaluate_loaded_integer(const Parameters&           parameters,
                                                   const Position&             position,
                                                   const LoadedAccumulatorSet& accumulators,
                                                   NativeIntegerStages&        stages) {
    return evaluate_integer(make_parameter_view(parameters), position, accumulators, stages);
}

constexpr std::string_view dense_simd_backend() {
#if defined(USE_AVX2)
    return "avx2";
#elif defined(USE_SSSE3)
    return "ssse3";
#else
    return "none";
#endif
}

std::optional<std::string> dense_simd_affine(const i8*  weights,
                                             const i32* biases,
                                             usize      outputs,
                                             usize      inputs,
                                             const i32* values,
                                             i32*       result) {
    alignas(CacheLineSize) std::array<u8, L1> packed{};
    if (inputs > packed.size())
        return "Alice native dense SIMD input exceeds the qualification buffer.";
    for (usize input = 0; input < inputs; ++input)
    {
        if (values[input] < 0 || values[input] > 127)
            return "Alice native dense SIMD input is outside unsigned seven-bit range.";
        packed[input] = u8(values[input]);
    }

#if defined(USE_AVX2)
    if (inputs % 32 != 0)
        return "Alice native AVX2 input width is not a multiple of 32.";
    for (usize output = 0; output < outputs; ++output)
    {
        __m256i   sum = _mm256_setzero_si256();
        const i8* row = weights + output * inputs;
        for (usize input = 0; input < inputs; input += 32)
        {
            const __m256i value =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed.data() + input));
            const __m256i weight =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + input));
            SIMD::m256_add_dpbusd_epi32(sum, value, weight);
        }
        result[output] = SIMD::m256_hadd(sum, biases[output]);
    }
#elif defined(USE_SSSE3)
    if (inputs % 16 != 0)
        return "Alice native SSSE3 input width is not a multiple of 16.";
    for (usize output = 0; output < outputs; ++output)
    {
        __m128i   sum = _mm_setzero_si128();
        const i8* row = weights + output * inputs;
        for (usize input = 0; input < inputs; input += 16)
        {
            const __m128i value =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed.data() + input));
            const __m128i weight = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + input));
            SIMD::m128_add_dpbusd_epi32(sum, value, weight);
        }
        result[output] = SIMD::m128_hadd(sum, biases[output]);
    }
#else
    (void) weights;
    (void) biases;
    (void) outputs;
    (void) values;
    (void) result;
    return "Alice native dense SIMD qualification is unavailable for this build.";
#endif
    return std::nullopt;
}

template<typename Parameters>
std::optional<std::string>
verify_dense_simd(const Parameters& parameters, const NativeIntegerStages& scalar, bool& verified) {
    verified = false;
    if (dense_simd_backend() == "none")
        return std::nullopt;

    const auto&         dense = parameters.dense[scalar.phase];
    std::array<i32, L2> z0{};
    if (auto error = dense_simd_affine(dense.fc0Weight.data(), dense.fc0Bias.data(), L2, L1,
                                       scalar.denseInput.data(), z0.data()))
        return error;
    if (z0 != scalar.z0)
        return "Alice native dense SIMD fc0 output differs from scalar.";

    std::array<i32, 64> y1{};
    for (usize output = 0; output < L2; ++output)
    {
        y1[output]      = scalar.s0[output];
        y1[L2 + output] = scalar.r0[output];
    }
    std::array<i32, L3> z1{};
    if (auto error = dense_simd_affine(dense.fc1Weight.data(), dense.fc1Bias.data(), L3, y1.size(),
                                       y1.data(), z1.data()))
        return error;
    if (z1 != scalar.z1)
        return "Alice native dense SIMD fc1 output differs from scalar.";

    std::array<i32, 128> y2{};
    for (usize output = 0; output < L3; ++output)
    {
        y2[output]               = scalar.s0[output];
        y2[L2 + output]          = scalar.r0[output];
        y2[2 * L2 + output]      = scalar.s1[output];
        y2[2 * L2 + L3 + output] = scalar.r1[output];
    }
    std::array<i32, 1> z2{};
    if (auto error = dense_simd_affine(dense.fc2Weight.data(), dense.fc2Bias.data(), 1, y2.size(),
                                       y2.data(), z2.data()))
        return error;
    if (z2[0] != scalar.z2)
        return "Alice native dense SIMD fc2 output differs from scalar.";

    verified = true;
    return std::nullopt;
}

}  // namespace

void WireValidator::reset() {
    ready   = false;
    current = {};
    lastError.clear();
}

std::optional<std::string>
WireValidator::validate(const std::filesystem::path&      file,
                        const std::optional<std::string>& expectedSha256) {
    reset();
    const auto reject = [&](std::string reason) -> std::optional<std::string> {
        lastError = std::move(reason);
        return lastError;
    };

    std::string expected;
    if (auto error = normalized_expected_sha(expectedSha256, expected))
        return reject(*error);

    std::ifstream input(file, std::ios::binary);
    if (!input)
        return reject("native wire file could not be opened: " + normalized_path(file));

    u32 version        = 0;
    u32 architecture   = 0;
    u32 manifestLength = 0;
    if (!read_u32(input, version) || !read_u32(input, architecture)
        || !read_u32(input, manifestLength))
        return reject("native wire header is truncated");
    if (version == LegacyWireVersion)
        return reject("legacy 0x7AF32F20 files are not native Alice networks");
    if (version != WireVersion)
        return reject("native wire version mismatch: expected " + hex32(WireVersion) + ", got "
                      + hex32(version));
    if (architecture != CompositeArchitectureHash)
        return reject("native architecture mismatch: expected " + hex32(CompositeArchitectureHash)
                      + ", got " + hex32(architecture));
    if (manifestLength > MaximumManifestBytes)
        return reject("native manifest length exceeds the 65536-byte limit");
    if (manifestLength != CanonicalManifestBytes)
        return reject("native manifest length mismatch: expected "
                      + std::to_string(CanonicalManifestBytes) + ", got "
                      + std::to_string(manifestLength));

    std::error_code sizeError;
    const u64       fileSize = std::filesystem::file_size(file, sizeError);
    if (sizeError)
        return reject("native wire size could not be read");
    if (fileSize < NativeWireBytes)
        return reject("native wire file is truncated: expected " + std::to_string(NativeWireBytes)
                      + " bytes, got " + std::to_string(fileSize));
    if (fileSize > NativeWireBytes)
        return reject("native wire file has trailing data: expected "
                      + std::to_string(NativeWireBytes) + " bytes, got "
                      + std::to_string(fileSize));

    std::string manifest(manifestLength, '\0');
    input.read(manifest.data(), std::streamsize(manifest.size()));
    if (!input)
        return reject("native manifest is truncated");
    const std::string manifestDigest = sha256(manifest);
    if (manifestDigest != ManifestSha256)
        return reject("native manifest SHA-256 mismatch: expected " + std::string(ManifestSha256)
                      + ", got " + manifestDigest);

    u32 transformerHash = 0;
    if (!read_u32(input, transformerHash))
        return reject("native feature-transformer hash is truncated");
    if (transformerHash != FeatureTransformerHash)
        return reject("native feature-transformer hash mismatch: expected "
                      + hex32(FeatureTransformerHash) + ", got " + hex32(transformerHash));

    input.seekg(std::streamoff(FeatureTensorBytes), std::ios::cur);
    if (!input)
        return reject("native feature tensors are truncated");
    for (IndexType stack = 0; stack < LayerStacks; ++stack)
    {
        u32 denseHash = 0;
        if (!read_u32(input, denseHash))
            return reject("native dense-stack hash is truncated at stack " + std::to_string(stack));
        if (denseHash != DenseArchitectureHash)
            return reject("native dense-stack hash mismatch at stack " + std::to_string(stack)
                          + ": expected " + hex32(DenseArchitectureHash) + ", got "
                          + hex32(denseHash));
        input.seekg(std::streamoff(DenseStackTensorBytes), std::ios::cur);
        if (!input)
            return reject("native dense tensors are truncated at stack " + std::to_string(stack));
    }
    if (input.tellg() != std::streampos(NativeWireBytes))
        return reject("native wire parser did not finish at the expected end of file");
    if (input.peek() != std::char_traits<char>::eof())
        return reject("native wire file contains trailing data");

    std::string fileDigest;
    if (auto error = sha256_file(file, fileDigest))
        return reject(*error);
    if (!expected.empty() && fileDigest != expected)
        return reject("native wire SHA-256 mismatch: expected " + expected + ", got " + fileDigest);

    WireMetadata accepted;
    accepted.normalizedPath = normalized_path(file);
    accepted.bytes          = fileSize;
    accepted.sha256         = fileDigest;
    accepted.manifestSha256 = manifestDigest;
    accepted.version        = version;
    accepted.architecture   = architecture;
    current                 = std::move(accepted);
    ready                   = true;
    return std::nullopt;
}

bool WireValidator::valid() const { return ready; }

const WireMetadata& WireValidator::metadata() const { return current; }

const std::string& WireValidator::last_error() const { return lastError; }

std::string WireValidator::status_line() const {
    if (!ready)
        return "Alice native wire is not validated"
             + (lastError.empty() ? std::string(".") : ": " + lastError);

    std::ostringstream out;
    out << "Alice native wire validated path=\"" << current.normalizedPath
        << "\" bytes=" << current.bytes << " sha256=" << current.sha256
        << " manifest_sha256=" << current.manifestSha256 << " version=" << hex32(current.version)
        << " architecture=" << hex32(current.architecture);
    return out.str();
}

QualificationNetwork::QualificationNetwork() = default;
QualificationNetwork::~QualificationNetwork() {
    assert(activeLeases.load(std::memory_order_acquire) == 0);
}

QualificationNetwork::Lease::Lease(const QualificationNetwork* leaseOwner,
                                   const Parameters*           parameters) noexcept :
    network(leaseOwner),
    pinned(parameters) {}

QualificationNetwork::Lease::~Lease() { reset(); }

QualificationNetwork::Lease::Lease(Lease&& other) noexcept :
    network(other.network),
    pinned(other.pinned) {
    other.network = nullptr;
    other.pinned  = nullptr;
}

QualificationNetwork::Lease& QualificationNetwork::Lease::operator=(Lease&& other) noexcept {
    if (this != &other)
    {
        reset();
        network       = other.network;
        pinned        = other.pinned;
        other.network = nullptr;
        other.pinned  = nullptr;
    }
    return *this;
}

QualificationNetwork::Lease::operator bool() const noexcept { return network && pinned; }

ParameterView QualificationNetwork::Lease::parameter_view() const noexcept {
    return pinned ? make_parameter_view(*pinned) : ParameterView{};
}

u64 QualificationNetwork::Lease::generation() const noexcept {
    return pinned ? pinned->generation : 0;
}

std::string_view QualificationNetwork::Lease::normalized_path() const noexcept {
    return pinned ? std::string_view(pinned->wire.normalizedPath) : std::string_view{};
}

std::string_view QualificationNetwork::Lease::sha256() const noexcept {
    return pinned ? std::string_view(pinned->wire.sha256) : std::string_view{};
}

u32 QualificationNetwork::Lease::version() const noexcept {
    return pinned ? pinned->wire.version : 0;
}

u32 QualificationNetwork::Lease::architecture() const noexcept {
    return pinned ? pinned->wire.architecture : 0;
}

void QualificationNetwork::Lease::reset() noexcept {
    if (network)
        network->release_lease();
    network = nullptr;
    pinned  = nullptr;
}

std::optional<QualificationNetwork::Lease>
QualificationNetwork::acquire_lease(std::string& error) const noexcept {
    error.clear();
    if (replacementInProgress.load(std::memory_order_acquire))
    {
        error = "Alice native parameter replacement is in progress.";
        return std::nullopt;
    }

    activeLeases.fetch_add(1, std::memory_order_acq_rel);
    if (replacementInProgress.load(std::memory_order_acquire))
    {
        release_lease();
        error = "Alice native parameter replacement is in progress.";
        return std::nullopt;
    }

    const Parameters* parameters = active.get();
    if (!parameters)
    {
        release_lease();
        error = "Alice native parameters are not loaded.";
        return std::nullopt;
    }

    Lease lease(this, parameters);
    return std::optional<Lease>(std::move(lease));
}

bool QualificationNetwork::has_active_lease() const noexcept {
    return activeLeases.load(std::memory_order_acquire) != 0;
}

void QualificationNetwork::release_lease() const noexcept {
    const u64 previous = activeLeases.fetch_sub(1, std::memory_order_acq_rel);
    assert(previous > 0);
    (void) previous;
}

std::optional<std::string> QualificationNetwork::load(const std::filesystem::path& file,
                                                      std::string_view             expectedSha256) {
    const auto reject = [&](std::string reason) -> std::optional<std::string> {
        lastError = std::move(reason);
        return lastError;
    };

    if (has_active_lease())
        return reject("native parameter replacement is rejected while a search lease is active");

    bool expectedReplacement = false;
    if (!replacementInProgress.compare_exchange_strong(expectedReplacement, true,
                                                       std::memory_order_acq_rel))
        return reject("another native parameter replacement is already in progress");

    struct ReplacementGuard {
        std::atomic_bool& flag;
        ~ReplacementGuard() { flag.store(false, std::memory_order_release); }
    } replacementGuard{replacementInProgress};

    if (has_active_lease())
        return reject("native parameter replacement is rejected while a search lease is active");

    std::string expected;
    if (auto error = normalized_required_sha(expectedSha256, expected))
        return reject(*error);
    if (active && active->generation == std::numeric_limits<u64>::max())
        return reject("native parameter generation is exhausted");

    std::ifstream input(file, std::ios::binary);
    if (!input)
        return reject("native parameter file could not be opened: " + normalized_path(file));

    input.seekg(0, std::ios::end);
    const std::streampos end = input.tellg();
    if (end == std::streampos(-1))
        return reject("native parameter size could not be derived from the open handle");
    const u64 fileSize = u64(end);
    if (fileSize < NativeWireBytes)
        return reject("native parameter file is truncated: expected "
                      + std::to_string(NativeWireBytes) + " bytes, got "
                      + std::to_string(fileSize));
    if (fileSize > NativeWireBytes)
        return reject("native parameter file has trailing data: expected "
                      + std::to_string(NativeWireBytes) + " bytes, got "
                      + std::to_string(fileSize));
    input.seekg(0, std::ios::beg);
    if (!input)
        return reject("native parameter handle could not seek to its beginning");

    AuthenticatingReader reader(input);
    u32                  version        = 0;
    u32                  architecture   = 0;
    u32                  manifestLength = 0;
    if (!reader.read_u32(version) || !reader.read_u32(architecture)
        || !reader.read_u32(manifestLength))
        return reject("native parameter header is truncated");
    if (version == LegacyWireVersion)
        return reject("legacy 0x7AF32F20 files are not native Alice parameters");
    if (version != WireVersion)
        return reject("native parameter version mismatch: expected " + hex32(WireVersion) + ", got "
                      + hex32(version));
    if (architecture != CompositeArchitectureHash)
        return reject("native parameter architecture mismatch: expected "
                      + hex32(CompositeArchitectureHash) + ", got " + hex32(architecture));
    if (manifestLength > MaximumManifestBytes)
        return reject("native parameter manifest length exceeds the 65536-byte limit");
    if (manifestLength != CanonicalManifestBytes)
        return reject("native parameter manifest length mismatch: expected "
                      + std::to_string(CanonicalManifestBytes) + ", got "
                      + std::to_string(manifestLength));

    std::string manifest(manifestLength, '\0');
    if (!reader.read(manifest.data(), manifest.size()))
        return reject("native parameter manifest is truncated");
    const std::string manifestDigest = sha256(manifest);
    if (manifestDigest != ManifestSha256)
        return reject("native parameter manifest SHA-256 mismatch: expected "
                      + std::string(ManifestSha256) + ", got " + manifestDigest);

    u32 transformerHash = 0;
    if (!reader.read_u32(transformerHash))
        return reject("native parameter feature-transformer hash is truncated");
    if (transformerHash != FeatureTransformerHash)
        return reject("native parameter feature-transformer hash mismatch: expected "
                      + hex32(FeatureTransformerHash) + ", got " + hex32(transformerHash));

    std::unique_ptr<Parameters> candidate(new (std::nothrow) Parameters());
    if (!candidate)
        return reject("native parameter candidate allocation failed");
    if (!candidate->allocate_features())
        return reject("native parameter feature allocation failed");

    std::array<Sha256, TensorCount> wireTensorHashes;
    std::string                     parseError;
    if (!read_i16_tensor(reader, candidate->ftBias.data(), FtBiasElements, wireTensorHashes[FtBias],
                         TensorNames[FtBias], 0, parseError)
        || !read_i8_tensor(reader, candidate->threatWeight.get(), ThreatWeightElements,
                           wireTensorHashes[ThreatWeight], TensorNames[ThreatWeight], 0, parseError)
        || !read_i32_tensor(reader, candidate->threatPsqt.get(), ThreatPsqtElements,
                            wireTensorHashes[ThreatPsqt], TensorNames[ThreatPsqt], 0, parseError)
        || !read_i16_tensor(reader, candidate->pieceSquareWeight.get(), PieceSquareWeightElements,
                            wireTensorHashes[PieceSquareWeight], TensorNames[PieceSquareWeight], 0,
                            parseError)
        || !read_i32_tensor(reader, candidate->pieceSquarePsqt.get(), PieceSquarePsqtElements,
                            wireTensorHashes[PieceSquarePsqt], TensorNames[PieceSquarePsqt], 0,
                            parseError))
        return reject(parseError);

    for (usize stack = 0; stack < LayerStacks; ++stack)
    {
        u32 denseHash = 0;
        if (!reader.read_u32(denseHash))
            return reject("native parameter dense hash is truncated at stack "
                          + std::to_string(stack));
        if (denseHash != DenseArchitectureHash)
            return reject("native parameter dense hash mismatch at stack " + std::to_string(stack)
                          + ": expected " + hex32(DenseArchitectureHash) + ", got "
                          + hex32(denseHash));

        auto& dense = candidate->dense[stack];
        if (!read_i32_tensor(reader, dense.fc0Bias.data(), Fc0BiasElementsPerStack,
                             wireTensorHashes[Fc0Bias], TensorNames[Fc0Bias],
                             stack * Fc0BiasElementsPerStack, parseError)
            || !read_i8_tensor(reader, dense.fc0Weight.data(), Fc0WeightElementsPerStack,
                               wireTensorHashes[Fc0Weight], TensorNames[Fc0Weight],
                               stack * Fc0WeightElementsPerStack, parseError)
            || !read_i32_tensor(reader, dense.fc1Bias.data(), Fc1BiasElementsPerStack,
                                wireTensorHashes[Fc1Bias], TensorNames[Fc1Bias],
                                stack * Fc1BiasElementsPerStack, parseError)
            || !read_i8_tensor(reader, dense.fc1Weight.data(), Fc1WeightElementsPerStack,
                               wireTensorHashes[Fc1Weight], TensorNames[Fc1Weight],
                               stack * Fc1WeightElementsPerStack, parseError)
            || !read_i32_tensor(reader, dense.fc2Bias.data(), Fc2BiasElementsPerStack,
                                wireTensorHashes[Fc2Bias], TensorNames[Fc2Bias],
                                stack * Fc2BiasElementsPerStack, parseError)
            || !read_i8_tensor(reader, dense.fc2Weight.data(), Fc2WeightElementsPerStack,
                               wireTensorHashes[Fc2Weight], TensorNames[Fc2Weight],
                               stack * Fc2WeightElementsPerStack, parseError))
            return reject(parseError);
    }

    if (reader.bytes_consumed() != NativeWireBytes || reader.bytes_consumed() != fileSize)
        return reject("native parameter parser did not consume the exact open-handle size");
    if (input.peek() != std::char_traits<char>::eof())
        return reject("native parameter handle contains trailing data");

    const std::string fileDigest = digest_string(reader.finish());
    if (fileDigest != expected)
        return reject("native parameter SHA-256 mismatch: expected " + expected + ", got "
                      + fileDigest);

    std::array<Sha256, TensorCount> runtimeTensorHashes;
    update_i16_digest(runtimeTensorHashes[FtBias], candidate->ftBias.data(), FtBiasElements);
    update_i8_digest(runtimeTensorHashes[ThreatWeight], candidate->threatWeight.get(),
                     ThreatWeightElements);
    update_i32_digest(runtimeTensorHashes[ThreatPsqt], candidate->threatPsqt.get(),
                      ThreatPsqtElements);
    update_i16_digest(runtimeTensorHashes[PieceSquareWeight], candidate->pieceSquareWeight.get(),
                      PieceSquareWeightElements);
    update_i32_digest(runtimeTensorHashes[PieceSquarePsqt], candidate->pieceSquarePsqt.get(),
                      PieceSquarePsqtElements);
    for (const auto& dense : candidate->dense)
    {
        update_i32_digest(runtimeTensorHashes[Fc0Bias], dense.fc0Bias.data(),
                          Fc0BiasElementsPerStack);
        update_i8_digest(runtimeTensorHashes[Fc0Weight], dense.fc0Weight.data(),
                         Fc0WeightElementsPerStack);
        update_i32_digest(runtimeTensorHashes[Fc1Bias], dense.fc1Bias.data(),
                          Fc1BiasElementsPerStack);
        update_i8_digest(runtimeTensorHashes[Fc1Weight], dense.fc1Weight.data(),
                         Fc1WeightElementsPerStack);
        update_i32_digest(runtimeTensorHashes[Fc2Bias], dense.fc2Bias.data(),
                          Fc2BiasElementsPerStack);
        update_i8_digest(runtimeTensorHashes[Fc2Weight], dense.fc2Weight.data(),
                         Fc2WeightElementsPerStack);
    }
    for (usize tensor = 0; tensor < TensorCount; ++tensor)
    {
        const std::array<u8, 32> wireDigest    = wireTensorHashes[tensor].finish();
        const std::array<u8, 32> runtimeDigest = runtimeTensorHashes[tensor].finish();
        if (runtimeDigest != wireDigest)
            return reject("native runtime traversal digest mismatch for "
                          + std::string(TensorNames[tensor]));
        candidate->tensorDigests[tensor] = wireDigest;
    }

    for (usize stack = 0; stack < LayerStacks; ++stack)
    {
        const auto&         dense = candidate->dense[stack];
        std::array<i64, L2> lower{};
        std::array<i64, L2> upper{};

        affine_bounds(dense.fc0Weight.data(), dense.fc0Bias.data(), L2, L1, lower, upper);
        if (!require_bounds("stack[" + std::to_string(stack) + "].fc0", lower, upper, L2,
                            std::numeric_limits<i16>::min(), std::numeric_limits<i16>::max(),
                            "signed i16", parseError))
            return reject(parseError);
        const auto fc0Lower = lower;
        const auto fc0Upper = upper;

        affine_bounds(dense.fc1Weight.data(), dense.fc1Bias.data(), L3, 64, lower, upper);
        if (!require_bounds("stack[" + std::to_string(stack) + "].fc1", lower, upper, L3,
                            std::numeric_limits<i16>::min(), std::numeric_limits<i16>::max(),
                            "signed i16", parseError))
            return reject(parseError);

        affine_bounds(dense.fc2Weight.data(), dense.fc2Bias.data(), 1, 128, lower, upper);
        if (!require_bounds("stack[" + std::to_string(stack) + "].fc2", lower, upper, 1,
                            std::numeric_limits<i32>::min(), std::numeric_limits<i32>::max(),
                            "signed i32", parseError))
            return reject(parseError);
        const i64 fwdLower = lower[0] + fc0Lower[30] - fc0Upper[31];
        const i64 fwdUpper = upper[0] + fc0Upper[30] - fc0Lower[31];
        lower[0]           = fwdLower;
        upper[0]           = fwdUpper;
        if (!require_bounds("stack[" + std::to_string(stack) + "].fwdOut", lower, upper, 1,
                            std::numeric_limits<i32>::min(), std::numeric_limits<i32>::max(),
                            "signed i32", parseError))
            return reject(parseError);
    }

    candidate->wire.normalizedPath = normalized_path(file);
    candidate->wire.bytes          = fileSize;
    candidate->wire.sha256         = fileDigest;
    candidate->wire.manifestSha256 = manifestDigest;
    candidate->wire.version        = version;
    candidate->wire.architecture   = architecture;
    candidate->generation          = active ? active->generation + 1 : 1;

    active.swap(candidate);
    lastError.clear();
    return std::nullopt;
}

std::optional<std::string> QualificationNetwork::verify_lease_contract(std::string& report) {
    report.clear();
    if (!active)
        return "Alice native lease verification requires qualification parameters.";

    const Parameters* originalPointer    = active.get();
    const u64         originalGeneration = active->generation;
    const std::string originalSha256     = active->wire.sha256;
    const auto        originalPath       = std::filesystem::path(active->wire.normalizedPath);

    std::string leaseError;
    auto        lease = acquire_lease(leaseError);
    if (!lease)
        return "Alice native lease acquisition failed: " + leaseError;
    if (!has_active_lease() || lease->generation() != originalGeneration
        || lease->sha256() != originalSha256 || lease->version() != WireVersion
        || lease->architecture() != CompositeArchitectureHash)
        return "Alice native lease identity did not match the active parameters.";

    const auto rejected = load(originalPath, originalSha256);
    if (!rejected || rejected->find("search lease is active") == std::string::npos)
        return "Alice native replacement was not rejected while a lease was active.";
    if (active.get() != originalPointer || active->generation != originalGeneration
        || active->wire.sha256 != originalSha256)
        return "Alice native active parameters changed after a rejected leased replacement.";

    lease.reset();
    if (has_active_lease())
        return "Alice native lease count remained active after release.";

    auto secondLease = acquire_lease(leaseError);
    if (!secondLease || secondLease->generation() != originalGeneration
        || secondLease->sha256() != originalSha256)
        return "Alice native parameters could not be leased again after rejection.";
    secondLease.reset();

    std::ostringstream out;
    out << "alice_native lease verified generation " << originalGeneration << " sha256 "
        << originalSha256 << " active_reload_rejections 1 reacquisitions 1";
    report = out.str();
    return std::nullopt;
}

bool QualificationNetwork::loaded() const { return bool(active); }

u64 QualificationNetwork::generation() const { return active ? active->generation : 0; }

const std::string& QualificationNetwork::last_error() const { return lastError; }

std::string QualificationNetwork::status_line() const {
    if (!active)
        return "Alice native qualification parameters are not loaded"
             + (lastError.empty() ? std::string(".") : ": " + lastError);

    std::ostringstream out;
    out << "Alice native qualification parameters loaded generation=" << active->generation
        << " path=\"" << active->wire.normalizedPath << "\" bytes=" << active->wire.bytes
        << " sha256=" << active->wire.sha256 << " manifest_sha256=" << active->wire.manifestSha256
        << " version=" << hex32(active->wire.version)
        << " architecture=" << hex32(active->wire.architecture) << " search=available";
    return out.str();
}

std::string QualificationNetwork::tensor_status_line() const {
    if (!active)
        return "Alice native qualification tensor identities are unavailable.";

    std::ostringstream out;
    out << "Alice native qualification tensors generation=" << active->generation;
    for (usize tensor = 0; tensor < TensorCount; ++tensor)
        out << ' ' << TensorNames[tensor] << "_bytes=" << TensorBytes[tensor] << ' '
            << TensorNames[tensor] << "_sha256=" << digest_string(active->tensorDigests[tensor]);
    return out.str();
}

std::optional<std::string>
QualificationNetwork::probe(std::string_view tensor, u64 index, std::string& report) const {
    if (!active)
        return "Alice native qualification parameters are not loaded.";

    i64 value = 0;
    if (tensor == TensorNames[FtBias])
    {
        if (index >= FtBiasElements)
            return "ft.bias probe index is out of range";
        value = active->ftBias[index];
    }
    else if (tensor == TensorNames[ThreatWeight])
    {
        if (index >= ThreatWeightElements)
            return "threat.weight probe index is out of range";
        value = active->threatWeight[index];
    }
    else if (tensor == TensorNames[ThreatPsqt])
    {
        if (index >= ThreatPsqtElements)
            return "threat.psqt probe index is out of range";
        value = active->threatPsqt[index];
    }
    else if (tensor == TensorNames[PieceSquareWeight])
    {
        if (index >= PieceSquareWeightElements)
            return "pieceSquare.weight probe index is out of range";
        value = active->pieceSquareWeight[index];
    }
    else if (tensor == TensorNames[PieceSquarePsqt])
    {
        if (index >= PieceSquarePsqtElements)
            return "pieceSquare.psqt probe index is out of range";
        value = active->pieceSquarePsqt[index];
    }
    else
    {
        usize tensorIndex = TensorCount;
        for (usize candidate = Fc0Bias; candidate < TensorCount; ++candidate)
            if (tensor == TensorNames[candidate])
                tensorIndex = candidate;
        if (tensorIndex == TensorCount)
            return "unknown Alice native qualification tensor: " + std::string(tensor);

        const std::array<u64, 6> elementsPerStack = {
          Fc0BiasElementsPerStack,   Fc0WeightElementsPerStack, Fc1BiasElementsPerStack,
          Fc1WeightElementsPerStack, Fc2BiasElementsPerStack,   Fc2WeightElementsPerStack,
        };
        const u64 perStack = elementsPerStack[tensorIndex - Fc0Bias];
        if (index >= u64(LayerStacks) * perStack)
            return std::string(tensor) + " probe index is out of range";
        const usize stack = usize(index / perStack);
        const usize local = usize(index % perStack);
        const auto& dense = active->dense[stack];
        switch (tensorIndex)
        {
        case Fc0Bias :
            value = dense.fc0Bias[local];
            break;
        case Fc0Weight :
            value = dense.fc0Weight[local];
            break;
        case Fc1Bias :
            value = dense.fc1Bias[local];
            break;
        case Fc1Weight :
            value = dense.fc1Weight[local];
            break;
        case Fc2Bias :
            value = dense.fc2Bias[local];
            break;
        case Fc2Weight :
            value = dense.fc2Weight[local];
            break;
        default :
            return "internal Alice native probe mapping failure";
        }
    }

    std::ostringstream out;
    out << "alice_native_parameter generation " << active->generation << " tensor " << tensor
        << " index " << index << " value " << value;
    report = out.str();
    return std::nullopt;
}

std::optional<std::string> QualificationNetwork::integer_trace(const Position& position,
                                                               std::string&    report) const {
    if (!active)
        return "Alice native integer trace requires loaded qualification parameters.";

    const usize pieceCount = popcount(position.pieces());
    if (pieceCount < 2 || pieceCount > 32)
        return "Alice native integer trace requires between 2 and 32 pieces.";

    const PositionTrace  trace = build_trace(position);
    LoadedAccumulatorSet qualificationAccumulators;
    for (Color perspective : {WHITE, BLACK})
        if (auto error = refresh_loaded_accumulator(*active, trace[perspective],
                                                    qualificationAccumulators[perspective]))
            return error;
    NativeIntegerStages qualificationStages;
    if (auto error = evaluate_loaded_integer(*active, position, qualificationAccumulators,
                                             qualificationStages))
        return error;
    bool featureSimdVerified = false;
    if (loaded_simd_available())
    {
        SimdLoadedAccumulatorSet simdAccumulators;
        for (Color perspective : {WHITE, BLACK})
        {
            refresh_loaded_simd_accumulator(*active, trace[perspective],
                                            simdAccumulators[perspective]);
            if (auto error =
                  compare_loaded_simd(simdAccumulators[perspective],
                                      qualificationAccumulators[perspective], perspective))
                return error;
        }
        featureSimdVerified = true;
    }
    bool denseSimdVerified = false;
    if (auto error = verify_dense_simd(*active, qualificationStages, denseSimdVerified))
        return error;

    const Color sideToMove      = qualificationStages.sideToMove;
    const usize phase           = qualificationStages.phase;
    const auto& accumulators    = qualificationAccumulators;
    const auto& transformed     = qualificationStages.transformed;
    const auto& denseInput      = qualificationStages.denseInput;
    const auto& z0              = qualificationStages.z0;
    const auto& s0              = qualificationStages.s0;
    const auto& r0              = qualificationStages.r0;
    const auto& z1              = qualificationStages.z1;
    const auto& s1              = qualificationStages.s1;
    const auto& r1              = qualificationStages.r1;
    const i32   z2              = qualificationStages.z2;
    const i32   skip            = qualificationStages.skip;
    const i32   fwdOut          = qualificationStages.fwdOut;
    const i32   positionalRaw16 = qualificationStages.positionalRaw;
    const i32   psqtRaw16       = qualificationStages.psqtRaw;
    const i32   positionalValue = qualificationStages.positional;
    const i32   psqtValue       = qualificationStages.psqt;
    const i32   nativeValue     = qualificationStages.value;

    std::ostringstream out;
    out << "{\"architecture\":\"" << ArchitectureId << "\",\"generation\":" << active->generation
        << ",\"networkSha256\":\"" << active->wire.sha256 << "\",\"denseSimd\":\""
        << (denseSimdVerified ? dense_simd_backend() : "none") << "\",\"featureSimd\":\""
        << (featureSimdVerified ? dense_simd_backend() : "none")
        << "\",\"sideToMove\":" << int(sideToMove) << ",\"pieceCount\":" << pieceCount
        << ",\"pieceFeatures\":[";
    for (Color perspective : {WHITE, BLACK})
    {
        if (perspective != WHITE)
            out << ',';
        write_feature_indices(out, trace[perspective].pieces);
    }
    out << "],\"threatFeatures\":[";
    for (Color perspective : {WHITE, BLACK})
    {
        if (perspective != WHITE)
            out << ',';
        write_feature_indices(out, trace[perspective].threats);
    }
    out << "],\"featureAccumulator\":[";
    write_integer_array(out, accumulators[WHITE].values);
    out << ',';
    write_integer_array(out, accumulators[BLACK].values);
    out << "],\"psqtAccumulator\":[";
    write_integer_array(out, accumulators[WHITE].psqt);
    out << ',';
    write_integer_array(out, accumulators[BLACK].psqt);
    out << "],\"transformedByPerspective\":[";
    write_integer_array(out, transformed[WHITE]);
    out << ',';
    write_integer_array(out, transformed[BLACK]);
    out << "],\"transformedInput\":";
    write_integer_array(out, denseInput);
    out << ",\"phase\":" << phase << ",\"fc0Raw\":";
    write_integer_array(out, z0);
    out << ",\"fc0Squared\":";
    write_integer_array(out, s0);
    out << ",\"fc0Linear\":";
    write_integer_array(out, r0);
    out << ",\"fc1Raw\":";
    write_integer_array(out, z1);
    out << ",\"fc1Squared\":";
    write_integer_array(out, s1);
    out << ",\"fc1Linear\":";
    write_integer_array(out, r1);
    out << ",\"fc2Raw\":" << z2 << ",\"skip\":" << skip << ",\"fwdOut\":" << fwdOut
        << ",\"positionalRaw16\":" << positionalRaw16 << ",\"psqtRaw16\":" << psqtRaw16
        << ",\"positionalValue\":" << positionalValue << ",\"psqtValue\":" << psqtValue
        << ",\"nativeNnueValue\":" << nativeValue << '}';
    report = out.str();
    return std::nullopt;
}

std::optional<std::string> QualificationNetwork::verify_incremental(
  Position& position, Depth depth, LoadedIncrementalVerificationStats& stats) const {
    stats = {};
    if (!active)
        return "Alice native loaded incremental verification requires qualification parameters.";
    if (depth < 0 || depth > 2)
        return "Alice native loaded incremental verification depth must be between 0 and 2.";

    const Parameters*   parameters         = active.get();
    const ParameterView parameterView      = make_parameter_view(*parameters);
    const u64           verifiedGeneration = active->generation;
    const std::string   rootFen            = position.fen();
    const Key           rootKey            = position.key();
    const PositionTrace rootTrace          = build_trace(position);
    FeatureSnapshot     rootSnapshot;
    if (auto error = build_fixed_snapshot(position, rootSnapshot))
        return error;
    LoadedAccumulatorSet rootAccumulators;
    LoadedAccumulatorSet rootFixedAccumulators;
    for (Color perspective : {WHITE, BLACK})
    {
        if (auto error = refresh_loaded_accumulator(*parameters, rootTrace[perspective],
                                                    rootAccumulators[perspective]))
            return error;
        if (auto error = refresh_integer_accumulator(parameterView, rootSnapshot[perspective],
                                                     rootFixedAccumulators[perspective]))
            return error;
    }
    SimdLoadedAccumulatorSet rootSimdAccumulators;
    if (loaded_simd_available())
        for (Color perspective : {WHITE, BLACK})
            refresh_loaded_simd_accumulator(*parameters, rootTrace[perspective],
                                            rootSimdAccumulators[perspective]);

    std::function<std::optional<std::string>(
      Depth, const PositionTrace&, const FeatureSnapshot&, const LoadedAccumulatorSet&,
      const LoadedAccumulatorSet&, const SimdLoadedAccumulatorSet&)>
      visit;
    visit =
      [&](
        Depth remaining, const PositionTrace& incrementalTrace,
        const FeatureSnapshot&          incrementalSnapshot,
        const LoadedAccumulatorSet&     incrementalAccumulators,
        const LoadedAccumulatorSet&     incrementalFixedAccumulators,
        const SimdLoadedAccumulatorSet& incrementalSimdAccumulators) -> std::optional<std::string> {
        ++stats.positions;

        LoadedAccumulatorSet     refreshedAccumulators;
        LoadedAccumulatorSet     refreshedFixedAccumulators;
        SimdLoadedAccumulatorSet refreshedSimdAccumulators;
        for (Color perspective : {WHITE, BLACK})
        {
            if (auto error = refresh_loaded_accumulator(*parameters, incrementalTrace[perspective],
                                                        refreshedAccumulators[perspective]))
                return error;
            if (incrementalAccumulators[perspective].values
                  != refreshedAccumulators[perspective].values
                || incrementalAccumulators[perspective].psqt
                     != refreshedAccumulators[perspective].psqt)
                return "Alice native loaded incremental accumulator mismatch at " + position.fen()
                     + ".";
            ++stats.accumulatorComparisons;
            if (auto error =
                  refresh_integer_accumulator(parameterView, incrementalSnapshot[perspective],
                                              refreshedFixedAccumulators[perspective]))
                return error;
            if (incrementalFixedAccumulators[perspective].values
                  != refreshedFixedAccumulators[perspective].values
                || incrementalFixedAccumulators[perspective].psqt
                     != refreshedFixedAccumulators[perspective].psqt
                || refreshedFixedAccumulators[perspective].values
                     != refreshedAccumulators[perspective].values
                || refreshedFixedAccumulators[perspective].psqt
                     != refreshedAccumulators[perspective].psqt)
                return "Alice native fixed snapshot accumulator mismatch at " + position.fen()
                     + ".";
            ++stats.fixedAccumulatorChecks;
            if (loaded_simd_available())
            {
                refresh_loaded_simd_accumulator(*parameters, incrementalTrace[perspective],
                                                refreshedSimdAccumulators[perspective]);
                if (auto error =
                      compare_loaded_simd(incrementalSimdAccumulators[perspective],
                                          refreshedAccumulators[perspective], perspective))
                    return error;
                if (auto error =
                      compare_loaded_simd(refreshedSimdAccumulators[perspective],
                                          refreshedAccumulators[perspective], perspective))
                    return error;
                ++stats.featureSimdComparisons;
            }
        }

        NativeIntegerStages incrementalStages;
        NativeIntegerStages refreshedStages;
        if (auto error = evaluate_loaded_integer(*parameters, position, incrementalAccumulators,
                                                 incrementalStages))
            return error;
        if (auto error = evaluate_loaded_integer(*parameters, position, refreshedAccumulators,
                                                 refreshedStages))
            return error;
        if (!same_integer_stages(incrementalStages, refreshedStages))
            return "Alice native loaded incremental integer-stage mismatch at " + position.fen()
                 + ".";
        ++stats.integerStageComparisons;
        bool denseSimdVerified = false;
        if (auto error = verify_dense_simd(*parameters, incrementalStages, denseSimdVerified))
            return error;
        stats.denseSimdComparisons += denseSimdVerified;

        if (remaining == 0)
            return std::nullopt;

        const std::string nodeFen = position.fen();
        const Key         nodeKey = position.key();
        std::vector<Move> legalMoves;
        for (Move move : MoveList<LEGAL>(position))
            legalMoves.push_back(move);

        for (Move move : legalMoves)
        {
            const Piece moved = position.moved_piece(move);
            stats.captures += position.capture(move);
            stats.promotions += move.type_of() == PROMOTION;
            stats.castlings += move.type_of() == CASTLING;
            stats.kingMoves += type_of(moved) == KING;

            StateInfo state;
            Dirties   dirties;
            position.do_move(move, state, position.gives_check(move), dirties, nullptr, nullptr);

            const PositionTrace        newTrace = build_trace(position);
            FeatureSnapshot            newSnapshot;
            auto                       snapshotError = build_fixed_snapshot(position, newSnapshot);
            LoadedAccumulatorSet       childAccumulators      = incrementalAccumulators;
            LoadedAccumulatorSet       childFixedAccumulators = incrementalFixedAccumulators;
            SimdLoadedAccumulatorSet   childSimdAccumulators  = incrementalSimdAccumulators;
            std::optional<std::string> error                  = snapshotError;
            for (Color perspective : {WHITE, BLACK})
            {
                if (error)
                    break;
                const auto& oldPerspective = incrementalTrace[perspective];
                const auto& newPerspective = newTrace[perspective];
                if (oldPerspective.kingSquare != newPerspective.kingSquare
                    || oldPerspective.kingBoard != newPerspective.kingBoard)
                {
                    ++stats.fullRefreshes[perspective];
                    error = refresh_loaded_accumulator(*parameters, newPerspective,
                                                       childAccumulators[perspective]);
                    if (!error && loaded_simd_available())
                        refresh_loaded_simd_accumulator(*parameters, newPerspective,
                                                        childSimdAccumulators[perspective]);
                    if (!error)
                        error = refresh_integer_accumulator(parameterView, newSnapshot[perspective],
                                                            childFixedAccumulators[perspective]);
                }
                else
                {
                    error = update_loaded_accumulator(*parameters, oldPerspective, newPerspective,
                                                      childAccumulators[perspective], stats);
                    if (!error && loaded_simd_available())
                        update_loaded_simd_accumulator(*parameters, oldPerspective, newPerspective,
                                                       childSimdAccumulators[perspective]);
                    if (!error)
                    {
                        AccumulatorDeltaStats fixedDelta;
                        error = update_integer_accumulator(
                          parameterView, incrementalSnapshot[perspective], newSnapshot[perspective],
                          childFixedAccumulators[perspective], fixedDelta);
                        ++stats.fixedDeltaUpdates;
                    }
                }
                if (error)
                    break;
            }
            ++stats.transitions;
            if (!error)
                error = visit(remaining - 1, newTrace, newSnapshot, childAccumulators,
                              childFixedAccumulators, childSimdAccumulators);

            position.undo_move(move);
            ++stats.undoChecks;
            if (position.fen() != nodeFen || position.key() != nodeKey)
                return "Alice native loaded incremental verification did not restore a parent position.";
            if (error)
                return error;
        }

        return std::nullopt;
    };

    auto error = visit(depth, rootTrace, rootSnapshot, rootAccumulators, rootFixedAccumulators,
                       rootSimdAccumulators);
    if (position.fen() != rootFen || position.key() != rootKey)
        return "Alice native loaded incremental verification did not restore the root position.";
    if (active.get() != parameters || active->generation != verifiedGeneration)
        return "Alice native qualification parameters changed during incremental verification.";
    return error;
}

std::optional<std::string>
QualificationNetwork::verify_session(Position& position, Depth depth, std::string& report) const {
    report.clear();
    if (depth < 0 || depth > 2)
        return "Alice native search-session verification depth must be between 0 and 2.";

    std::string leaseError;
    auto        lease = acquire_lease(leaseError);
    if (!lease)
        return "Alice native search-session verification requires qualification parameters: "
             + leaseError;

    const Parameters*   parameters         = lease->pinned;
    const ParameterView parameterView      = lease->parameter_view();
    const u64           verifiedGeneration = lease->generation();
    const std::string   verifiedSha256(lease->sha256());
    const std::string   rootFen        = position.fen();
    const Key           rootKey        = position.key();
    const Bitboard      rootBoardB     = position.state()->boardB;
    const Color         rootSideToMove = position.side_to_move();
    const int           rootPieceCount = position.count<ALL_PIECES>();

    std::unique_ptr<SearchSession> session(new (std::nothrow) SearchSession(
      parameterView, verifiedGeneration, verifiedSha256, position));
    if (!session)
        return "Alice native search-session allocation failed.";

    const auto describe_failure = [](std::string_view                action,
                                     const AliceSearch::EvalFailure& failure) {
        std::ostringstream out;
        out << "Alice native search session " << action
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
    u64 valueChecks       = 0;
    u64 undoChecks        = 0;

    std::function<std::optional<std::string>(Depth)> visit;
    visit = [&](Depth remaining) -> std::optional<std::string> {
        ++positions;
        if (!session->matches_current(position))
            return "Alice native search session did not match the current position.";

        Value                    sessionValue = VALUE_ZERO;
        AliceSearch::EvalFailure evaluationFailure;
        if (!session->evaluate(position, sessionValue, evaluationFailure))
            return describe_failure("evaluation", evaluationFailure);

        const PositionTrace   trace = build_trace(position);
        IntegerAccumulatorSet refreshed;
        for (Color perspective : {WHITE, BLACK})
        {
            if (auto error = refresh_integer_accumulator(parameterView, trace[perspective],
                                                         refreshed[perspective]))
                return error;
            const auto& current = session->current_accumulators()[perspective];
            if (current.values != refreshed[perspective].values
                || current.psqt != refreshed[perspective].psqt)
                return "Alice native search-session accumulator mismatch at " + position.fen()
                     + ".";
            ++accumulatorChecks;
        }

        NativeIntegerStages stages;
        if (auto error = evaluate_integer(parameterView, position, refreshed, stages))
            return error;
        if (sessionValue != stages.value)
            return "Alice native search-session value mismatch at " + position.fen() + ".";
        ++valueChecks;

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
                return "Alice native search session did not restore a parent position.";
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
        return "Alice native search session did not restore its root position.";
    if (active.get() != parameters || active->generation != verifiedGeneration
        || active->wire.sha256 != verifiedSha256)
        return "Alice native qualification parameters changed during search-session verification.";

    const RuntimeSessionStats& runtime = session->stats();
    if (runtime.evaluations != positions || runtime.pushes != transitions
        || runtime.pops != transitions || accumulatorChecks != 2 * positions
        || valueChecks != positions || undoChecks != transitions)
        return "Alice native search-session counters violated their traversal invariants.";

    std::ostringstream out;
    out << "alice_native session verified generation " << verifiedGeneration << " positions "
        << positions << " transitions " << transitions << " captures " << captures << " promotions "
        << promotions << " castlings " << castlings << " king_moves " << kingMoves
        << " evaluations " << runtime.evaluations << " pushes " << runtime.pushes << " pops "
        << runtime.pops << " refreshes " << runtime.fullRefreshes[WHITE] << ','
        << runtime.fullRefreshes[BLACK] << " piece_adds " << runtime.pieceAdds << " piece_removes "
        << runtime.pieceRemoves << " threat_adds " << runtime.threatAdds << " threat_removes "
        << runtime.threatRemoves << " max_piece_events " << runtime.maxPieceEvents
        << " max_threat_events " << runtime.maxThreatEvents << " accumulator_checks "
        << accumulatorChecks << " value_checks " << valueChecks << " undo_checks " << undoChecks
        << " depth " << depth << " search available";
    report = out.str();
    return std::nullopt;
}

}  // namespace Stockfish::Eval::NNUE::AliceNative
