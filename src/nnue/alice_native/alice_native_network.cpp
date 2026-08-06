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
#include <cctype>
#include <cstring>
#include <fstream>
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
#include "../../position.h"
#include "alice_native_features.h"

namespace Stockfish::Eval::NNUE::AliceNative {

struct QualificationNetwork::Parameters {
    struct DenseStack {
        std::array<i32, Fc0BiasElementsPerStack>   fc0Bias{};
        std::array<i8, Fc0WeightElementsPerStack>  fc0Weight{};
        std::array<i32, Fc1BiasElementsPerStack>   fc1Bias{};
        std::array<i8, Fc1WeightElementsPerStack>  fc1Weight{};
        std::array<i32, Fc2BiasElementsPerStack>   fc2Bias{};
        std::array<i8, Fc2WeightElementsPerStack>  fc2Weight{};
    };

    bool allocate_features() {
        threatWeight.reset(new (std::nothrow) i8[ThreatWeightElements]);
        threatPsqt.reset(new (std::nothrow) i32[ThreatPsqtElements]);
        pieceSquareWeight.reset(new (std::nothrow) i16[PieceSquareWeightElements]);
        pieceSquarePsqt.reset(new (std::nothrow) i32[PieceSquarePsqtElements]);
        return threatWeight && threatPsqt && pieceSquareWeight && pieceSquarePsqt;
    }

    std::array<i16, FtBiasElements> ftBias{};
    std::unique_ptr<i8[]>           threatWeight;
    std::unique_ptr<i32[]>          threatPsqt;
    std::unique_ptr<i16[]>          pieceSquareWeight;
    std::unique_ptr<i32[]>          pieceSquarePsqt;
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
  "ft.bias",          "threat.weight",    "threat.psqt",     "pieceSquare.weight",
  "pieceSquare.psqt", "stack.fc0.bias",   "stack.fc0.weight", "stack.fc1.bias",
  "stack.fc1.weight", "stack.fc2.bias",   "stack.fc2.weight",
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
        value = u32(bytes[0]) | (u32(bytes[1]) << 8) | (u32(bytes[2]) << 16)
              | (u32(bytes[3]) << 24);
        return true;
    }

    u64                    bytes_consumed() const { return consumed; }
    std::array<u8, 32> finish() { return wholeHash.finish(); }

   private:
    std::istream& input;
    Sha256       wholeHash;
    u64          consumed = 0;
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
                                                   std::string&    normalized) {
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
                          | (u32(buffer[4 * index + 2]) << 16)
                          | (u32(buffer[4 * index + 3]) << 24);
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

void affine_bounds(const i8*              weights,
                   const i32*             biases,
                   usize                  outputs,
                   usize                  inputs,
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

bool require_bounds(std::string_view          label,
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

QualificationNetwork::QualificationNetwork()  = default;
QualificationNetwork::~QualificationNetwork() = default;

std::optional<std::string> QualificationNetwork::load(const std::filesystem::path& file,
                                                      std::string_view expectedSha256) {
    const auto reject = [&](std::string reason) -> std::optional<std::string> {
        lastError = std::move(reason);
        return lastError;
    };

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
    if (!read_i16_tensor(reader, candidate->ftBias.data(), FtBiasElements,
                         wireTensorHashes[FtBias], TensorNames[FtBias], 0, parseError)
        || !read_i8_tensor(reader, candidate->threatWeight.get(), ThreatWeightElements,
                           wireTensorHashes[ThreatWeight], TensorNames[ThreatWeight], 0, parseError)
        || !read_i32_tensor(reader, candidate->threatPsqt.get(), ThreatPsqtElements,
                            wireTensorHashes[ThreatPsqt], TensorNames[ThreatPsqt], 0, parseError)
        || !read_i16_tensor(reader, candidate->pieceSquareWeight.get(),
                            PieceSquareWeightElements, wireTensorHashes[PieceSquareWeight],
                            TensorNames[PieceSquareWeight], 0, parseError)
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
        const auto&          dense = candidate->dense[stack];
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
        << " sha256=" << active->wire.sha256
        << " manifest_sha256=" << active->wire.manifestSha256
        << " version=" << hex32(active->wire.version)
        << " architecture=" << hex32(active->wire.architecture)
        << " search=disabled";
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

std::optional<std::string> QualificationNetwork::probe(std::string_view tensor,
                                                       u64              index,
                                                       std::string&     report) const {
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
          Fc0BiasElementsPerStack, Fc0WeightElementsPerStack, Fc1BiasElementsPerStack,
          Fc1WeightElementsPerStack, Fc2BiasElementsPerStack, Fc2WeightElementsPerStack,
        };
        const u64 perStack = elementsPerStack[tensorIndex - Fc0Bias];
        if (index >= u64(LayerStacks) * perStack)
            return std::string(tensor) + " probe index is out of range";
        const usize stack = usize(index / perStack);
        const usize local = usize(index % perStack);
        const auto& dense = active->dense[stack];
        switch (tensorIndex)
        {
        case Fc0Bias:
            value = dense.fc0Bias[local];
            break;
        case Fc0Weight:
            value = dense.fc0Weight[local];
            break;
        case Fc1Bias:
            value = dense.fc1Bias[local];
            break;
        case Fc1Weight:
            value = dense.fc1Weight[local];
            break;
        case Fc2Bias:
            value = dense.fc2Bias[local];
            break;
        case Fc2Weight:
            value = dense.fc2Weight[local];
            break;
        default:
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

    const PositionTrace trace = build_trace(position);
    std::array<std::array<i16, L1>, COLOR_NB> accumulators{};
    std::array<std::array<i32, PsqtBuckets>, COLOR_NB> psqtAccumulators{};

    for (Color perspective : {WHITE, BLACK})
    {
        std::array<i64, L1>          lanes{};
        std::array<i64, PsqtBuckets> psqt{};
        for (usize lane = 0; lane < L1; ++lane)
            lanes[lane] = active->ftBias[lane];

        for (const auto& feature : trace[perspective].pieces)
        {
            if (feature.index >= PieceSquareDimensions)
                return "Alice native piece feature index is outside the loaded tensor.";
            const u64 row = u64(feature.index) * L1;
            for (usize lane = 0; lane < L1; ++lane)
                lanes[lane] += active->pieceSquareWeight[row + lane];
            const u64 psqtRow = u64(feature.index) * PsqtBuckets;
            for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
                psqt[bucket] += active->pieceSquarePsqt[psqtRow + bucket];
        }
        for (const auto& feature : trace[perspective].threats)
        {
            if (feature.index >= ThreatDimensions)
                return "Alice native threat feature index is outside the loaded tensor.";
            const u64 row = u64(feature.index) * L1;
            for (usize lane = 0; lane < L1; ++lane)
                lanes[lane] += active->threatWeight[row + lane];
            const u64 psqtRow = u64(feature.index) * PsqtBuckets;
            for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
                psqt[bucket] += active->threatPsqt[psqtRow + bucket];
        }

        for (usize lane = 0; lane < L1; ++lane)
        {
            if (lanes[lane] < std::numeric_limits<i16>::min()
                || lanes[lane] > std::numeric_limits<i16>::max())
                return "Alice native feature accumulator exceeds signed i16 at perspective "
                     + std::to_string(perspective) + " lane " + std::to_string(lane) + ": "
                     + std::to_string(lanes[lane]);
            accumulators[perspective][lane] = i16(lanes[lane]);
        }
        for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
        {
            if (psqt[bucket] < std::numeric_limits<i32>::min()
                || psqt[bucket] > std::numeric_limits<i32>::max())
                return "Alice native PSQT accumulator exceeds signed i32 at perspective "
                     + std::to_string(perspective) + " bucket " + std::to_string(bucket) + ": "
                     + std::to_string(psqt[bucket]);
            psqtAccumulators[perspective][bucket] = i32(psqt[bucket]);
        }
    }

    std::array<std::array<i32, L1 / 2>, COLOR_NB> transformed{};
    for (Color perspective : {WHITE, BLACK})
        for (usize lane = 0; lane < L1 / 2; ++lane)
        {
            const i32 left  = std::clamp<i32>(accumulators[perspective][lane], 0, 255);
            const i32 right =
              std::clamp<i32>(accumulators[perspective][lane + L1 / 2], 0, 255);
            transformed[perspective][lane] = left * right / 512;
        }

    const Color sideToMove = position.side_to_move();
    std::array<i32, L1> denseInput{};
    for (usize lane = 0; lane < L1 / 2; ++lane)
    {
        denseInput[lane]          = transformed[sideToMove][lane];
        denseInput[lane + L1 / 2] = transformed[~sideToMove][lane];
    }

    const usize phase = (pieceCount - 1) / 4;
    const auto& dense = active->dense[phase];
    const auto affine = [](const i8*       weights,
                           const i32*      biases,
                           usize           outputs,
                           usize           inputs,
                           const i32*      values,
                           i32*            result,
                           std::string_view label,
                           bool            requireI16) -> std::optional<std::string> {
        for (usize output = 0; output < outputs; ++output)
        {
            i64 total = biases[output];
            for (usize input = 0; input < inputs; ++input)
                total += i64(weights[output * inputs + input]) * values[input];
            if (total < std::numeric_limits<i32>::min()
                || total > std::numeric_limits<i32>::max())
                return std::string(label) + " exceeds signed i32 at row "
                     + std::to_string(output) + ": " + std::to_string(total);
            if (requireI16
                && (total < std::numeric_limits<i16>::min()
                    || total > std::numeric_limits<i16>::max()))
                return std::string(label) + " exceeds signed i16 at row "
                     + std::to_string(output) + ": " + std::to_string(total);
            result[output] = i32(total);
        }
        return std::nullopt;
    };
    const auto activate = [](i32 value, int shift, bool square) {
        const i64 raw = square ? i64(value) * value / (i64(1) << (2 * shift + 7))
                               : value / (i64(1) << shift);
        return i32(std::clamp<i64>(raw, 0, 127));
    };

    std::array<i32, L2> z0{};
    if (auto error = affine(dense.fc0Weight.data(), dense.fc0Bias.data(), L2, L1,
                            denseInput.data(), z0.data(), "fc0", true))
        return error;
    std::array<i32, L2> s0{};
    std::array<i32, L2> r0{};
    std::array<i32, 64> y1{};
    for (usize output = 0; output < L2; ++output)
    {
        s0[output]      = activate(z0[output], 7, true);
        r0[output]      = activate(z0[output], 7, false);
        y1[output]      = s0[output];
        y1[L2 + output] = r0[output];
    }

    std::array<i32, L3> z1{};
    if (auto error = affine(dense.fc1Weight.data(), dense.fc1Bias.data(), L3, y1.size(),
                            y1.data(), z1.data(), "fc1", true))
        return error;
    std::array<i32, L3>  s1{};
    std::array<i32, L3>  r1{};
    std::array<i32, 128> y2{};
    for (usize output = 0; output < L3; ++output)
    {
        s1[output]                = activate(z1[output], 6, true);
        r1[output]                = activate(z1[output], 6, false);
        y2[output]                = s0[output];
        y2[L2 + output]           = r0[output];
        y2[2 * L2 + output]       = s1[output];
        y2[2 * L2 + L3 + output] = r1[output];
    }

    std::array<i32, 1> z2{};
    if (auto error = affine(dense.fc2Weight.data(), dense.fc2Bias.data(), 1, y2.size(),
                            y2.data(), z2.data(), "fc2", false))
        return error;
    const i64 skip64   = i64(z0[30]) - z0[31];
    const i64 fwdOut64 = i64(z2[0]) + skip64;
    if (fwdOut64 < std::numeric_limits<i32>::min()
        || fwdOut64 > std::numeric_limits<i32>::max())
        return "fwdOut exceeds signed i32: " + std::to_string(fwdOut64);
    const i32 skip   = i32(skip64);
    const i32 fwdOut = i32(fwdOut64);

    const i64 positionalRaw64 = fwdOut64 * 9600 / 16384;
    const i64 psqtDifference =
      i64(psqtAccumulators[sideToMove][phase]) - psqtAccumulators[~sideToMove][phase];
    const i64 psqtRaw64 = psqtDifference / 2;
    if (positionalRaw64 < std::numeric_limits<i32>::min()
        || positionalRaw64 > std::numeric_limits<i32>::max())
        return "positionalRaw16 exceeds signed i32: " + std::to_string(positionalRaw64);
    if (psqtRaw64 < std::numeric_limits<i32>::min()
        || psqtRaw64 > std::numeric_limits<i32>::max())
        return "psqtRaw16 exceeds signed i32: " + std::to_string(psqtRaw64);
    const i32 positionalRaw16 = i32(positionalRaw64);
    const i32 psqtRaw16       = i32(psqtRaw64);
    const i32 positionalValue = positionalRaw16 / 16;
    const i32 psqtValue       = psqtRaw16 / 16;
    const i32 nativeValue     = positionalValue + psqtValue;

    std::ostringstream out;
    out << "{\"architecture\":\"" << ArchitectureId << "\",\"generation\":"
        << active->generation << ",\"networkSha256\":\"" << active->wire.sha256
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
    write_integer_array(out, accumulators[WHITE]);
    out << ',';
    write_integer_array(out, accumulators[BLACK]);
    out << "],\"psqtAccumulator\":[";
    write_integer_array(out, psqtAccumulators[WHITE]);
    out << ',';
    write_integer_array(out, psqtAccumulators[BLACK]);
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
    out << ",\"fc2Raw\":" << z2[0] << ",\"skip\":" << skip << ",\"fwdOut\":"
        << fwdOut << ",\"positionalRaw16\":" << positionalRaw16 << ",\"psqtRaw16\":"
        << psqtRaw16 << ",\"positionalValue\":" << positionalValue << ",\"psqtValue\":"
        << psqtValue << ",\"nativeNnueValue\":" << nativeValue << '}';
    report = out.str();
    return std::nullopt;
}

}  // namespace Stockfish::Eval::NNUE::AliceNative
