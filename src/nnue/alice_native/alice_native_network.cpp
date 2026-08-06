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
#include <sstream>
#include <string_view>
#include <system_error>

#include "../../misc.h"

namespace Stockfish::Eval::NNUE::AliceNative {

namespace {

constexpr u32 LegacyWireVersion    = 0x7AF32F20u;
constexpr u64 MaximumManifestBytes = 65536;

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

}  // namespace Stockfish::Eval::NNUE::AliceNative
