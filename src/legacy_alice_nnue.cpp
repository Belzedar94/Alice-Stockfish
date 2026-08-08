/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Alice-Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.
*/

#include "legacy_alice_nnue.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(USE_AVX2)
    #include <immintrin.h>
#endif

#include "bitboard.h"
#include "misc.h"
#include "position.h"

namespace Stockfish {

namespace {

constexpr usize FeatureDimensions    = 45056;
constexpr usize TransformedHalf      = 512;
constexpr usize TransformedInput     = 1024;
constexpr usize PsqtBuckets          = 8;
constexpr usize LayerStacks          = 8;
constexpr usize PieceFeatureStride   = 704;
constexpr usize MaxDescriptionLength = 4096;
constexpr usize ExpectedPayloadSize =
  4 + TransformedHalf * sizeof(i16) + FeatureDimensions * TransformedHalf * sizeof(i16)
  + FeatureDimensions * PsqtBuckets * sizeof(i32)
  + LayerStacks
      * (4 + 16 * sizeof(i32) + 16 * TransformedInput * sizeof(i8) + 32 * sizeof(i32)
         + 32 * 32 * sizeof(i8) + sizeof(i32) + 32 * sizeof(i8));

constexpr u32 ExpectedTransformerHash = 0x5F2348B8U;
constexpr u32 ExpectedNetworkHash     = 0x633376CAU;

u32 rotate_right(u32 value, unsigned shift) { return (value >> shift) | (value << (32 - shift)); }

std::array<u8, 32> sha256(const std::vector<u8>& input) {
    constexpr std::array<u32, 64> constants = {
      0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU, 0x59F111F1U, 0x923F82A4U,
      0xAB1C5ED5U, 0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U, 0x72BE5D74U, 0x80DEB1FEU,
      0x9BDC06A7U, 0xC19BF174U, 0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU,
      0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU, 0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U,
      0xC6E00BF3U, 0xD5A79147U, 0x06CA6351U, 0x14292967U, 0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU,
      0x53380D13U, 0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U, 0xA2BFE8A1U, 0xA81A664BU,
      0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U, 0x19A4C116U,
      0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU, 0x682E6FF3U,
      0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U, 0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U,
      0xC67178F2U};

    std::array<u32, 8> state = {0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
                                0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U};

    const auto process_block = [&](const u8* block) {
        std::array<u32, 64> words{};
        for (usize i = 0; i < 16; ++i)
            words[i] = (u32(block[4 * i]) << 24) | (u32(block[4 * i + 1]) << 16)
                     | (u32(block[4 * i + 2]) << 8) | u32(block[4 * i + 3]);
        for (usize i = 16; i < words.size(); ++i)
        {
            const u32 s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18)
                         ^ (words[i - 15] >> 3);
            const u32 s1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19)
                         ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
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
            const u32 choice   = (e & f) ^ (~e & g);
            const u32 majority = (a & b) ^ (a & c) ^ (b & c);
            const u32 sigma0   = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const u32 sigma1   = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const u32 temp1    = h + sigma1 + choice + constants[i] + words[i];
            const u32 temp2    = sigma0 + majority;
            h                  = g;
            g                  = f;
            f                  = e;
            e                  = d + temp1;
            d                  = c;
            c                  = b;
            b                  = a;
            a                  = temp1 + temp2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    };

    usize offset = 0;
    while (input.size() - offset >= 64)
    {
        process_block(input.data() + offset);
        offset += 64;
    }

    std::array<u8, 128> tail{};
    const usize         remainder = input.size() - offset;
    std::memcpy(tail.data(), input.data() + offset, remainder);
    tail[remainder]    = 0x80;
    const usize blocks = remainder < 56 ? 1 : 2;
    const u64   bits   = u64(input.size()) * 8;
    for (usize i = 0; i < 8; ++i)
        tail[blocks * 64 - 1 - i] = u8(bits >> (8 * i));
    for (usize i = 0; i < blocks; ++i)
        process_block(tail.data() + 64 * i);

    std::array<u8, 32> digest{};
    for (usize i = 0; i < state.size(); ++i)
        for (usize j = 0; j < 4; ++j)
            digest[4 * i + j] = u8(state[i] >> (24 - 8 * j));
    return digest;
}

std::string digest_string(const std::array<u8, 32>& digest) {
    std::ostringstream out;
    out << std::hex << std::uppercase << std::setfill('0');
    for (u8 byte : digest)
        out << std::setw(2) << unsigned(byte);
    return out.str();
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

class ByteReader {
   public:
    explicit ByteReader(const std::vector<u8>& source) :
        bytes(source) {}

    template<typename T>
    bool read(T& value) {
        static_assert(std::is_integral_v<T>);
        if (remaining() < sizeof(T))
            return false;

        using Unsigned = std::make_unsigned_t<T>;
        Unsigned raw   = 0;
        for (usize i = 0; i < sizeof(T); ++i)
            raw |= Unsigned(bytes[position + i]) << (8 * i);
        std::memcpy(&value, &raw, sizeof(T));
        position += sizeof(T);
        return true;
    }

    template<typename T, usize Size>
    bool read(std::array<T, Size>& values) {
        return read(values.data(), values.size());
    }

    template<typename T>
    bool read(std::vector<T>& values, usize count) {
        values.resize(count);
        return read(values.data(), values.size());
    }

    bool read_string(std::string& value, usize size) {
        if (remaining() < size)
            return false;
        value.assign(reinterpret_cast<const char*>(bytes.data() + position), size);
        position += size;
        return true;
    }

    usize remaining() const { return bytes.size() - position; }

   private:
    template<typename T>
    bool read(T* values, usize count) {
        const usize size = count * sizeof(T);
        if (remaining() < size)
            return false;
        if (IsLittleEndian)
        {
            std::memcpy(values, bytes.data() + position, size);
            position += size;
            return true;
        }
        for (usize i = 0; i < count; ++i)
            if (!read(values[i]))
                return false;
        return true;
    }

    const std::vector<u8>& bytes;
    usize                  position = 0;
};

#if !defined(USE_AVX2)
i16 add_wrapped(i16 left, i16 right) {
    const u16 raw = u16(u16(left) + u16(right));
    i16       result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

i16 subtract_wrapped(i16 left, i16 right) {
    const u16 raw = u16(u16(left) - u16(right));
    i16       result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}
#endif

int truncate_division(int value, int divisor) { return value / divisor; }

usize piece_feature_offset(Color perspective, Piece piece) {
    usize offset = 0;
    switch (type_of(piece))
    {
    case PAWN :
        offset = 0;
        break;
    case KNIGHT :
        offset = 128;
        break;
    case BISHOP :
        offset = 256;
        break;
    case ROOK :
        offset = 384;
        break;
    case QUEEN :
        offset = 512;
        break;
    case KING :
        offset = 640;
        break;
    default :
        assert(false && "Unsupported legacy Alice feature piece");
        break;
    }
    if (type_of(piece) != KING && color_of(piece) != perspective)
        offset += 64;
    return offset;
}

template<usize OutputDimensions>
std::array<i32, OutputDimensions> affine(const u8*  input,
                                         usize      inputDimensions,
                                         usize      paddedInputDimensions,
                                         const i32* biases,
                                         const i8*  weights) {
    std::array<i32, OutputDimensions> output{};
    for (usize out = 0; out < OutputDimensions; ++out)
    {
        i32 sum = biases[out];
        for (usize in = 0; in < inputDimensions; ++in)
            sum += i32(input[in]) * i32(weights[out * paddedInputDimensions + in]);
        output[out] = sum;
    }
    return output;
}

std::array<i32, 16> affine_16x1024(const u8*                        input,
                                   const std::array<i32, 16>&       biases,
                                   const std::array<i8, 16 * 1024>& rowMajorWeights,
                                   const std::array<i8, 16 * 1024>& interleavedWeights) {
#if defined(USE_AVX2)
    (void) rowMajorWeights;
    constexpr usize ChunkCount  = 1024 / 4;
    constexpr usize ChunkStride = 16 * 4;

    const __m256i ones    = _mm256_set1_epi16(1);
    __m256i       sums[2] = {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(biases.data())),
                             _mm256_loadu_si256(reinterpret_cast<const __m256i*>(biases.data() + 8))};

    const auto multiply_chunk = [&](usize chunk, __m256i packedInput, usize group) {
        const auto* weights = reinterpret_cast<const __m256i*>(interleavedWeights.data()
                                                               + chunk * ChunkStride + group * 32);
        __m256i     product = _mm256_maddubs_epi16(packedInput, _mm256_loadu_si256(weights));
        return product;
    };

    for (usize chunk = 0; chunk < ChunkCount; chunk += 4)
    {
        i32 packed[4];
        for (usize i = 0; i < 4; ++i)
            std::memcpy(&packed[i], input + 4 * (chunk + i), sizeof(packed[i]));

        const __m256i in[4] = {_mm256_set1_epi32(packed[0]), _mm256_set1_epi32(packed[1]),
                               _mm256_set1_epi32(packed[2]), _mm256_set1_epi32(packed[3])};
        for (usize group = 0; group < 2; ++group)
        {
            __m256i product0 = multiply_chunk(chunk, in[0], group);
            __m256i product1 = multiply_chunk(chunk + 1, in[1], group);
            __m256i product2 = multiply_chunk(chunk + 2, in[2], group);
            __m256i product3 = multiply_chunk(chunk + 3, in[3], group);
            product0         = _mm256_adds_epi16(product0, product1);
            product2         = _mm256_adds_epi16(product2, product3);
            product0         = _mm256_madd_epi16(product0, ones);
            product2         = _mm256_madd_epi16(product2, ones);
            sums[group]      = _mm256_add_epi32(sums[group], _mm256_add_epi32(product0, product2));
        }
    }

    std::array<i32, 16> output{};
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output.data()), sums[0]);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output.data() + 8), sums[1]);
    return output;
#else
    (void) interleavedWeights;
    return affine<16>(input, 1024, 1024, biases.data(), rowMajorWeights.data());
#endif
}

template<usize Size>
std::array<u8, Size> clipped_relu(const std::array<i32, Size>& input) {
    std::array<u8, Size> output{};
    for (usize i = 0; i < Size; ++i)
        output[i] = u8(std::clamp(input[i] >> 6, 0, 127));
    return output;
}

struct LegacyAccumulatorState {
    std::array<std::array<i16, TransformedHalf>, COLOR_NB> accumulation{};
    std::array<std::array<i32, PsqtBuckets>, COLOR_NB>     psqt{};
    std::array<Square, COLOR_NB>                           kingSquares{SQ_NONE, SQ_NONE};
};

void apply_feature(LegacyAccumulatorState& state,
                   Color                   perspective,
                   Piece                   piece,
                   Square                  square,
                   bool                    add,
                   const std::vector<i16>& weights,
                   const std::vector<i32>& psqtWeights) {
    assert(piece != NO_PIECE && is_ok(square));
    const Square orientedSquare = relative_square(perspective, square);
    const usize  feature        = usize(state.kingSquares[perspective]) * PieceFeatureStride
                        + piece_feature_offset(perspective, piece) + usize(orientedSquare);
    assert(feature < FeatureDimensions);

    const usize weightOffset = feature * TransformedHalf;
#if defined(USE_AVX2)
    for (usize i = 0; i < TransformedHalf; i += 16)
    {
        const __m256i current = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(state.accumulation[perspective].data() + i));
        const __m256i delta =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights.data() + weightOffset + i));
        const __m256i updated =
          add ? _mm256_add_epi16(current, delta) : _mm256_sub_epi16(current, delta);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(state.accumulation[perspective].data() + i),
                            updated);
    }
#else
    for (usize i = 0; i < TransformedHalf; ++i)
        state.accumulation[perspective][i] =
          add ? add_wrapped(state.accumulation[perspective][i], weights[weightOffset + i])
              : subtract_wrapped(state.accumulation[perspective][i], weights[weightOffset + i]);
#endif

    const usize psqtOffset = feature * PsqtBuckets;
#if defined(USE_AVX2)
    const __m256i currentPsqt =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(state.psqt[perspective].data()));
    const __m256i psqtDelta =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(psqtWeights.data() + psqtOffset));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(state.psqt[perspective].data()),
                        add ? _mm256_add_epi32(currentPsqt, psqtDelta)
                            : _mm256_sub_epi32(currentPsqt, psqtDelta));
#else
    for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
        state.psqt[perspective][bucket] +=
          add ? psqtWeights[psqtOffset + bucket] : -psqtWeights[psqtOffset + bucket];
#endif
}

void refresh_perspective(LegacyAccumulatorState&                 state,
                         Color                                   perspective,
                         const Position&                         pos,
                         const std::array<i16, TransformedHalf>& biases,
                         const std::vector<i16>&                 weights,
                         const std::vector<i32>&                 psqtWeights) {
    state.accumulation[perspective] = biases;
    state.psqt[perspective].fill(0);
    state.kingSquares[perspective] = relative_square(perspective, pos.square<KING>(perspective));

    Bitboard occupied = pos.pieces();
    while (occupied)
    {
        const Square square = pop_lsb(occupied);
        apply_feature(state, perspective, pos.piece_on(square), square, true, weights, psqtWeights);
    }
}

void refresh_state(LegacyAccumulatorState&                 state,
                   const Position&                         pos,
                   const std::array<i16, TransformedHalf>& biases,
                   const std::vector<i16>&                 weights,
                   const std::vector<i32>&                 psqtWeights) {
    refresh_perspective(state, WHITE, pos, biases, weights, psqtWeights);
    refresh_perspective(state, BLACK, pos, biases, weights, psqtWeights);
}

template<typename Stack>
Value evaluate_state(const Position&                       pos,
                     const LegacyAccumulatorState&         state,
                     const std::array<Stack, LayerStacks>& stacks,
                     bool                                  adjusted) {
    std::array<u8, TransformedInput> transformed{};
    const std::array<Color, 2>       perspectives = {pos.side_to_move(), ~pos.side_to_move()};
    for (usize p = 0; p < perspectives.size(); ++p)
        for (usize i = 0; i < TransformedHalf; ++i)
            transformed[p * TransformedHalf + i] =
              u8(std::clamp(int(state.accumulation[perspectives[p]][i]), 0, 127));

    const usize pieceCount = popcount(pos.pieces());
    assert(pieceCount > 0);
    const usize bucket   = std::min((pieceCount - 1) * 8 / 32, usize(7));
    const i32   material = truncate_division(
      state.psqt[perspectives[0]][bucket] - state.psqt[perspectives[1]][bucket], 2);

    const Stack& stack   = stacks[bucket];
    const auto   hidden1 = clipped_relu(
      affine_16x1024(transformed.data(), stack.bias1, stack.weight1, stack.weight1Interleaved));
    const auto hidden2 = clipped_relu(
      affine<32>(hidden1.data(), hidden1.size(), 32, stack.bias2.data(), stack.weight2.data()));
    const i32 positional =
      affine<1>(hidden2.data(), hidden2.size(), 32, &stack.bias3, stack.weight3.data())[0];

    const int delta         = std::abs(pos.non_pawn_material(WHITE) - pos.non_pawn_material(BLACK));
    const int entertainment = adjusted && delta <= BishopValue - KnightValue ? 7 : 0;
    const int sum = ((128 - entertainment) * material + (128 + entertainment) * positional) / 128;
    return Value(sum / 16);
}

}  // namespace

struct LegacyAliceExact::Accumulator::Impl {
    std::vector<LegacyAccumulatorState> states;
};

LegacyAliceExact::Accumulator::Accumulator() :
    impl(std::make_unique<Impl>()) {}
LegacyAliceExact::Accumulator::~Accumulator() = default;

struct LegacyAliceExact::Impl {
    struct Stack {
        std::array<i32, 16>       bias1{};
        std::array<i8, 16 * 1024> weight1{};
        std::array<i8, 16 * 1024> weight1Interleaved{};
        std::array<i32, 32>       bias2{};
        std::array<i8, 32 * 32>   weight2{};
        i32                       bias3 = 0;
        std::array<i8, 32>        weight3{};
    };

    std::array<i16, TransformedHalf> biases{};
    std::vector<i16>                 weights;
    std::vector<i32>                 psqtWeights;
    std::array<Stack, LayerStacks>   stacks{};
    Metadata                         metadata;
    std::string                      lastError;
    bool                             ready = false;

    void clear() {
        ready = false;
        weights.clear();
        psqtWeights.clear();
        metadata = {};
    }
};

LegacyAliceExact::LegacyAliceExact() :
    impl(std::make_unique<Impl>()) {}
LegacyAliceExact::~LegacyAliceExact() = default;

void LegacyAliceExact::reset() {
    impl->clear();
    impl->lastError.clear();
}

std::optional<std::string> LegacyAliceExact::load(const std::filesystem::path& file,
                                                  LoadPolicy                   policy) {
    impl->clear();
    impl->lastError.clear();

    const auto reject = [this](std::string message) -> std::optional<std::string> {
        impl->clear();
        impl->lastError = std::move(message);
        return impl->lastError;
    };

    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    if (!stream)
        return reject("Unable to open EvalFile: " + path_string(file) + ".");

    const std::streampos end = stream.tellg();
    if (end < 0)
        return reject("Unable to determine EvalFile length: " + path_string(file) + ".");
    const u64 fileSize = u64(end);
    if (fileSize < 12 || fileSize > 64ULL * 1024 * 1024)
        return reject("EvalFile length is outside the accepted legacy Alice format bounds.");

    std::array<u8, 12> header{};
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(header.data()), std::streamsize(header.size()));
    if (!stream || usize(stream.gcount()) != header.size())
        return reject("EvalFile header is truncated.");

    const auto header_word = [&header](usize offset) {
        return u32(header[offset]) | (u32(header[offset + 1]) << 8)
             | (u32(header[offset + 2]) << 16) | (u32(header[offset + 3]) << 24);
    };
    const u32 version         = header_word(0);
    const u32 architecture    = header_word(4);
    const u32 descriptionSize = header_word(8);
    if (version != ExpectedVersion)
        return reject("Unsupported NNUE serialization version " + hex32(version) + "; expected "
                      + hex32(ExpectedVersion) + ".");
    if (architecture != ExpectedArchitecture)
        return reject("Unsupported NNUE architecture " + hex32(architecture) + "; expected "
                      + hex32(ExpectedArchitecture) + ".");
    if (descriptionSize > MaxDescriptionLength)
        return reject("EvalFile description is unreasonably large.");
    if (fileSize != 12 + u64(descriptionSize) + ExpectedPayloadSize)
        return reject("EvalFile structural length does not match the exact legacy architecture.");

    std::vector<u8> bytes(static_cast<usize>(fileSize), u8{});
    stream.clear();
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!stream || usize(stream.gcount()) != bytes.size())
        return reject("EvalFile could not be read completely.");

    const std::string checksum = digest_string(sha256(bytes));
    ByteReader        reader(bytes);
    u32               parsedVersion         = 0;
    u32               parsedArchitecture    = 0;
    u32               parsedDescriptionSize = 0;
    if (!reader.read(parsedVersion) || !reader.read(parsedArchitecture)
        || !reader.read(parsedDescriptionSize))
        return reject("EvalFile header is truncated.");
    if (parsedVersion != version || parsedArchitecture != architecture
        || parsedDescriptionSize != descriptionSize)
        return reject("EvalFile changed while it was being read.");

    std::string description;
    if (!reader.read_string(description, descriptionSize))
        return reject("EvalFile description is truncated.");
    if (policy == LoadPolicy::FrozenBaseline && checksum != FrozenSha256)
        return reject("EvalFile SHA-256 does not match the frozen Alice baseline.");

    u32 transformerHash = 0;
    if (!reader.read(transformerHash) || transformerHash != ExpectedTransformerHash)
        return reject("EvalFile feature-transformer hash is incompatible.");
    if (!reader.read(impl->biases)
        || !reader.read(impl->weights, FeatureDimensions * TransformedHalf)
        || !reader.read(impl->psqtWeights, FeatureDimensions * PsqtBuckets))
        return reject("EvalFile feature-transformer parameters are truncated.");

    for (auto& stack : impl->stacks)
    {
        u32 networkHash = 0;
        if (!reader.read(networkHash) || networkHash != ExpectedNetworkHash)
            return reject("EvalFile layer-stack hash is incompatible.");
        if (!reader.read(stack.bias1) || !reader.read(stack.weight1) || !reader.read(stack.bias2)
            || !reader.read(stack.weight2) || !reader.read(stack.bias3)
            || !reader.read(stack.weight3))
            return reject("EvalFile layer-stack parameters are truncated.");

        for (usize i = 0; i < stack.weight1.size(); ++i)
        {
            const usize interleaved = (i / 4) % (1024 / 4) * 16 * 4 + i / 1024 * 4 + i % 4;
            stack.weight1Interleaved[interleaved] = stack.weight1[i];
        }
    }
    if (reader.remaining() != 0)
        return reject("EvalFile has unexpected trailing data.");

    std::error_code ec;
    auto            normalized = std::filesystem::weakly_canonical(file, ec);
    if (ec)
    {
        ec.clear();
        normalized = std::filesystem::absolute(file, ec);
        if (ec)
            normalized = file.lexically_normal();
    }

    impl->metadata.normalizedPath = path_string(normalized);
    impl->metadata.sha256         = checksum;
    impl->metadata.description    = std::move(description);
    impl->metadata.version        = version;
    impl->metadata.architecture   = architecture;
    impl->metadata.fileSize       = fileSize;
    impl->metadata.frozen         = checksum == FrozenSha256;
    impl->ready                   = true;
    return std::nullopt;
}

bool LegacyAliceExact::loaded() const { return impl->ready; }

std::optional<Value> LegacyAliceExact::evaluate(const Position& pos, bool adjusted) const {
    if (!impl->ready)
        return std::nullopt;

    LegacyAccumulatorState state;
    refresh_state(state, pos, impl->biases, impl->weights, impl->psqtWeights);
    return evaluate_state(pos, state, impl->stacks, adjusted);
}

std::unique_ptr<LegacyAliceExact::Accumulator>
LegacyAliceExact::make_accumulator(const Position& pos) const {
    if (!impl->ready)
        return nullptr;

    auto accumulator = std::unique_ptr<Accumulator>(new Accumulator());
    accumulator->impl->states.emplace_back();
    refresh_state(accumulator->impl->states.back(), pos, impl->biases, impl->weights,
                  impl->psqtWeights);
    return accumulator;
}

std::optional<Value> LegacyAliceExact::evaluate(const Position&    pos,
                                                const Accumulator& accumulator,
                                                bool               adjusted) const {
    if (!impl->ready || accumulator.impl->states.empty())
        return std::nullopt;
    return evaluate_state(pos, accumulator.impl->states.back(), impl->stacks, adjusted);
}

void LegacyAliceExact::push(Accumulator&    accumulator,
                            const Position& pos,
                            const Dirties&  dirties) const {
    assert(impl->ready && !accumulator.impl->states.empty());
    LegacyAccumulatorState next = accumulator.impl->states.back();
    accumulator.impl->states.push_back(std::move(next));
    LegacyAccumulatorState& state = accumulator.impl->states.back();

    for (Color perspective : {WHITE, BLACK})
    {
        const Square kingSquare = relative_square(perspective, pos.square<KING>(perspective));
        if (kingSquare != state.kingSquares[perspective])
        {
            refresh_perspective(state, perspective, pos, impl->biases, impl->weights,
                                impl->psqtWeights);
            continue;
        }

        const DirtyPiece& dirty = dirties.dirtyPiece;
        apply_feature(state, perspective, dirty.pc, dirty.from, false, impl->weights,
                      impl->psqtWeights);
        if (dirty.to != SQ_NONE)
            apply_feature(state, perspective, dirty.pc, dirty.to, true, impl->weights,
                          impl->psqtWeights);
        if (dirty.remove_sq != SQ_NONE)
            apply_feature(state, perspective, dirty.remove_pc, dirty.remove_sq, false,
                          impl->weights, impl->psqtWeights);
        if (dirty.add_sq != SQ_NONE)
            apply_feature(state, perspective, dirty.add_pc, dirty.add_sq, true, impl->weights,
                          impl->psqtWeights);
    }
}

void LegacyAliceExact::pop(Accumulator& accumulator) const {
    assert(accumulator.impl->states.size() > 1);
    accumulator.impl->states.pop_back();
}

const LegacyAliceExact::Metadata& LegacyAliceExact::metadata() const { return impl->metadata; }
const std::string&                LegacyAliceExact::last_error() const { return impl->lastError; }

std::string LegacyAliceExact::status_line() const {
    if (!impl->ready)
        return "LegacyAliceExact is not loaded"
             + (impl->lastError.empty() ? std::string(".") : ": " + impl->lastError);

    std::ostringstream out;
    out << "LegacyAliceExact loaded path=\"" << impl->metadata.normalizedPath
        << "\" mode=" << (impl->metadata.frozen ? "frozen-baseline" : "format-compatible")
        << " sha256=" << impl->metadata.sha256 << " version=" << hex32(impl->metadata.version)
        << " architecture=" << hex32(impl->metadata.architecture);
    return out.str();
}

}  // namespace Stockfish
