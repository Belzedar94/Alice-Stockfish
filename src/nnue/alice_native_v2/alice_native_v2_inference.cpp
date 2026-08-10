/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "alice_native_v2_inference.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string_view>

#if defined(USE_AVX2)
    #include <immintrin.h>
#endif

#include "../../bitboard.h"
#include "../../evaluate.h"
#include "../../position.h"

namespace Stockfish::Eval::NNUE::AliceNativeV2 {

namespace {

std::optional<std::string> validate_accumulator(const IntegerAccumulator& accumulator,
                                                Color                     perspective) {
    (void) accumulator;
    (void) perspective;
    return std::nullopt;
}

template<typename Update>
void apply_piece_delta(const AliceNative::FixedIndexList<AliceNative::MaximumPieceFeatures>& before,
                       const AliceNative::FixedIndexList<AliceNative::MaximumPieceFeatures>& after,
                       Update&& update,
                       u64&     adds,
                       u64&     removes) {
    // Remove first so every intermediate accumulator represents at most the
    // sealed 32 active rows. A merged add-before-remove sequence can transiently
    // represent 33 rows even for a quiet move and invalidate the i16 envelope.
    usize beforeIndex = 0;
    usize afterIndex  = 0;
    while (beforeIndex < before.size())
    {
        while (afterIndex < after.size() && after[afterIndex] < before[beforeIndex])
            ++afterIndex;
        if (afterIndex == after.size() || before[beforeIndex] < after[afterIndex])
        {
            update(before[beforeIndex], -1);
            ++removes;
        }
        ++beforeIndex;
    }

    beforeIndex = 0;
    afterIndex  = 0;
    while (afterIndex < after.size())
    {
        while (beforeIndex < before.size() && before[beforeIndex] < after[afterIndex])
            ++beforeIndex;
        if (beforeIndex == before.size() || after[afterIndex] < before[beforeIndex])
        {
            update(after[afterIndex], 1);
            ++adds;
        }
        ++afterIndex;
    }
}

template<typename Input>
std::optional<std::string> affine(const i8*        weights,
                                  const i32*       biases,
                                  usize            outputs,
                                  usize            wireInputs,
                                  usize            logicalInputs,
                                  const Input*     values,
                                  i32*             result,
                                  std::string_view label) {
    for (usize output = 0; output < outputs; ++output)
    {
        i64 total = biases[output];
        for (usize input = 0; input < logicalInputs; ++input)
            total += i64(weights[output * wireInputs + input]) * values[input];
        if (total < std::numeric_limits<i32>::min() || total > std::numeric_limits<i32>::max())
            return std::string(label) + " exceeds signed i32 at row " + std::to_string(output)
                 + ": " + std::to_string(total);
        result[output] = i32(total);
    }
    return std::nullopt;
}

std::optional<std::string> affine_fc0(const DenseParameterView& dense,
                                      const u8*                 values,
                                      i32*                      result) {
#if defined(USE_AVX2)
    if (!dense.fc0InterleavedWeight)
        return "AliceNative-v2 fc0 interleaved weights are unavailable.";
    constexpr usize ChunkCount  = DenseInput / 4;
    constexpr usize ChunkStride = Hidden1 * 4;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i sums[2] = {
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dense.fc0Bias)),
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dense.fc0Bias + 8))};

    const auto multiply_chunk = [&](usize chunk, __m256i packedInput, usize group) {
        const auto* weights = reinterpret_cast<const __m256i*>(
          dense.fc0InterleavedWeight + chunk * ChunkStride + group * 32);
        return _mm256_maddubs_epi16(packedInput, _mm256_loadu_si256(weights));
    };

    for (usize chunk = 0; chunk < ChunkCount; chunk += 4)
    {
        i32 packed[4];
        for (usize index = 0; index < 4; ++index)
            std::memcpy(&packed[index], values + 4 * (chunk + index), sizeof(packed[index]));
        const __m256i input[4] = {
          _mm256_set1_epi32(packed[0]), _mm256_set1_epi32(packed[1]),
          _mm256_set1_epi32(packed[2]), _mm256_set1_epi32(packed[3])};
        for (usize group = 0; group < 2; ++group)
        {
            __m256i product0 = multiply_chunk(chunk, input[0], group);
            __m256i product1 = multiply_chunk(chunk + 1, input[1], group);
            __m256i product2 = multiply_chunk(chunk + 2, input[2], group);
            __m256i product3 = multiply_chunk(chunk + 3, input[3], group);
            product0 = _mm256_adds_epi16(product0, product1);
            product2 = _mm256_adds_epi16(product2, product3);
            product0 = _mm256_madd_epi16(product0, ones);
            product2 = _mm256_madd_epi16(product2, ones);
            sums[group] =
              _mm256_add_epi32(sums[group], _mm256_add_epi32(product0, product2));
        }
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), sums[0]);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + 8), sums[1]);
    return std::nullopt;
#else
    return affine(dense.fc0Weight, dense.fc0Bias, Hidden1, DenseInput, DenseInput, values,
                  result, "AliceNative-v2 fc0");
#endif
}

#if defined(USE_AVX2)
void apply_feature_vector(i16* destination, const i16* source, bool add) {
    for (usize lane = 0; lane < TransformerWidth; lane += 16)
    {
        const __m256i current =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(destination + lane));
        const __m256i delta =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + lane));
        const __m256i updated =
          add ? _mm256_add_epi16(current, delta) : _mm256_sub_epi16(current, delta);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + lane), updated);
    }
}

void apply_psqt_vector(i32* destination, const i32* source, bool add) {
    const __m256i current =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(destination));
    const __m256i delta = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));
    const __m256i updated = add ? _mm256_add_epi32(current, delta)
                                : _mm256_sub_epi32(current, delta);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), updated);
}
#endif

i32 hidden_activation(i32 value) {
    if (value <= 0)
        return 0;
    return std::min<i32>(value / (1 << HiddenShift), FeatureScale);
}

bool checked_i32(i64 value, i32& destination) {
    if (value < std::numeric_limits<i32>::min() || value > std::numeric_limits<i32>::max())
        return false;
    destination = i32(value);
    return true;
}

}  // namespace

std::optional<std::string>
refresh_accumulator(const ParameterView&                         parameters,
                    const AliceNative::PerspectivePieceSnapshot& snapshot,
                    IntegerAccumulator&                           accumulator) {
    accumulator = {};
    std::array<i64, PsqtBuckets>      psqt{};
#if defined(USE_AVX2)
    std::copy_n(parameters.ftBias, TransformerWidth, accumulator.values.begin());
#else
    std::array<i32, TransformerWidth> values{};
    for (usize lane = 0; lane < TransformerWidth; ++lane)
        values[lane] = parameters.ftBias[lane];
#endif

    for (IndexType index : snapshot.pieces)
    {
        if (index >= PieceSquareDimensions)
            return "AliceNative-v2 piece feature index is outside the loaded tensor.";
        const u64 row = u64(index) * TransformerWidth;
#if defined(USE_AVX2)
        apply_feature_vector(accumulator.values.data(), parameters.pieceSquareWeight + row, true);
#else
        for (usize lane = 0; lane < TransformerWidth; ++lane)
            values[lane] += parameters.pieceSquareWeight[row + lane];
#endif
        const u64 psqtRow = u64(index) * PsqtBuckets;
        for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
            psqt[bucket] += parameters.pieceSquarePsqt[psqtRow + bucket];
    }
#if !defined(USE_AVX2)
    for (usize lane = 0; lane < TransformerWidth; ++lane)
    {
        if (values[lane] < std::numeric_limits<i16>::min()
            || values[lane] > std::numeric_limits<i16>::max())
            return "AliceNative-v2 feature refresh exceeds signed i16 at perspective "
                 + std::to_string(int(snapshot.perspective)) + " lane " + std::to_string(lane);
        accumulator.values[lane] = i16(values[lane]);
    }
#endif
    for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
    {
        if (psqt[bucket] < std::numeric_limits<i32>::min()
            || psqt[bucket] > std::numeric_limits<i32>::max())
            return "AliceNative-v2 PSQT refresh exceeds signed i32 at perspective "
                 + std::to_string(int(snapshot.perspective)) + " bucket "
                 + std::to_string(bucket);
        accumulator.psqt[bucket] = i32(psqt[bucket]);
    }
    return std::nullopt;
}

std::optional<std::string>
update_accumulator(const ParameterView&                         parameters,
                   const AliceNative::PerspectivePieceSnapshot& before,
                   const AliceNative::PerspectivePieceSnapshot& after,
                   IntegerAccumulator&                           accumulator,
                   AccumulatorDeltaStats&                        stats) {
    stats = {};
    bool overflow = false;
    apply_piece_delta(
      before.pieces, after.pieces,
      [&](IndexType index, i32 sign) {
          const u64 row = u64(index) * TransformerWidth;
#if defined(USE_AVX2)
          apply_feature_vector(accumulator.values.data(), parameters.pieceSquareWeight + row,
                               sign > 0);
#else
          for (usize lane = 0; lane < TransformerWidth; ++lane)
          {
              const i32 next = i32(accumulator.values[lane])
                             + sign * i32(parameters.pieceSquareWeight[row + lane]);
              if (next < std::numeric_limits<i16>::min()
                  || next > std::numeric_limits<i16>::max())
              {
                  overflow = true;
                  accumulator.values[lane] = std::numeric_limits<i16>::min();
              }
              else
                  accumulator.values[lane] = i16(next);
          }
#endif
          const u64 psqtRow = u64(index) * PsqtBuckets;
#if defined(USE_AVX2)
          apply_psqt_vector(accumulator.psqt.data(), parameters.pieceSquarePsqt + psqtRow,
                            sign > 0);
#else
          for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
          {
              const i64 next = i64(accumulator.psqt[bucket])
                             + sign * i64(parameters.pieceSquarePsqt[psqtRow + bucket]);
              if (next < std::numeric_limits<i32>::min()
                  || next > std::numeric_limits<i32>::max())
              {
                  overflow = true;
                  accumulator.psqt[bucket] = std::numeric_limits<i32>::min();
              }
              else
                  accumulator.psqt[bucket] = i32(next);
          }
#endif
      },
      stats.pieceAdds, stats.pieceRemoves);
    if (overflow)
        return "AliceNative-v2 incremental accumulator update overflowed its sealed integer domain.";
    return std::nullopt;
}

std::optional<std::string>
update_accumulator_events(const ParameterView&    parameters,
                          const PieceEventIndices& removals,
                          usize                    removalCount,
                          const PieceEventIndices& additions,
                          usize                    additionCount,
                          IntegerAccumulator&      accumulator,
                          AccumulatorDeltaStats&   stats) {
    stats = {};
    if (removalCount > removals.size() || additionCount > additions.size())
        return "AliceNative-v2 piece event count exceeds the sealed transition capacity.";
    bool overflow = false;
    const auto apply = [&](IndexType index, bool add) {
        if (index >= PieceSquareDimensions)
        {
            overflow = true;
            return;
        }
        const u64 row = u64(index) * TransformerWidth;
#if defined(USE_AVX2)
        apply_feature_vector(accumulator.values.data(), parameters.pieceSquareWeight + row, add);
        apply_psqt_vector(accumulator.psqt.data(),
                          parameters.pieceSquarePsqt + u64(index) * PsqtBuckets, add);
#else
        for (usize lane = 0; lane < TransformerWidth; ++lane)
        {
            const i32 next = i32(accumulator.values[lane])
                           + (add ? 1 : -1) * i32(parameters.pieceSquareWeight[row + lane]);
            if (next < std::numeric_limits<i16>::min()
                || next > std::numeric_limits<i16>::max())
                overflow = true;
            else
                accumulator.values[lane] = i16(next);
        }
        const u64 psqtRow = u64(index) * PsqtBuckets;
        for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
        {
            const i64 next = i64(accumulator.psqt[bucket])
                           + (add ? 1 : -1) * i64(parameters.pieceSquarePsqt[psqtRow + bucket]);
            if (next < std::numeric_limits<i32>::min()
                || next > std::numeric_limits<i32>::max())
                overflow = true;
            else
                accumulator.psqt[bucket] = i32(next);
        }
#endif
    };

    // This ordering is part of the integer-domain proof: never materialize a
    // transient 33rd feature while replacing one piece event with another.
    for (usize index = 0; index < removalCount; ++index)
        apply(removals[index], false);
    for (usize index = 0; index < additionCount; ++index)
        apply(additions[index], true);
    stats.pieceRemoves = removalCount;
    stats.pieceAdds    = additionCount;
    if (overflow)
        return "AliceNative-v2 direct piece event update exceeded its sealed integer domain.";
    return std::nullopt;
}

std::optional<std::string> evaluate(const ParameterView&         parameters,
                                    const Position&              position,
                                    const IntegerAccumulatorSet& accumulators,
                                    IntegerStages&               stages) {
    stages            = {};
    stages.pieceCount = popcount(position.pieces());
    if (stages.pieceCount < 2 || stages.pieceCount > 32)
        return "AliceNative-v2 integer evaluation requires between 2 and 32 pieces.";

    for (Color perspective : {WHITE, BLACK})
    {
        if (auto error = validate_accumulator(accumulators[perspective], perspective))
            return error;
        for (usize lane = 0; lane < TransformerWidth; ++lane)
            stages.transformed[perspective][lane] = i32(std::clamp<i64>(
              accumulators[perspective].values[lane], 0, FeatureScale));
    }

    stages.sideToMove = position.side_to_move();
    for (usize lane = 0; lane < TransformerWidth; ++lane)
    {
        stages.denseInput[lane] = stages.transformed[stages.sideToMove][lane];
        stages.denseInput[TransformerWidth + lane] =
          stages.transformed[~stages.sideToMove][lane];
    }

    stages.phase      = (stages.pieceCount - 1) / 4;
    const auto& dense = parameters.dense[stages.phase];
    if (auto error = affine(dense.fc0Weight, dense.fc0Bias, Hidden1, DenseInput, DenseInput,
                            stages.denseInput.data(), stages.fc0Raw.data(), "AliceNative-v2 fc0"))
        return error;
    for (usize output = 0; output < Hidden1; ++output)
        stages.fc0[output] = hidden_activation(stages.fc0Raw[output]);

    if (auto error = affine(dense.fc1Weight, dense.fc1Bias, Hidden2, Hidden2, Hidden1,
                            stages.fc0.data(), stages.fc1Raw.data(), "AliceNative-v2 fc1"))
        return error;
    for (usize output = 0; output < Hidden2; ++output)
        stages.fc1[output] = hidden_activation(stages.fc1Raw[output]);

    if (auto error = affine(dense.fc2Weight, dense.fc2Bias, 1, Hidden2, Hidden2,
                            stages.fc1.data(), &stages.fc2Raw, "AliceNative-v2 fc2"))
        return error;

    const i64 materialDifference = i64(accumulators[stages.sideToMove].psqt[stages.phase])
                                 - i64(accumulators[~stages.sideToMove].psqt[stages.phase]);
    if (!checked_i32(materialDifference / 2, stages.materialRaw))
        return "AliceNative-v2 material raw value exceeds signed i32.";
    if (!checked_i32(i64(stages.fc2Raw) + stages.materialRaw, stages.combinedRaw))
        return "AliceNative-v2 combined raw value exceeds signed i32.";
    stages.rawValue = stages.combinedRaw / EngineOutputDivisor;

    const int delta =
      std::abs(position.non_pawn_material(WHITE) - position.non_pawn_material(BLACK));
    stages.entertainment = delta <= BishopValue - KnightValue ? 7 : 0;
    const i64 adjustedNumerator =
      i64(128 - stages.entertainment) * stages.materialRaw
      + i64(128 + stages.entertainment) * stages.fc2Raw;
    if (!checked_i32(adjustedNumerator / 128, stages.adjustedRaw))
        return "AliceNative-v2 adjusted raw value exceeds signed i32.";
    stages.adjustedValue = stages.adjustedRaw / EngineOutputDivisor;
    return std::nullopt;
}

bool same_stages(const IntegerStages& left, const IntegerStages& right) {
    return left.sideToMove == right.sideToMove && left.pieceCount == right.pieceCount
        && left.phase == right.phase && left.transformed == right.transformed
        && left.denseInput == right.denseInput && left.fc0Raw == right.fc0Raw
        && left.fc0 == right.fc0 && left.fc1Raw == right.fc1Raw && left.fc1 == right.fc1
        && left.fc2Raw == right.fc2Raw && left.materialRaw == right.materialRaw
        && left.combinedRaw == right.combinedRaw && left.entertainment == right.entertainment
        && left.adjustedRaw == right.adjustedRaw && left.rawValue == right.rawValue
        && left.adjustedValue == right.adjustedValue;
}

std::optional<std::string> evaluate_value(const ParameterView&         parameters,
                                          const Position&              position,
                                          const IntegerAccumulatorSet& accumulators,
                                          bool                         adjusted,
                                          i32&                         value) {
    const usize pieceCount = popcount(position.pieces());
    if (pieceCount < 2 || pieceCount > 32)
        return "AliceNative-v2 integer evaluation requires between 2 and 32 pieces.";
    const Color sideToMove = position.side_to_move();
    std::array<u8, DenseInput> denseInput{};
    for (usize lane = 0; lane < TransformerWidth; ++lane)
    {
        denseInput[lane] = u8(std::clamp<i32>(accumulators[sideToMove].values[lane], 0,
                                               FeatureScale));
        denseInput[TransformerWidth + lane] =
          u8(std::clamp<i32>(accumulators[~sideToMove].values[lane], 0, FeatureScale));
    }
    const usize phase = (pieceCount - 1) / 4;
    const auto& dense = parameters.dense[phase];
    std::array<i32, Hidden1> fc0Raw{};
    std::array<u8, Hidden1>  fc0{};
    if (auto error = affine_fc0(dense, denseInput.data(), fc0Raw.data()))
        return error;
    for (usize output = 0; output < Hidden1; ++output)
        fc0[output] = u8(hidden_activation(fc0Raw[output]));
    std::array<i32, Hidden2> fc1Raw{};
    std::array<u8, Hidden2>  fc1{};
    if (auto error = affine(dense.fc1Weight, dense.fc1Bias, Hidden2, Hidden2, Hidden1,
                            fc0.data(), fc1Raw.data(), "AliceNative-v2 fc1"))
        return error;
    for (usize output = 0; output < Hidden2; ++output)
        fc1[output] = u8(hidden_activation(fc1Raw[output]));
    i32 positionalRaw = 0;
    if (auto error = affine(dense.fc2Weight, dense.fc2Bias, 1, Hidden2, Hidden2, fc1.data(),
                            &positionalRaw, "AliceNative-v2 fc2"))
        return error;
    const i64 materialDifference = i64(accumulators[sideToMove].psqt[phase])
                                 - i64(accumulators[~sideToMove].psqt[phase]);
    const i64 materialRaw = materialDifference / 2;
    i64 mixedRaw = positionalRaw + materialRaw;
    if (adjusted)
    {
        const int delta =
          std::abs(position.non_pawn_material(WHITE) - position.non_pawn_material(BLACK));
        const int entertainment = delta <= BishopValue - KnightValue ? 7 : 0;
        mixedRaw = (i64(128 - entertainment) * materialRaw
                    + i64(128 + entertainment) * positionalRaw)
                 / 128;
    }
    const i64 result = mixedRaw / EngineOutputDivisor;
    if (result < std::numeric_limits<i32>::min() || result > std::numeric_limits<i32>::max())
        return "AliceNative-v2 value exceeds signed i32.";
    value = i32(result);
    return std::nullopt;
}

}  // namespace Stockfish::Eval::NNUE::AliceNativeV2
