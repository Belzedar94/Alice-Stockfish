/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "alice_native_inference.h"

#include <algorithm>
#include <limits>
#include <string_view>

#include "../../bitboard.h"
#include "../../position.h"

namespace Stockfish::Eval::NNUE::AliceNative {

namespace {

std::optional<std::string> validate_accumulator(const IntegerAccumulator& accumulator,
                                                Color                     perspective) {
    for (usize lane = 0; lane < L1; ++lane)
        if (accumulator.values[lane] < std::numeric_limits<i16>::min()
            || accumulator.values[lane] > std::numeric_limits<i16>::max())
            return "Alice native feature accumulator exceeds signed i16 at perspective "
                 + std::to_string(int(perspective)) + " lane " + std::to_string(lane) + ": "
                 + std::to_string(accumulator.values[lane]);

    for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
        if (accumulator.psqt[bucket] < std::numeric_limits<i32>::min()
            || accumulator.psqt[bucket] > std::numeric_limits<i32>::max())
            return "Alice native PSQT accumulator exceeds signed i32 at perspective "
                 + std::to_string(int(perspective)) + " bucket " + std::to_string(bucket) + ": "
                 + std::to_string(accumulator.psqt[bucket]);

    return std::nullopt;
}

IndexType feature_index(IndexType index) { return index; }

template<typename Feature>
IndexType feature_index(const Feature& feature) {
    return feature.index;
}

template<typename FeatureRange, typename Update>
void apply_feature_delta(
  const FeatureRange& before, const FeatureRange& after, Update&& update, u64& adds, u64& removes) {
    usize beforeIndex = 0;
    usize afterIndex  = 0;
    while (beforeIndex < before.size() || afterIndex < after.size())
    {
        if (afterIndex == after.size()
            || (beforeIndex < before.size()
                && feature_index(before[beforeIndex]) < feature_index(after[afterIndex])))
        {
            update(feature_index(before[beforeIndex++]), -1);
            ++removes;
        }
        else if (beforeIndex == before.size()
                 || feature_index(after[afterIndex]) < feature_index(before[beforeIndex]))
        {
            update(feature_index(after[afterIndex++]), 1);
            ++adds;
        }
        else
        {
            ++beforeIndex;
            ++afterIndex;
        }
    }
}

std::optional<std::string> affine(const i8*        weights,
                                  const i32*       biases,
                                  usize            outputs,
                                  usize            inputs,
                                  const i32*       values,
                                  i32*             result,
                                  std::string_view label,
                                  bool             requireI16) {
    for (usize output = 0; output < outputs; ++output)
    {
        i64 total = biases[output];
        for (usize input = 0; input < inputs; ++input)
            total += i64(weights[output * inputs + input]) * values[input];
        if (total < std::numeric_limits<i32>::min() || total > std::numeric_limits<i32>::max())
            return std::string(label) + " exceeds signed i32 at row " + std::to_string(output)
                 + ": " + std::to_string(total);
        if (requireI16
            && (total < std::numeric_limits<i16>::min() || total > std::numeric_limits<i16>::max()))
            return std::string(label) + " exceeds signed i16 at row " + std::to_string(output)
                 + ": " + std::to_string(total);
        result[output] = i32(total);
    }
    return std::nullopt;
}

i32 activate(i32 value, int shift, bool square) {
    const i64 raw =
      square ? i64(value) * value / (i64(1) << (2 * shift + 7)) : value / (i64(1) << shift);
    return i32(std::clamp<i64>(raw, 0, 127));
}

template<typename PieceRange, typename ThreatRange>
std::optional<std::string> refresh_integer_accumulator_impl(const ParameterView& parameters,
                                                            Color                perspective,
                                                            const PieceRange&    pieces,
                                                            const ThreatRange&   threats,
                                                            IntegerAccumulator&  accumulator) {
    accumulator = {};
    for (usize lane = 0; lane < L1; ++lane)
        accumulator.values[lane] = parameters.ftBias[lane];

    for (const auto& feature : pieces)
    {
        const IndexType index = feature_index(feature);
        if (index >= PieceSquareDimensions)
            return "Alice native piece feature index is outside the loaded tensor.";
        const u64 row = u64(index) * L1;
        for (usize lane = 0; lane < L1; ++lane)
            accumulator.values[lane] += parameters.pieceSquareWeight[row + lane];
        const u64 psqtRow = u64(index) * PsqtBuckets;
        for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
            accumulator.psqt[bucket] += parameters.pieceSquarePsqt[psqtRow + bucket];
    }

    for (const auto& feature : threats)
    {
        const IndexType index = feature_index(feature);
        if (index >= ThreatDimensions)
            return "Alice native threat feature index is outside the loaded tensor.";
        const u64 row = u64(index) * L1;
        for (usize lane = 0; lane < L1; ++lane)
            accumulator.values[lane] += parameters.threatWeight[row + lane];
        const u64 psqtRow = u64(index) * PsqtBuckets;
        for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
            accumulator.psqt[bucket] += parameters.threatPsqt[psqtRow + bucket];
    }

    return validate_accumulator(accumulator, perspective);
}

template<typename PieceRange, typename ThreatRange>
std::optional<std::string> update_integer_accumulator_impl(const ParameterView&   parameters,
                                                           Color                  perspective,
                                                           const PieceRange&      beforePieces,
                                                           const PieceRange&      afterPieces,
                                                           const ThreatRange&     beforeThreats,
                                                           const ThreatRange&     afterThreats,
                                                           IntegerAccumulator&    accumulator,
                                                           AccumulatorDeltaStats& stats) {
    stats = {};
    apply_feature_delta(
      beforePieces, afterPieces,
      [&](IndexType index, i32 sign) {
          const u64 row = u64(index) * L1;
          for (usize lane = 0; lane < L1; ++lane)
              accumulator.values[lane] += sign * i64(parameters.pieceSquareWeight[row + lane]);
          const u64 psqtRow = u64(index) * PsqtBuckets;
          for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
              accumulator.psqt[bucket] += sign * i64(parameters.pieceSquarePsqt[psqtRow + bucket]);
      },
      stats.pieceAdds, stats.pieceRemoves);

    apply_feature_delta(
      beforeThreats, afterThreats,
      [&](IndexType index, i32 sign) {
          const u64 row = u64(index) * L1;
          for (usize lane = 0; lane < L1; ++lane)
              accumulator.values[lane] += sign * i64(parameters.threatWeight[row + lane]);
          const u64 psqtRow = u64(index) * PsqtBuckets;
          for (usize bucket = 0; bucket < PsqtBuckets; ++bucket)
              accumulator.psqt[bucket] += sign * i64(parameters.threatPsqt[psqtRow + bucket]);
      },
      stats.threatAdds, stats.threatRemoves);

    return validate_accumulator(accumulator, perspective);
}

}  // namespace

std::optional<std::string> refresh_integer_accumulator(const ParameterView&    parameters,
                                                       const PerspectiveTrace& trace,
                                                       IntegerAccumulator&     accumulator) {
    return refresh_integer_accumulator_impl(parameters, trace.perspective, trace.pieces,
                                            trace.threats, accumulator);
}

std::optional<std::string> refresh_integer_accumulator(const ParameterView&              parameters,
                                                       const PerspectiveFeatureSnapshot& snapshot,
                                                       IntegerAccumulator& accumulator) {
    return refresh_integer_accumulator_impl(parameters, snapshot.perspective, snapshot.pieces,
                                            snapshot.threats, accumulator);
}

std::optional<std::string> update_integer_accumulator(const ParameterView&    parameters,
                                                      const PerspectiveTrace& before,
                                                      const PerspectiveTrace& after,
                                                      IntegerAccumulator&     accumulator,
                                                      AccumulatorDeltaStats&  stats) {
    return update_integer_accumulator_impl(parameters, after.perspective, before.pieces,
                                           after.pieces, before.threats, after.threats, accumulator,
                                           stats);
}

std::optional<std::string> update_integer_accumulator(const ParameterView&              parameters,
                                                      const PerspectiveFeatureSnapshot& before,
                                                      const PerspectiveFeatureSnapshot& after,
                                                      IntegerAccumulator&               accumulator,
                                                      AccumulatorDeltaStats&            stats) {
    return update_integer_accumulator_impl(parameters, after.perspective, before.pieces,
                                           after.pieces, before.threats, after.threats, accumulator,
                                           stats);
}

std::optional<std::string> evaluate_integer(const ParameterView&         parameters,
                                            const Position&              position,
                                            const IntegerAccumulatorSet& accumulators,
                                            NativeIntegerStages&         stages) {
    stages            = {};
    stages.pieceCount = popcount(position.pieces());
    if (stages.pieceCount < 2 || stages.pieceCount > 32)
        return "Alice native integer evaluation requires between 2 and 32 pieces.";

    for (Color perspective : {WHITE, BLACK})
    {
        if (auto error = validate_accumulator(accumulators[perspective], perspective))
            return error;
        for (usize lane = 0; lane < L1 / 2; ++lane)
        {
            const i32 left = std::clamp<i32>(i32(accumulators[perspective].values[lane]), 0, 255);
            const i32 right =
              std::clamp<i32>(i32(accumulators[perspective].values[lane + L1 / 2]), 0, 255);
            stages.transformed[perspective][lane] = left * right / 512;
        }
    }

    stages.sideToMove = position.side_to_move();
    for (usize lane = 0; lane < L1 / 2; ++lane)
    {
        stages.denseInput[lane]          = stages.transformed[stages.sideToMove][lane];
        stages.denseInput[lane + L1 / 2] = stages.transformed[~stages.sideToMove][lane];
    }

    stages.phase      = (stages.pieceCount - 1) / 4;
    const auto& dense = parameters.dense[stages.phase];
    if (auto error = affine(dense.fc0Weight, dense.fc0Bias, L2, L1, stages.denseInput.data(),
                            stages.z0.data(), "fc0", true))
        return error;
    std::array<i32, 64> y1{};
    for (usize output = 0; output < L2; ++output)
    {
        stages.s0[output] = activate(stages.z0[output], 7, true);
        stages.r0[output] = activate(stages.z0[output], 7, false);
        y1[output]        = stages.s0[output];
        y1[L2 + output]   = stages.r0[output];
    }

    if (auto error = affine(dense.fc1Weight, dense.fc1Bias, L3, y1.size(), y1.data(),
                            stages.z1.data(), "fc1", true))
        return error;
    std::array<i32, 128> y2{};
    for (usize output = 0; output < L3; ++output)
    {
        stages.s1[output]        = activate(stages.z1[output], 6, true);
        stages.r1[output]        = activate(stages.z1[output], 6, false);
        y2[output]               = stages.s0[output];
        y2[L2 + output]          = stages.r0[output];
        y2[2 * L2 + output]      = stages.s1[output];
        y2[2 * L2 + L3 + output] = stages.r1[output];
    }

    if (auto error =
          affine(dense.fc2Weight, dense.fc2Bias, 1, y2.size(), y2.data(), &stages.z2, "fc2", false))
        return error;
    const i64 skip64   = i64(stages.z0[30]) - stages.z0[31];
    const i64 fwdOut64 = i64(stages.z2) + skip64;
    if (fwdOut64 < std::numeric_limits<i32>::min() || fwdOut64 > std::numeric_limits<i32>::max())
        return "fwdOut exceeds signed i32: " + std::to_string(fwdOut64);
    stages.skip   = i32(skip64);
    stages.fwdOut = i32(fwdOut64);

    const i64 positionalRaw64 = fwdOut64 * 9600 / 16384;
    const i64 psqtDifference  = accumulators[stages.sideToMove].psqt[stages.phase]
                             - accumulators[~stages.sideToMove].psqt[stages.phase];
    const i64 psqtRaw64 = psqtDifference / 2;
    if (positionalRaw64 < std::numeric_limits<i32>::min()
        || positionalRaw64 > std::numeric_limits<i32>::max())
        return "positionalRaw16 exceeds signed i32: " + std::to_string(positionalRaw64);
    if (psqtRaw64 < std::numeric_limits<i32>::min() || psqtRaw64 > std::numeric_limits<i32>::max())
        return "psqtRaw16 exceeds signed i32: " + std::to_string(psqtRaw64);
    stages.positionalRaw = i32(positionalRaw64);
    stages.psqtRaw       = i32(psqtRaw64);
    stages.positional    = stages.positionalRaw / 16;
    stages.psqt          = stages.psqtRaw / 16;
    const i64 value64    = i64(stages.positional) + stages.psqt;
    if (value64 < std::numeric_limits<i32>::min() || value64 > std::numeric_limits<i32>::max())
        return "native value exceeds signed i32: " + std::to_string(value64);
    stages.value = i32(value64);
    return std::nullopt;
}

bool same_integer_stages(const NativeIntegerStages& left, const NativeIntegerStages& right) {
    return left.sideToMove == right.sideToMove && left.pieceCount == right.pieceCount
        && left.phase == right.phase && left.transformed == right.transformed
        && left.denseInput == right.denseInput && left.z0 == right.z0 && left.s0 == right.s0
        && left.r0 == right.r0 && left.z1 == right.z1 && left.s1 == right.s1 && left.r1 == right.r1
        && left.z2 == right.z2 && left.skip == right.skip && left.fwdOut == right.fwdOut
        && left.positionalRaw == right.positionalRaw && left.psqtRaw == right.psqtRaw
        && left.positional == right.positional && left.psqt == right.psqt
        && left.value == right.value;
}

}  // namespace Stockfish::Eval::NNUE::AliceNative
