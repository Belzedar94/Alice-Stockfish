/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef NNUE_ALICE_NATIVE_V2_INFERENCE_H_INCLUDED
#define NNUE_ALICE_NATIVE_V2_INFERENCE_H_INCLUDED

#include <array>
#include <optional>
#include <string>

#include "../../types.h"
#include "../alice_native/alice_native_features.h"
#include "manifest.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::AliceNativeV2 {

struct DenseParameterView {
    const i32* fc0Bias   = nullptr;
    const i8*  fc0Weight = nullptr;
    const i8*  fc0InterleavedWeight = nullptr;
    const i32* fc1Bias   = nullptr;
    const i8*  fc1Weight = nullptr;
    const i32* fc2Bias   = nullptr;
    const i8*  fc2Weight = nullptr;
};

struct ParameterView {
    const i16* ftBias            = nullptr;
    const i16* pieceSquareWeight = nullptr;
    const i32* pieceSquarePsqt   = nullptr;
    std::array<DenseParameterView, LayerStacks> dense{};
};

struct IntegerAccumulator {
    std::array<i16, TransformerWidth> values{};
    std::array<i32, PsqtBuckets>      psqt{};
};

using IntegerAccumulatorSet = std::array<IntegerAccumulator, COLOR_NB>;

struct AccumulatorDeltaStats {
    u64 pieceAdds    = 0;
    u64 pieceRemoves = 0;
};

struct IntegerStages {
    Color                                                    sideToMove = WHITE;
    usize                                                    pieceCount = 0;
    usize                                                    phase      = 0;
    std::array<std::array<i32, TransformerWidth>, COLOR_NB> transformed{};
    std::array<i32, DenseInput>                             denseInput{};
    std::array<i32, Hidden1>                                fc0Raw{};
    std::array<i32, Hidden1>                                fc0{};
    std::array<i32, Hidden2>                                fc1Raw{};
    std::array<i32, Hidden2>                                fc1{};
    i32                                                      fc2Raw       = 0;
    i32                                                      materialRaw  = 0;
    i32                                                      combinedRaw  = 0;
    i32                                                      entertainment = 0;
    i32                                                      adjustedRaw  = 0;
    i32                                                      rawValue      = 0;
    i32                                                      adjustedValue = 0;
};

std::optional<std::string>
refresh_accumulator(const ParameterView&                         parameters,
                    const AliceNative::PerspectivePieceSnapshot& snapshot,
                    IntegerAccumulator&                           accumulator);

std::optional<std::string>
update_accumulator(const ParameterView&                         parameters,
                   const AliceNative::PerspectivePieceSnapshot& before,
                   const AliceNative::PerspectivePieceSnapshot& after,
                   IntegerAccumulator&                           accumulator,
                   AccumulatorDeltaStats&                        stats);

using PieceEventIndices = std::array<IndexType, 2>;

std::optional<std::string>
update_accumulator_events(const ParameterView&    parameters,
                          const PieceEventIndices& removals,
                          usize                    removalCount,
                          const PieceEventIndices& additions,
                          usize                    additionCount,
                          IntegerAccumulator&      accumulator,
                          AccumulatorDeltaStats&   stats);

std::optional<std::string> evaluate(const ParameterView&         parameters,
                                    const Position&              position,
                                    const IntegerAccumulatorSet& accumulators,
                                    IntegerStages&               stages);

// Compact production path. It deliberately avoids materializing the large
// diagnostic stage record used by cross-language qualification.
std::optional<std::string> evaluate_value(const ParameterView&         parameters,
                                          const Position&              position,
                                          const IntegerAccumulatorSet& accumulators,
                                          bool                         adjusted,
                                          i32&                         value);

bool same_stages(const IntegerStages& left, const IntegerStages& right);

}  // namespace Stockfish::Eval::NNUE::AliceNativeV2

#endif  // NNUE_ALICE_NATIVE_V2_INFERENCE_H_INCLUDED
