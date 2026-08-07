/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef NNUE_ALICE_NATIVE_INFERENCE_H_INCLUDED
#define NNUE_ALICE_NATIVE_INFERENCE_H_INCLUDED

#include <array>
#include <optional>
#include <string>

#include "../../types.h"
#include "alice_native_features.h"
#include "manifest.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::AliceNative {

struct DenseParameterView {
    const i32* fc0Bias   = nullptr;
    const i8*  fc0Weight = nullptr;
    const i32* fc1Bias   = nullptr;
    const i8*  fc1Weight = nullptr;
    const i32* fc2Bias   = nullptr;
    const i8*  fc2Weight = nullptr;
};

struct ParameterView {
    const i16* ftBias            = nullptr;
    const i8*  threatWeight      = nullptr;
    const i32* threatPsqt        = nullptr;
    const i16* pieceSquareWeight = nullptr;
    const i32* pieceSquarePsqt   = nullptr;

    std::array<DenseParameterView, LayerStacks> dense{};
};

struct IntegerAccumulator {
    std::array<i64, L1>          values{};
    std::array<i64, PsqtBuckets> psqt{};
};

using IntegerAccumulatorSet = std::array<IntegerAccumulator, COLOR_NB>;

struct AccumulatorDeltaStats {
    u64 pieceAdds     = 0;
    u64 pieceRemoves  = 0;
    u64 threatAdds    = 0;
    u64 threatRemoves = 0;
};

struct NativeIntegerStages {
    Color                                         sideToMove = WHITE;
    usize                                         pieceCount = 0;
    usize                                         phase      = 0;
    std::array<std::array<i32, L1 / 2>, COLOR_NB> transformed{};
    std::array<i32, L1>                           denseInput{};
    std::array<i32, L2>                           z0{};
    std::array<i32, L2>                           s0{};
    std::array<i32, L2>                           r0{};
    std::array<i32, L3>                           z1{};
    std::array<i32, L3>                           s1{};
    std::array<i32, L3>                           r1{};
    i32                                           z2            = 0;
    i32                                           skip          = 0;
    i32                                           fwdOut        = 0;
    i32                                           positionalRaw = 0;
    i32                                           psqtRaw       = 0;
    i32                                           positional    = 0;
    i32                                           psqt          = 0;
    i32                                           value         = 0;
};

std::optional<std::string> refresh_integer_accumulator(const ParameterView&    parameters,
                                                       const PerspectiveTrace& trace,
                                                       IntegerAccumulator&     accumulator);

std::optional<std::string> refresh_integer_accumulator(const ParameterView&              parameters,
                                                       const PerspectiveFeatureSnapshot& snapshot,
                                                       IntegerAccumulator& accumulator);

std::optional<std::string> update_integer_accumulator(const ParameterView&    parameters,
                                                      const PerspectiveTrace& before,
                                                      const PerspectiveTrace& after,
                                                      IntegerAccumulator&     accumulator,
                                                      AccumulatorDeltaStats&  stats);

std::optional<std::string> update_integer_accumulator(const ParameterView&              parameters,
                                                      const PerspectiveFeatureSnapshot& before,
                                                      const PerspectiveFeatureSnapshot& after,
                                                      IntegerAccumulator&               accumulator,
                                                      AccumulatorDeltaStats&            stats);

std::optional<std::string> evaluate_integer(const ParameterView&         parameters,
                                            const Position&              position,
                                            const IntegerAccumulatorSet& accumulators,
                                            NativeIntegerStages&         stages);

bool same_integer_stages(const NativeIntegerStages& left, const NativeIntegerStages& right);

}  // namespace Stockfish::Eval::NNUE::AliceNative

#endif  // NNUE_ALICE_NATIVE_INFERENCE_H_INCLUDED
