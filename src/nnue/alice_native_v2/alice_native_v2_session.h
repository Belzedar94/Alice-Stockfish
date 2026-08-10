/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef NNUE_ALICE_NATIVE_V2_SESSION_H_INCLUDED
#define NNUE_ALICE_NATIVE_V2_SESSION_H_INCLUDED

#include <array>
#include <string_view>

#include "../../alice_search.h"
#include "../../types.h"
#include "../alice_native/alice_native_features.h"
#include "alice_native_v2_inference.h"

namespace Stockfish {
class Position;
struct StateInfo;

namespace Eval::NNUE::AliceNativeV2 {

struct RuntimeSessionStats {
    u64 evaluations             = 0;
    u64 pushes                  = 0;
    u64 pops                    = 0;
    u64 fullRefreshes[COLOR_NB] = {};
    u64 pieceAdds               = 0;
    u64 pieceRemoves            = 0;
    u64 maxPieceEvents          = 0;
};

class SearchSession final: public AliceSearch::Evaluator {
   public:
    SearchSession(const ParameterView& parameters,
                  u64                  generation,
                  std::string_view     sha256,
                  const Position&      root) noexcept;

    SearchSession(const SearchSession&)            = delete;
    SearchSession(SearchSession&&)                 = delete;
    SearchSession& operator=(const SearchSession&) = delete;
    SearchSession& operator=(SearchSession&&)      = delete;

    AliceSearch::EvaluatorIdentity identity() const noexcept override;
    bool evaluate(const Position& position, Value& value,
                  AliceSearch::EvalFailure& failure) noexcept override;
    bool push(const Position& position, const Dirties& dirties,
              AliceSearch::EvalFailure& failure) noexcept override;
    bool pop(const Position& restoredParent, AliceSearch::EvalFailure& failure) noexcept override;

    bool ready() const noexcept;
    bool matches_current(const Position& position) const noexcept;
    usize ply() const noexcept;
    const IntegerAccumulatorSet& current_accumulators() const noexcept;
    const RuntimeSessionStats& stats() const noexcept;

   private:
    struct PositionIdentity {
        const StateInfo* state      = nullptr;
        Key              key        = 0;
        Bitboard         boardB     = 0;
        Color            sideToMove = WHITE;
        int              pieceCount = 0;
    };
    struct Frame {
        PositionIdentity               position;
        std::array<Square, COLOR_NB>   kingSquares{SQ_NONE, SQ_NONE};
        std::array<Board, COLOR_NB>    kingBoards{BOARD_A, BOARD_A};
        IntegerAccumulatorSet          accumulators;
    };

    bool initialize(const Position& root) noexcept;
    bool fail(AliceSearch::EvalFailure& failure, AliceSearch::EvalFailureCode code,
              AliceSearch::EvalStage stage, int perspective = -1) const noexcept;
    bool inference_fail(AliceSearch::EvalFailure& failure, std::string_view message,
                        AliceSearch::EvalStage defaultStage, int perspective = -1) const noexcept;
    static PositionIdentity capture(const Position& position) noexcept;
    static bool same_position(const PositionIdentity& expected, const Position& position) noexcept;
    static bool complete(const ParameterView& parameters) noexcept;

    ParameterView                  parameters;
    u64                            parameterGeneration = 0;
    std::string_view               parameterSha256;
    std::array<Frame, MAX_PLY + 1> frames{};
    usize                          currentPly  = 0;
    bool                           initialized = false;
    AliceSearch::EvalFailure       initializationFailure;
    RuntimeSessionStats            counters;
};

}  // namespace Eval::NNUE::AliceNativeV2
}  // namespace Stockfish

#endif  // NNUE_ALICE_NATIVE_V2_SESSION_H_INCLUDED
