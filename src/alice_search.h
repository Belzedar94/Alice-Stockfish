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

#ifndef ALICE_SEARCH_H_INCLUDED
#define ALICE_SEARCH_H_INCLUDED

#include <atomic>
#include <functional>
#include <string_view>
#include <vector>

#include "misc.h"
#include "search.h"
#include "types.h"

namespace Stockfish {

class Position;

namespace AliceSearch {

struct Limits {
    Depth     depth    = 1;
    u64       nodes    = 0;
    TimePoint deadline = 0;
};

enum class EvalFailureCode : u8 {
    NONE,
    NOT_READY,
    UNSUPPORTED_POSITION,
    STACK_OVERFLOW,
    STACK_UNDERFLOW,
    POSITION_MISMATCH,
    PARAMETER_IDENTITY_MISMATCH,
    FEATURE_CAPACITY_EXCEEDED,
    FEATURE_INDEX_OUT_OF_RANGE,
    ACCUMULATOR_OUT_OF_RANGE,
    PSQT_OUT_OF_RANGE,
    DENSE_ARITHMETIC_OUT_OF_RANGE,
    STATIC_VALUE_OUT_OF_RANGE,
    INTERNAL_INVARIANT
};

enum class EvalStage : u8 {
    NONE,
    ROOT_REFRESH,
    FEATURE_EXTRACTION,
    PIECE_DELTA,
    THREAT_DELTA,
    FEATURE_TRANSFORM,
    FC0,
    FC1,
    FC2,
    OUTPUT_SCALING,
    EVALUATE,
    PUSH,
    POP
};

struct EvalFailure {
    EvalFailureCode code        = EvalFailureCode::NONE;
    EvalStage       stage       = EvalStage::NONE;
    u64             generation  = 0;
    int             ply         = 0;
    int             perspective = -1;
    u32             index       = 0;
    i64             observed    = 0;
    i64             minimum     = 0;
    i64             maximum     = 0;

    constexpr explicit operator bool() const noexcept { return code != EvalFailureCode::NONE; }
};

struct EvaluatorIdentity {
    std::string_view backend;
    u64              generation = 0;
    std::string_view sha256;
};

class Evaluator {
   public:
    virtual ~Evaluator() = default;

    virtual EvaluatorIdentity identity() const noexcept                                    = 0;
    virtual bool              evaluate(const Position&, Value&, EvalFailure&) noexcept     = 0;
    virtual bool              push(const Position&, const Dirties&, EvalFailure&) noexcept = 0;
    virtual bool              pop(const Position& restoredParent, EvalFailure&) noexcept   = 0;
};

enum class Completion : u8 {
    COMPLETED,
    STOPPED,
    FAILED
};

enum class Terminal : u8 {
    NONE,
    CHECKMATE,
    STALEMATE,
    RULE_DRAW
};

std::string_view failure_code_name(EvalFailureCode code) noexcept;
std::string_view failure_stage_name(EvalStage stage) noexcept;

struct Result {
    Move            bestMove = Move::none();
    Value           score    = VALUE_ZERO;
    Depth           depth    = 0;
    u64             nodes    = 0;
    Search::PVMoves pv;
    Completion      completion = Completion::COMPLETED;
    Terminal        terminal   = Terminal::NONE;
    EvalFailure     failure;
    bool            rootRestored = true;
};

using IterationCallback = std::function<void(const Result&)>;

Result search(Position&                pos,
              const std::vector<Move>& rootMoves,
              const Limits&            limits,
              Evaluator&               evaluator,
              std::atomic_bool&        stop,
              const IterationCallback& onIteration = {});

}  // namespace AliceSearch
}  // namespace Stockfish

#endif  // ALICE_SEARCH_H_INCLUDED
