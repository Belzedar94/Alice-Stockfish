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

#include "alice_search.h"

#include <algorithm>

#include "movegen.h"
#include "position.h"

namespace Stockfish::AliceSearch {

std::string_view failure_code_name(EvalFailureCode code) noexcept {
    switch (code)
    {
    case EvalFailureCode::NONE :
        return "none";
    case EvalFailureCode::NOT_READY :
        return "not-ready";
    case EvalFailureCode::UNSUPPORTED_POSITION :
        return "unsupported-position";
    case EvalFailureCode::STACK_OVERFLOW :
        return "stack-overflow";
    case EvalFailureCode::STACK_UNDERFLOW :
        return "stack-underflow";
    case EvalFailureCode::POSITION_MISMATCH :
        return "position-mismatch";
    case EvalFailureCode::PARAMETER_IDENTITY_MISMATCH :
        return "parameter-identity-mismatch";
    case EvalFailureCode::FEATURE_CAPACITY_EXCEEDED :
        return "feature-capacity-exceeded";
    case EvalFailureCode::FEATURE_INDEX_OUT_OF_RANGE :
        return "feature-index-out-of-range";
    case EvalFailureCode::ACCUMULATOR_OUT_OF_RANGE :
        return "accumulator-out-of-range";
    case EvalFailureCode::PSQT_OUT_OF_RANGE :
        return "psqt-out-of-range";
    case EvalFailureCode::DENSE_ARITHMETIC_OUT_OF_RANGE :
        return "dense-arithmetic-out-of-range";
    case EvalFailureCode::STATIC_VALUE_OUT_OF_RANGE :
        return "static-value-out-of-range";
    case EvalFailureCode::INTERNAL_INVARIANT :
        return "internal-invariant";
    }
    return "unknown";
}

std::string_view failure_stage_name(EvalStage stage) noexcept {
    switch (stage)
    {
    case EvalStage::NONE :
        return "none";
    case EvalStage::ROOT_REFRESH :
        return "root-refresh";
    case EvalStage::FEATURE_EXTRACTION :
        return "feature-extraction";
    case EvalStage::PIECE_DELTA :
        return "piece-delta";
    case EvalStage::THREAT_DELTA :
        return "threat-delta";
    case EvalStage::FEATURE_TRANSFORM :
        return "feature-transform";
    case EvalStage::FC0 :
        return "fc0";
    case EvalStage::FC1 :
        return "fc1";
    case EvalStage::FC2 :
        return "fc2";
    case EvalStage::OUTPUT_SCALING :
        return "output-scaling";
    case EvalStage::EVALUATE :
        return "evaluate";
    case EvalStage::PUSH :
        return "push";
    case EvalStage::POP :
        return "pop";
    }
    return "unknown";
}

namespace {

class Searcher {
   public:
    Searcher(Position&                position,
             const std::vector<Move>& allowedRootMoves,
             const Limits&            searchLimits,
             Evaluator&               staticEvaluator,
             std::atomic_bool&        stopFlag) :
        pos(position),
        rootMoves(allowedRootMoves),
        limits(searchLimits),
        evaluator(staticEvaluator),
        stop(stopFlag),
        rootState(position.state()),
        rootKey(position.key()),
        rootBoardB(position.state()->boardB),
        rootSideToMove(position.side_to_move()),
        rootPieceCount(position.count<ALL_PIECES>()) {}

    Result iterative_deepening(const IterationCallback& onIteration) {
        Result completed;

        if (rootMoves.empty())
        {
            completed.score    = pos.checkers() ? mated_in(0) : VALUE_DRAW;
            completed.terminal = pos.checkers() ? Terminal::CHECKMATE : Terminal::STALEMATE;
            finalize(completed);
            return completed;
        }

        if (pos.is_draw(0))
        {
            completed.score    = VALUE_DRAW;
            completed.terminal = Terminal::RULE_DRAW;
            finalize(completed);
            return completed;
        }

        // A stopped depth-one search must still return a legal move.
        completed.bestMove = rootMoves.front();
        completed.pv.push_back(completed.bestMove);

        for (Depth depth = 1; depth <= limits.depth && !should_stop(); ++depth)
        {
            Result iteration = search_root(depth);
            if (failed || stopped)
                break;

            completed       = iteration;
            completed.depth = depth;
            if (onIteration)
                onIteration(completed);
        }

        finalize(completed);
        return completed;
    }

   private:
    bool should_stop() {
        if (failed)
            return true;
        if (stop.load(std::memory_order_relaxed))
        {
            stopped = true;
            return true;
        }
        if (limits.nodes && nodes >= limits.nodes)
        {
            stopped = true;
            return true;
        }
        if (limits.deadline && now() >= limits.deadline)
        {
            stopped = true;
            return true;
        }
        return false;
    }

    void record_failure(EvalFailure error, EvalStage defaultStage, int ply) {
        if (failed)
            return;
        if (!error)
            error.code = EvalFailureCode::INTERNAL_INVARIANT;
        if (error.stage == EvalStage::NONE)
            error.stage = defaultStage;
        error.ply = ply;
        failure   = error;
        failed    = true;
    }

    bool root_is_restored() const {
        return pos.state() == rootState && pos.key() == rootKey && pos.state()->boardB == rootBoardB
            && pos.side_to_move() == rootSideToMove && pos.count<ALL_PIECES>() == rootPieceCount;
    }

    void finalize(Result& result) {
        result.nodes        = nodes;
        result.rootRestored = root_is_restored();
        if (!result.rootRestored)
        {
            EvalFailure error;
            error.code = EvalFailureCode::INTERNAL_INVARIANT;
            record_failure(error, EvalStage::POP, 0);
        }

        if (failed)
        {
            result.completion = Completion::FAILED;
            result.failure    = failure;
        }
        else if (stopped)
            result.completion = Completion::STOPPED;
    }

    Value negamax(Depth depth, int ply, Value alpha, Value beta, Search::PVMoves& pv) {
        pv.clear();
        if (should_stop())
            return VALUE_DRAW;
        ++nodes;

        if (pos.is_draw(ply))
            return VALUE_DRAW;

        MoveList<LEGAL> moves(pos);
        if (moves.size() == 0)
            return pos.checkers() ? mated_in(ply) : VALUE_DRAW;
        if (depth == 0)
        {
            Value       value = VALUE_ZERO;
            EvalFailure error;
            if (!evaluator.evaluate(pos, value, error))
            {
                record_failure(error, EvalStage::EVALUATE, ply);
                return VALUE_DRAW;
            }
            return value;
        }

        Value best = -VALUE_INFINITE;
        for (Move move : moves)
        {
            StateInfo       state;
            Dirties         dirties;
            Search::PVMoves childPv;
            pos.do_move(move, state, pos.gives_check(move), dirties, nullptr, nullptr);
            EvalFailure pushError;
            if (!evaluator.push(pos, dirties, pushError))
            {
                record_failure(pushError, EvalStage::PUSH, ply + 1);
                pos.undo_move(move);
                return VALUE_DRAW;
            }
            const Value score = -negamax(depth - 1, ply + 1, -beta, -alpha, childPv);
            pos.undo_move(move);
            EvalFailure popError;
            if (!evaluator.pop(pos, popError))
                record_failure(popError, EvalStage::POP, ply);

            if (failed || stopped)
                return VALUE_DRAW;
            if (score > best)
            {
                best = score;
                pv.update(move, &childPv);
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta)
                break;
        }

        return best;
    }

    Result search_root(Depth depth) {
        Result result;
        result.depth = depth;
        result.score = -VALUE_INFINITE;

        for (Move move : rootMoves)
        {
            if (should_stop())
                break;

            StateInfo       state;
            Dirties         dirties;
            Search::PVMoves childPv;
            pos.do_move(move, state, pos.gives_check(move), dirties, nullptr, nullptr);
            EvalFailure pushError;
            if (!evaluator.push(pos, dirties, pushError))
            {
                record_failure(pushError, EvalStage::PUSH, 1);
                pos.undo_move(move);
                break;
            }
            // Every root move receives an exact score. The temporary safe
            // search deliberately gives up aspiration and PVS shortcuts until
            // they have Alice-specific validation.
            const Value score = -negamax(depth - 1, 1, -VALUE_INFINITE, VALUE_INFINITE, childPv);
            pos.undo_move(move);
            EvalFailure popError;
            if (!evaluator.pop(pos, popError))
                record_failure(popError, EvalStage::POP, 0);

            if (failed || stopped)
                break;
            if (score > result.score)
            {
                result.score    = score;
                result.bestMove = move;
                result.pv.update(move, &childPv);
            }
        }

        result.nodes = nodes;
        return result;
    }

    Position&                pos;
    const std::vector<Move>& rootMoves;
    const Limits&            limits;
    Evaluator&               evaluator;
    std::atomic_bool&        stop;
    const StateInfo*         rootState;
    Key                      rootKey;
    Bitboard                 rootBoardB;
    Color                    rootSideToMove;
    int                      rootPieceCount;
    u64                      nodes   = 0;
    bool                     stopped = false;
    bool                     failed  = false;
    EvalFailure              failure;
};

}  // namespace

Result search(Position&                pos,
              const std::vector<Move>& rootMoves,
              const Limits&            limits,
              Evaluator&               evaluator,
              std::atomic_bool&        stop,
              const IterationCallback& onIteration) {
    Searcher searcher(pos, rootMoves, limits, evaluator, stop);
    return searcher.iterative_deepening(onIteration);
}

}  // namespace Stockfish::AliceSearch
