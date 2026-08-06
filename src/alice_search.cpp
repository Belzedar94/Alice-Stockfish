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

namespace {

class Searcher {
   public:
    Searcher(Position&                position,
             const std::vector<Move>& allowedRootMoves,
             const Limits&            searchLimits,
             const Evaluator&         staticEvaluator,
             std::atomic_bool&        stopFlag) :
        pos(position),
        rootMoves(allowedRootMoves),
        limits(searchLimits),
        evaluator(staticEvaluator),
        stop(stopFlag) {}

    Result iterative_deepening(const IterationCallback& onIteration) {
        Result completed;

        if (rootMoves.empty())
        {
            completed.score = pos.checkers() ? mated_in(0) : VALUE_DRAW;
            return completed;
        }

        // A stopped depth-one search must still return a legal move.
        completed.bestMove = rootMoves.front();
        completed.pv.push_back(completed.bestMove);

        for (Depth depth = 1; depth <= limits.depth && !should_stop(); ++depth)
        {
            aborted          = false;
            Result iteration = search_root(depth);
            if (aborted)
                break;

            completed       = iteration;
            completed.depth = depth;
            if (onIteration)
                onIteration(completed);
        }

        completed.nodes = nodes;
        return completed;
    }

   private:
    bool should_stop() {
        if (stop.load(std::memory_order_relaxed))
            return true;
        if (limits.nodes && nodes >= limits.nodes)
            return true;
        if (limits.deadline && now() >= limits.deadline)
            return true;
        return false;
    }

    Value negamax(Depth depth, int ply, Value alpha, Value beta, Search::PVMoves& pv) {
        pv.clear();
        if (should_stop())
        {
            aborted = true;
            return VALUE_DRAW;
        }
        ++nodes;

        if (pos.is_draw(ply))
            return VALUE_DRAW;

        MoveList<LEGAL> moves(pos);
        if (moves.size() == 0)
            return pos.checkers() ? mated_in(ply) : VALUE_DRAW;
        if (depth == 0)
            return evaluator.value ? evaluator.value(pos) : VALUE_ZERO;

        Value best = -VALUE_INFINITE;
        for (Move move : moves)
        {
            StateInfo       state;
            Dirties         dirties;
            Search::PVMoves childPv;
            pos.do_move(move, state, pos.gives_check(move), dirties, nullptr, nullptr);
            if (evaluator.push)
                evaluator.push(pos, dirties);
            const Value score = -negamax(depth - 1, ply + 1, -beta, -alpha, childPv);
            if (evaluator.pop)
                evaluator.pop();
            pos.undo_move(move);

            if (aborted)
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
            {
                aborted = true;
                break;
            }

            StateInfo       state;
            Dirties         dirties;
            Search::PVMoves childPv;
            pos.do_move(move, state, pos.gives_check(move), dirties, nullptr, nullptr);
            if (evaluator.push)
                evaluator.push(pos, dirties);
            // Every root move receives an exact score. The temporary safe
            // search deliberately gives up aspiration and PVS shortcuts until
            // they have Alice-specific validation.
            const Value score = -negamax(depth - 1, 1, -VALUE_INFINITE, VALUE_INFINITE, childPv);
            if (evaluator.pop)
                evaluator.pop();
            pos.undo_move(move);

            if (aborted)
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
    const Evaluator&         evaluator;
    std::atomic_bool&        stop;
    u64                      nodes   = 0;
    bool                     aborted = false;
};

}  // namespace

Result search(Position&                pos,
              const std::vector<Move>& rootMoves,
              const Limits&            limits,
              const Evaluator&         evaluator,
              std::atomic_bool&        stop,
              const IterationCallback& onIteration) {
    Searcher searcher(pos, rootMoves, limits, evaluator, stop);
    return searcher.iterative_deepening(onIteration);
}

}  // namespace Stockfish::AliceSearch
