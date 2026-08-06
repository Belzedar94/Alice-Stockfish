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

struct Result {
    Move            bestMove = Move::none();
    Value           score    = VALUE_ZERO;
    Depth           depth    = 0;
    u64             nodes    = 0;
    Search::PVMoves pv;
};

using IterationCallback = std::function<void(const Result&)>;

Result search(Position&                pos,
              const std::vector<Move>& rootMoves,
              const Limits&            limits,
              std::atomic_bool&        stop,
              const IterationCallback& onIteration = {});

}  // namespace AliceSearch
}  // namespace Stockfish

#endif  // ALICE_SEARCH_H_INCLUDED
