/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef NNUE_ALICE_NATIVE_FEATURES_H_INCLUDED
#define NNUE_ALICE_NATIVE_FEATURES_H_INCLUDED

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "../../types.h"
#include "../nnue_common.h"
#include "manifest.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::AliceNative {

enum class Relation : u8 {
    SAME,
    OTHER
};

constexpr Relation relation_of(Board subject, Board perspectiveKing) {
    return subject == perspectiveKing ? Relation::SAME : Relation::OTHER;
}

class PieceSquareFeatures {
   public:
    static constexpr u32       HashValue  = PieceSquareHash;
    static constexpr IndexType Dimensions = PieceSquareDimensions;

    // Bucket IDs after vertically orienting the perspective king.
    static constexpr std::array<u8, SQUARE_NB> KingBuckets = {
      28, 29, 30, 31, 31, 30, 29, 28, 24, 25, 26, 27, 27, 26, 25, 24, 20, 21, 22, 23, 23, 22,
      21, 20, 16, 17, 18, 19, 19, 18, 17, 16, 12, 13, 14, 15, 15, 14, 13, 12, 8,  9,  10, 11,
      11, 10, 9,  8,  4,  5,  6,  7,  7,  6,  5,  4,  0,  1,  2,  3,  3,  2,  1,  0,
    };

    // Mirror files a-d so that the perspective king is represented on e-h.
    static constexpr std::array<u8, SQUARE_NB> HorizontalMirror = {
      7, 7, 7, 7, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0,
      0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 7, 7, 7, 7,
      0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0,
    };

    static constexpr IndexType piece_plane(Color perspective, Piece piece) {
        return type_of(piece) == KING ? 10
                                      : 2 * (IndexType(type_of(piece)) - IndexType(PAWN))
                                          + IndexType(color_of(piece) != perspective);
    }

    static constexpr IndexType make_index(Color  perspective,
                                          Square pieceSquare,
                                          Piece  piece,
                                          Board  pieceBoard,
                                          Square kingSquare,
                                          Board  kingBoard) {
        const IndexType verticalFlip = 56 * IndexType(perspective);
        const IndexType orientedSquare =
          IndexType(pieceSquare) ^ HorizontalMirror[kingSquare] ^ verticalFlip;
        const IndexType bucket   = KingBuckets[IndexType(kingSquare) ^ verticalFlip];
        const IndexType relation = pieceBoard == kingBoard ? 0 : 1;

        return orientedSquare + PiecePlaneStride * piece_plane(perspective, piece)
             + RelationStride * relation + KingBucketStride * bucket;
    }
};

static_assert(PieceSquareFeatures::make_index(WHITE, SQ_E2, W_PAWN, BOARD_A, SQ_E1, BOARD_A)
              == 43660);
static_assert(PieceSquareFeatures::make_index(WHITE, SQ_E4, W_PAWN, BOARD_B, SQ_E1, BOARD_A)
              == 44380);

class ThreatFeatures {
   public:
    static constexpr u32       HashValue  = ThreatHash;
    static constexpr IndexType Dimensions = ThreatDimensions;

    static IndexType make_index(Color  perspective,
                                Piece  attacker,
                                Square from,
                                Square to,
                                Piece  attacked,
                                Board  edgeBoard,
                                Square kingSquare,
                                Board  kingBoard);
};

struct PieceFeatureTrace {
    IndexType index;
    Piece     piece;
    Square    square;
    Board     board;
    Relation  relation;
};

struct ThreatFeatureTrace {
    IndexType index;
    Piece     attacker;
    Square    from;
    Piece     attacked;
    Square    to;
    Board     board;
    Relation  relation;
};

enum class DeltaOperation : u8 {
    REMOVE,
    ADD
};

struct AlicePieceDelta {
    DeltaOperation operation;
    Piece          piece;
    Square         square;
    Board          board;
};

struct AliceThreatDelta {
    DeltaOperation operation;
    Piece          attacker;
    Square         from;
    Piece          attacked;
    Square         to;
    Board          board;
};

struct NativeFeatureDelta {
    std::vector<AlicePieceDelta>  pieces;
    std::vector<AliceThreatDelta> threats;
};

struct IncrementalVerificationStats {
    u64 positions               = 0;
    u64 transitions             = 0;
    u64 captures                = 0;
    u64 promotions              = 0;
    u64 castlings               = 0;
    u64 kingMoves               = 0;
    u64 fullRefreshes[COLOR_NB] = {};
    u64 maxPieceEvents          = 0;
    u64 maxThreatEvents         = 0;
    u64 cacheChecks             = 0;
    u64 cachePieceAdds          = 0;
    u64 cachePieceRemoves       = 0;
    u64 cacheBoardBEvents       = 0;
    u64 simdChecks              = 0;
    u64 fixedSnapshotChecks     = 0;
};

struct PerspectiveTrace {
    Color                           perspective;
    Square                          kingSquare;
    Board                           kingBoard;
    std::vector<PieceFeatureTrace>  pieces;
    std::vector<ThreatFeatureTrace> threats;
};

using PositionTrace = std::array<PerspectiveTrace, COLOR_NB>;

constexpr usize MaximumPieceFeatures  = 32;
constexpr usize MaximumThreatFeatures = 1024;

template<usize Capacity>
class FixedIndexList {
   public:
    bool push_back(IndexType index) noexcept {
        if (used == Capacity)
            return false;
        indices[used++] = index;
        return true;
    }

    void clear() noexcept { used = 0; }

    usize size() const noexcept { return used; }
    bool  empty() const noexcept { return used == 0; }

    IndexType&       operator[](usize index) noexcept { return indices[index]; }
    const IndexType& operator[](usize index) const noexcept { return indices[index]; }

    IndexType*       begin() noexcept { return indices.data(); }
    IndexType*       end() noexcept { return indices.data() + used; }
    const IndexType* begin() const noexcept { return indices.data(); }
    const IndexType* end() const noexcept { return indices.data() + used; }

   private:
    std::array<IndexType, Capacity> indices{};
    usize                           used = 0;
};

struct PerspectiveFeatureSnapshot {
    Color                                 perspective = WHITE;
    Square                                kingSquare  = SQ_NONE;
    Board                                 kingBoard   = BOARD_A;
    FixedIndexList<MaximumPieceFeatures>  pieces;
    FixedIndexList<MaximumThreatFeatures> threats;
};

using FeatureSnapshot = std::array<PerspectiveFeatureSnapshot, COLOR_NB>;

PositionTrace              build_trace(const Position& position);
std::optional<std::string> build_fixed_snapshot(const Position& position, FeatureSnapshot& result);
std::string                trace_json(const Position& position);
std::optional<std::string>
verify_incremental(Position& position, Depth depth, IncrementalVerificationStats& stats);

}  // namespace Stockfish::Eval::NNUE::AliceNative

#endif  // NNUE_ALICE_NATIVE_FEATURES_H_INCLUDED
