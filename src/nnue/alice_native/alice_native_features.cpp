/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "alice_native_features.h"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

#include "../../attacks.h"
#include "../../bitboard.h"
#include "../../movegen.h"
#include "../../position.h"
#include "../features/full_threats.h"
#include "../simd.h"

namespace Stockfish::Eval::NNUE::AliceNative {

IndexType ThreatFeatures::make_index(Color  perspective,
                                     Piece  attacker,
                                     Square from,
                                     Square to,
                                     Piece  attacked,
                                     Board  edgeBoard,
                                     Square kingSquare,
                                     Board  kingBoard) {
    const IndexType base =
      Features::FullThreats::make_index(perspective, attacker, from, to, attacked, kingSquare);
    if (base >= BaseThreatDimensions)
        return Dimensions;

    return base + (edgeBoard == kingBoard ? 0 : BaseThreatDimensions);
}

namespace {

struct SemanticPiece {
    Piece  piece;
    Square square;
    Board  board;
};

struct SemanticThreat {
    Piece  attacker;
    Square from;
    Piece  attacked;
    Square to;
    Board  board;
};

struct SemanticState {
    std::vector<SemanticPiece>  pieces;
    std::vector<SemanticThreat> threats;
};

bool semantic_piece_less(const SemanticPiece& left, const SemanticPiece& right) {
    return std::tie(left.piece, left.square, left.board)
         < std::tie(right.piece, right.square, right.board);
}

bool semantic_threat_less(const SemanticThreat& left, const SemanticThreat& right) {
    return std::tie(left.attacker, left.from, left.attacked, left.to, left.board)
         < std::tie(right.attacker, right.from, right.attacked, right.to, right.board);
}

SemanticState collect_semantic_state(const Position& position) {
    SemanticState state;

    for (Square square = SQ_A1; square <= SQ_H8; ++square)
    {
        const Piece piece = position.piece_on(square);
        if (piece != NO_PIECE)
            state.pieces.push_back({piece, square, position.board_of(square)});
    }

    for (Board board : {BOARD_A, BOARD_B})
    {
        const Bitboard occupied           = position.occupancy_on(board);
        const Bitboard pawnTargets        = position.pieces_on(board, KNIGHT, ROOK);
        const Bitboard minorSliderTargets = position.pieces_on(board, PAWN, KNIGHT, BISHOP, ROOK);
        const Bitboard queenTargets = position.pieces_on(board, PAWN, KNIGHT, BISHOP, ROOK, QUEEN);

        for (Color color : {WHITE, BLACK})
        {
            {
                const Piece    attacker             = make_piece(color, PAWN);
                const Bitboard pawns                = position.pieces_on(board, color, PAWN);
                auto           process_pawn_attacks = [&](Bitboard attacks, Direction direction) {
                    while (attacks)
                    {
                        const Square to = pop_lsb(attacks);
                        state.threats.push_back(
                          {attacker, to - direction, position.piece_on(board, to), to, board});
                    }
                };

                if (color == WHITE)
                {
                    process_pawn_attacks(shift<NORTH_EAST>(pawns) & pawnTargets, NORTH_EAST);
                    process_pawn_attacks(shift<NORTH_WEST>(pawns) & pawnTargets, NORTH_WEST);
                }
                else
                {
                    process_pawn_attacks(shift<SOUTH_WEST>(pawns) & pawnTargets, SOUTH_WEST);
                    process_pawn_attacks(shift<SOUTH_EAST>(pawns) & pawnTargets, SOUTH_EAST);
                }
            }

            for (PieceType type = KNIGHT; type < KING; ++type)
            {
                const Piece    attacker  = make_piece(color, type);
                Bitboard       attackers = position.pieces_on(board, color, type);
                const Bitboard targets =
                  type == KNIGHT || type == QUEEN ? queenTargets : minorSliderTargets;

                while (attackers)
                {
                    const Square from    = pop_lsb(attackers);
                    Bitboard     attacks = Attacks::attacks_bb(type, from, occupied) & targets;
                    while (attacks)
                    {
                        const Square to = pop_lsb(attacks);
                        state.threats.push_back(
                          {attacker, from, position.piece_on(board, to), to, board});
                    }
                }
            }
        }
    }

    std::sort(state.pieces.begin(), state.pieces.end(), semantic_piece_less);
    std::sort(state.threats.begin(), state.threats.end(), semantic_threat_less);
    return state;
}

NativeFeatureDelta derive_delta(const SemanticState& before, const SemanticState& after) {
    NativeFeatureDelta delta;

    usize beforeIndex = 0;
    usize afterIndex  = 0;
    while (beforeIndex < before.pieces.size() || afterIndex < after.pieces.size())
    {
        if (afterIndex == after.pieces.size()
            || (beforeIndex < before.pieces.size()
                && semantic_piece_less(before.pieces[beforeIndex], after.pieces[afterIndex])))
        {
            const auto& piece = before.pieces[beforeIndex++];
            delta.pieces.push_back(
              {DeltaOperation::REMOVE, piece.piece, piece.square, piece.board});
        }
        else if (beforeIndex == before.pieces.size()
                 || semantic_piece_less(after.pieces[afterIndex], before.pieces[beforeIndex]))
        {
            const auto& piece = after.pieces[afterIndex++];
            delta.pieces.push_back({DeltaOperation::ADD, piece.piece, piece.square, piece.board});
        }
        else
        {
            ++beforeIndex;
            ++afterIndex;
        }
    }

    beforeIndex = 0;
    afterIndex  = 0;
    while (beforeIndex < before.threats.size() || afterIndex < after.threats.size())
    {
        if (afterIndex == after.threats.size()
            || (beforeIndex < before.threats.size()
                && semantic_threat_less(before.threats[beforeIndex], after.threats[afterIndex])))
        {
            const auto& threat = before.threats[beforeIndex++];
            delta.threats.push_back({DeltaOperation::REMOVE, threat.attacker, threat.from,
                                     threat.attacked, threat.to, threat.board});
        }
        else if (beforeIndex == before.threats.size()
                 || semantic_threat_less(after.threats[afterIndex], before.threats[beforeIndex]))
        {
            const auto& threat = after.threats[afterIndex++];
            delta.threats.push_back({DeltaOperation::ADD, threat.attacker, threat.from,
                                     threat.attacked, threat.to, threat.board});
        }
        else
        {
            ++beforeIndex;
            ++afterIndex;
        }
    }

    return delta;
}

u32 mix_fixture(u32 value) {
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    return value ^ (value >> 16);
}

i32 fixture_value(u32 tag, IndexType index, IndexType lane, i32 radius) {
    const u32 mixed = mix_fixture(tag ^ (index * 0x9E3779B9u) ^ (lane * 0x85EBCA6Bu));
    return i32(mixed % u32(2 * radius + 1)) - radius;
}

i32 fixture_bias(IndexType lane) { return fixture_value(0xA11CE101u, 0, lane, 31); }

i32 fixture_piece_weight(IndexType index, IndexType lane) {
    return fixture_value(0xA11CE201u, index, lane, 15);
}

i32 fixture_threat_weight(IndexType index, IndexType lane) {
    return fixture_value(0xA11CE301u, index, lane, 7);
}

i32 fixture_piece_psqt(IndexType index, IndexType bucket) {
    return fixture_value(0xA11CE401u, index, bucket, 127);
}

i32 fixture_threat_psqt(IndexType index, IndexType bucket) {
    return fixture_value(0xA11CE501u, index, bucket, 63);
}

struct FixtureAccumulator {
    std::array<i32, L1>          values{};
    std::array<i64, PsqtBuckets> psqt{};
};

struct alignas(CacheLineSize) SimdFixtureAccumulator {
    alignas(CacheLineSize) std::array<i16, L1> values{};
    alignas(CacheLineSize) std::array<i32, PsqtBuckets> psqt{};
};

struct FixtureCacheEntry {
    FixtureAccumulator           pieceAccumulator;
    std::array<Piece, SQUARE_NB> pieces{};
    Bitboard                     pieceBB = 0;
    Bitboard                     boardB  = 0;

    FixtureCacheEntry() {
        for (IndexType lane = 0; lane < L1; ++lane)
            pieceAccumulator.values[lane] = fixture_bias(lane);
        pieces.fill(NO_PIECE);
    }
};

struct FixtureCache {
    std::array<std::array<std::array<FixtureCacheEntry, SQUARE_NB>, BOARD_NB>, COLOR_NB> entries;
};

void update_fixture_piece(FixtureAccumulator& accumulator, IndexType index, i32 sign) {
    for (IndexType lane = 0; lane < L1; ++lane)
        accumulator.values[lane] += sign * fixture_piece_weight(index, lane);
    for (IndexType bucket = 0; bucket < PsqtBuckets; ++bucket)
        accumulator.psqt[bucket] += sign * fixture_piece_psqt(index, bucket);
}

void update_fixture_threat(FixtureAccumulator& accumulator, IndexType index, i32 sign) {
    for (IndexType lane = 0; lane < L1; ++lane)
        accumulator.values[lane] += sign * fixture_threat_weight(index, lane);
    for (IndexType bucket = 0; bucket < PsqtBuckets; ++bucket)
        accumulator.psqt[bucket] += sign * fixture_threat_psqt(index, bucket);
}

template<typename WeightFunction, typename PsqtFunction>
void update_simd_fixture(SimdFixtureAccumulator& accumulator,
                         IndexType               index,
                         i32                     sign,
                         WeightFunction          weight,
                         PsqtFunction            psqt) {
    using namespace SIMD;

    alignas(CacheLineSize) std::array<i16, L1>          laneWeights;
    alignas(CacheLineSize) std::array<i32, PsqtBuckets> psqtWeights;
    for (IndexType lane = 0; lane < L1; ++lane)
        laneWeights[lane] = i16(weight(index, lane));
    for (IndexType bucket = 0; bucket < PsqtBuckets; ++bucket)
        psqtWeights[bucket] = psqt(index, bucket);

    static_assert(L1 % (sizeof(vec_t) / sizeof(i16)) == 0);
    static_assert(PsqtBuckets % (sizeof(psqt_vec_t) / sizeof(i32)) == 0);

    auto*       destination = reinterpret_cast<vec_t*>(accumulator.values.data());
    const auto* source      = reinterpret_cast<const vec_t*>(laneWeights.data());
    for (usize block = 0; block < L1 / (sizeof(vec_t) / sizeof(i16)); ++block)
    {
        const vec_t current = vec_load(&destination[block]);
        const vec_t column  = vec_load(&source[block]);
        vec_store(&destination[block],
                  sign > 0 ? vec_add_16(current, column) : vec_sub_16(current, column));
    }

    auto*       psqtDestination = reinterpret_cast<psqt_vec_t*>(accumulator.psqt.data());
    const auto* psqtSource      = reinterpret_cast<const psqt_vec_t*>(psqtWeights.data());
    for (usize block = 0; block < PsqtBuckets / (sizeof(psqt_vec_t) / sizeof(i32)); ++block)
    {
        const psqt_vec_t current = vec_load_psqt(&psqtDestination[block]);
        const psqt_vec_t column  = vec_load_psqt(&psqtSource[block]);
        vec_store_psqt(&psqtDestination[block], sign > 0 ? vec_add_psqt_32(current, column)
                                                         : vec_sub_psqt_32(current, column));
    }
}

FixtureAccumulator build_fixture_accumulator(const PerspectiveTrace& trace) {
    FixtureAccumulator accumulator;
    for (IndexType lane = 0; lane < L1; ++lane)
        accumulator.values[lane] = fixture_bias(lane);
    for (const auto& feature : trace.pieces)
        update_fixture_piece(accumulator, feature.index, 1);
    for (const auto& feature : trace.threats)
        update_fixture_threat(accumulator, feature.index, 1);
    return accumulator;
}

SimdFixtureAccumulator build_simd_fixture_accumulator(const PerspectiveTrace& trace) {
    SimdFixtureAccumulator accumulator;
    for (IndexType lane = 0; lane < L1; ++lane)
        accumulator.values[lane] = i16(fixture_bias(lane));
    for (const auto& feature : trace.pieces)
        update_simd_fixture(accumulator, feature.index, 1, fixture_piece_weight,
                            fixture_piece_psqt);
    for (const auto& feature : trace.threats)
        update_simd_fixture(accumulator, feature.index, 1, fixture_threat_weight,
                            fixture_threat_psqt);
    return accumulator;
}

FixtureAccumulator refresh_fixture_cache(const Position&               position,
                                         const PerspectiveTrace&       trace,
                                         FixtureCache&                 cache,
                                         IncrementalVerificationStats& stats) {
    FixtureCacheEntry& entry = cache.entries[trace.perspective][trace.kingBoard][trace.kingSquare];
    const Bitboard     currentPieceBB = position.pieces();
    const Bitboard     currentBoardB  = position.occupancy_on(BOARD_B);

    for (Square square = SQ_A1; square <= SQ_H8; ++square)
    {
        const Bitboard squareBB   = square_bb(square);
        const bool     oldPresent = bool(entry.pieceBB & squareBB);
        const Piece    oldPiece   = oldPresent ? entry.pieces[square] : NO_PIECE;
        const Board    oldBoard   = entry.boardB & squareBB ? BOARD_B : BOARD_A;
        const Piece    newPiece   = position.piece_on(square);
        const bool     newPresent = newPiece != NO_PIECE;
        const Board    newBoard   = newPresent ? position.board_of(square) : BOARD_A;
        const bool     unchanged =
          oldPresent && newPresent && oldPiece == newPiece && oldBoard == newBoard;

        if (oldPresent && !unchanged)
        {
            const IndexType index = PieceSquareFeatures::make_index(
              trace.perspective, square, oldPiece, oldBoard, trace.kingSquare, trace.kingBoard);
            update_fixture_piece(entry.pieceAccumulator, index, -1);
            ++stats.cachePieceRemoves;
            stats.cacheBoardBEvents += oldBoard == BOARD_B;
        }
        if (newPresent && !unchanged)
        {
            const IndexType index = PieceSquareFeatures::make_index(
              trace.perspective, square, newPiece, newBoard, trace.kingSquare, trace.kingBoard);
            update_fixture_piece(entry.pieceAccumulator, index, 1);
            ++stats.cachePieceAdds;
            stats.cacheBoardBEvents += newBoard == BOARD_B;
        }

        entry.pieces[square] = newPiece;
    }

    entry.pieceBB = currentPieceBB;
    entry.boardB  = currentBoardB;

    FixtureAccumulator result = entry.pieceAccumulator;
    for (const auto& feature : trace.threats)
        update_fixture_threat(result, feature.index, 1);
    ++stats.cacheChecks;
    return result;
}

std::optional<std::string> verify_refresh_routes(const Position&               position,
                                                 const PositionTrace&          trace,
                                                 FixtureCache&                 cache,
                                                 IncrementalVerificationStats& stats) {
    for (Color perspective : {WHITE, BLACK})
    {
        const FixtureAccumulator     scalar = build_fixture_accumulator(trace[perspective]);
        const SimdFixtureAccumulator simd   = build_simd_fixture_accumulator(trace[perspective]);
        for (IndexType lane = 0; lane < L1; ++lane)
            if (scalar.values[lane] != simd.values[lane])
                return "Alice native SIMD accumulator mismatch at " + position.fen() + ".";
        for (IndexType bucket = 0; bucket < PsqtBuckets; ++bucket)
            if (scalar.psqt[bucket] != simd.psqt[bucket])
                return "Alice native SIMD PSQT mismatch at " + position.fen() + ".";
        ++stats.simdChecks;

        const FixtureAccumulator cached =
          refresh_fixture_cache(position, trace[perspective], cache, stats);
        if (cached.values != scalar.values || cached.psqt != scalar.psqt)
            return "Alice native board-aware cache mismatch at " + position.fen() + ".";
    }
    return std::nullopt;
}

bool erase_index(std::vector<IndexType>& indices, IndexType index) {
    const auto found = std::find(indices.begin(), indices.end(), index);
    if (found == indices.end())
        return false;
    indices.erase(found);
    return true;
}

std::vector<IndexType> piece_indices(const PerspectiveTrace& trace) {
    std::vector<IndexType> result;
    result.reserve(trace.pieces.size());
    for (const auto& feature : trace.pieces)
        result.push_back(feature.index);
    return result;
}

std::vector<IndexType> threat_indices(const PerspectiveTrace& trace) {
    std::vector<IndexType> result;
    result.reserve(trace.threats.size());
    for (const auto& feature : trace.threats)
        result.push_back(feature.index);
    return result;
}

std::optional<std::string> verify_transition(const NativeFeatureDelta&     delta,
                                             const PositionTrace&          before,
                                             const PositionTrace&          after,
                                             const std::string&            resultingFen,
                                             IncrementalVerificationStats& stats) {
    if (delta.pieces.size() > 8)
        return "Alice native piece delta exceeded eight events at " + resultingFen + ".";
    if (delta.threats.size() > 512)
        return "Alice native threat delta exceeded 512 events at " + resultingFen + ".";

    stats.maxPieceEvents  = std::max<u64>(stats.maxPieceEvents, delta.pieces.size());
    stats.maxThreatEvents = std::max<u64>(stats.maxThreatEvents, delta.threats.size());

    for (Color perspective : {WHITE, BLACK})
    {
        const auto& oldTrace = before[perspective];
        const auto& newTrace = after[perspective];
        const bool  ownKingTransferred =
          oldTrace.kingSquare != newTrace.kingSquare || oldTrace.kingBoard != newTrace.kingBoard;

        FixtureAccumulator incremental = build_fixture_accumulator(oldTrace);
        if (ownKingTransferred)
        {
            ++stats.fullRefreshes[perspective];
            incremental = build_fixture_accumulator(newTrace);
        }
        else
        {
            std::vector<IndexType> incrementalPieces  = piece_indices(oldTrace);
            std::vector<IndexType> incrementalThreats = threat_indices(oldTrace);

            for (const auto& event : delta.pieces)
            {
                const IndexType index = PieceSquareFeatures::make_index(
                  perspective, event.square, event.piece, event.board, oldTrace.kingSquare,
                  oldTrace.kingBoard);
                const i32 sign = event.operation == DeltaOperation::ADD ? 1 : -1;
                if (sign < 0)
                {
                    if (!erase_index(incrementalPieces, index))
                        return "Alice native piece removal was absent for "
                             + std::string(perspective == WHITE ? "white" : "black") + " at "
                             + resultingFen + ".";
                }
                else
                    incrementalPieces.push_back(index);
                update_fixture_piece(incremental, index, sign);
            }

            for (const auto& event : delta.threats)
            {
                const IndexType index = ThreatFeatures::make_index(
                  perspective, event.attacker, event.from, event.to, event.attacked, event.board,
                  oldTrace.kingSquare, oldTrace.kingBoard);
                if (index >= ThreatDimensions)
                    continue;

                const i32 sign = event.operation == DeltaOperation::ADD ? 1 : -1;
                if (sign < 0)
                {
                    if (!erase_index(incrementalThreats, index))
                        return "Alice native threat removal was absent for "
                             + std::string(perspective == WHITE ? "white" : "black") + " at "
                             + resultingFen + ".";
                }
                else
                    incrementalThreats.push_back(index);
                update_fixture_threat(incremental, index, sign);
            }

            std::sort(incrementalPieces.begin(), incrementalPieces.end());
            std::sort(incrementalThreats.begin(), incrementalThreats.end());
            if (incrementalPieces != piece_indices(newTrace))
                return "Alice native incremental piece trace mismatch at " + resultingFen + ".";
            if (incrementalThreats != threat_indices(newTrace))
                return "Alice native incremental threat trace mismatch at " + resultingFen + ".";
        }

        const FixtureAccumulator refreshed = build_fixture_accumulator(newTrace);
        if (incremental.values != refreshed.values || incremental.psqt != refreshed.psqt)
            return "Alice native scalar accumulator mismatch at " + resultingFen + ".";
    }

    return std::nullopt;
}

template<typename Sink>
bool enumerate_piece_features(
  const Position& position, Color perspective, Square kingSquare, Board kingBoard, Sink&& sink) {
    for (Square square = SQ_A1; square <= SQ_H8; ++square)
    {
        const Piece piece = position.piece_on(square);
        if (piece == NO_PIECE)
            continue;

        const Board     board = position.board_of(square);
        const IndexType index =
          PieceSquareFeatures::make_index(perspective, square, piece, board, kingSquare, kingBoard);
        if (!sink(index, piece, square, board, relation_of(board, kingBoard)))
            return false;
    }
    return true;
}

void append_piece_features(const Position& position, PerspectiveTrace& trace) {
    const bool complete = enumerate_piece_features(
      position, trace.perspective, trace.kingSquare, trace.kingBoard,
      [&](IndexType index, Piece piece, Square square, Board board, Relation relation) {
          trace.pieces.push_back({index, piece, square, board, relation});
          return true;
      });
    (void) complete;
    assert(complete);
}

template<typename Sink>
bool enumerate_threat_features(
  const Position& position, Color perspective, Square kingSquare, Board kingBoard, Sink&& sink) {
    for (Board board : {BOARD_A, BOARD_B})
    {
        const Bitboard occupied           = position.occupancy_on(board);
        const Bitboard pawnTargets        = position.pieces_on(board, KNIGHT, ROOK);
        const Bitboard minorSliderTargets = position.pieces_on(board, PAWN, KNIGHT, BISHOP, ROOK);
        const Bitboard queenTargets = position.pieces_on(board, PAWN, KNIGHT, BISHOP, ROOK, QUEEN);

        for (Color relative : {WHITE, BLACK})
        {
            const Color color = Color(perspective ^ relative);

            {
                const Piece    attacker             = make_piece(color, PAWN);
                const Bitboard pawns                = position.pieces_on(board, color, PAWN);
                auto           process_pawn_attacks = [&](Bitboard attacks, Direction direction) {
                    while (attacks)
                    {
                        const Square    to       = pop_lsb(attacks);
                        const Square    from     = to - direction;
                        const Piece     attacked = position.piece_on(board, to);
                        const IndexType index    = ThreatFeatures::make_index(
                          perspective, attacker, from, to, attacked, board, kingSquare, kingBoard);
                        if (index < ThreatDimensions)
                        {
                            if (!sink(index, attacker, from, attacked, to, board,
                                                relation_of(board, kingBoard)))
                                return false;
                        }
                    }
                    return true;
                };

                if (color == WHITE)
                {
                    if (!process_pawn_attacks(shift<NORTH_EAST>(pawns) & pawnTargets, NORTH_EAST)
                        || !process_pawn_attacks(shift<NORTH_WEST>(pawns) & pawnTargets,
                                                 NORTH_WEST))
                        return false;
                }
                else
                {
                    if (!process_pawn_attacks(shift<SOUTH_WEST>(pawns) & pawnTargets, SOUTH_WEST)
                        || !process_pawn_attacks(shift<SOUTH_EAST>(pawns) & pawnTargets,
                                                 SOUTH_EAST))
                        return false;
                }
            }

            for (PieceType type = KNIGHT; type < KING; ++type)
            {
                const Piece    attacker  = make_piece(color, type);
                Bitboard       attackers = position.pieces_on(board, color, type);
                const Bitboard targets =
                  type == KNIGHT || type == QUEEN ? queenTargets : minorSliderTargets;

                while (attackers)
                {
                    const Square from    = pop_lsb(attackers);
                    Bitboard     attacks = Attacks::attacks_bb(type, from, occupied) & targets;
                    while (attacks)
                    {
                        const Square    to       = pop_lsb(attacks);
                        const Piece     attacked = position.piece_on(board, to);
                        const IndexType index    = ThreatFeatures::make_index(
                          perspective, attacker, from, to, attacked, board, kingSquare, kingBoard);
                        if (index < ThreatDimensions)
                        {
                            if (!sink(index, attacker, from, attacked, to, board,
                                      relation_of(board, kingBoard)))
                                return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

void append_threat_features(const Position& position, PerspectiveTrace& trace) {
    const bool complete = enumerate_threat_features(
      position, trace.perspective, trace.kingSquare, trace.kingBoard,
      [&](IndexType index, Piece attacker, Square from, Piece attacked, Square to, Board board,
          Relation relation) {
          trace.threats.push_back({index, attacker, from, attacked, to, board, relation});
          return true;
      });
    (void) complete;
    assert(complete);
}

const char* color_name(Color color) { return color == WHITE ? "white" : "black"; }

const char* board_name(Board board) { return board == BOARD_A ? "A" : "B"; }

const char* relation_name(Relation relation) {
    return relation == Relation::SAME ? "SAME" : "OTHER";
}

const char* piece_name(Piece piece) {
    static constexpr const char* Names[PIECE_NB] = {
      "none", "wP", "wN", "wB", "wR", "wQ", "wK", "none",
      "none", "bP", "bN", "bB", "bR", "bQ", "bK", "none",
    };
    return Names[piece];
}

std::string square_name(Square square) {
    std::string name(2, ' ');
    name[0] = char('a' + int(file_of(square)));
    name[1] = char('1' + int(rank_of(square)));
    return name;
}

void write_piece(std::ostream& out, const PieceFeatureTrace& feature) {
    out << "{\"board\":\"" << board_name(feature.board) << "\",\"index\":" << feature.index
        << ",\"piece\":\"" << piece_name(feature.piece) << "\",\"relation\":\""
        << relation_name(feature.relation) << "\",\"square\":\"" << square_name(feature.square)
        << "\"}";
}

void write_threat(std::ostream& out, const ThreatFeatureTrace& feature) {
    out << "{\"attacked\":\"" << piece_name(feature.attacked) << "\",\"attacker\":\""
        << piece_name(feature.attacker) << "\",\"board\":\"" << board_name(feature.board)
        << "\",\"from\":\"" << square_name(feature.from) << "\",\"index\":" << feature.index
        << ",\"relation\":\"" << relation_name(feature.relation) << "\",\"to\":\""
        << square_name(feature.to) << "\"}";
}

}  // namespace

PositionTrace build_trace(const Position& position) {
    PositionTrace result;

    for (Color perspective : {WHITE, BLACK})
    {
        PerspectiveTrace& trace = result[perspective];
        trace.perspective       = perspective;
        trace.kingSquare        = position.square<KING>(perspective);
        trace.kingBoard         = position.board_of(trace.kingSquare);

        append_piece_features(position, trace);
        append_threat_features(position, trace);

        std::sort(trace.pieces.begin(), trace.pieces.end(),
                  [](const auto& left, const auto& right) {
                      return std::tie(left.index, left.piece, left.square, left.board)
                           < std::tie(right.index, right.piece, right.square, right.board);
                  });
        std::sort(trace.threats.begin(), trace.threats.end(),
                  [](const auto& left, const auto& right) {
                      return std::tie(left.index, left.attacker, left.from, left.attacked, left.to,
                                      left.board)
                           < std::tie(right.index, right.attacker, right.from, right.attacked,
                                      right.to, right.board);
                  });
    }

    return result;
}

std::optional<std::string> build_fixed_snapshot(const Position& position, FeatureSnapshot& result) {
    result = {};

    if (position.count<KING>(WHITE) != 1 || position.count<KING>(BLACK) != 1)
        return "Alice native feature snapshots require exactly one king per color.";

    const int pieceCount = popcount(position.pieces());
    if (pieceCount < 2 || pieceCount > int(MaximumPieceFeatures))
        return "Alice native feature snapshots require between 2 and 32 pieces.";

    for (Color perspective : {WHITE, BLACK})
    {
        PerspectiveFeatureSnapshot& snapshot = result[perspective];
        snapshot.perspective                 = perspective;
        snapshot.kingSquare                  = position.square<KING>(perspective);
        snapshot.kingBoard                   = position.board_of(snapshot.kingSquare);

        if (!enumerate_piece_features(
              position, snapshot.perspective, snapshot.kingSquare, snapshot.kingBoard,
              [&](IndexType index, Piece, Square, Board, Relation) {
                  return index < PieceSquareDimensions && snapshot.pieces.push_back(index);
              }))
            return "Alice native fixed piece feature capacity or index range was exceeded.";

        if (!enumerate_threat_features(
              position, snapshot.perspective, snapshot.kingSquare, snapshot.kingBoard,
              [&](IndexType index, Piece, Square, Piece, Square, Board, Relation) {
                  return index < ThreatDimensions && snapshot.threats.push_back(index);
              }))
            return "Alice native fixed threat feature capacity or index range was exceeded.";

        std::sort(snapshot.pieces.begin(), snapshot.pieces.end());
        std::sort(snapshot.threats.begin(), snapshot.threats.end());
    }

    return std::nullopt;
}

namespace {

std::optional<std::string> verify_fixed_snapshot(const Position&               position,
                                                 const PositionTrace&          trace,
                                                 IncrementalVerificationStats& stats) {
    FeatureSnapshot snapshot;
    if (auto error = build_fixed_snapshot(position, snapshot))
        return error;

    for (Color perspective : {WHITE, BLACK})
    {
        const auto& fixed = snapshot[perspective];
        if (fixed.perspective != perspective || fixed.kingSquare != trace[perspective].kingSquare
            || fixed.kingBoard != trace[perspective].kingBoard)
            return "Alice native fixed snapshot identity differed from the semantic trace.";

        const std::vector<IndexType> fixedPieces(fixed.pieces.begin(), fixed.pieces.end());
        const std::vector<IndexType> fixedThreats(fixed.threats.begin(), fixed.threats.end());
        if (fixedPieces != piece_indices(trace[perspective]))
            return "Alice native fixed piece snapshot differed from the semantic trace.";
        if (fixedThreats != threat_indices(trace[perspective]))
            return "Alice native fixed threat snapshot differed from the semantic trace.";

        ++stats.fixedSnapshotChecks;
    }

    return std::nullopt;
}

}  // namespace

std::string trace_json(const Position& position) {
    const PositionTrace trace = build_trace(position);
    std::ostringstream  out;

    out << "{\"architecture\":\"" << ArchitectureId << "\",\"compositeHash\":\"EC7CCD50\""
        << ",\"featureTransformerHash\":\"8F4FBC46\",\"pairFeature\":\"" << PairFeatureId
        << "\",\"perspectives\":[";

    for (Color perspective : {WHITE, BLACK})
    {
        if (perspective != WHITE)
            out << ',';

        const PerspectiveTrace& current = trace[perspective];
        out << "{\"color\":\"" << color_name(perspective) << "\",\"kingBoard\":\""
            << board_name(current.kingBoard) << "\",\"kingSquare\":\""
            << square_name(current.kingSquare) << "\",\"pieceFeatures\":[";

        for (usize i = 0; i < current.pieces.size(); ++i)
        {
            if (i)
                out << ',';
            write_piece(out, current.pieces[i]);
        }

        out << "],\"threatFeatures\":[";
        for (usize i = 0; i < current.threats.size(); ++i)
        {
            if (i)
                out << ',';
            write_threat(out, current.threats[i]);
        }
        out << "]}";
    }

    out << "],\"pieceSquareDimensions\":" << PieceSquareDimensions << ",\"pieceSquareFeature\":\""
        << PieceSquareFeatureId << "\",\"threatDimensions\":" << ThreatDimensions
        << ",\"threatFeature\":\"" << ThreatFeatureId << "\",\"wireVersion\":\"A11CE001\"}";
    return out.str();
}

std::optional<std::string>
verify_incremental(Position& position, Depth depth, IncrementalVerificationStats& stats) {
    stats = {};
    if (depth < 0 || depth > 2)
        return "Alice native incremental verification depth must be between 0 and 2.";

    const std::string rootFen = position.fen();
    const Key         rootKey = position.key();
    auto              cache   = std::make_unique<FixtureCache>();

    std::function<std::optional<std::string>(Depth)> visit;
    visit = [&](Depth remaining) -> std::optional<std::string> {
        ++stats.positions;
        const PositionTrace currentTrace = build_trace(position);
        if (auto error = verify_fixed_snapshot(position, currentTrace, stats))
            return error;
        if (auto error = verify_refresh_routes(position, currentTrace, *cache, stats))
            return error;
        if (remaining == 0)
            return std::nullopt;

        const std::string   nodeFen  = position.fen();
        const Key           nodeKey  = position.key();
        const SemanticState before   = collect_semantic_state(position);
        const PositionTrace oldTrace = currentTrace;
        std::vector<Move>   legalMoves;
        for (Move move : MoveList<LEGAL>(position))
            legalMoves.push_back(move);

        for (Move move : legalMoves)
        {
            const Piece moved = position.moved_piece(move);
            stats.captures += position.capture(move);
            stats.promotions += move.type_of() == PROMOTION;
            stats.castlings += move.type_of() == CASTLING;
            stats.kingMoves += type_of(moved) == KING;

            StateInfo state;
            Dirties   dirties;
            position.do_move(move, state, position.gives_check(move), dirties, nullptr, nullptr);

            const SemanticState      after    = collect_semantic_state(position);
            const PositionTrace      newTrace = build_trace(position);
            const NativeFeatureDelta delta    = derive_delta(before, after);
            ++stats.transitions;

            auto error = verify_transition(delta, oldTrace, newTrace, position.fen(), stats);
            if (!error)
                error = visit(remaining - 1);

            position.undo_move(move);
            if (position.fen() != nodeFen || position.key() != nodeKey)
                return "Alice native incremental verification did not restore a parent position.";
            if (error)
                return error;
        }

        return std::nullopt;
    };

    auto error = visit(depth);
    if (position.fen() != rootFen || position.key() != rootKey)
        return "Alice native incremental verification did not restore the root position.";
    return error;
}

}  // namespace Stockfish::Eval::NNUE::AliceNative
