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
#include <sstream>
#include <tuple>

#include "../../attacks.h"
#include "../../bitboard.h"
#include "../../position.h"
#include "../features/full_threats.h"

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

void append_piece_features(const Position& position, PerspectiveTrace& trace) {
    for (Square square = SQ_A1; square <= SQ_H8; ++square)
    {
        const Piece piece = position.piece_on(square);
        if (piece == NO_PIECE)
            continue;

        const Board board = position.board_of(square);
        trace.pieces.push_back(
          {PieceSquareFeatures::make_index(trace.perspective, square, piece, board,
                                           trace.kingSquare, trace.kingBoard),
           piece, square, board, relation_of(board, trace.kingBoard)});
    }
}

void append_threat_features(const Position& position, PerspectiveTrace& trace) {
    for (Board board : {BOARD_A, BOARD_B})
    {
        const Bitboard occupied           = position.occupancy_on(board);
        const Bitboard pawnTargets        = position.pieces_on(board, KNIGHT, ROOK);
        const Bitboard minorSliderTargets = position.pieces_on(board, PAWN, KNIGHT, BISHOP, ROOK);
        const Bitboard queenTargets = position.pieces_on(board, PAWN, KNIGHT, BISHOP, ROOK, QUEEN);

        for (Color relative : {WHITE, BLACK})
        {
            const Color color = Color(trace.perspective ^ relative);

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
                          trace.perspective, attacker, from, to, attacked, board, trace.kingSquare,
                          trace.kingBoard);
                        if (index < ThreatDimensions)
                            trace.threats.push_back({index, attacker, from, attacked, to, board,
                                                     relation_of(board, trace.kingBoard)});
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
                        const Square    to       = pop_lsb(attacks);
                        const Piece     attacked = position.piece_on(board, to);
                        const IndexType index    = ThreatFeatures::make_index(
                          trace.perspective, attacker, from, to, attacked, board, trace.kingSquare,
                          trace.kingBoard);
                        if (index < ThreatDimensions)
                            trace.threats.push_back({index, attacker, from, attacked, to, board,
                                                     relation_of(board, trace.kingBoard)});
                    }
                }
            }
        }
    }
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

}  // namespace Stockfish::Eval::NNUE::AliceNative
