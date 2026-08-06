/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "movegen.h"

#include <initializer_list>

#include "attacks.h"
#include "bitboard.h"
#include "position.h"

namespace Stockfish {

namespace {

Move* add_promotions(Move* moveList, Square from, Square to) {
    *moveList++ = Move::make<PROMOTION>(from, to, QUEEN);
    *moveList++ = Move::make<PROMOTION>(from, to, ROOK);
    *moveList++ = Move::make<PROMOTION>(from, to, BISHOP);
    *moveList++ = Move::make<PROMOTION>(from, to, KNIGHT);
    return moveList;
}

bool target_is_available(const Position& pos, Square to, Board source, Color us) {
    if (pos.empty(to))
        return true;
    if (pos.board_of(to) != source)
        return false;

    const Piece victim = pos.piece_on(to);
    return color_of(victim) != us && type_of(victim) != KING;
}

Move* generate_pawns(const Position& pos, Move* moveList, Color us) {
    Bitboard pawns = pos.pieces(us, PAWN);
    while (pawns)
    {
        const Square    from        = pop_lsb(pawns);
        const Board     source      = pos.board_of(from);
        const Direction push        = pawn_push(us);
        const Rank      startRank   = relative_rank(us, RANK_2);
        const Rank      promoteRank = relative_rank(us, RANK_8);
        const Square    one         = from + push;

        // The intermediate square of a double push is tested only on the
        // source board. Every actual arrival coordinate must be empty on both.
        if (is_ok(one) && pos.empty_on(source, one))
        {
            if (pos.empty(one))
            {
                if (rank_of(one) == promoteRank)
                    moveList = add_promotions(moveList, from, one);
                else
                    *moveList++ = Move(from, one);
            }

            if (rank_of(from) == startRank)
            {
                const Square two = one + push;
                if (is_ok(two) && pos.empty_on(source, two) && pos.empty(two))
                    *moveList++ = Move(from, two);
            }
        }

        // Captures exist only against an opponent on the source board. The
        // captured coordinate is then free for transfer to the other board.
        Bitboard captures = Attacks::attacks_bb<PAWN>(from, us) & pos.pieces_on(source, ~us);
        while (captures)
        {
            const Square to = pop_lsb(captures);
            if (type_of(pos.piece_on(to)) == KING)
                continue;
            if (rank_of(to) == promoteRank)
                moveList = add_promotions(moveList, from, to);
            else
                *moveList++ = Move(from, to);
        }
    }

    return moveList;
}

Move* generate_pieces(const Position& pos, Move* moveList, Color us, PieceType type) {
    Bitboard sources = pos.pieces(us, type);
    while (sources)
    {
        const Square from    = pop_lsb(sources);
        const Board  source  = pos.board_of(from);
        Bitboard     targets = Attacks::attacks_bb(type, from, pos.occupancy_on(source));

        while (targets)
        {
            const Square to = pop_lsb(targets);
            if (target_is_available(pos, to, source, us))
                *moveList++ = Move(from, to);
        }
    }

    return moveList;
}

Move* generate_castling(const Position& pos, Move* moveList, Color us) {
    const Square kingSquare = relative_square(us, SQ_E1);
    if (pos.piece_on(kingSquare) != make_piece(us, KING))
        return moveList;

    const Board source = pos.board_of(kingSquare);
    for (CastlingRights right : {us & KING_SIDE, us & QUEEN_SIDE})
    {
        if (!pos.can_castle(right))
            continue;

        const bool   kingSide = bool(right & KING_SIDE);
        const Square rookFrom = relative_square(us, kingSide ? SQ_H1 : SQ_A1);
        const Square kingTo   = relative_square(us, kingSide ? SQ_G1 : SQ_C1);
        const Square rookTo   = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
        if (pos.piece_on(source, rookFrom) != make_piece(us, ROOK))
            continue;

        const Bitboard sourcePath =
          Attacks::between_bb(kingSquare, rookFrom) & ~(kingSquare | rookFrom);
        if (sourcePath & pos.occupancy_on(source))
            continue;
        if (!pos.empty(kingTo) || !pos.empty(rookTo))
            continue;

        // Internal castling keeps Stockfish's king-to-rook encoding. Public
        // UCI conversion emits the orthodox king destination coordinate.
        *moveList++ = Move::make<CASTLING>(kingSquare, rookFrom);
    }

    return moveList;
}

Move* generate_candidates(const Position& pos, Move* moveList) {
    const Color us = pos.side_to_move();
    moveList       = generate_pawns(pos, moveList, us);
    moveList       = generate_pieces(pos, moveList, us, KNIGHT);
    moveList       = generate_pieces(pos, moveList, us, BISHOP);
    moveList       = generate_pieces(pos, moveList, us, ROOK);
    moveList       = generate_pieces(pos, moveList, us, QUEEN);
    moveList       = generate_pieces(pos, moveList, us, KING);
    return generate_castling(pos, moveList, us);
}

template<GenType Type>
bool include_move(const Position& pos, Move move) {
    if constexpr (Type == NON_EVASIONS)
        return true;
    else if constexpr (Type == EVASIONS || Type == LEGAL)
        return pos.legal(move);
    else
    {
        const bool capture = pos.capture(move);
        if constexpr (Type == CAPTURES)
            return capture || (move.type_of() == PROMOTION && move.promotion_type() == QUEEN);
        else
        {
            static_assert(Type == QUIETS);
            return move.type_of() != PROMOTION ? !capture
                                               : !capture && move.promotion_type() != QUEEN;
        }
    }
}

template<GenType Type>
Move* generate_alice(const Position& pos, Move* moveList) {
    Move  candidates[MAX_MOVES];
    Move* end = generate_candidates(pos, candidates);

    for (Move* current = candidates; current != end; ++current)
        if (include_move<Type>(pos, *current))
            *moveList++ = *current;

    return moveList;
}

}  // namespace


// CAPTURES preserves Stockfish's staging convention: captures and queen
// promotions. QUIETS contains ordinary non-captures and quiet underpromotions.
// EVASIONS and LEGAL both use the complete Alice legality filter.
template<GenType Type>
Move* generate(const Position& pos, Move* moveList) {
    static_assert(Type != LEGAL, "LEGAL has an explicit specialization");
    return generate_alice<Type>(pos, moveList);
}

template Move* generate<CAPTURES>(const Position&, Move*);
template Move* generate<QUIETS>(const Position&, Move*);
template Move* generate<EVASIONS>(const Position&, Move*);
template Move* generate<NON_EVASIONS>(const Position&, Move*);

template<>
Move* generate<LEGAL>(const Position& pos, Move* moveList) {
    return generate_alice<LEGAL>(pos, moveList);
}

}  // namespace Stockfish
