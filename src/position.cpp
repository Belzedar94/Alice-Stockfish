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

#include "position.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "movegen.h"
#include "uci.h"

using std::string;

namespace Stockfish {

using namespace Attacks;

namespace Zobrist {

Key psq[PIECE_NB][SQUARE_NB];
Key boardB[PIECE_NB][SQUARE_NB];
Key enpassant[FILE_NB];
Key castling[CASTLING_RIGHT_NB];
Key side, noPawns;

}

namespace {

constexpr std::string_view PieceToChar(" PNBRQK  pnbrqk");

static constexpr Piece Pieces[] = {W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                                   B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING};

Key alice_piece_key(Piece piece, Square square, Board board) {
    return Zobrist::psq[piece][square]
         ^ (board == BOARD_B ? Zobrist::boardB[piece][square] : Key(0));
}

struct AliceFenCell {
    Piece piece = NO_PIECE;
    Board board = BOARD_A;
};

struct ParsedAliceFen {
    std::array<AliceFenCell, SQUARE_NB> cells{};
    Color                               sideToMove = WHITE;
    std::string                         castling;
    int                                 rule50         = 0;
    int                                 fullmoveNumber = 1;
};

bool parse_decimal(std::string_view text, int& value) {
    if (text.empty())
        return false;

    int parsed = 0;
    for (char token : text)
    {
        if (token < '0' || token > '9')
            return false;

        const int digit = token - '0';
        if (parsed > (std::numeric_limits<int>::max() - digit) / 10)
            return false;
        parsed = parsed * 10 + digit;
    }

    value = parsed;
    return true;
}

std::optional<PositionSetError>
parse_empty_run(std::string_view rank, std::size_t& index, int maximum, int& run) {
    const std::size_t start = index;
    while (index < rank.size() && rank[index] >= '0' && rank[index] <= '9')
        ++index;

    const std::string_view digits = rank.substr(start, index - start);
    if (digits.front() == '0')
        return PositionSetError(
          "Invalid Alice FEN. Empty-square runs cannot contain a leading zero.");
    if (!parse_decimal(digits, run) || run < 1 || run > maximum)
        return PositionSetError(
          "Invalid Alice FEN. Empty-square run is outside the accepted width.");

    return std::nullopt;
}

std::optional<PositionSetError> split_alice_ranks(std::string_view                       placement,
                                                  std::array<std::string_view, RANK_NB>& ranks) {
    std::size_t start = 0;
    for (int rank = 0; rank < RANK_NB; ++rank)
    {
        const std::size_t slash = placement.find('/', start);
        const bool        last  = rank == RANK_NB - 1;

        if ((!last && slash == std::string_view::npos) || (last && slash != std::string_view::npos))
            return PositionSetError(
              "Invalid Alice FEN. Piece placement must contain exactly eight ranks.");

        const std::size_t end = last ? placement.size() : slash;
        ranks[rank]           = placement.substr(start, end - start);
        if (ranks[rank].empty())
            return PositionSetError("Invalid Alice FEN. Empty rank encoding.");
        start = end + 1;
    }

    return std::nullopt;
}

std::optional<PositionSetError> parse_compact_rank(std::string_view             rank,
                                                   std::array<AliceFenCell, 8>& cells) {
    std::size_t index  = 0;
    int         file   = 0;
    bool        boardB = false;

    while (index < rank.size())
    {
        const char token = rank[index];

        if (token == '|')
        {
            // The frozen opening book has one historical start-position rank
            // ending in a redundant layer marker. The legacy parser ignored
            // it after the rank had already expanded to all eight files. Keep
            // that input compatibility without accepting a marker in place of
            // a missing coordinate or before any other non-piece token.
            if (file == FILE_NB && index + 1 == rank.size())
            {
                ++index;
                continue;
            }
            if (boardB || index + 1 >= rank.size()
                || PieceToChar.find(rank[index + 1]) == std::string_view::npos
                || rank[index + 1] == ' ')
                return PositionSetError(
                  "Invalid Alice FEN. The layer marker must immediately precede a piece.");

            boardB = true;
            ++index;
            continue;
        }

        if (token >= '0' && token <= '9')
        {
            if (boardB)
                return PositionSetError(
                  "Invalid Alice FEN. A layer marker cannot precede an empty-square run.");

            int run = 0;
            if (auto error = parse_empty_run(rank, index, 8, run))
                return error;
            file += run;
        }
        else
        {
            const std::size_t pieceIndex = PieceToChar.find(token);
            if (pieceIndex == std::string_view::npos || token == ' ')
                return PositionSetError(std::string("Invalid Alice FEN. Invalid placement token: ")
                                        + token);
            if (file >= FILE_NB)
                return PositionSetError(
                  "Invalid Alice FEN. Rank expands beyond eight coordinates.");

            cells[file++] = {Piece(pieceIndex), boardB ? BOARD_B : BOARD_A};
            boardB        = false;
            ++index;
        }

        if (file > FILE_NB)
            return PositionSetError("Invalid Alice FEN. Rank expands beyond eight coordinates.");
    }

    if (boardB)
        return PositionSetError("Invalid Alice FEN. A layer marker cannot terminate a rank.");
    if (file != FILE_NB)
        return PositionSetError(
          "Invalid Alice FEN. Compact rank must expand to eight coordinates.");

    return std::nullopt;
}

std::optional<PositionSetError>
parse_unmarked_rank(std::string_view rank, std::array<Piece, 16>& cells, int& width) {
    std::size_t index = 0;
    width             = 0;

    while (index < rank.size())
    {
        const char token = rank[index];
        if (token >= '0' && token <= '9')
        {
            int run = 0;
            if (auto error = parse_empty_run(rank, index, 16, run))
                return error;
            width += run;
        }
        else
        {
            const std::size_t pieceIndex = PieceToChar.find(token);
            if (pieceIndex == std::string_view::npos || token == ' ' || token == '|')
                return PositionSetError(std::string("Invalid Alice FEN. Invalid placement token: ")
                                        + token);
            if (width >= 16)
                return PositionSetError("Invalid Alice FEN. Rank expands beyond sixteen cells.");
            cells[width++] = Piece(pieceIndex);
            ++index;
        }

        if (width > 16)
            return PositionSetError("Invalid Alice FEN. Rank expands beyond sixteen cells.");
    }

    return std::nullopt;
}

std::optional<PositionSetError> validate_alice_material(const ParsedAliceFen& parsed) {
    std::array<int, PIECE_NB> counts{};
    int                       total = 0;

    for (Square square = SQ_A1; square <= SQ_H8; ++square)
    {
        const Piece piece = parsed.cells[square].piece;
        if (piece == NO_PIECE)
            continue;

        ++counts[piece];
        ++total;
        if (type_of(piece) == PAWN && (rank_of(square) == RANK_1 || rank_of(square) == RANK_8))
            return PositionSetError(
              "Unsupported Alice position. Pawns cannot occupy the first or eighth rank.");
    }

    if (counts[W_KING] != 1 || counts[B_KING] != 1)
        return PositionSetError(
          "Unsupported Alice position. Exactly one king of each color is required.");
    if (total > 32)
        return PositionSetError("Unsupported Alice position. More than 32 pieces.");

    for (Color color : {WHITE, BLACK})
    {
        int colorTotal = 0;
        for (PieceType type = PAWN; type <= KING; ++type)
            colorTotal += counts[make_piece(color, type)];

        if (colorTotal > 16)
            return PositionSetError(
              "Unsupported Alice position. More than 16 pieces for one color.");

        const int pawns = counts[make_piece(color, PAWN)];
        if (pawns > 8)
            return PositionSetError(
              "Unsupported Alice position. More than eight pawns for one color.");

        const int promotedSurplus = std::max(counts[make_piece(color, KNIGHT)] - 2, 0)
                                  + std::max(counts[make_piece(color, BISHOP)] - 2, 0)
                                  + std::max(counts[make_piece(color, ROOK)] - 2, 0)
                                  + std::max(counts[make_piece(color, QUEEN)] - 1, 0);
        if (promotedSurplus > 8 - pawns)
            return PositionSetError(
              "Unsupported Alice position. Promoted material exceeds the missing-pawn allowance.");
    }

    return std::nullopt;
}

std::optional<PositionSetError> validate_alice_castling(const ParsedAliceFen& parsed) {
    if (parsed.castling == "-")
        return std::nullopt;
    if (parsed.castling.empty())
        return PositionSetError("Invalid Alice FEN. Missing castling field.");

    std::array<bool, 4> seen{};
    for (char right : parsed.castling)
    {
        std::size_t index;
        Piece       king;
        Piece       rook;
        Square      kingSquare;
        Square      rookSquare;

        switch (right)
        {
        case 'K' :
            index = 0, king = W_KING, rook = W_ROOK, kingSquare = SQ_E1, rookSquare = SQ_H1;
            break;
        case 'Q' :
            index = 1, king = W_KING, rook = W_ROOK, kingSquare = SQ_E1, rookSquare = SQ_A1;
            break;
        case 'k' :
            index = 2, king = B_KING, rook = B_ROOK, kingSquare = SQ_E8, rookSquare = SQ_H8;
            break;
        case 'q' :
            index = 3, king = B_KING, rook = B_ROOK, kingSquare = SQ_E8, rookSquare = SQ_A8;
            break;
        default :
            return PositionSetError(
              "Invalid Alice FEN. Castling rights must use unique KQkq symbols or '-'.");
        }

        if (seen[index])
            return PositionSetError("Invalid Alice FEN. Duplicate castling right.");
        seen[index] = true;

        const AliceFenCell& kingCell = parsed.cells[kingSquare];
        const AliceFenCell& rookCell = parsed.cells[rookSquare];
        if (kingCell.piece != king || rookCell.piece != rook)
            return PositionSetError(
              "Unsupported Alice position. Castling right lacks its orthodox king or rook.");
        if (kingCell.board != rookCell.board)
            return PositionSetError(
              "Unsupported Alice position. Castling king and rook must share a board.");
    }

    return std::nullopt;
}

std::optional<PositionSetError>
parse_alice_fen(const std::string& fen, bool isChess960, ParsedAliceFen& parsed) {
    if (isChess960)
        return PositionSetError(
          "Unsupported Alice position. Chess960 is not part of Alice rules v1.");

    std::array<std::string, 6> fields;
    std::istringstream         stream(fen);
    for (std::string& field : fields)
        if (!(stream >> field))
            return PositionSetError("Invalid Alice FEN. Exactly six fields are required.");

    std::string trailing;
    if (stream >> trailing)
        return PositionSetError("Invalid Alice FEN. Exactly six fields are required.");

    std::array<std::string_view, RANK_NB> ranks;
    if (auto error = split_alice_ranks(fields[0], ranks))
        return error;

    if (fields[0].find('|') != std::string::npos)
    {
        for (int fenRank = 0; fenRank < RANK_NB; ++fenRank)
        {
            std::array<AliceFenCell, 8> cells{};
            if (auto error = parse_compact_rank(ranks[fenRank], cells))
                return error;

            const Rank rank = Rank(RANK_8 - fenRank);
            for (File file = FILE_A; file <= FILE_H; ++file)
                parsed.cells[make_square(file, rank)] = cells[file];
        }
    }
    else
    {
        int expectedWidth = 0;
        for (int fenRank = 0; fenRank < RANK_NB; ++fenRank)
        {
            std::array<Piece, 16> cells{};
            int                   width = 0;
            if (auto error = parse_unmarked_rank(ranks[fenRank], cells, width))
                return error;
            if (width != 8 && width != 16)
                return PositionSetError(
                  "Invalid Alice FEN. Unmarked ranks must expand to eight or sixteen cells.");
            if (expectedWidth && width != expectedWidth)
                return PositionSetError("Invalid Alice FEN. Mixed rank widths are not accepted.");
            expectedWidth = width;

            const Rank rank = Rank(RANK_8 - fenRank);
            for (File file = FILE_A; file <= FILE_H; ++file)
            {
                const Piece boardA = cells[file];
                const Piece boardB = width == 16 ? cells[file + 8] : NO_PIECE;
                if (boardA != NO_PIECE && boardB != NO_PIECE)
                    return PositionSetError(
                      "Invalid Alice FEN. A coordinate cannot be occupied on both boards.");
                parsed.cells[make_square(file, rank)] = boardA != NO_PIECE
                                                        ? AliceFenCell{boardA, BOARD_A}
                                                        : AliceFenCell{boardB, BOARD_B};
            }
        }
    }

    if (fields[1] == "w")
        parsed.sideToMove = WHITE;
    else if (fields[1] == "b")
        parsed.sideToMove = BLACK;
    else
        return PositionSetError("Invalid Alice FEN. Active color must be 'w' or 'b'.");

    parsed.castling = fields[2];
    if (fields[3] != "-")
        return PositionSetError("Invalid Alice FEN. The en-passant field must be '-'.");
    if (!parse_decimal(fields[4], parsed.rule50) || parsed.rule50 > 32767)
        return PositionSetError("Unsupported Alice position. Rule50 counter is out of range.");
    if (!parse_decimal(fields[5], parsed.fullmoveNumber) || parsed.fullmoveNumber < 1
        || parsed.fullmoveNumber > 100000)
        return PositionSetError("Unsupported Alice position. Fullmove number is out of range.");

    if (auto error = validate_alice_material(parsed))
        return error;
    return validate_alice_castling(parsed);
}

struct AliceCastlingLayout {
    Square rookFrom;
    Square transit;
    Square kingTo;
    Square rookTo;
};

AliceCastlingLayout alice_castling_layout(Color color, Move move) {
    const bool kingSide = move.to_sq() > move.from_sq();
    return {move.to_sq(), relative_square(color, kingSide ? SQ_F1 : SQ_D1),
            relative_square(color, kingSide ? SQ_G1 : SQ_C1),
            relative_square(color, kingSide ? SQ_F1 : SQ_D1)};
}

}  // namespace


// Returns an ASCII representation of the position
std::ostream& operator<<(std::ostream& os, const Position& pos) {

    os << "\n +---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_8;; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
            os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];

        os << " | " << (1 + r) << "\n +---+---+---+---+---+---+---+---+\n";

        if (r == RANK_1)
            break;
    }

    os << "   a   b   c   d   e   f   g   h\n"
       << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase << std::setfill('0')
       << std::setw(16) << pos.key() << "\nPawn key: " << std::setw(16) << pos.pawn_key()
       << "\nMinor key: " << std::setw(16) << pos.minor_piece_key()
       << "\nWhite non-pawn key: " << std::setw(16) << pos.non_pawn_key(WHITE)
       << "\nBlack non-pawn key: " << std::setw(16) << pos.non_pawn_key(BLACK)
       << "\nMaterial key: " << std::setw(16) << pos.material_key() << std::setfill(' ') << std::dec
       << "\nCheckers: ";

    for (Bitboard b = pos.checkers(); b;)
        os << UCIEngine::square(pop_lsb(b)) << " ";

    return os;
}


// Implements Marcel van Kervinck's cuckoo algorithm to detect repetition of positions
// for 3-fold repetition draws. The algorithm uses two hash tables with Zobrist hashes
// to allow fast detection of recurring positions. For details see:
// http://web.archive.org/web/20201107002606/https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf

// First and second hash functions for indexing the cuckoo tables
inline int H1(Key h) { return h & 0x1fff; }
inline int H2(Key h) { return (h >> 16) & 0x1fff; }

// Cuckoo tables with Zobrist hashes of valid reversible moves, and the moves themselves
static std::array<Key, 8192>  cuckoo;
static std::array<Move, 8192> cuckooMove;

// Initializes at startup the various arrays used to compute hash keys
void Position::init() {

    PRNG rng(1070372);

    for (Piece pc : Pieces)
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
            Zobrist::psq[pc][s] = rng.rand<Key>();
    // pawns on these squares will promote
    std::fill_n(Zobrist::psq[W_PAWN] + SQ_A8, 8, 0);
    std::fill_n(Zobrist::psq[B_PAWN], 8, 0);

    for (File f = FILE_A; f <= FILE_H; ++f)
        Zobrist::enpassant[f] = rng.rand<Key>();

    for (int cr = NO_CASTLING; cr <= ANY_CASTLING; ++cr)
        Zobrist::castling[cr] = rng.rand<Key>();

    Zobrist::side    = rng.rand<Key>();
    Zobrist::noPawns = rng.rand<Key>();

    PRNG aliceRng(0xA11CEB04DULL);
    for (Piece pc : Pieces)
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
            Zobrist::boardB[pc][s] = aliceRng.rand<Key>();

    // Prepare the cuckoo tables
    cuckoo.fill(0);
    cuckooMove.fill(Move::none());
    [[maybe_unused]] int count = 0;
    for (Piece pc : Pieces)
        for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
            for (Square s2 = Square(s1 + 1); s2 <= SQ_H8; ++s2)
                if ((type_of(pc) != PAWN) && (attacks_bb(type_of(pc), s1, 0) & s2))
                {
                    Move move = Move(s1, s2);
                    Key  key  = Zobrist::psq[pc][s1] ^ Zobrist::psq[pc][s2] ^ Zobrist::side;
                    int  i    = H1(key);
                    while (true)
                    {
                        std::swap(cuckoo[i], key);
                        std::swap(cuckooMove[i], move);
                        if (move == Move::none())  // Arrived at empty slot?
                            break;
                        i = (i == H1(key)) ? H2(key) : H1(key);  // Push victim to alternative slot
                    }
                    count++;
                }
    assert(count == 3668);
}


// Initializes an Alice position from canonical compact FEN or legacy 16-wide FEN.
// Parsing and validation complete before the live position is modified.
std::optional<PositionSetError>
Position::set(const string& fenStr, bool isChess960, StateInfo* si) {
    ParsedAliceFen parsed;
    if (auto error = parse_alice_fen(fenStr, isChess960, parsed))
        return error;

    std::memset(reinterpret_cast<char*>(this), 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    for (Square square = SQ_A1; square <= SQ_H8; ++square)
    {
        const AliceFenCell& cell = parsed.cells[square];
        if (cell.piece != NO_PIECE)
            put_piece(cell.piece, square, cell.board);
    }

    sideToMove   = parsed.sideToMove;
    st->epSquare = SQ_NONE;
    st->rule50   = parsed.rule50;
    gamePly      = 2 * (parsed.fullmoveNumber - 1) + (sideToMove == BLACK);
    chess960     = false;

    if (parsed.castling != "-")
        for (char right : parsed.castling)
        {
            const Color  color      = right == 'K' || right == 'Q' ? WHITE : BLACK;
            const Square rookSquare = right == 'K' ? SQ_H1
                                    : right == 'Q' ? SQ_A1
                                    : right == 'k' ? SQ_H8
                                                   : SQ_A8;
            set_castling_right(color, rookSquare);
        }

    set_state();
    assert(pos_is_ok());
    return std::nullopt;
}


#if 0
// Frozen orthodox parser retained only as local porting context. It is excluded
// from every Alice build and is not reachable from the public loading path.
std::optional<PositionSetError>
Position::set_orthodox_fen_legacy(const string& fenStr, bool isChess960, StateInfo* si) {
    /*
   A FEN string defines a particular position using only the ASCII character set.

   A FEN string contains six fields separated by a space. The fields are:

   1) Piece placement (from white's perspective). Each rank is described, starting
      with rank 8 and ending with rank 1. Within each rank, the contents of each
      square are described from file A through file H. Following the Standard
      Algebraic Notation (SAN), each piece is identified by a single letter taken
      from the standard English names. White pieces are designated using upper-case
      letters ("PNBRQK") whilst Black uses lowercase ("pnbrqk"). Blank squares are
      noted using digits 1 through 8 (the number of blank squares), and "/"
      separates ranks.

   2) Active color. "w" means white moves next, "b" means black.

   3) Castling availability. If neither side can castle, this is "-". Otherwise,
      this has one or more letters: "K" (White can castle kingside), "Q" (White
      can castle queenside), "k" (Black can castle kingside), and/or "q" (Black
      can castle queenside).

   4) En passant target square (in algebraic notation). If there's no en passant
      target square, this is "-". If a pawn has just made a 2-square move, this
      is the position "behind" the pawn. Following X-FEN standard, this is recorded
      only if there is a pawn in position to make an en passant capture, and if
      there really is a pawn that might have advanced two squares.

   5) Halfmove clock. This is the number of halfmoves since the last pawn advance
      or capture. This is used to determine if a draw can be claimed under the
      fifty-move rule.

   6) Fullmove number. The number of the full move. It starts at 1, and is
      incremented after Black's move.
*/

    unsigned char      token;
    std::istringstream ss(fenStr);

    std::memset(reinterpret_cast<char*>(this), 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    ss >> std::noskipws;

    int numPieces = 0;
    int file      = FILE_A;
    int rank      = RANK_8;

    // 1. Piece placement
    for (;;)
    {
        if (!(ss >> token))
            return PositionSetError("Invalid FEN. Unexpected end of stream.");

        if (isspace(token))
            break;

        if (isdigit(token))
        {
            const int diff = (token - '0');
            if (diff < 1 || diff > 8)
                return PositionSetError("Invalid FEN. Invalid number of squares to skip.");

            file += diff;
            if (file > FILE_NB)
                return PositionSetError("Invalid FEN. Invalid file reached.");
        }
        else if (token == '/')
        {
            if (file != FILE_NB)
                return PositionSetError(
                  "Invalid FEN. Trying to end rank when not at the end of it.");

            --rank;
            file = FILE_A;

            if (rank < RANK_1)
                return PositionSetError("Invalid FEN. Invalid rank reached.");
        }
        else
        {
            if (file >= FILE_NB)
                return PositionSetError("Invalid FEN. Invalid file reached.");

            const usize idx = PieceToChar.find(token);
            if (idx == string::npos)
                return PositionSetError(std::string("Invalid FEN. Invalid piece: ")
                                        + std::string(1, token));

            if (++numPieces > 32)
                return PositionSetError("Invalid FEN. More than 32 pieces on the board.");

            const Square sq = make_square(File(file), Rank(rank));
            put_piece(Piece(idx), sq);

            ++file;
        }
    }

    if (rank != RANK_1 || file != FILE_NB)
        return PositionSetError("Invalid FEN. Board state encoding ended but cursor not at end.");

    if (pieces(PAWN) & (Rank1BB | Rank8BB))
        return PositionSetError("Unsupported position. Pawns on the first or eighth rank.");

    if (count<KING>(WHITE) != 1 || count<KING>(BLACK) != 1)
        return PositionSetError("Unsupported position. Incorrect number of kings.");

    for (Color c : {WHITE, BLACK})
    {
        if (count<PAWN>(c) > 8)
            return PositionSetError(std::string("Unsupported position. ")
                                    + (c == WHITE ? "WHITE" : "BLACK") + " has more than 8 pawns.");

        int additional = std::max(count<KNIGHT>(c) - 2, 0) + std::max(count<BISHOP>(c) - 2, 0)
                       + std::max(count<ROOK>(c) - 2, 0) + std::max(count<QUEEN>(c) - 1, 0);
        if (additional > 8 - count<PAWN>(c))
            return PositionSetError(std::string("Unsupported position. Too many pieces for ")
                                    + (c == WHITE ? "WHITE." : "BLACK."));
    }

    // 2. Active color
    if (!(ss >> token))
        return PositionSetError("Invalid FEN. Unexpected end of stream.");
    if (token != 'w' && token != 'b')
        return PositionSetError(std::string("Invalid FEN. Invalid side to move: ")
                                + std::string(1, token));
    sideToMove = (token == 'w' ? WHITE : BLACK);
    if (!(ss >> token) || !isspace(token) || ss.eof())
        return PositionSetError("Invalid FEN. Expected whitespace after side to move.");

    // 3. Castling availability. Compatible with 3 standards: Normal FEN standard,
    // Shredder-FEN that uses the letters of the columns on which the rooks began
    // the game instead of KQkq and also X-FEN standard that, in case of Chess960,
    // if an inner rook is associated with the castling right, the castling tag is
    // replaced by the file letter of the involved rook, as for the Shredder-FEN.
    //
    // NOTE: Due to the prevalence of incorrect (or missing) castling rights the
    // validation is less strict. However, incorrect castling rights are still sanitized.
    int num_castling_rights = 0;
    for (;;)
    {
        if (!(ss >> token))
            break;

        if (isspace(token))
            break;

        if (num_castling_rights == 0 && token == '-')
        {
            ss >> std::ws;
            break;
        }

        if (++num_castling_rights > 4)
            return PositionSetError("Invalid FEN. Maximum of 4 castling rights can be specified.");

        Square rsq  = SQ_NONE;
        Square ksq  = SQ_NONE;
        Color  c    = islower(token) ? BLACK : WHITE;
        Piece  rook = make_piece(c, ROOK);
        Piece  king = make_piece(c, KING);

        token = char(toupper(token));

        if (token == 'K' || token == 'Q')
        {
            const int dir = token == 'K' ? -1 : 1;
            Square    sq  = relative_square(c, token == 'K' ? SQ_H1 : SQ_A1);
            // Look for a rook and a king for the castling. King must come later.
            // Only the first rook is noted.
            // If the castling rights are available the king must always be between files 2 and 7 inclusive
            // so there is no need to check the last square.
            for (int i = 0; i < 7; ++i, sq = Square(sq + dir))
            {
                const Piece pc = piece_on(sq);
                if (pc == king)
                {
                    ksq = sq;
                    break;
                }
                else if (pc == rook && rsq == SQ_NONE)
                {
                    rsq = sq;
                }
            }
        }
        else if (token >= 'A' && token <= 'H')
        {
            const Square rsqCandidate = make_square(File(token - 'A'), relative_rank(c, RANK_1));
            if (piece_on(rsqCandidate) == rook)
                rsq = rsqCandidate;

            // If the castling rights are available the king must always be between files 2 and 7 inclusive.
            Square sq = relative_square(c, SQ_B1);
            for (int i = 0; i < 6; ++i, ++sq)
            {
                if (piece_on(sq) == king)
                    ksq = sq;
            }
        }
        else
        {
            return PositionSetError(std::string("Invalid FEN. Expected castling rights. Got: ")
                                    + std::string(1, token));
        }

        // Only apply castling rights if they can be valid.
        if (ksq != SQ_NONE && rsq != SQ_NONE)
            set_castling_right(c, rsq);
    }

    // 4. En passant square.
    // Ignore if square is invalid or not on side to move relative rank 6.
    bool          enpassant = false, legalEP = false;
    unsigned char col = '-', row;
    ss >> col;
    if (col != '-')
    {
        if (!(ss >> row))
            return PositionSetError("Invalid FEN. Unexpected end of stream.");

        if ((col >= 'a' && col <= 'h') && (row == (sideToMove == WHITE ? '6' : '3')))
        {
            st->epSquare = make_square(File(col - 'a'), Rank(row - '1'));

            Bitboard pawns = attacks_bb<PAWN>(st->epSquare, ~sideToMove) & pieces(sideToMove, PAWN);
            Bitboard target = (pieces(~sideToMove, PAWN) & (st->epSquare + pawn_push(~sideToMove)));
            Bitboard occ    = pieces() ^ target ^ st->epSquare;

            // En passant square will be considered only if
            // a) side to move have a pawn threatening epSquare
            // b) there is an enemy pawn in front of epSquare
            // c) there is no piece on epSquare or behind epSquare
            enpassant = pawns && target
                     && !(pieces() & (st->epSquare | (st->epSquare + pawn_push(sideToMove))));

            // If no pawn can execute the en passant capture without leaving the king in check, don't record the epSquare
            while (pawns)
                legalEP |= !(attackers_to(square<KING>(sideToMove), occ ^ pop_lsb(pawns))
                             & pieces(~sideToMove) & ~target);
        }
        else
            return PositionSetError("Invalid FEN. Invalid en-passant square.");
    }

    if (!enpassant || !legalEP)
        st->epSquare = SQ_NONE;

    // 5-6. Halfmove clock and fullmove number
    ss >> std::skipws >> st->rule50 >> gamePly;

    // Normally values larger than 99 would be pointless but we do support ignoring 50 move rule for TB purposes.
    // Limit at 2**15 as it's used multiplicatively with position evaluation during search.
    if (st->rule50 < 0 || st->rule50 > 32767)
        return PositionSetError("Unsupported position. Rule50 counter out of range.");

    if (gamePly < 0 || gamePly > 100000)
        return PositionSetError("Unsupported position. Game ply out of range.");

    // Convert from fullmove starting from 1 to gamePly starting from 0,
    // handle also common incorrect FEN with fullmove = 0.
    gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

    chess960 = isChess960;
    set_state();

    if (attackers_to_exist(square<KING>(~sideToMove), pieces(), sideToMove))
        return PositionSetError("Unsupported position. King can be captured.");

    assert(pos_is_ok());

    return std::nullopt;
}
#endif


// Helper function used to set castling
// rights given the corresponding color and the rook starting square.
void Position::set_castling_right(Color c, Square rfrom) {

    Square         kfrom = square<KING>(c);
    CastlingRights cr    = c & (kfrom < rfrom ? KING_SIDE : QUEEN_SIDE);

    st->castlingRights |= cr;
    castlingRightsMask[kfrom] |= cr;
    castlingRightsMask[rfrom] |= cr;
    castlingRookSquare[cr] = rfrom;

    Square kto = relative_square(c, cr & KING_SIDE ? SQ_G1 : SQ_C1);
    Square rto = relative_square(c, cr & KING_SIDE ? SQ_F1 : SQ_D1);

    castlingPath[cr] = (between_bb(rfrom, rto) | between_bb(kfrom, kto)) & ~(kfrom | rfrom);
}


// Sets king attacks to detect if a move gives check
void Position::set_check_info() const {

    update_slider_blockers(WHITE);
    update_slider_blockers(BLACK);

    Square ksq                              = square<KING>(~sideToMove);
    Board  layer                            = board_of(ksq);
    const auto [bishopAttacks, rookAttacks] = both_attacks_bb(ksq, occupancy_on(layer));

    st->checkSquares[PAWN]   = attacks_bb<PAWN>(ksq, ~sideToMove);
    st->checkSquares[KNIGHT] = attacks_bb<KNIGHT>(ksq);
    st->checkSquares[BISHOP] = bishopAttacks;
    st->checkSquares[ROOK]   = rookAttacks;
    st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
    st->checkSquares[KING]   = 0;
}


// Computes the hash keys of the position, and other
// data that once computed is updated incrementally as moves are made.
// The function is only used when a new position is set up
void Position::set_state() const {

    st->key               = 0;
    st->minorPieceKey     = 0;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = 0;
    st->pawnKey                                   = Zobrist::noPawns;
    st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = VALUE_ZERO;
    const Square kingSquare                                 = square<KING>(sideToMove);
    const Board  kingBoard                                  = board_of(kingSquare);
    st->checkersBB = attackers_to(kingSquare, kingBoard) & pieces_on(kingBoard, ~sideToMove);

    set_check_info();

    for (Bitboard b = pieces(); b;)
    {
        Square    s        = pop_lsb(b);
        Piece     pc       = piece_on(s);
        const Key pieceKey = alice_piece_key(pc, s, board_of(s));
        st->key ^= pieceKey;

        if (type_of(pc) == PAWN)
            st->pawnKey ^= pieceKey;

        else
        {
            st->nonPawnKey[color_of(pc)] ^= pieceKey;

            if (type_of(pc) != KING)
            {
                st->nonPawnMaterial[color_of(pc)] += PieceValue[pc];

                if (type_of(pc) <= BISHOP)
                    st->minorPieceKey ^= pieceKey;
            }
        }
    }

    if (st->epSquare != SQ_NONE)
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;

    st->key ^= Zobrist::castling[st->castlingRights];
    st->materialKey = compute_material_key();
}

Key Position::compute_material_key() const {
    Key k = 0;
    for (Piece pc : Pieces)
        for (int cnt = 0; cnt < pieceCount[pc]; ++cnt)
            k ^= Zobrist::psq[pc][8 + cnt];
    return k;
}


// Overload to initialize the position object with the given endgame code string
// like "KBPKN". It's mainly a helper to get the material key out of an endgame code.
std::optional<PositionSetError> Position::set(const string& code, Color c, StateInfo* si) {

    assert(code[0] == 'K');

    string sides[] = {code.substr(code.find('K', 1)),                                // Weak
                      code.substr(0, std::min(code.find('v'), code.find('K', 1)))};  // Strong

    assert(sides[0].length() > 0 && sides[0].length() < 8);
    assert(sides[1].length() > 0 && sides[1].length() < 8);

    std::transform(sides[c].begin(), sides[c].end(), sides[c].begin(), tolower);

    string fenStr = "8/" + sides[0] + char(8 - sides[0].length() + '0') + "/8/8/8/8/" + sides[1]
                  + char(8 - sides[1].length() + '0') + "/8 w - - 0 10";

    return set(fenStr, false, si);
}


// Returns the canonical compact Alice FEN representation.
string Position::fen() const {

    int                emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_8;; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
        {
            for (emptyCnt = 0; f <= FILE_H && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_H)
            {
                const Square square = make_square(f, r);
                if (board_of(square) == BOARD_B)
                    ss << '|';
                ss << PieceToChar[piece_on(square)];
            }
        }

        if (r == RANK_1)
            break;
        ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    if (can_castle(WHITE_OO))
        ss << 'K';

    if (can_castle(WHITE_OOO))
        ss << 'Q';

    if (can_castle(BLACK_OO))
        ss << 'k';

    if (can_castle(BLACK_OOO))
        ss << 'q';

    if (!can_castle(ANY_CASTLING))
        ss << '-';

    ss << (ep_square() == SQ_NONE ? " - " : " " + UCIEngine::square(ep_square()) + " ")
       << st->rule50 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    return ss.str();
}

// Calculates st->blockersForKing[c] and st->pinners[~c],
// which store respectively the pieces preventing king of color c from being in check
// and the slider pieces of color ~c pinning pieces of color c to the king.
void Position::update_slider_blockers(Color c) const {

    Square ksq   = square<KING>(c);
    Board  layer = board_of(ksq);

    st->blockersForKing[c] = 0;
    st->pinners[~c]        = 0;

    // Snipers are sliders that attack 's' when a piece and other snipers are removed
    Bitboard snipers   = ((attacks_bb<ROOK>(ksq) & pieces_on(layer, ~c, QUEEN, ROOK))
                        | (attacks_bb<BISHOP>(ksq) & pieces_on(layer, ~c, QUEEN, BISHOP)));
    Bitboard occupancy = occupancy_on(layer) ^ snipers;

    while (snipers)
    {
        Square   sniperSq = pop_lsb(snipers);
        Bitboard b        = between_bb(ksq, sniperSq) & occupancy;

        if (b && !more_than_one(b))
        {
            st->blockersForKing[c] |= b;
            if (b & pieces_on(layer, c))
                st->pinners[~c] |= sniperSq;
        }
    }
}


// Computes a bitboard of all pieces which attack a given square.
// Slider attacks use the occupied bitboard to indicate occupancy.
Bitboard Position::attackers_to(Square s, Bitboard occupied) const {

    const auto [bishopAttacks, rookAttacks] = both_attacks_bb(s, occupied);

    return (rookAttacks & pieces(ROOK, QUEEN)) | (bishopAttacks & pieces(BISHOP, QUEEN))
         | (attacks_bb<PAWN>(s, BLACK) & pieces(WHITE, PAWN))
         | (attacks_bb<PAWN>(s, WHITE) & pieces(BLACK, PAWN))
         | (attacks_bb<KNIGHT>(s) & pieces(KNIGHT)) | (attacks_bb<KING>(s) & pieces(KING));
}

Bitboard Position::attackers_to(Square s, Board layer) const {
    return attackers_to(s, layer, occupancy_on(layer));
}

Bitboard Position::attackers_to(Square s, Board layer, Bitboard occupied) const {
    const auto [bishopAttacks, rookAttacks] = both_attacks_bb(s, occupied);

    return (rookAttacks & pieces_on(layer, ROOK, QUEEN))
         | (bishopAttacks & pieces_on(layer, BISHOP, QUEEN))
         | (attacks_bb<PAWN>(s, BLACK) & pieces_on(layer, WHITE, PAWN))
         | (attacks_bb<PAWN>(s, WHITE) & pieces_on(layer, BLACK, PAWN))
         | (attacks_bb<KNIGHT>(s) & pieces_on(layer, KNIGHT))
         | (attacks_bb<KING>(s) & pieces_on(layer, KING));
}

bool Position::attackers_to_exist(Square s, Bitboard occupied, Color c) const {

    return (attacks_bb<ROOK>(s, occupied) & pieces(c, ROOK, QUEEN))
        || (attacks_bb<BISHOP>(s, occupied) & pieces(c, BISHOP, QUEEN))
        || (attacks_bb<PAWN>(s, ~c) & pieces(c, PAWN))
        || (attacks_bb<KNIGHT>(s) & pieces(c, KNIGHT)) || (attacks_bb<KING>(s) & pieces(c, KING));
}

bool Position::attackers_to_exist(Square s, Board layer, Bitboard occupied, Color c) const {
    return (attacks_bb<ROOK>(s, occupied) & pieces_on(layer, c, ROOK, QUEEN))
        || (attacks_bb<BISHOP>(s, occupied) & pieces_on(layer, c, BISHOP, QUEEN))
        || (attacks_bb<PAWN>(s, ~c) & pieces_on(layer, c, PAWN))
        || (attacks_bb<KNIGHT>(s) & pieces_on(layer, c, KNIGHT))
        || (attacks_bb<KING>(s) & pieces_on(layer, c, KING));
}

// Tests whether a pseudo-legal move is legal
bool Position::legal(Move m) const {
    if (!m.is_ok() || m.type_of() == EN_PASSANT)
        return false;

    const Color  us    = sideToMove;
    const Square from  = m.from_sq();
    const Piece  mover = piece_on(from);
    if (mover == NO_PIECE || color_of(mover) != us)
        return false;

    const auto attacked = [&](Square target, Board layer, Bitboard occupied,
                              Bitboard removedEnemy = 0) {
        return bool(attackers_to(target, layer, occupied) & pieces_on(layer, ~us) & ~removedEnemy);
    };

    if (m.type_of() == CASTLING)
    {
        const AliceCastlingLayout layout  = alice_castling_layout(us, m);
        const Board               source  = board_of(from);
        const Board               arrival = opposite(source);
        if (type_of(mover) != KING || piece_on(source, layout.rookFrom) != make_piece(us, ROOK)
            || checkers() || !empty(layout.transit) || !empty(layout.kingTo)
            || !empty(layout.rookTo))
            return false;

        Bitboard transit = (occupancy_on(source) & ~square_bb(from)) | layout.transit;
        if (attacked(layout.transit, source, transit))
            return false;

        Bitboard provisional =
          occupancy_on(source) & ~square_bb(from) & ~square_bb(layout.rookFrom);
        provisional |= layout.kingTo | layout.rookTo;
        if (attacked(layout.kingTo, source, provisional))
            return false;

        const Bitboard finalOccupancy = occupancy_on(arrival) | layout.kingTo | layout.rookTo;
        return !attacked(layout.kingTo, arrival, finalOccupancy);
    }

    const Square to = m.to_sq();
    if (!is_ok(to))
        return false;

    const Board    source       = board_of(from);
    const Board    arrival      = opposite(source);
    const bool     capture      = !empty(to);
    const Bitboard removedEnemy = capture ? square_bb(to) : Bitboard(0);
    const Square   currentKing  = square<KING>(us);
    const Board    currentLayer = board_of(currentKing);

    if (currentLayer == source)
    {
        const Square provisionalKing = type_of(mover) == KING ? to : currentKing;
        Bitboard     provisional     = occupancy_on(source) & ~square_bb(from) & ~removedEnemy;
        provisional |= to;
        if (attacked(provisionalKing, source, provisional, removedEnemy))
            return false;
    }

    const Square finalKing      = type_of(mover) == KING ? to : currentKing;
    const Board  finalLayer     = type_of(mover) == KING ? arrival : currentLayer;
    Bitboard     finalOccupancy = occupancy_on(finalLayer);
    if (finalLayer == source)
        finalOccupancy &= ~square_bb(from) & ~removedEnemy;
    if (finalLayer == arrival)
        finalOccupancy |= to;

    return !attacked(finalKing, finalLayer, finalOccupancy,
                     finalLayer == source ? removedEnemy : Bitboard(0));
}


// Takes a random move and tests whether the move is
// pseudo-legal. It is used to validate moves from TT that can be corrupted
// due to SMP concurrent access or hash position key aliasing.
bool Position::pseudo_legal(const Move m) const {
    if (!m.is_ok() || m.type_of() == EN_PASSANT)
        return false;

    const Square from  = m.from_sq();
    const Square to    = m.to_sq();
    const Piece  mover = piece_on(from);
    if (mover == NO_PIECE || color_of(mover) != sideToMove || !is_ok(to))
        return false;

    const Color us     = sideToMove;
    const Board source = board_of(from);
    if (m.type_of() == CASTLING)
    {
        if (type_of(mover) != KING || piece_on(source, to) != make_piece(us, ROOK))
            return false;
        const CastlingRights right = us & (to > from ? KING_SIDE : QUEEN_SIDE);
        if (!can_castle(right))
            return false;
        const AliceCastlingLayout layout     = alice_castling_layout(us, m);
        const Bitboard            sourcePath = between_bb(from, to) & ~(from | to);
        return !(sourcePath & occupancy_on(source)) && empty(layout.kingTo) && empty(layout.rookTo);
    }

    if (!empty(to)
        && (board_of(to) != source || color_of(piece_on(to)) == us
            || type_of(piece_on(to)) == KING))
        return false;

    const PieceType type = type_of(mover);
    if (type == PAWN)
    {
        const Rank promotionRank = relative_rank(us, RANK_8);
        if ((rank_of(to) == promotionRank) != (m.type_of() == PROMOTION))
            return false;
        if (m.type_of() == PROMOTION && (m.promotion_type() < KNIGHT || m.promotion_type() > QUEEN))
            return false;

        if (!empty(to))
            return bool(attacks_bb<PAWN>(from, us) & to);

        const Direction push = pawn_push(us);
        if (to == from + push)
            return empty_on(source, to);
        if (rank_of(from) == relative_rank(us, RANK_2) && to == from + 2 * push)
            return empty_on(source, from + push) && empty_on(source, to);
        return false;
    }

    return m.type_of() == NORMAL && bool(attacks_bb(type, from, occupancy_on(source)) & to);
}


// Tests whether a pseudo-legal move gives a check
bool Position::gives_check(Move m) const {
    assert(m.is_ok() && color_of(moved_piece(m)) == sideToMove);

    const Color  us        = sideToMove;
    const Square king      = square<KING>(~us);
    const Board  kingLayer = board_of(king);
    const Square from      = m.from_sq();
    const Board  source    = board_of(from);
    const Board  arrival   = opposite(source);

    std::array<Bitboard, PIECE_TYPE_NB> attackers{};
    for (PieceType type : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING})
        attackers[type] = pieces_on(kingLayer, us, type);

    Bitboard occupied = occupancy_on(kingLayer);
    if (m.type_of() == CASTLING)
    {
        const AliceCastlingLayout layout = alice_castling_layout(us, m);
        if (kingLayer == source)
        {
            occupied &= ~square_bb(from) & ~square_bb(layout.rookFrom);
            attackers[KING] &= ~square_bb(from);
            attackers[ROOK] &= ~square_bb(layout.rookFrom);
        }
        if (kingLayer == arrival)
        {
            occupied |= layout.kingTo | layout.rookTo;
            attackers[KING] |= layout.kingTo;
            attackers[ROOK] |= layout.rookTo;
        }
    }
    else
    {
        const Square    to         = m.to_sq();
        const PieceType moverType  = type_of(moved_piece(m));
        const PieceType resultType = m.type_of() == PROMOTION ? m.promotion_type() : moverType;
        if (kingLayer == source)
        {
            occupied &= ~square_bb(from);
            attackers[moverType] &= ~square_bb(from);
            if (!empty(to))
                occupied &= ~square_bb(to);
        }
        if (kingLayer == arrival)
        {
            occupied |= to;
            attackers[resultType] |= to;
        }
    }

    const auto [bishopAttacks, rookAttacks] = both_attacks_bb(king, occupied);
    return bool((rookAttacks & (attackers[ROOK] | attackers[QUEEN]))
                | (bishopAttacks & (attackers[BISHOP] | attackers[QUEEN]))
                | (attacks_bb<PAWN>(king, ~us) & attackers[PAWN])
                | (attacks_bb<KNIGHT>(king) & attackers[KNIGHT])
                | (attacks_bb<KING>(king) & attackers[KING]));
}


// Makes an Alice move and recomputes every derived state field from the result.
// The full recomputation is the correctness baseline for later incremental work.
void Position::do_move(Move                      m,
                       StateInfo&                newSt,
                       bool                      givesCheck,
                       Dirties&                  dirties,
                       const TranspositionTable* tt      = nullptr,
                       const SharedHistories*    history = nullptr) {
    assert(m.is_ok());
    assert(&newSt != st);
    assert(pseudo_legal(m));
    assert(legal(m));

    (void) givesCheck;
    (void) tt;
    (void) history;

    new (&dirties.dirtyThreats) DirtyThreats;
    new (&dirties.dirtyPawnPairs) DirtyPawnPairs;

    auto& dirtyPiece        = dirties.dirtyPiece;
    auto& pawnPairs         = dirties.dirtyPawnPairs;
    pawnPairs.before[WHITE] = pieces(WHITE, PAWN);
    pawnPairs.before[BLACK] = pieces(BLACK, PAWN);

    const Color  us       = sideToMove;
    const Square from     = m.from_sq();
    const Piece  mover    = piece_on(from);
    const Board  source   = board_of(from);
    const Board  arrival  = opposite(source);
    Piece        captured = NO_PIECE;

    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st             = &newSt;

    ++gamePly;
    ++st->rule50;
    ++st->pliesFromNull;
    st->epSquare = SQ_NONE;

    dirtyPiece.pc        = mover;
    dirtyPiece.from      = from;
    dirtyPiece.add_sq    = SQ_NONE;
    dirtyPiece.remove_sq = SQ_NONE;

    if (m.type_of() == CASTLING)
    {
        const AliceCastlingLayout layout = alice_castling_layout(us, m);
        dirtyPiece.to                    = layout.kingTo;
        dirtyPiece.remove_pc = dirtyPiece.add_pc = make_piece(us, ROOK);
        dirtyPiece.remove_sq                     = layout.rookFrom;
        dirtyPiece.add_sq                        = layout.rookTo;

        remove_piece(from);
        remove_piece(layout.rookFrom);
        put_piece(make_piece(us, KING), layout.kingTo, arrival);
        put_piece(make_piece(us, ROOK), layout.rookTo, arrival);
    }
    else
    {
        const Square to = m.to_sq();
        captured        = piece_on(to);
        dirtyPiece.to   = to;

        if (captured != NO_PIECE)
        {
            dirtyPiece.remove_pc = captured;
            dirtyPiece.remove_sq = to;
            remove_piece(to);
        }

        remove_piece(from);
        Piece result = mover;
        if (m.type_of() == PROMOTION)
        {
            result            = make_piece(us, m.promotion_type());
            dirtyPiece.to     = SQ_NONE;
            dirtyPiece.add_pc = result;
            dirtyPiece.add_sq = to;
        }
        put_piece(result, to, arrival);
    }

    st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[m.to_sq()]);
    if (type_of(mover) == PAWN || captured != NO_PIECE)
        st->rule50 = 0;

    sideToMove = ~sideToMove;
    set_state();
    st->capturedPiece = captured;

    st->repetition = 0;
    const int end  = std::min(st->rule50, st->pliesFromNull);
    if (end >= 4)
    {
        StateInfo* previous = st->previous->previous;
        for (int distance = 4; distance <= end; distance += 2)
        {
            previous = previous->previous->previous;
            if (previous->key == st->key)
            {
                st->repetition = previous->repetition ? -distance : distance;
                break;
            }
        }
    }

    pawnPairs.after[WHITE] = pieces(WHITE, PAWN);
    pawnPairs.after[BLACK] = pieces(BLACK, PAWN);
    assert(pos_is_ok());
}


#if 0
// Frozen orthodox transition retained only as local porting context.
void Position::do_move_orthodox_legacy(Move                      m,
                                       StateInfo&                newSt,
                                       bool                      givesCheck,
                                       Dirties&                  dirties,
                                       const TranspositionTable* tt,
                                       const SharedHistories*    history) {

    assert(m.is_ok());
    assert(&newSt != st);

    Key k = st->key ^ Zobrist::side;

    // Copy some fields of the old state to our new StateInfo object except the
    // ones which are going to be recalculated from scratch anyway and then switch
    // our state pointer to point to the new (ready to be updated) state.
    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st             = &newSt;

    // Increment ply counters. In particular, rule50 will be reset to zero later on
    // in case of a capture or a pawn move.
    ++gamePly;
    ++st->rule50;
    ++st->pliesFromNull;

    auto& dpps = dirties.dirtyPawnPairs;
    auto& dts  = dirties.dirtyThreats;
    auto& dp   = dirties.dirtyPiece;

    dpps.before[WHITE] = pieces(WHITE, PAWN);
    dpps.before[BLACK] = pieces(BLACK, PAWN);

    Color  us       = sideToMove;
    Color  them     = ~us;
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = m.type_of() == EN_PASSANT ? make_piece(them, PAWN) : piece_on(to);

    dp.pc     = pc;
    dp.from   = from;
    dp.to     = to;
    dp.add_sq = SQ_NONE;

    assert(color_of(pc) == us);
    assert(captured == NO_PIECE || color_of(captured) == (m.type_of() != CASTLING ? them : us));
    assert(type_of(captured) != KING);

    if (m.type_of() == CASTLING)
    {
        assert(pc == make_piece(us, KING));
        assert(captured == make_piece(us, ROOK));

        Square rfrom, rto;
        do_castling<true>(us, from, to, rfrom, rto, &dts, &dp);

        k ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
        st->nonPawnKey[us] ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
        captured = NO_PIECE;
    }
    else if (captured)
    {
        Square capsq = to;

        // If the captured piece is a pawn, update pawn hash key, otherwise
        // update non-pawn material.
        if (type_of(captured) == PAWN)
        {
            if (m.type_of() == EN_PASSANT)
            {
                capsq -= pawn_push(us);

                assert(pc == make_piece(us, PAWN));
                assert(to == st->epSquare);
                assert(relative_rank(us, to) == RANK_6);
                assert(piece_on(to) == NO_PIECE);
                assert(piece_on(capsq) == make_piece(them, PAWN));

                // Update board and piece lists in ep case, normal captures are updated later
                remove_piece(capsq, &dts);
            }

            st->pawnKey ^= Zobrist::psq[captured][capsq];
        }
        else
        {
            st->nonPawnMaterial[them] -= PieceValue[captured];
            st->nonPawnKey[them] ^= Zobrist::psq[captured][capsq];

            if (type_of(captured) <= BISHOP)
                st->minorPieceKey ^= Zobrist::psq[captured][capsq];
        }

        dp.remove_pc = captured;
        dp.remove_sq = capsq;

        k ^= Zobrist::psq[captured][capsq];
        st->materialKey ^=
          Zobrist::psq[captured][8 + pieceCount[captured] - (m.type_of() != EN_PASSANT)];

        // Reset rule 50 counter
        st->rule50 = 0;
    }
    else
        dp.remove_sq = SQ_NONE;

    // Update hash key
    k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

    // Reset en passant square
    if (st->epSquare != SQ_NONE)
    {
        k ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    // Update castling rights.
    k ^= Zobrist::castling[st->castlingRights];
    st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[to]);
    k ^= Zobrist::castling[st->castlingRights];

    // If the moving piece is a pawn do some special extra work
    if (type_of(pc) == PAWN)
    {
        // Check if the en passant square needs to be set. Accurate e.p. info is needed
        // for correct zobrist key generation and 3-fold checking.
        if ((int(to) ^ int(from)) == 16)
        {
            Square   epSquare = to - pawn_push(us);
            Bitboard pawns    = attacks_bb<PAWN>(epSquare, us) & pieces(them, PAWN);

            // If there are no pawns attacking the ep square, ep is not possible.
            if (pawns)
            {
                Square   ksq         = square<KING>(them);
                Bitboard notBlockers = ~st->previous->blockersForKing[them];
                bool     noDiscovery = (from & notBlockers) || file_of(from) == file_of(ksq);

                // If the pawn gives discovered check, ep is never legal. Else, if at least one
                // pawn was not a blocker for the enemy king or lies on the same line as the
                // enemy king and en passant square, a legal capture exists.
                if (noDiscovery && (pawns & (notBlockers | line_bb(epSquare, ksq))))
                {
                    st->epSquare = epSquare;
                    k ^= Zobrist::enpassant[file_of(epSquare)];
                }
            }
        }

        else if (m.type_of() == PROMOTION)
        {
            PieceType pt        = m.promotion_type();
            Piece     promotion = make_piece(us, pt);

            assert(relative_rank(us, to) == RANK_8);
            assert(pt >= KNIGHT && pt <= QUEEN);

            dp.add_pc = promotion;
            dp.add_sq = to;
            dp.to     = SQ_NONE;

            // Update hash keys
            // Zobrist::psq[pc][to] is zero, so we don't need to clear it
            k ^= Zobrist::psq[promotion][to];
            st->materialKey ^= Zobrist::psq[promotion][8 + pieceCount[promotion]]
                             ^ Zobrist::psq[pc][8 + pieceCount[pc] - 1];
            st->nonPawnKey[us] ^= Zobrist::psq[promotion][to];

            if (pt <= BISHOP)
                st->minorPieceKey ^= Zobrist::psq[promotion][to];

            // Update material
            st->nonPawnMaterial[us] += PieceValue[promotion];
        }

        // Update pawn hash key
        st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

        // Reset rule 50 draw counter
        st->rule50 = 0;
    }

    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

        if (type_of(pc) <= BISHOP)
            st->minorPieceKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    }

    if (tt)
        prefetch(tt->first_entry(adjust_key50(k)));
    // Update the key with the final value
    st->key = k;

    if (history)
    {
        prefetch(&history->pawn_entry(*this)[pc][to]);
        prefetch(&history->pawn_correction_entry(*this));
        prefetch(&history->minor_piece_correction_entry(*this));
        prefetch(&history->nonpawn_correction_entry<WHITE>(*this));
        prefetch(&history->nonpawn_correction_entry<BLACK>(*this));
    }

    // Move the piece. The tricky Chess960 castling is handled earlier
    if (m.type_of() != CASTLING)
    {
        Piece toPc = pc;
        if (m.type_of() == PROMOTION)
            toPc = make_piece(us, m.promotion_type());

        if (captured && m.type_of() != EN_PASSANT)
        {
            remove_piece(from, &dts);
            swap_piece(to, toPc, &dts);
        }
        else if (pc == toPc)
            move_piece(from, to, &dts);
        else
        {
            remove_piece(from, &dts);
            put_piece(toPc, to, &dts);
        }
    }

    // Set capture piece
    st->capturedPiece = captured;

    // Calculate checkers only on the layer occupied by the opposing king.
    const Square opposingKing  = square<KING>(them);
    const Board  opposingBoard = board_of(opposingKing);
    st->checkersBB =
      givesCheck ? attackers_to(opposingKing, opposingBoard) & pieces_on(opposingBoard, us) : 0;

    sideToMove = ~sideToMove;

    // Update king attacks used for fast check detection
    set_check_info();

    // Calculate the repetition info. It is the ply distance from the previous
    // occurrence of the same position, negative in the 3-fold case, or zero
    // if the position was not repeated.
    st->repetition = 0;
    int end        = std::min(st->rule50, st->pliesFromNull);
    if (end >= 4)
    {
        StateInfo* stp = st->previous->previous;
        for (int i = 4; i <= end; i += 2)
        {
            stp = stp->previous->previous;
            if (stp->key == st->key)
            {
                st->repetition = stp->repetition ? -i : i;
                break;
            }
        }
    }

    assert(pos_is_ok());

    dpps.after[WHITE] = pieces(WHITE, PAWN);
    dpps.after[BLACK] = pieces(BLACK, PAWN);

    assert(dp.pc != NO_PIECE);
    assert(!(bool(captured) || m.type_of() == CASTLING) ^ (dp.remove_sq != SQ_NONE));
    assert(dp.from != SQ_NONE);
    assert(!(dp.add_sq != SQ_NONE) ^ (m.type_of() == PROMOTION || m.type_of() == CASTLING));
}
#endif


// Unmakes an Alice move and restores the exact prior state object.
void Position::undo_move(Move m) {

    assert(m.is_ok());
    assert(st->previous);

    sideToMove     = ~sideToMove;
    const Color us = sideToMove;

    if (m.type_of() == CASTLING)
    {
        const AliceCastlingLayout layout  = alice_castling_layout(us, m);
        const Board               arrival = board_of(layout.kingTo);
        const Board               source  = opposite(arrival);

        assert(piece_on(layout.kingTo) == make_piece(us, KING));
        assert(piece_on(layout.rookTo) == make_piece(us, ROOK));
        remove_piece(layout.kingTo);
        remove_piece(layout.rookTo);
        put_piece(make_piece(us, KING), m.from_sq(), source);
        put_piece(make_piece(us, ROOK), layout.rookFrom, source);
    }
    else
    {
        const Square to      = m.to_sq();
        const Piece  result  = piece_on(to);
        const Board  arrival = board_of(to);
        const Board  source  = opposite(arrival);
        const Piece  mover   = m.type_of() == PROMOTION ? make_piece(us, PAWN) : result;

        remove_piece(to);
        put_piece(mover, m.from_sq(), source);
        if (st->capturedPiece != NO_PIECE)
            put_piece(st->capturedPiece, to, source);
    }

    st = st->previous;
    --gamePly;
    assert(pos_is_ok());
}


#if 0
// Frozen orthodox undo retained only as local porting context.
void Position::undo_move_orthodox_legacy(Move m) {

    assert(m.is_ok());

    sideToMove = ~sideToMove;

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = piece_on(to);

    assert(empty(from) || m.type_of() == CASTLING);
    assert(type_of(st->capturedPiece) != KING);

    if (m.type_of() == PROMOTION)
    {
        assert(relative_rank(us, to) == RANK_8);
        assert(type_of(pc) == m.promotion_type());
        assert(type_of(pc) >= KNIGHT && type_of(pc) <= QUEEN);

        pc = make_piece(us, PAWN);
        swap_piece(to, pc);
    }

    if (m.type_of() == CASTLING)
    {
        Square rfrom, rto;
        do_castling<false>(us, from, to, rfrom, rto);
    }
    else
    {
        move_piece(to, from);  // Put the piece back at the source square

        if (st->capturedPiece)
        {
            Square capsq = to;

            if (m.type_of() == EN_PASSANT)
            {
                capsq -= pawn_push(us);

                assert(type_of(pc) == PAWN);
                assert(to == st->previous->epSquare);
                assert(relative_rank(us, to) == RANK_6);
                assert(piece_on(capsq) == NO_PIECE);
                assert(st->capturedPiece == make_piece(~us, PAWN));
            }

            put_piece(st->capturedPiece, capsq);  // Restore the captured piece
        }
    }

    // Finally point our state pointer back to the previous state
    st = st->previous;
    --gamePly;

    assert(pos_is_ok());
}
#endif

inline void add_dirty_threat(DirtyThreats* const dts,
                             bool                putPiece,
                             Piece               pc,
                             Piece               threatened,
                             Square              s,
                             Square              threatenedSq) {
    dts->list.push_back({pc, threatened, s, threatenedSq, putPiece});
}

#ifdef USE_AVX512ICL
// Given a DirtyThreat template and bit offsets to insert the piece type and square, write the threats
// present at the given bitboard.
template<int SqShift, int PcShift>
void write_multiple_dirties(const Position& p,
                            Bitboard        mask,
                            DirtyThreat     dt_template,
                            DirtyThreats*   dts) {
    static_assert(sizeof(DirtyThreat) == 4);

    const __m512i board    = _mm512_loadu_si512(p.piece_array().data());
    const int     dt_count = popcount(mask);
    assert(dt_count <= 16);

    const __m512i template_v = _mm512_set1_epi32(dt_template.raw());
    auto*         write      = dts->list.make_space(dt_count);

    // Extract the list of squares and upconvert to 32 bits. There are never more than 16
    // incoming threats so this is sufficient.
    __m512i threat_squares = _mm512_maskz_compress_epi8(mask, AllSquares);
    threat_squares         = _mm512_cvtepi8_epi32(_mm512_castsi512_si128(threat_squares));

    __m512i threat_pieces =
      _mm512_maskz_permutexvar_epi8(0x1111111111111111ULL, threat_squares, board);

    // Shift the piece and square into place
    threat_squares = _mm512_slli_epi32(threat_squares, SqShift);
    threat_pieces  = _mm512_slli_epi32(threat_pieces, PcShift);

    const __m512i dirties =
      _mm512_ternarylogic_epi32(template_v, threat_squares, threat_pieces, 254 /* A | B | C */);
    _mm512_storeu_si512(write, dirties);
}
#endif

constexpr bool can_slider_threat(Piece pc, Piece slider) {
    return type_of(pc) != QUEEN || type_of(slider) == QUEEN;
}

template<bool ComputeRay>
void Position::update_piece_threats(Piece               pc,
                                    bool                putPiece,
                                    Square              s,
                                    DirtyThreats* const dts,
                                    // Silence spurious warning on GCC 10
                                    [[maybe_unused]] Bitboard noRaysContaining) const {
    const Bitboard occupied     = pieces();
    const Bitboard rookQueens   = pieces(ROOK, QUEEN);
    const Bitboard bishopQueens = pieces(BISHOP, QUEEN);
    const auto     attacks      = both_attacks_bb(s, occupied);
    const Bitboard bAttacks     = attacks.first;
    const Bitboard rAttacks     = attacks.second;
    const Bitboard occupiedNoK  = occupied ^ pieces(KING);

    Bitboard sliders       = (rookQueens & rAttacks) | (bishopQueens & bAttacks);
    Bitboard directSliders = type_of(pc) == QUEEN ? sliders & pieces(QUEEN) : sliders;

    auto process_sliders = [&](bool addDirectAttacks) {
        while (sliders)
        {
            Square sliderSq = pop_lsb(sliders);
            Piece  slider   = piece_on(sliderSq);

            const Bitboard ray        = ray_pass_bb(sliderSq, s);
            const Bitboard discovered = ray & (rAttacks | bAttacks) & occupiedNoK;

            assert(!more_than_one(discovered));
            if (discovered && (ray & noRaysContaining) != noRaysContaining)
            {
                const Square threatenedSq = lsb(discovered);
                const Piece  threatenedPc = piece_on(threatenedSq);
                if (can_slider_threat(threatenedPc, slider))
                    add_dirty_threat(dts, !putPiece, slider, threatenedPc, sliderSq, threatenedSq);
            }

            if (addDirectAttacks && can_slider_threat(pc, slider))
                add_dirty_threat(dts, putPiece, slider, pc, sliderSq, s);
        }
    };

    if (type_of(pc) == KING)
    {
        if constexpr (ComputeRay)
            process_sliders(false);
        return;
    }


    const Bitboard knights    = pieces(KNIGHT);
    const Bitboard whitePawns = pieces(WHITE, PAWN);
    const Bitboard blackPawns = pieces(BLACK, PAWN);


    Bitboard threatened       = attacks_bb(pc, s, occupied) & occupiedNoK;
    Bitboard incoming_threats = PseudoAttacks[KNIGHT][s] & knights;

    if (type_of(pc) == KNIGHT || type_of(pc) == ROOK)
        incoming_threats |=
          (attacks_bb<PAWN>(s, WHITE) & blackPawns) | (attacks_bb<PAWN>(s, BLACK) & whitePawns);

    switch (type_of(pc))
    {
    case PAWN :
        threatened &= pieces(KNIGHT, ROOK);
        break;
    case BISHOP :
    case ROOK :
        threatened &= pieces(PAWN, KNIGHT, BISHOP, ROOK);
        break;
    default :
        threatened &= occupiedNoK;
        break;
    }

#ifdef USE_AVX512ICL
    DirtyThreat dt_template{pc, NO_PIECE, s, Square(0), putPiece};
    write_multiple_dirties<DirtyThreat::ThreatenedSqOffset, DirtyThreat::ThreatenedPcOffset>(
      *this, threatened, dt_template, dts);

    Bitboard all_attackers = directSliders | incoming_threats;

    dt_template = {NO_PIECE, pc, Square(0), s, putPiece};
    write_multiple_dirties<DirtyThreat::PcSqOffset, DirtyThreat::PcOffset>(*this, all_attackers,
                                                                           dt_template, dts);
#else
    while (threatened)
    {
        Square threatenedSq = pop_lsb(threatened);
        Piece  threatenedPc = piece_on(threatenedSq);

        assert(threatenedSq != s);
        assert(threatenedPc);

        add_dirty_threat(dts, putPiece, pc, threatenedPc, s, threatenedSq);
    }
#endif

    if constexpr (ComputeRay)
    {
#ifndef USE_AVX512ICL
        process_sliders(true);
#else  // for ICL, direct threats were processed earlier (all_attackers)
        process_sliders(false);
#endif
    }
    else
    {
        incoming_threats |= directSliders;
    }

#ifndef USE_AVX512ICL
    while (incoming_threats)
    {
        Square srcSq = pop_lsb(incoming_threats);
        Piece  srcPc = piece_on(srcSq);

        assert(srcSq != s);
        assert(srcPc != NO_PIECE);

        add_dirty_threat(dts, putPiece, srcPc, pc, srcSq, s);
    }
#endif
}

Key Position::prefetch_key(Move m) const {
    (void) m;
    // Alice transitions change the board layer as well as the square. Until a
    // proven incremental predictor exists, prefetch the current bucket only.
    return key();
}

// Helper used to do/undo a castling move. This is a bit
// tricky in Chess960 where from/to squares can overlap.
template<bool Do>
void Position::do_castling(Color               us,
                           Square              from,
                           Square&             to,
                           Square&             rfrom,
                           Square&             rto,
                           DirtyThreats* const dts,
                           DirtyPiece* const   dp) {

    bool kingSide = to > from;
    rfrom         = to;  // Castling is encoded as "king captures friendly rook"
    rto           = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to            = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

    assert(!Do || dp);

    if (Do)
    {
        dp->to        = to;
        dp->remove_pc = dp->add_pc = make_piece(us, ROOK);
        dp->remove_sq              = rfrom;
        dp->add_sq                 = rto;
    }

    // Remove both pieces first since squares could overlap in Chess960
    remove_piece(Do ? from : to, dts);
    remove_piece(Do ? rfrom : rto, dts);
    put_piece(make_piece(us, KING), Do ? to : from, dts);
    put_piece(make_piece(us, ROOK), Do ? rto : rfrom, dts);
}


// Used to do a "null move": it flips
// the side to move without executing any move on the board.
void Position::do_null_move(StateInfo& newSt) {

    assert(!checkers());
    assert(&newSt != st);

    std::memcpy(&newSt, st, sizeof(StateInfo));

    newSt.previous = st;
    st             = &newSt;

    if (st->epSquare != SQ_NONE)
    {
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    st->key ^= Zobrist::side;

    st->pliesFromNull = 0;

    st->capturedPiece = NO_PIECE;

    sideToMove = ~sideToMove;

    set_check_info();

    st->repetition = 0;

    assert(pos_is_ok());
}


// Must be used to undo a "null move"
void Position::undo_null_move() {

    assert(!checkers());

    st         = st->previous;
    sideToMove = ~sideToMove;
}


// Tests if the SEE (Static Exchange Evaluation)
// value of the move is greater or equal to the given threshold. We'll use an
// algorithm similar to alpha-beta pruning with a null window.
bool Position::see_ge(Move m, int threshold) const {

    (void) m;
    (void) threshold;
    // Classical SEE cannot model recaptures that transfer between boards. Its
    // pruning consumers must treat every candidate as admissible until alice_see
    // replaces this conservative gate.
    return true;

#if 0
    assert(m.is_ok());

    // Only deal with normal moves, assume others pass a simple SEE
    if (m.type_of() != NORMAL)
        return VALUE_ZERO >= threshold;

    Square from = m.from_sq(), to = m.to_sq();

    assert(piece_on(from) != NO_PIECE);

    int swap = PieceValue[piece_on(to)] - threshold;
    if (swap < 0)
        return false;

    swap = PieceValue[piece_on(from)] - swap;
    if (swap <= 0)
        return true;

    assert(color_of(piece_on(from)) == sideToMove);
    Bitboard occupied  = pieces() ^ from ^ to;  // xoring to is important for pinned piece logic
    Color    stm       = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);
    Bitboard stmAttackers, bb;
    int      res = 1;

    while (true)
    {
        stm = ~stm;
        attackers &= occupied;

        // If stm has no more attackers then give up: stm loses
        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        // Don't allow pinned pieces to attack as long as there are
        // pinners on their original square.
        if (pinners(~stm) & occupied)
        {
            stmAttackers &= ~blockers_for_king(stm);

            if (!stmAttackers)
                break;
        }

        res ^= 1;

        // Locate and remove the next least valuable attacker, and add to
        // the bitboard 'attackers' any X-ray attackers behind it.
        if ((bb = stmAttackers & pieces(PAWN)))
        {
            if ((swap = PawnValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(KNIGHT)))
        {
            if ((swap = KnightValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }

        else if ((bb = stmAttackers & pieces(BISHOP)))
        {
            if ((swap = BishopValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(ROOK)))
        {
            if ((swap = RookValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(QUEEN)))
        {
            swap = QueenValue - swap;
            //  implies that the previous recapture was done by a higher rated piece than a Queen (King is excluded)
            assert(swap >= res);
            occupied ^= least_significant_square_bb(bb);

            const auto [bishopAttacks, rookAttacks] = both_attacks_bb(to, occupied);
            attackers |=
              (bishopAttacks & pieces(BISHOP, QUEEN)) | (rookAttacks & pieces(ROOK, QUEEN));
        }

        else  // KING
              // If we "capture" with the king but the opponent still has attackers,
              // reverse the result.
            return (attackers & ~pieces(stm)) ? res ^ 1 : res;
    }

    return bool(res);
#endif
}

// Tests whether the position is drawn by 50-move rule
// or by repetition. It does not detect stalemates.
bool Position::is_draw(int ply) const {

    if (st->rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size()))
        return true;

    return is_repetition(ply);
}

// Return a draw score if a position repeats once earlier but strictly
// after the root, or repeats twice before or at the root.
bool Position::is_repetition(int ply) const { return st->repetition && st->repetition < ply; }

// Tests whether there has been at least one repetition
// of positions since the last capture or pawn move.
bool Position::has_repeated() const {

    StateInfo* stc = st;
    int        end = std::min(st->rule50, st->pliesFromNull);
    while (end-- >= 4)
    {
        if (stc->repetition)
            return true;

        stc = stc->previous;
    }
    return false;
}


// Tests if the position has a move which draws by repetition.
// This function accurately matches the outcome of is_draw() over all legal moves.
bool Position::upcoming_repetition(int ply) const {

    (void) ply;
    // The classical cuckoo table encodes same-board reversible moves. Exact
    // StateInfo key comparison remains active; only the speculative shortcut is off.
    return false;

#if 0
    int j;

    int end = std::min(st->rule50, st->pliesFromNull);

    if (end < 3)
        return false;

    Key        originalKey = st->key;
    StateInfo* stp         = st->previous;
    Key        other       = originalKey ^ stp->key ^ Zobrist::side;

    for (int i = 3; i <= end; i += 2)
    {
        stp = stp->previous;
        other ^= stp->key ^ stp->previous->key ^ Zobrist::side;
        stp = stp->previous;

        if (other != 0)
            continue;

        Key moveKey = originalKey ^ stp->key;
        if ((j = H1(moveKey), cuckoo[j] == moveKey) || (j = H2(moveKey), cuckoo[j] == moveKey))
        {
            Move   move = cuckooMove[j];
            Square s1   = move.from_sq();
            Square s2   = move.to_sq();

            if (!((between_bb(s1, s2) ^ s2) & pieces()))
            {
                if (ply > i)
                    return true;

                // For nodes before or at the root, check that the move is a
                // repetition rather than a move to the current position.
                if (stp->repetition)
                    return true;
            }
        }
    }
    return false;
#endif
}


// Flips position with the white and black sides reversed. This
// is only useful for debugging e.g. for finding evaluation symmetry bugs.
std::optional<PositionSetError> Position::flip() {

    string            f, token;
    std::stringstream ss(fen());

    for (Rank r = RANK_8;; --r)  // Piece placement
    {
        std::getline(ss, token, r > RANK_1 ? '/' : ' ');
        f.insert(0, token + (f.empty() ? " " : "/"));

        if (r == RANK_1)
            break;
    }

    ss >> token;                        // Active color
    f += (token == "w" ? "B " : "W ");  // Will be lowercased later

    ss >> token;  // Castling availability
    f += token + " ";

    std::transform(f.begin(), f.end(), f.begin(),
                   [](char c) { return char(islower(c) ? toupper(c) : tolower(c)); });

    ss >> token;  // En passant square
    f += (token == "-" ? token : token.replace(1, 1, token[1] == '3' ? "6" : "3"));

    std::getline(ss, token);  // Half and full moves
    f += token;

    return set(f, is_chess960(), st);
}


bool Position::material_key_is_ok() const { return compute_material_key() == st->materialKey; }


// Performs some consistency checks for the position object
// and raise an assert if something wrong is detected.
// This is meant to be helpful when debugging.
bool Position::pos_is_ok() const {

    if (st->boardB & ~pieces())
        assert(0 && "pos_is_ok: boardB contains empty coordinates");

    if ((sideToMove != WHITE && sideToMove != BLACK) || piece_on(square<KING>(WHITE)) != W_KING
        || piece_on(square<KING>(BLACK)) != B_KING
        || (ep_square() != SQ_NONE && relative_rank(sideToMove, ep_square()) != RANK_6))
        assert(0 && "pos_is_ok: Default");

    const Square opposingKing  = square<KING>(~sideToMove);
    const Board  opposingBoard = board_of(opposingKing);
    if (count<KING>(WHITE) != 1 || count<KING>(BLACK) != 1
        || attackers_to_exist(opposingKing, opposingBoard, occupancy_on(opposingBoard), sideToMove))
        assert(0 && "pos_is_ok: Kings");

    if ((pieces(PAWN) & (Rank1BB | Rank8BB)) || count<PAWN>(WHITE) > 8 || count<PAWN>(BLACK) > 8)
        assert(0 && "pos_is_ok: Pawns");


    if (ep_square() != SQ_NONE)
    {
        Square ksq = square<KING>(sideToMove);

        Bitboard captured = (ep_square() + pawn_push(~sideToMove)) & pieces(~sideToMove, PAWN);
        Bitboard pawns    = attacks_bb<PAWN>(ep_square(), ~sideToMove) & pieces(sideToMove, PAWN);
        Bitboard potentialCheckers = pieces(~sideToMove) ^ captured;

        if (!captured || !pawns
            || ((attackers_to(ksq, pieces() ^ captured ^ ep_square() ^ lsb(pawns))
                 & potentialCheckers)
                && (attackers_to(ksq, pieces() ^ captured ^ ep_square() ^ msb(pawns))
                    & potentialCheckers)))
            assert(0 && "pos_is_ok: En passant square");
    }

    if ((pieces(WHITE) & pieces(BLACK)) || (pieces(WHITE) | pieces(BLACK)) != pieces()
        || popcount(pieces(WHITE)) > 16 || popcount(pieces(BLACK)) > 16)
        assert(0 && "pos_is_ok: Bitboards");

    for (PieceType p1 = PAWN; p1 <= KING; ++p1)
        for (PieceType p2 = PAWN; p2 <= KING; ++p2)
            if (p1 != p2 && (pieces(p1) & pieces(p2)))
                assert(0 && "pos_is_ok: Bitboards");

    for (Piece pc : Pieces)
        if (pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc)))
            || pieceCount[pc] != std::count(board.begin(), board.end(), pc))
            assert(0 && "pos_is_ok: Pieces");

    for (Color c : {WHITE, BLACK})
        for (CastlingRights cr : {c & KING_SIDE, c & QUEEN_SIDE})
        {
            if (!can_castle(cr))
                continue;

            if (piece_on(castling_rook_square(cr)) != make_piece(c, ROOK)
                || castlingRightsMask[castlingRookSquare[cr]] != cr
                || (castlingRightsMask[square<KING>(c)] & cr) != cr)
                assert(0 && "pos_is_ok: Castling");
        }

    assert(material_key_is_ok() && "pos_is_ok: materialKey");

    return true;
}

}  // namespace Stockfish
