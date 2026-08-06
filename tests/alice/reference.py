"""Slow, specification-first Alice Chess reference implementation.

This module intentionally favors clarity over speed.  It is independent from
the engine's bitboards and search code and exists to validate rules, FEN
normalization, legal move sets, and perft results.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable, Iterator


FILES = "abcdefgh"
PIECE_LETTERS = frozenset("PNBRQKpnbrqk")
PROMOTIONS = ("q", "r", "b", "n")
LAYERS = ("A", "B")
UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


class FenError(ValueError):
    """Raised when an Alice FEN violates the public position contract."""


class MoveResolutionError(ValueError):
    """Raised when a UCI string does not identify exactly one legal move."""


def opposite_layer(layer: str) -> str:
    return "B" if layer == "A" else "A"


def piece_color(symbol: str) -> str:
    return "w" if symbol.isupper() else "b"


def square_name(square: int) -> str:
    return FILES[square % 8] + str(square // 8 + 1)


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] not in FILES or name[1] not in "12345678":
        raise ValueError(f"Invalid square: {name}")
    return FILES.index(name[0]) + 8 * (int(name[1]) - 1)


@dataclass(frozen=True)
class Piece:
    symbol: str
    layer: str

    def __post_init__(self) -> None:
        if self.symbol not in PIECE_LETTERS:
            raise ValueError(f"Invalid piece: {self.symbol}")
        if self.layer not in LAYERS:
            raise ValueError(f"Invalid layer: {self.layer}")

    @property
    def color(self) -> str:
        return piece_color(self.symbol)

    @property
    def kind(self) -> str:
        return self.symbol.upper()


@dataclass(frozen=True)
class Move:
    from_square: int
    to_square: int
    promotion: str | None = None
    castling: bool = False

    def uci(self) -> str:
        return square_name(self.from_square) + square_name(self.to_square) + (self.promotion or "")


def _consume_number(text: str, start: int) -> tuple[int, int]:
    end = start
    while end < len(text) and text[end].isdigit():
        end += 1
    if text[start] == "0":
        raise FenError("Empty-square runs cannot contain a leading zero.")
    value = int(text[start:end])
    if value < 1:
        raise FenError("Empty-square runs must be positive.")
    return value, end


def _expand_compact_rank(text: str) -> list[Piece | None]:
    cells: list[Piece | None] = []
    index = 0
    layer_b = False

    while index < len(text):
        token = text[index]
        if token == "|":
            # The frozen book's first position contains one redundant marker
            # after a rank that is already complete. The historical parser
            # ignored it and canonical output never emits it.
            if len(cells) == 8 and index + 1 == len(text):
                index += 1
                continue
            if layer_b or index + 1 >= len(text) or text[index + 1] not in PIECE_LETTERS:
                raise FenError("The compact layer marker must immediately precede a piece.")
            layer_b = True
            index += 1
            continue

        if token.isdigit():
            if layer_b:
                raise FenError("A layer marker cannot precede an empty-square run.")
            run, index = _consume_number(text, index)
            if run > 8:
                raise FenError("A compact empty-square run cannot exceed eight squares.")
            cells.extend([None] * run)
            continue

        if token not in PIECE_LETTERS:
            raise FenError(f"Invalid placement token: {token}")
        cells.append(Piece(token, "B" if layer_b else "A"))
        layer_b = False
        index += 1

    if layer_b:
        raise FenError("A compact layer marker cannot terminate a rank.")
    if len(cells) != 8:
        raise FenError(f"A compact rank must expand to eight coordinates, got {len(cells)}.")
    return cells


def _expand_unmarked_rank(text: str) -> list[str | None]:
    cells: list[str | None] = []
    index = 0
    while index < len(text):
        token = text[index]
        if token.isdigit():
            run, index = _consume_number(text, index)
            if run > 16:
                raise FenError("A legacy empty-square run cannot exceed sixteen cells.")
            cells.extend([None] * run)
            continue
        if token not in PIECE_LETTERS:
            raise FenError(f"Invalid placement token: {token}")
        cells.append(token)
        index += 1
    return cells


class Position:
    """A complete Alice position using one coordinate slot plus a piece layer."""

    def __init__(
        self,
        board: list[Piece | None],
        side_to_move: str,
        castling_rights: Iterable[str],
        halfmove_clock: int,
        fullmove_number: int,
    ) -> None:
        if len(board) != 64:
            raise ValueError("The board must contain exactly 64 shared coordinates.")
        self.board = board
        self.side_to_move = side_to_move
        self.castling_rights = set(castling_rights)
        self.halfmove_clock = halfmove_clock
        self.fullmove_number = fullmove_number

    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        fields = fen.split()
        if len(fields) != 6:
            raise FenError("Alice FEN must contain exactly six fields.")

        placement, side, castling, ep, halfmove, fullmove = fields
        ranks = placement.split("/")
        if len(ranks) != 8:
            raise FenError("Piece placement must contain exactly eight ranks.")

        board: list[Piece | None] = [None] * 64
        compact_marked = "|" in placement

        if compact_marked:
            expanded: list[list[Piece | None]] = [_expand_compact_rank(rank) for rank in ranks]
        else:
            raw = [_expand_unmarked_rank(rank) for rank in ranks]
            widths = {len(rank) for rank in raw}
            if len(widths) != 1 or next(iter(widths), 0) not in (8, 16):
                raise FenError("All unmarked ranks must consistently expand to eight or sixteen cells.")

            width = next(iter(widths))
            expanded = []
            for rank in raw:
                if width == 8:
                    expanded.append([Piece(symbol, "A") if symbol else None for symbol in rank])
                    continue

                folded: list[Piece | None] = [None] * 8
                for file_index in range(8):
                    left = rank[file_index]
                    right = rank[file_index + 8]
                    if left and right:
                        raise FenError(
                            f"Layers A and B both occupy file {FILES[file_index]} on one rank."
                        )
                    if left:
                        folded[file_index] = Piece(left, "A")
                    elif right:
                        folded[file_index] = Piece(right, "B")
                expanded.append(folded)

        for fen_rank, cells in enumerate(expanded):
            rank_index = 7 - fen_rank
            for file_index, piece in enumerate(cells):
                board[file_index + 8 * rank_index] = piece

        if side not in ("w", "b"):
            raise FenError("The active color must be w or b.")
        if ep != "-":
            raise FenError("The en-passant field must be -.")

        if castling == "-":
            rights: set[str] = set()
        else:
            if any(right not in "KQkq" for right in castling) or len(set(castling)) != len(castling):
                raise FenError("Castling rights must be unique KQkq symbols or -.")
            rights = set(castling)

        try:
            halfmove_value = int(halfmove)
            fullmove_value = int(fullmove)
        except ValueError as exc:
            raise FenError("Move counters must be decimal integers.") from exc
        if halfmove_value < 0:
            raise FenError("The halfmove clock cannot be negative.")
        if fullmove_value < 1:
            raise FenError("The fullmove number must be at least one.")

        position = cls(board, side, rights, halfmove_value, fullmove_value)
        position._validate_material()
        position._validate_castling_rights()
        return position

    def clone(self) -> "Position":
        return Position(
            list(self.board),
            self.side_to_move,
            set(self.castling_rights),
            self.halfmove_clock,
            self.fullmove_number,
        )

    def _validate_material(self) -> None:
        symbols = [piece.symbol for piece in self.board if piece]
        if symbols.count("K") != 1 or symbols.count("k") != 1:
            raise FenError("A position must contain exactly one king of each color.")
        if len(symbols) > 32:
            raise FenError("A position cannot contain more than 32 pieces.")

        for color in ("w", "b"):
            color_symbols = [symbol for symbol in symbols if piece_color(symbol) == color]
            if len(color_symbols) > 16:
                raise FenError("A color cannot contain more than 16 pieces.")
            pawn = "P" if color == "w" else "p"
            pawn_count = color_symbols.count(pawn)
            if pawn_count > 8:
                raise FenError("A color cannot contain more than eight pawns.")
            normalized = [symbol.upper() for symbol in color_symbols]
            promoted_surplus = (
                max(normalized.count("N") - 2, 0)
                + max(normalized.count("B") - 2, 0)
                + max(normalized.count("R") - 2, 0)
                + max(normalized.count("Q") - 1, 0)
            )
            if promoted_surplus > 8 - pawn_count:
                raise FenError("Promoted material exceeds the number of missing pawns.")

        for square, piece in enumerate(self.board):
            if piece and piece.kind == "P" and square // 8 in (0, 7):
                raise FenError("Unpromoted pawns cannot occupy the first or eighth rank.")

    def _validate_castling_rights(self) -> None:
        specifications = {
            "K": ("K", "R", "e1", "h1"),
            "Q": ("K", "R", "e1", "a1"),
            "k": ("k", "r", "e8", "h8"),
            "q": ("k", "r", "e8", "a8"),
        }
        for right in self.castling_rights:
            king_symbol, rook_symbol, king_name, rook_name = specifications[right]
            king = self.board[parse_square(king_name)]
            rook = self.board[parse_square(rook_name)]
            if not king or not rook or king.symbol != king_symbol or rook.symbol != rook_symbol:
                raise FenError(f"Castling right {right} lacks its orthodox king or rook.")
            if king.layer != rook.layer:
                raise FenError(f"Castling right {right} requires king and rook on the same layer.")

    def piece_at(self, layer: str, square: int) -> Piece | None:
        piece = self.board[square]
        return piece if piece and piece.layer == layer else None

    def occupancy(self, layer: str) -> set[int]:
        return {square for square, piece in enumerate(self.board) if piece and piece.layer == layer}

    def king_square(self, color: str) -> int:
        symbol = "K" if color == "w" else "k"
        for square, piece in enumerate(self.board):
            if piece and piece.symbol == symbol:
                return square
        raise AssertionError("Validated positions always contain both kings.")

    def _ray_attacks(self, source: int, target: int, layer: str, directions: Iterable[tuple[int, int]]) -> bool:
        source_file, source_rank = source % 8, source // 8
        target_file, target_rank = target % 8, target // 8
        for file_step, rank_step in directions:
            file_index = source_file + file_step
            rank_index = source_rank + rank_step
            while 0 <= file_index < 8 and 0 <= rank_index < 8:
                square = file_index + 8 * rank_index
                if square == target:
                    return True
                if self.piece_at(layer, square):
                    break
                file_index += file_step
                rank_index += rank_step
        return False

    def _piece_attacks(self, source: int, piece: Piece, target: int) -> bool:
        source_file, source_rank = source % 8, source // 8
        target_file, target_rank = target % 8, target // 8
        file_delta = target_file - source_file
        rank_delta = target_rank - source_rank

        if piece.kind == "P":
            forward = 1 if piece.color == "w" else -1
            return rank_delta == forward and abs(file_delta) == 1
        if piece.kind == "N":
            return (abs(file_delta), abs(rank_delta)) in ((1, 2), (2, 1))
        if piece.kind == "K":
            return max(abs(file_delta), abs(rank_delta)) == 1
        if piece.kind == "B":
            return self._ray_attacks(source, target, piece.layer, ((1, 1), (1, -1), (-1, 1), (-1, -1)))
        if piece.kind == "R":
            return self._ray_attacks(source, target, piece.layer, ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if piece.kind == "Q":
            return self._ray_attacks(
                source,
                target,
                piece.layer,
                ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)),
            )
        raise AssertionError(f"Unsupported piece kind: {piece.kind}")

    def attackers(self, square: int, layer: str, by_color: str) -> list[int]:
        return [
            source
            for source, piece in enumerate(self.board)
            if piece
            and piece.layer == layer
            and piece.color == by_color
            and self._piece_attacks(source, piece, square)
        ]

    def in_check(self, color: str) -> bool:
        king_square = self.king_square(color)
        king = self.board[king_square]
        assert king is not None
        return bool(self.attackers(king_square, king.layer, "b" if color == "w" else "w"))

    def _slider_moves(self, source: int, piece: Piece, directions: Iterable[tuple[int, int]]) -> Iterator[Move]:
        source_file, source_rank = source % 8, source // 8
        for file_step, rank_step in directions:
            file_index = source_file + file_step
            rank_index = source_rank + rank_step
            while 0 <= file_index < 8 and 0 <= rank_index < 8:
                target = file_index + 8 * rank_index
                occupant = self.board[target]
                if occupant and occupant.layer == piece.layer:
                    if occupant.color != piece.color and occupant.kind != "K":
                        yield Move(source, target)
                    break
                if occupant is None:
                    yield Move(source, target)
                # A piece on the other layer does not block the ray, but it does
                # make this coordinate an invalid transfer destination.
                file_index += file_step
                rank_index += rank_step

    def _step_moves(self, source: int, piece: Piece, deltas: Iterable[tuple[int, int]]) -> Iterator[Move]:
        source_file, source_rank = source % 8, source // 8
        for file_step, rank_step in deltas:
            file_index = source_file + file_step
            rank_index = source_rank + rank_step
            if not (0 <= file_index < 8 and 0 <= rank_index < 8):
                continue
            target = file_index + 8 * rank_index
            occupant = self.board[target]
            if occupant is None:
                yield Move(source, target)
            elif occupant.layer == piece.layer and occupant.color != piece.color and occupant.kind != "K":
                yield Move(source, target)

    def _pawn_moves(self, source: int, piece: Piece) -> Iterator[Move]:
        source_file, source_rank = source % 8, source // 8
        rank_step = 1 if piece.color == "w" else -1
        start_rank = 1 if piece.color == "w" else 6
        promotion_rank = 7 if piece.color == "w" else 0

        one_rank = source_rank + rank_step
        if 0 <= one_rank < 8:
            one = source_file + 8 * one_rank
            if self.piece_at(piece.layer, one) is None:
                if self.board[one] is None:
                    if one_rank == promotion_rank:
                        for promotion in PROMOTIONS:
                            yield Move(source, one, promotion)
                    else:
                        yield Move(source, one)

                two_rank = source_rank + 2 * rank_step
                if source_rank == start_rank and 0 <= two_rank < 8:
                    two = source_file + 8 * two_rank
                    if self.piece_at(piece.layer, two) is None and self.board[two] is None:
                        yield Move(source, two)

        target_rank = source_rank + rank_step
        if not 0 <= target_rank < 8:
            return
        for file_step in (-1, 1):
            target_file = source_file + file_step
            if not 0 <= target_file < 8:
                continue
            target = target_file + 8 * target_rank
            victim = self.piece_at(piece.layer, target)
            if not victim or victim.color == piece.color or victim.kind == "K":
                continue
            if target_rank == promotion_rank:
                for promotion in PROMOTIONS:
                    yield Move(source, target, promotion)
            else:
                yield Move(source, target)

    def _castling_candidates(self, color: str) -> Iterator[Move]:
        rank = 0 if color == "w" else 7
        king_square = 4 + 8 * rank
        king = self.board[king_square]
        if not king or king.color != color or king.kind != "K":
            return

        definitions = (
            ("K" if color == "w" else "k", 7, 6, (5, 6), (5, 6)),
            ("Q" if color == "w" else "q", 0, 2, (1, 2, 3), (2, 3)),
        )
        for right, rook_file, king_to_file, source_clear_files, arrival_files in definitions:
            if right not in self.castling_rights:
                continue
            rook_square = rook_file + 8 * rank
            rook = self.board[rook_square]
            if not rook or rook.color != color or rook.kind != "R" or rook.layer != king.layer:
                continue
            if any(self.piece_at(king.layer, file_index + 8 * rank) for file_index in source_clear_files):
                continue
            if any(self.board[file_index + 8 * rank] is not None for file_index in arrival_files):
                continue
            yield Move(king_square, king_to_file + 8 * rank, castling=True)

    def pseudo_legal_moves(self) -> Iterator[Move]:
        color = self.side_to_move
        for source, piece in enumerate(self.board):
            if not piece or piece.color != color:
                continue
            if piece.kind == "P":
                yield from self._pawn_moves(source, piece)
            elif piece.kind == "N":
                yield from self._step_moves(
                    source, piece, ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
                )
            elif piece.kind == "B":
                yield from self._slider_moves(source, piece, ((1, 1), (1, -1), (-1, 1), (-1, -1)))
            elif piece.kind == "R":
                yield from self._slider_moves(source, piece, ((1, 0), (-1, 0), (0, 1), (0, -1)))
            elif piece.kind == "Q":
                yield from self._slider_moves(
                    source,
                    piece,
                    ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)),
                )
            else:
                yield from self._step_moves(
                    source, piece, ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1))
                )
        yield from self._castling_candidates(color)

    def _ordinary_provisional(self, move: Move) -> "Position":
        result = self.clone()
        mover = result.board[move.from_square]
        assert mover is not None
        victim = result.board[move.to_square]
        if victim:
            assert victim.layer == mover.layer and victim.color != mover.color
        result.board[move.from_square] = None
        promoted_symbol = mover.symbol
        if move.promotion:
            promoted_symbol = move.promotion.upper() if mover.color == "w" else move.promotion
        result.board[move.to_square] = Piece(promoted_symbol, mover.layer)
        return result

    def _castle_provisional(self, move: Move) -> tuple["Position", int, int, str]:
        result = self.clone()
        king = result.board[move.from_square]
        assert king is not None and king.kind == "K"
        rank = move.from_square // 8
        king_side = move.to_square % 8 == 6
        rook_from = (7 if king_side else 0) + 8 * rank
        rook_to = (5 if king_side else 3) + 8 * rank
        rook = result.board[rook_from]
        assert rook is not None and rook.kind == "R" and rook.layer == king.layer
        result.board[move.from_square] = None
        result.board[rook_from] = None
        result.board[move.to_square] = king
        result.board[rook_to] = rook
        return result, rook_from, rook_to, king.layer

    def _is_legal(self, move: Move) -> bool:
        mover = self.board[move.from_square]
        if not mover or mover.color != self.side_to_move:
            return False

        if move.castling:
            if self.in_check(mover.color):
                return False
            rank = move.from_square // 8
            transit = (5 if move.to_square % 8 == 6 else 3) + 8 * rank
            transit_position = self.clone()
            transit_position.board[move.from_square] = None
            transit_position.board[transit] = mover
            if transit_position.in_check(mover.color):
                return False

            provisional, _rook_from, rook_to, source_layer = self._castle_provisional(move)
            if provisional.in_check(mover.color):
                return False
            final = provisional.clone()
            king = final.board[move.to_square]
            rook = final.board[rook_to]
            assert king is not None and rook is not None
            final.board[move.to_square] = Piece(king.symbol, opposite_layer(source_layer))
            final.board[rook_to] = Piece(rook.symbol, opposite_layer(source_layer))
            return not final.in_check(mover.color)

        provisional = self._ordinary_provisional(move)
        king_square = provisional.king_square(mover.color)
        king = provisional.board[king_square]
        assert king is not None
        if king.layer == mover.layer and provisional.in_check(mover.color):
            return False

        final = provisional.clone()
        moved = final.board[move.to_square]
        assert moved is not None
        final.board[move.to_square] = Piece(moved.symbol, opposite_layer(mover.layer))
        return not final.in_check(mover.color)

    def legal_moves(self) -> list[Move]:
        return [move for move in self.pseudo_legal_moves() if self._is_legal(move)]

    def resolve_uci(self, text: str) -> Move:
        if not UCI_RE.fullmatch(text):
            raise MoveResolutionError(f"Invalid UCI move syntax: {text}")
        matches = [move for move in self.legal_moves() if move.uci() == text]
        if len(matches) != 1:
            raise MoveResolutionError(
                f"UCI move {text} identifies {len(matches)} legal moves; exactly one is required."
            )
        return matches[0]

    def is_uci_legal(self, text: str) -> bool:
        try:
            self.resolve_uci(text)
        except MoveResolutionError:
            return False
        return True

    def _clear_castling_for_move(self, result: "Position", move: Move, mover: Piece, victim: Piece | None) -> None:
        if mover.kind == "K":
            result.castling_rights -= {"K", "Q"} if mover.color == "w" else {"k", "q"}

        rook_rights = {
            (parse_square("h1"), "w"): "K",
            (parse_square("a1"), "w"): "Q",
            (parse_square("h8"), "b"): "k",
            (parse_square("a8"), "b"): "q",
        }
        right = rook_rights.get((move.from_square, mover.color))
        if mover.kind == "R" and right:
            result.castling_rights.discard(right)

        if victim and victim.kind == "R":
            captured_right = rook_rights.get((move.to_square, victim.color))
            if captured_right:
                result.castling_rights.discard(captured_right)

    def after(self, move: Move) -> "Position":
        if move not in self.legal_moves():
            raise MoveResolutionError(f"Move is not legal: {move.uci()}")

        mover = self.board[move.from_square]
        assert mover is not None
        victim = None if move.castling else self.board[move.to_square]

        if move.castling:
            result, _rook_from, rook_to, source_layer = self._castle_provisional(move)
            king = result.board[move.to_square]
            rook = result.board[rook_to]
            assert king is not None and rook is not None
            result.board[move.to_square] = Piece(king.symbol, opposite_layer(source_layer))
            result.board[rook_to] = Piece(rook.symbol, opposite_layer(source_layer))
        else:
            result = self._ordinary_provisional(move)
            moved = result.board[move.to_square]
            assert moved is not None
            result.board[move.to_square] = Piece(moved.symbol, opposite_layer(mover.layer))

        self._clear_castling_for_move(result, move, mover, victim)
        result.halfmove_clock = 0 if mover.kind == "P" or victim else self.halfmove_clock + 1
        result.fullmove_number = self.fullmove_number + (1 if self.side_to_move == "b" else 0)
        result.side_to_move = "b" if self.side_to_move == "w" else "w"
        return result

    def push_uci(self, text: str) -> "Position":
        return self.after(self.resolve_uci(text))

    def identity(self) -> tuple[object, ...]:
        pieces = tuple(
            (square, piece.symbol, piece.layer)
            for square, piece in enumerate(self.board)
            if piece is not None
        )
        return pieces, self.side_to_move, tuple(sorted(self.castling_rights))

    def pawn_identity(self) -> tuple[tuple[int, str, str], ...]:
        return tuple(
            (square, piece.symbol, piece.layer)
            for square, piece in enumerate(self.board)
            if piece is not None and piece.kind == "P"
        )

    def minor_piece_identity(self) -> tuple[tuple[int, str, str], ...]:
        return tuple(
            (square, piece.symbol, piece.layer)
            for square, piece in enumerate(self.board)
            if piece is not None and piece.kind in ("N", "B")
        )

    def non_pawn_identity(self, color: str) -> tuple[tuple[int, str, str], ...]:
        return tuple(
            (square, piece.symbol, piece.layer)
            for square, piece in enumerate(self.board)
            if piece is not None and piece.color == color and piece.kind != "P"
        )

    def material_identity(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(Counter(piece.symbol for piece in self.board if piece).items()))

    def fen(self) -> str:
        rank_texts: list[str] = []
        for rank in range(7, -1, -1):
            fields: list[str] = []
            empty_count = 0
            for file_index in range(8):
                piece = self.board[file_index + 8 * rank]
                if piece is None:
                    empty_count += 1
                    continue
                if empty_count:
                    fields.append(str(empty_count))
                    empty_count = 0
                fields.append(("|" if piece.layer == "B" else "") + piece.symbol)
            if empty_count:
                fields.append(str(empty_count))
            rank_texts.append("".join(fields))

        castling = "".join(right for right in "KQkq" if right in self.castling_rights) or "-"
        return (
            "/".join(rank_texts)
            + f" {self.side_to_move} {castling} - {self.halfmove_clock} {self.fullmove_number}"
        )


def perft(position: Position, depth: int) -> int:
    """Count legal leaf nodes without caching or engine shortcuts."""

    if depth < 0:
        raise ValueError("Perft depth cannot be negative.")
    if depth == 0:
        return 1
    return sum(perft(position.after(move), depth - 1) for move in position.legal_moves())
