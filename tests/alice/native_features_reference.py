"""Independent scalar reference for the Alice-native v1 sparse features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from reference import FILES, Piece, Position, square_name


WHITE = 0
BLACK = 1
PIECE_SQUARE_DIMENSIONS = 45_056
BASE_THREAT_DIMENSIONS = 59_808
THREAT_DIMENSIONS = 119_616

PIECE_CODE = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 9,
    "n": 10,
    "b": 11,
    "r": 12,
    "q": 13,
    "k": 14,
}

PIECE_NAME = {
    code: ("w" if symbol.isupper() else "b") + symbol.upper()
    for symbol, code in PIECE_CODE.items()
}

KING_BUCKETS = (
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
    8, 9, 10, 11, 11, 10, 9, 8,
    4, 5, 6, 7, 7, 6, 5, 4,
    0, 1, 2, 3, 3, 2, 1, 0,
)

NUM_VALID_TARGETS = (0, 4, 10, 8, 8, 10, 0, 0, 0, 4, 10, 8, 8, 10, 0, 0)
TARGET_MAP = (
    (-1, 0, -1, 1, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (-1, -1, -1, -1, -1, -1),
)
ALL_PIECES = (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14)

KNIGHT_DELTAS = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)
BISHOP_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ROOK_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _piece_type(piece: int) -> int:
    return piece & 7


def _piece_color(piece: int) -> int:
    return BLACK if piece >= 8 else WHITE


def _inside(file_index: int, rank_index: int) -> bool:
    return 0 <= file_index < 8 and 0 <= rank_index < 8


def _jump_attacks(square: int, deltas: Iterable[tuple[int, int]]) -> set[int]:
    file_index, rank_index = square % 8, square // 8
    result: set[int] = set()
    for file_delta, rank_delta in deltas:
        target_file = file_index + file_delta
        target_rank = rank_index + rank_delta
        if _inside(target_file, target_rank):
            result.add(target_file + 8 * target_rank)
    return result


def _ray_attacks(square: int, directions: Iterable[tuple[int, int]]) -> set[int]:
    result: set[int] = set()
    for file_delta, rank_delta in directions:
        file_index = square % 8 + file_delta
        rank_index = square // 8 + rank_delta
        while _inside(file_index, rank_index):
            result.add(file_index + 8 * rank_index)
            file_index += file_delta
            rank_index += rank_delta
    return result


def pseudo_attacks(piece: int, square: int) -> set[int]:
    piece_type = _piece_type(piece)
    if piece_type == 1:
        rank_delta = 1 if _piece_color(piece) == WHITE else -1
        return _jump_attacks(square, ((-1, rank_delta), (1, rank_delta)))
    if piece_type == 2:
        return _jump_attacks(square, KNIGHT_DELTAS)
    if piece_type == 3:
        return _ray_attacks(square, BISHOP_DIRECTIONS)
    if piece_type == 4:
        return _ray_attacks(square, ROOK_DIRECTIONS)
    if piece_type == 5:
        return _ray_attacks(square, BISHOP_DIRECTIONS + ROOK_DIRECTIONS)
    if piece_type == 6:
        return _jump_attacks(square, BISHOP_DIRECTIONS + ROOK_DIRECTIONS)
    return set()


def occupied_attacks(piece: int, square: int, occupied: set[int]) -> set[int]:
    piece_type = _piece_type(piece)
    if piece_type in (1, 2):
        return pseudo_attacks(piece, square)

    if piece_type == 3:
        directions = BISHOP_DIRECTIONS
    elif piece_type == 4:
        directions = ROOK_DIRECTIONS
    elif piece_type == 5:
        directions = BISHOP_DIRECTIONS + ROOK_DIRECTIONS
    else:
        return set()

    result: set[int] = set()
    for file_delta, rank_delta in directions:
        file_index = square % 8 + file_delta
        rank_index = square // 8 + rank_delta
        while _inside(file_index, rank_index):
            target = file_index + 8 * rank_index
            result.add(target)
            if target in occupied:
                break
            file_index += file_delta
            rank_index += rank_delta
    return result


@dataclass(frozen=True)
class ThreatOffsets:
    piece_span: int
    cumulative: int


def _build_threat_offsets() -> tuple[dict[int, ThreatOffsets], dict[int, tuple[int, ...]]]:
    helpers: dict[int, ThreatOffsets] = {}
    offsets: dict[int, tuple[int, ...]] = {}
    cumulative = 0

    for piece in ALL_PIECES:
        per_square: list[int] = []
        piece_span = 0
        for square in range(64):
            per_square.append(piece_span)
            if _piece_type(piece) != 1 or 8 <= square <= 55:
                piece_span += len(pseudo_attacks(piece, square))
        helpers[piece] = ThreatOffsets(piece_span, cumulative)
        offsets[piece] = tuple(per_square)
        cumulative += NUM_VALID_TARGETS[piece] * piece_span

    if cumulative != BASE_THREAT_DIMENSIONS:
        raise AssertionError(f"Unexpected FullThreats dimension: {cumulative}")
    return helpers, offsets


THREAT_HELPERS, THREAT_OFFSETS = _build_threat_offsets()


def piece_feature_index(
    perspective: int,
    piece: int,
    square: int,
    piece_layer: str,
    king_square: int,
    king_layer: str,
) -> int:
    piece_type = _piece_type(piece)
    plane = 10 if piece_type == 6 else 2 * (piece_type - 1) + (_piece_color(piece) != perspective)
    vertical_flip = 56 * perspective
    horizontal_mirror = 7 if king_square % 8 < 4 else 0
    oriented_square = square ^ vertical_flip ^ horizontal_mirror
    relation = int(piece_layer != king_layer)
    bucket = KING_BUCKETS[king_square ^ vertical_flip]
    return oriented_square + 64 * plane + 704 * relation + 1_408 * bucket


def base_threat_index(
    perspective: int,
    attacker: int,
    from_square: int,
    to_square: int,
    attacked: int,
    king_square: int,
) -> int:
    horizontal_mirror = 0 if king_square % 8 < 4 else 7
    orientation = horizontal_mirror ^ (56 * perspective)
    oriented_from = from_square ^ orientation
    oriented_to = to_square ^ orientation
    oriented_attacker = attacker ^ (8 * perspective)
    oriented_attacked = attacked ^ (8 * perspective)

    attacker_type = _piece_type(oriented_attacker)
    attacked_type = _piece_type(oriented_attacked)
    target_class = TARGET_MAP[attacker_type - 1][attacked_type - 1]
    if target_class < 0:
        return BASE_THREAT_DIMENSIONS

    enemy = (oriented_attacker ^ oriented_attacked) == 8
    semi_excluded = attacker_type == attacked_type and (enemy or attacker_type != 1)
    if semi_excluded and oriented_from < oriented_to:
        return BASE_THREAT_DIMENSIONS

    helper = THREAT_HELPERS[oriented_attacker]
    base = helper.cumulative + (
        _piece_color(oriented_attacked) * (NUM_VALID_TARGETS[oriented_attacker] // 2)
        + target_class
    ) * helper.piece_span
    ordinal = sum(
        candidate < oriented_to for candidate in pseudo_attacks(oriented_attacker, oriented_from)
    )
    return base + THREAT_OFFSETS[oriented_attacker][oriented_from] + ordinal


def _king(position: Position, perspective: int) -> tuple[int, Piece]:
    symbol = "K" if perspective == WHITE else "k"
    for square, piece in enumerate(position.board):
        if piece is not None and piece.symbol == symbol:
            return square, piece
    raise AssertionError(f"Missing {symbol} king")


def _piece_trace(position: Position, perspective: int, king_square: int, king: Piece) -> list[dict]:
    sortable: list[tuple[tuple[int, int, int, int], dict]] = []
    for square, piece in enumerate(position.board):
        if piece is None:
            continue
        code = PIECE_CODE[piece.symbol]
        index = piece_feature_index(
            perspective, code, square, piece.layer, king_square, king.layer
        )
        value = {
            "board": piece.layer,
            "index": index,
            "piece": PIECE_NAME[code],
            "relation": "SAME" if piece.layer == king.layer else "OTHER",
            "square": square_name(square),
        }
        sortable.append(((index, code, square, 0 if piece.layer == "A" else 1), value))
    return [value for _, value in sorted(sortable)]


def _allowed_target_types(attacker_type: int) -> set[int]:
    if attacker_type == 1:
        return {2, 4}
    if attacker_type in (2, 5):
        return {1, 2, 3, 4, 5}
    if attacker_type in (3, 4):
        return {1, 2, 3, 4}
    return set()


def _threat_trace(
    position: Position, perspective: int, king_square: int, king: Piece
) -> list[dict]:
    sortable: list[tuple[tuple[int, int, int, int, int, int], dict]] = []

    for layer in ("A", "B"):
        occupied = {
            square
            for square, piece in enumerate(position.board)
            if piece is not None and piece.layer == layer
        }
        for from_square, attacker_piece in enumerate(position.board):
            if attacker_piece is None or attacker_piece.layer != layer:
                continue
            attacker = PIECE_CODE[attacker_piece.symbol]
            attacker_type = _piece_type(attacker)
            allowed_targets = _allowed_target_types(attacker_type)
            if not allowed_targets:
                continue

            for to_square in occupied_attacks(attacker, from_square, occupied):
                attacked_piece = position.board[to_square]
                if (
                    attacked_piece is None
                    or attacked_piece.layer != layer
                    or _piece_type(PIECE_CODE[attacked_piece.symbol]) not in allowed_targets
                ):
                    continue
                attacked = PIECE_CODE[attacked_piece.symbol]
                base = base_threat_index(
                    perspective,
                    attacker,
                    from_square,
                    to_square,
                    attacked,
                    king_square,
                )
                if base >= BASE_THREAT_DIMENSIONS:
                    continue
                index = base + (BASE_THREAT_DIMENSIONS if layer != king.layer else 0)
                value = {
                    "attacked": PIECE_NAME[attacked],
                    "attacker": PIECE_NAME[attacker],
                    "board": layer,
                    "from": square_name(from_square),
                    "index": index,
                    "relation": "SAME" if layer == king.layer else "OTHER",
                    "to": square_name(to_square),
                }
                sortable.append(
                    (
                        (
                            index,
                            attacker,
                            from_square,
                            attacked,
                            to_square,
                            0 if layer == "A" else 1,
                        ),
                        value,
                    )
                )

    return [value for _, value in sorted(sortable)]


def perspective_trace(position: Position, perspective: int) -> dict:
    king_square, king = _king(position, perspective)
    return {
        "color": "white" if perspective == WHITE else "black",
        "kingBoard": king.layer,
        "kingSquare": square_name(king_square),
        "pieceFeatures": _piece_trace(position, perspective, king_square, king),
        "threatFeatures": _threat_trace(position, perspective, king_square, king),
    }


def position_trace(position: Position) -> list[dict]:
    return [perspective_trace(position, WHITE), perspective_trace(position, BLACK)]


def swap_board_names(position: Position) -> Position:
    swapped = [
        None
        if piece is None
        else Piece(piece.symbol, "B" if piece.layer == "A" else "A")
        for piece in position.board
    ]
    return Position(
        swapped,
        position.side_to_move,
        position.castling_rights,
        position.halfmove_clock,
        position.fullmove_number,
    )
