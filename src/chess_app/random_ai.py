from __future__ import annotations

import random

import chess

from chess_app.opening_book import OpeningBook, build_default_opening_book

# ---------------------------------------------------------------------------
# Piece-square tables  (from White's perspective; mirror rank for Black)
# Values are in centipawn-like units.  Index 0 = rank 1 (white back rank),
# index 7 = rank 8 (black back rank).  Files a-h = indices 0-7.
# ---------------------------------------------------------------------------

_PST_KNIGHT = (
    (-165, -95, -30, -10, -10, -30, -95, -165),
    (-80, -30,  10,  25,  25,  10, -30, -80),
    (-50,  10,  30,  45,  45,  30,  10, -50),
    (-25,  20,  45,  60,  60,  45,  20, -25),
    (-25,  20,  45,  60,  60,  45,  20, -25),
    (-50,  10,  30,  45,  45,  30,  10, -50),
    (-80, -30,  10,  25,  25,  10, -30, -80),
    (-165, -95, -30, -10, -10, -30, -95, -165),
)

_PST_BISHOP = (
    (-35, -20, -10, -5, -5, -10, -20, -35),
    (-20,  -5,   5, 10, 10,   5,  -5, -20),
    (-10,  10,  20, 30, 30,  20,  10, -10),
    ( -5,  15,  25, 35, 35,  25,  15,  -5),
    ( -5,  15,  25, 35, 35,  25,  15,  -5),
    (-10,  10,  20, 30, 30,  20,  10, -10),
    (-20,  -5,   5, 10, 10,   5,  -5, -20),
    (-35, -20, -10, -5, -5, -10, -20, -35),
)

_PST_ROOK = (
    (-10,  -5,  0, 10, 10,  0,  -5, -10),
    (-10,  -5,  0, 10, 10,  0,  -5, -10),
    ( -5,   0,  5, 10, 10,  5,   0,  -5),
    ( -5,   0,  5, 10, 10,  5,   0,  -5),
    ( -5,   0,  5, 10, 10,  5,   0,  -5),
    ( -5,   0,  5, 10, 10,  5,   0,  -5),
    (  5,  10, 15, 25, 25, 15,  10,   5),
    (-10,  -5,  0,  5,  5,  0,  -5, -10),
)

_PST_QUEEN = (
    (-20, -10, -5, -5, -5,  -5, -10, -20),
    (-10,   0,  0,  0,  0,   0,   0, -10),
    ( -5,   0,  5,  5,  5,   5,   0,  -5),
    ( -5,   0,  5,  5,  5,   5,   0,  -5),
    ( -5,   0,  5,  5,  5,   5,   0,  -5),
    ( -5,   0,  5,  5,  5,   5,   0,  -5),
    (-10,   0,  5,  5,  5,   5,   0, -10),
    (-20, -10, -5, -5, -5,  -5, -10, -20),
)

_PST_KING_MG = (
    (  0,  40,  15, -25, -25,  15,  40,   0),
    (-20, -10, -25, -40, -40, -25, -10, -20),
    (-30, -20, -40, -55, -55, -40, -20, -30),
    (-40, -30, -55, -70, -70, -55, -30, -40),
    (-50, -40, -60, -75, -75, -60, -40, -50),
    (-50, -40, -60, -75, -75, -60, -40, -50),
    (-50, -40, -60, -75, -75, -60, -40, -50),
    (-50, -40, -60, -75, -75, -60, -40, -50),
)

_PST_KING_EG = (
    (-50, -30, -15,  0,  0, -15, -30, -50),
    (-30, -10,   5, 15, 15,   5, -10, -30),
    (-15,   5,  20, 30, 30,  20,   5, -15),
    (  0,  15,  30, 40, 40,  30,  15,   0),
    (  0,  15,  30, 40, 40,  30,  15,   0),
    (-15,   5,  20, 30, 30,  20,   5, -15),
    (-30, -10,   5, 15, 15,   5, -10, -30),
    (-50, -30, -15,  0,  0, -15, -30, -50),
)

_PST_PAWN = (
    (  0,   0,   0,   0,   0,   0,   0,   0),
    ( -5,  -2,   2,   5,   5,   2,  -2,  -5),
    ( -3,   0,   3,   6,   6,   3,   0,  -3),
    (  0,   3,   6,  10,  10,   6,   3,   0),
    (  5,   8,  12,  18,  18,  12,   8,   5),
    ( 15,  20,  28,  35,  35,  28,  20,  15),
    ( 40,  50,  60,  70,  70,  60,  50,  40),
    (  0,   0,   0,   0,   0,   0,   0,   0),
)

_PST = {
    chess.PAWN: _PST_PAWN,
    chess.KNIGHT: _PST_KNIGHT,
    chess.BISHOP: _PST_BISHOP,
    chess.ROOK: _PST_ROOK,
    chess.QUEEN: _PST_QUEEN,
}

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
NEAR_CENTER_SQUARES = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4, chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}

# Bishop pair bonus
BISHOP_PAIR_BONUS = 30

# Rook on open / semi-open file
ROOK_OPEN_FILE_BONUS = 20
ROOK_SEMI_OPEN_FILE_BONUS = 10

# Passed pawn bonus per rank advanced from rank 2 (white) / rank 7 (black)
PASSED_PAWN_BONUS_PER_RANK = 15

# Tempo bonus
TEMPO_BONUS = 15
HANGING_MOVE_PENALTY_SCALE = 3
MINOR_PIECES = {chess.KNIGHT, chess.BISHOP}
RIM_KNIGHT_PENALTY = 55
UNSAFE_MINOR_PAWN_CAPTURE_PENALTY = 140
URGENT_THREAT_THRESHOLD = 120
WING_PAWN_DISTRACTION_PENALTY = 280

EVAL_WEIGHTS = {
    "material": 1.00,
    "pst": 0.85,
    "pawn_structure": 0.90,
    "pawn_chain": 0.55,
    "backward_pawns": 0.75,
    "space": 0.35,
    "mobility": 0.35,
    "rook_activity": 0.65,
    "bishop_pair": 0.75,
    "piece_coordination": 0.70,
    "piece_protection": 0.60,
    "piece_safety": 1.25,
    "king_safety": 0.85,
    "king_attack": 0.65,
    "tactics": 0.80,
    "capture_opportunity": 0.65,
    "discovered": 0.35,
    "tempo": 1.00,
}

EVAL_CAPS = {
    "pst": 260,
    "pawn_structure": 220,
    "pawn_chain": 120,
    "backward_pawns": 120,
    "space": 180,
    "mobility": 140,
    "rook_activity": 90,
    "bishop_pair": 40,
    "piece_coordination": 220,
    "piece_protection": 180,
    "piece_safety": 900,
    "king_safety": 320,
    "king_attack": 260,
    "tactics": 320,
    "capture_opportunity": 360,
    "discovered": 90,
}

# Files adjacent to the king's file for pawn shield
_KING_FILE_OFFSETS = (-1, 0, 1)


def _clamp(value: int, limit: int) -> int:
    return max(-limit, min(limit, value))


def _weighted_component(name: str, white_cp: int) -> int:
    capped = _clamp(white_cp, EVAL_CAPS.get(name, 10_000))
    return int(round(capped * EVAL_WEIGHTS.get(name, 1.0)))


def _pst_value(piece_type: int, square: int, color: bool) -> int:
    """Return piece-square table value for *piece_type* on *square*."""
    if piece_type == chess.KING:
        return 0  # king PST handled separately via phase blend
    table = _PST.get(piece_type)
    if table is None:
        return 0
    rank = chess.square_rank(square)
    file_idx = chess.square_file(square)
    if color == chess.WHITE:
        return table[rank][file_idx]
    return table[7 - rank][file_idx]


def _king_pst_value(square: int, color: bool, phase: float) -> int:
    rank = chess.square_rank(square)
    file_idx = chess.square_file(square)
    if color == chess.WHITE:
        mg = _PST_KING_MG[rank][file_idx]
        eg = _PST_KING_EG[rank][file_idx]
    else:
        mg = _PST_KING_MG[7 - rank][file_idx]
        eg = _PST_KING_EG[7 - rank][file_idx]
    return int(mg * phase + eg * (1.0 - phase))


# ---------------------------------------------------------------------------
# Pawn-structure helpers
# ---------------------------------------------------------------------------

def _pawn_structure_score(board: chess.Board) -> int:
    """Return pawn-structure score from White's perspective (centipawns)."""
    score = 0
    pawns_by_color = {chess.WHITE: [], chess.BLACK: []}
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            pawns_by_color[piece.color].append(square)

    for color in (chess.WHITE, chess.BLACK):
        pawns = pawns_by_color[color]
        enemy_pawns = pawns_by_color[not color]
        sign = 1 if color == chess.WHITE else -1
        direction = 1 if color == chess.WHITE else -1
        file_counts: dict[int, int] = {}
        for sq in pawns:
            f = chess.square_file(sq)
            file_counts[f] = file_counts.get(f, 0) + 1

        for f, count in file_counts.items():
            if count > 1:
                score += sign * -15 * (count - 1)
            if count:
                front_rank = max(chess.square_rank(sq) for sq in pawns if chess.square_file(sq) == f)
                if color == chess.BLACK:
                    front_rank = min(chess.square_rank(sq) for sq in pawns if chess.square_file(sq) == f)
                front_sq = chess.square(f, front_rank + direction) if 0 <= front_rank + direction <= 7 else None
                if front_sq is not None and board.piece_at(front_sq) is not None:
                    score += sign * -8

        occupied_files = {chess.square_file(sq) for sq in pawns}
        for sq in pawns:
            f = chess.square_file(sq)
            rank = chess.square_rank(sq)
            if (f - 1) not in occupied_files and (f + 1) not in occupied_files:
                score += sign * -12

            blockers = [
                enemy_sq
                for enemy_sq in enemy_pawns
                if abs(chess.square_file(enemy_sq) - f) <= 1
                and ((chess.square_rank(enemy_sq) > rank) if color == chess.WHITE else (chess.square_rank(enemy_sq) < rank))
            ]
            advanced = rank - 1 if color == chess.WHITE else 6 - rank
            if not blockers:
                bonus = PASSED_PAWN_BONUS_PER_RANK * max(0, advanced)
                if board.attackers(color, sq):
                    bonus += 8
                if rank in (5, 6) if color == chess.WHITE else rank in (1, 2):
                    bonus += 18
                score += sign * bonus
            else:
                front_rank = rank + direction
                if 0 <= front_rank <= 7:
                    front_sq = chess.square(f, front_rank)
                    if not board.is_attacked_by(not color, front_sq):
                        score += sign * max(0, advanced) * 3

    return score


# ---------------------------------------------------------------------------
# King-safety helper
# ---------------------------------------------------------------------------

def _king_safety_score(board: chess.Board) -> int:
    """Return king-safety score from White's perspective."""
    score = 0
    phase = _phase(board)
    attack_weight = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 8,
        chess.KING: 0,
    }

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        king_sq = board.king(color)
        if king_sq is None:
            continue
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        local = 0

        if phase > 0.25:
            is_center_file = king_file in (3, 4, 5)
            is_center_rank = (color == chess.WHITE and king_rank <= 2) or (color == chess.BLACK and king_rank >= 5)
            if is_center_file and is_center_rank:
                local -= int(85 * phase)
            elif is_center_file or is_center_rank:
                local -= int(35 * phase)

            home_rank = 0 if color == chess.WHITE else 7
            if king_rank == home_rank and king_file in (3, 4, 5):
                if not board.has_kingside_castling_rights(color) and not board.has_queenside_castling_rights(color):
                    local -= int(45 * phase)

        if king_file in (2, 6):
            local += int(38 * phase)

        shield_bonus = 0
        for df in _KING_FILE_OFFSETS:
            f = king_file + df
            if f < 0 or f > 7:
                continue
            # Look for pawns 1-2 ranks ahead of the king
            for dr in (1, 2):
                r = king_rank + dr if color == chess.WHITE else king_rank - dr
                if r < 0 or r > 7:
                    continue
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                    shield_bonus += 9 if dr == 1 else 5
        local += int(shield_bonus * phase)

        attack_penalty = 0
        for df in _KING_FILE_OFFSETS:
            f = king_file + df
            if f < 0 or f > 7:
                continue
            for dr in (1, 2):
                r = king_rank + dr if color == chess.WHITE else king_rank - dr
                if r < 0 or r > 7:
                    continue
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece is not None and piece.piece_type == chess.PAWN and piece.color != color:
                    attack_penalty += 9 if dr == 1 else 5
        local -= int(attack_penalty * phase)

        for df in _KING_FILE_OFFSETS:
            f = king_file + df
            if f < 0 or f > 7:
                continue
            has_own_pawn = False
            has_enemy_pawn = False
            for r in range(8):
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece is not None and piece.piece_type == chess.PAWN:
                    if piece.color == color:
                        has_own_pawn = True
                    else:
                        has_enemy_pawn = True
            if not has_own_pawn:
                local -= int((18 if has_enemy_pawn else 28) * phase)

        king_zone = []
        for df in (-2, -1, 0, 1, 2):
            for dr in (-2, -1, 0, 1, 2):
                f = king_file + df
                r = king_rank + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    king_zone.append(chess.square(f, r))

        attacker_units = 0
        weak_squares = 0
        for sq in king_zone:
            if board.is_attacked_by(not color, sq):
                attackers = board.attackers(not color, sq)
                for attacker_sq in attackers:
                    attacker = board.piece_at(attacker_sq)
                    if attacker is not None:
                        attacker_units += attack_weight[attacker.piece_type]
            if sq != king_sq and not board.is_attacked_by(color, sq):
                weak_squares += 1

        local -= int(min(130, attacker_units * 3) * (0.45 + 0.55 * phase))
        local -= int(min(60, weak_squares * 3) * phase)
        score += sign * local

    return score


# ---------------------------------------------------------------------------
# Mobility helper
# ---------------------------------------------------------------------------

def _mobility_score(board: chess.Board) -> int:
    """Return mobility score from White's perspective (centipawns).

    Counts legal moves for both sides by temporarily switching the turn.
    """
    original_turn = board.turn
    board.turn = chess.WHITE
    white_moves = board.legal_moves.count()
    board.turn = chess.BLACK
    black_moves = board.legal_moves.count()
    board.turn = original_turn
    return 4 * (white_moves - black_moves)


# ---------------------------------------------------------------------------
# Piece-activity helpers
# ---------------------------------------------------------------------------

def _rook_activity(board: chess.Board) -> int:
    """Rook bonuses for open/semi-open files."""
    score = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.ROOK:
            continue
        sign = 1 if piece.color == chess.WHITE else -1
        f = chess.square_file(square)
        own_pawn_on_file = False
        any_pawn_on_file = False
        for r in range(8):
            p = board.piece_at(chess.square(f, r))
            if p is not None and p.piece_type == chess.PAWN:
                any_pawn_on_file = True
                if p.color == piece.color:
                    own_pawn_on_file = True
        if not any_pawn_on_file:
            score += sign * ROOK_OPEN_FILE_BONUS
        elif not own_pawn_on_file:
            score += sign * ROOK_SEMI_OPEN_FILE_BONUS
    return score


def _bishop_pair_bonus(board: chess.Board) -> int:
    score = 0
    white_bishops = sum(1 for p in board.piece_map().values() if p == chess.Piece(chess.BISHOP, chess.WHITE))
    black_bishops = sum(1 for p in board.piece_map().values() if p == chess.Piece(chess.BISHOP, chess.BLACK))
    if white_bishops >= 2:
        score += BISHOP_PAIR_BONUS
    if black_bishops >= 2:
        score -= BISHOP_PAIR_BONUS
    return score


def _cheapest_attacker_value(board: chess.Board, color: bool, square: int) -> int | None:
    values = [
        PIECE_VALUES[piece.piece_type]
        for attacker_sq in board.attackers(color, square)
        if (piece := board.piece_at(attacker_sq)) is not None
    ]
    return min(values) if values else None


def _cheapest_defender_value(board: chess.Board, color: bool, square: int) -> int | None:
    values = [
        PIECE_VALUES[piece.piece_type]
        for defender_sq in board.attackers(color, square)
        if defender_sq != square and (piece := board.piece_at(defender_sq)) is not None
    ]
    return min(values) if values else None


def _defender_values(board: chess.Board, color: bool, square: int) -> list[int]:
    return [
        PIECE_VALUES[piece.piece_type]
        for defender_sq in board.attackers(color, square)
        if defender_sq != square and (piece := board.piece_at(defender_sq)) is not None
    ]


def _captured_piece_for_move(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    if board.is_en_passant(move):
        return chess.Piece(chess.PAWN, not board.turn)
    return board.piece_at(move.to_square)


def _is_immediate_checkmate(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        return board.is_checkmate()
    finally:
        board.pop()


def _piece_threat_penalty(board: chess.Board, square: int, piece: chess.Piece) -> int:
    if piece.piece_type == chess.KING:
        return 0

    value = PIECE_VALUES[piece.piece_type]
    attackers = [
        PIECE_VALUES[attacker.piece_type]
        for attacker_sq in board.attackers(not piece.color, square)
        if (attacker := board.piece_at(attacker_sq)) is not None
    ]
    if not attackers:
        return 0

    defenders = _defender_values(board, piece.color, square)
    cheapest_attacker = min(attackers)
    cheapest_defender = min(defenders) if defenders else None
    penalty = 0

    if cheapest_defender is None:
        penalty += min(value, max(value // 2, value - cheapest_attacker // 2))
    elif cheapest_attacker < value:
        penalty += max(0, value - cheapest_attacker - cheapest_defender // 3)

    if len(attackers) > len(defenders):
        penalty += min(90, (len(attackers) - len(defenders)) * 30)

    if piece.piece_type in MINOR_PIECES and cheapest_attacker <= PIECE_VALUES[chess.PAWN]:
        penalty = max(penalty, value - 25)

    if piece.piece_type in (chess.ROOK, chess.QUEEN) and cheapest_attacker <= PIECE_VALUES[chess.BISHOP]:
        penalty = max(penalty + 70, value - cheapest_attacker // 2)

    return min(value, penalty)


def _side_threat_penalty(board: chess.Board, color: bool) -> int:
    return sum(
        _piece_threat_penalty(board, square, piece)
        for square, piece in board.piece_map().items()
        if piece.color == color and piece.piece_type != chess.KING
    )


def _capture_gain(board: chess.Board, move: chess.Move) -> int:
    moving_piece = board.piece_at(move.from_square)
    captured_piece = _captured_piece_for_move(board, move)
    if moving_piece is None or captured_piece is None:
        return 0

    captured_value = PIECE_VALUES[captured_piece.piece_type]
    board.push(move)
    try:
        moved_piece = board.piece_at(move.to_square)
        if moved_piece is None:
            return captured_value
        return captured_value - _piece_threat_penalty(board, move.to_square, moved_piece)
    finally:
        board.pop()


def _move_safety_swing(board: chess.Board, move: chess.Move, color: bool | None = None) -> int:
    color = board.turn if color is None else color
    before = _side_threat_penalty(board, color)
    board.push(move)
    try:
        after = _side_threat_penalty(board, color)
    finally:
        board.pop()
    return before - after


def _is_quiet_wing_pawn_push(board: chess.Board, move: chess.Move) -> bool:
    moving_piece = board.piece_at(move.from_square)
    return (
        moving_piece is not None
        and moving_piece.piece_type == chess.PAWN
        and chess.square_file(move.from_square) in (0, 7)
        and not board.is_capture(move)
        and not move.promotion
    )


def _urgent_safety_move_score(board: chess.Board, move: chess.Move) -> int:
    own_threat = _side_threat_penalty(board, board.turn)
    if own_threat < URGENT_THREAT_THRESHOLD:
        return -45 if _is_quiet_wing_pawn_push(board, move) and board.fullmove_number <= 16 else 0

    swing = _move_safety_swing(board, move, board.turn)
    score = 4 * swing

    if swing <= 0 and not board.is_capture(move) and not move.promotion:
        score -= min(1600, 3 * own_threat)
        if _is_quiet_wing_pawn_push(board, move):
            score -= WING_PAWN_DISTRACTION_PENALTY
    elif swing < own_threat // 3 and _is_quiet_wing_pawn_push(board, move):
        score -= WING_PAWN_DISTRACTION_PENALTY

    return score


def _move_tactical_bonus(board: chess.Board, move: chess.Move) -> int:
    """Return a bounded bonus for moves that create immediate tactical threats."""
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        return 0

    board.push(move)
    try:
        moved_piece = board.piece_at(move.to_square)
        if moved_piece is None or moved_piece.piece_type == chess.KING:
            return 0

        attacked_values = []
        for target_sq in board.attacks(move.to_square):
            target = board.piece_at(target_sq)
            if target is None or target.color == moved_piece.color:
                continue
            if target.piece_type == chess.KING:
                attacked_values.append(1000)
            elif PIECE_VALUES[target.piece_type] >= PIECE_VALUES[moved_piece.piece_type] - 100:
                attacked_values.append(PIECE_VALUES[target.piece_type])

        if len(attacked_values) < 2:
            return 0

        attacked_values.sort(reverse=True)
        fork_bonus = min(380, (attacked_values[0] + attacked_values[1]) // 3)
        if moved_piece.piece_type == chess.KNIGHT:
            fork_bonus = int(fork_bonus * 1.25)
        elif moved_piece.piece_type == chess.PAWN:
            fork_bonus = int(fork_bonus * 1.35)

        if board.is_check() and attacked_values[1] >= PIECE_VALUES[chess.ROOK]:
            fork_bonus += 160

        return min(620, fork_bonus)
    finally:
        board.pop()


def _move_hanging_penalty(board: chess.Board, move: chess.Move) -> int:
    """Return an ordering penalty for moves that leave the moved piece loose."""
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None or moving_piece.piece_type == chess.KING:
        return 0

    captured_piece = _captured_piece_for_move(board, move)
    captured_value = PIECE_VALUES[captured_piece.piece_type] if captured_piece is not None else 0
    is_minor_pawn_capture = moving_piece.piece_type in MINOR_PIECES and captured_value == PIECE_VALUES[chess.PAWN]

    board.push(move)
    try:
        moved_piece = board.piece_at(move.to_square)
        if moved_piece is None or board.is_checkmate():
            return 0

        moved_value = PIECE_VALUES[moved_piece.piece_type]
        mover_color = moved_piece.color
        enemy_color = not mover_color
        cheapest_attacker = _cheapest_attacker_value(board, enemy_color, move.to_square)
        penalty = 0

        if moved_piece.piece_type == chess.KNIGHT and chess.square_file(move.to_square) in (0, 7):
            penalty += RIM_KNIGHT_PENALTY

        if cheapest_attacker is None:
            return penalty

        cheapest_defender = _cheapest_defender_value(board, mover_color, move.to_square)
        exchange_loss = max(0, moved_value - captured_value)
        if exchange_loss == 0:
            return penalty

        if cheapest_defender is None:
            penalty += exchange_loss
        else:
            trade_loss = max(0, moved_value - captured_value - cheapest_attacker)
            if cheapest_attacker < moved_value:
                penalty += max(trade_loss, exchange_loss // 3)
            else:
                penalty += max(0, trade_loss // 2)

        if is_minor_pawn_capture and cheapest_attacker <= PIECE_VALUES[chess.PAWN]:
            penalty += UNSAFE_MINOR_PAWN_CAPTURE_PENALTY

        return penalty
    finally:
        board.pop()


def _piece_safety_score(board: chess.Board) -> int:
    """Evaluate loose or tactically vulnerable pieces from White's perspective."""
    score = 0
    phase = _phase(board)
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue

        sign = 1 if piece.color == chess.WHITE else -1
        penalty = _piece_threat_penalty(board, square, piece)

        if piece.piece_type == chess.KNIGHT and chess.square_file(square) in (0, 7):
            penalty += int(RIM_KNIGHT_PENALTY * phase)

        score -= sign * penalty

    return score


def _piece_protection_score(board: chess.Board) -> int:
    """Reward pieces that are mutually protected and penalize loose pieces."""
    score = 0
    phase = _phase(board)
    loose_penalty = {
        chess.PAWN: 2,
        chess.KNIGHT: 16,
        chess.BISHOP: 16,
        chess.ROOK: 22,
        chess.QUEEN: 32,
    }

    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue

        sign = 1 if piece.color == chess.WHITE else -1
        defenders = _defender_values(board, piece.color, square)
        attackers = _defender_values(board, not piece.color, square)
        pawn_defended = any(
            (defender := board.piece_at(defender_sq)) is not None and defender.piece_type == chess.PAWN
            for defender_sq in board.attackers(piece.color, square)
            if defender_sq != square
        )

        if defenders:
            bonus = min(24, len(defenders) * 6)
            if pawn_defended:
                bonus += 8
            if attackers and min(defenders) <= min(attackers):
                bonus += 6
            if piece.piece_type in MINOR_PIECES and chess.square_file(square) not in (0, 7):
                bonus += int(5 * phase)
            score += sign * bonus
        else:
            penalty = loose_penalty.get(piece.piece_type, 0)
            if piece.piece_type in MINOR_PIECES and chess.square_file(square) in (2, 3, 4, 5):
                penalty += 8
            score -= sign * penalty

    return score


def _capture_opportunity_score(board: chess.Board) -> int:
    """Reward the side to move for immediate favorable captures."""
    gains = [
        _capture_gain(board, move)
        for move in list(board.legal_moves)
        if board.is_capture(move)
    ]
    best_gain = max(gains, default=0)
    if best_gain <= 0:
        return 0

    sign = 1 if board.turn == chess.WHITE else -1
    return sign * min(360, best_gain)


def _space_advantage(board: chess.Board) -> int:
    """Evaluate space advantage based on controlled squares and advanced pawns.

    Returns score from White's perspective.
    """
    score = 0
    phase = _phase(board)

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        controlled_space = 0

        for sq in chess.SQUARES:
            rank = chess.square_rank(sq)
            file_idx = chess.square_file(sq)
            is_enemy_half = rank >= 4 if color == chess.WHITE else rank <= 3
            is_center_band = 2 <= file_idx <= 5
            occupant = board.piece_at(sq)

            if occupant is not None and occupant.color == color:
                continue
            if is_enemy_half and is_center_band and board.is_attacked_by(color, sq):
                if not board.is_attacked_by(not color, sq):
                    controlled_space += 2
                else:
                    controlled_space += 1

        score += sign * int(controlled_space * (0.45 + 0.55 * phase))

    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.PAWN:
            continue

        rank = chess.square_rank(square)
        sign = 1 if piece.color == chess.WHITE else -1

        if piece.color == chess.WHITE and rank >= 4:
            score += sign * int((rank - 3) * 6 * (0.6 + phase))
        elif piece.color == chess.BLACK and rank <= 3:
            score += sign * int((4 - rank) * 6 * (0.6 + phase))

    return score


def _piece_coordination(board: chess.Board) -> int:
    """Evaluate piece coordination: knight outposts, rook on 7th rank, bishop activity.

    Returns score from White's perspective.
    """
    score = 0
    phase = _phase(board)

    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        file_idx = chess.square_file(square)
        rank = chess.square_rank(square)

        if piece.piece_type == chess.KNIGHT:
            if file_idx in (0, 7):
                score -= sign * int(45 * phase)
            if file_idx in (2, 3, 4, 5) and rank not in (0, 7):
                score += sign * int(12 * phase)

            own_pawn_defended = any(
                (attacker := board.piece_at(attacker_sq)) is not None and attacker.piece_type == chess.PAWN
                for attacker_sq in board.attackers(piece.color, square)
            )
            enemy_pawn_attack = any(
                (attacker := board.piece_at(attacker_sq)) is not None and attacker.piece_type == chess.PAWN
                for attacker_sq in board.attackers(not piece.color, square)
            )
            is_advanced = rank >= 4 if piece.color == chess.WHITE else rank <= 3
            if is_advanced and own_pawn_defended and not enemy_pawn_attack:
                score += sign * 34

        if piece.piece_type == chess.ROOK:
            if piece.color == chess.WHITE and rank == 6:  # 7th rank for white
                score += sign * 25
            elif piece.color == chess.BLACK and rank == 1:  # 2nd rank for black
                score += sign * 25

        if piece.piece_type == chess.BISHOP:
            if file_idx == rank or file_idx == 7 - rank:
                score += sign * 10
            if file_idx in (0, 7) or rank in (0, 7):
                score -= sign * int(8 * phase)

        if piece.piece_type == chess.QUEEN:
            if 2 <= file_idx <= 5 and 2 <= rank <= 5:
                score += sign * 15

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        rooks = list(board.pieces(chess.ROOK, color))
        if len(rooks) >= 2:
            files = [chess.square_file(sq) for sq in rooks]
            ranks = [chess.square_rank(sq) for sq in rooks]
            if len(set(files)) < len(files) or len(set(ranks)) < len(ranks):
                score += sign * 14

    return score


def _pawn_chain_strength(board: chess.Board) -> int:
    """Evaluate pawn chains and pawn islands.

    Returns score from White's perspective.
    """
    score = 0

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        pawns = [sq for sq, p in board.piece_map().items()
                 if p.piece_type == chess.PAWN and p.color == color]

        # Group pawns by file
        files_with_pawns = {}
        for sq in pawns:
            f = chess.square_file(sq)
            if f not in files_with_pawns:
                files_with_pawns[f] = []
            files_with_pawns[f].append(sq)

        # Pawn chains: connected pawns protecting each other
        chain_bonus = 0
        for f in sorted(files_with_pawns.keys()):
            # Check if this pawn is protected by a pawn on adjacent file
            for pawn_sq in files_with_pawns[f]:
                rank = chess.square_rank(pawn_sq)
                # Look for protecting pawns on adjacent files
                for df in [-1, 1]:
                    adj_file = f + df
                    if adj_file in files_with_pawns:
                        for adj_pawn in files_with_pawns[adj_file]:
                            adj_rank = chess.square_rank(adj_pawn)
                            # Pawn is protected if adjacent pawn is one rank behind
                            if color == chess.WHITE and adj_rank == rank - 1:
                                chain_bonus += 5
                            elif color == chess.BLACK and adj_rank == rank + 1:
                                chain_bonus += 5

        score += sign * chain_bonus

        # Pawn islands: fewer islands is better
        # An island is a group of pawns on consecutive files
        sorted_files = sorted(files_with_pawns.keys())
        islands = 1
        for i in range(1, len(sorted_files)):
            if sorted_files[i] - sorted_files[i-1] > 1:
                islands += 1

        # Penalty for each island (fewer islands = better coordination)
        score -= sign * (islands - 1) * 10

    return score


def _backward_pawns(board: chess.Board) -> int:
    """Detect backward pawns (pawns that can't advance safely).

    Returns score from White's perspective.
    """
    score = 0

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        pawns = [sq for sq, p in board.piece_map().items()
                 if p.piece_type == chess.PAWN and p.color == color]

        for pawn_sq in pawns:
            file_idx = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)

            # Check if pawn is backward
            is_backward = False

            # Check adjacent files for friendly pawns
            has_adjacent_pawn = False
            for df in [-1, 1]:
                adj_file = file_idx + df
                if 0 <= adj_file <= 7:
                    for adj_sq in pawns:
                        if chess.square_file(adj_sq) == adj_file:
                            adj_rank = chess.square_rank(adj_sq)
                            # Adjacent pawn is ahead or on same rank
                            if color == chess.WHITE and adj_rank >= rank:
                                has_adjacent_pawn = True
                            elif color == chess.BLACK and adj_rank <= rank:
                                has_adjacent_pawn = True

            if not has_adjacent_pawn:
                # Check if the square in front is controlled by enemy
                forward_rank = rank + 1 if color == chess.WHITE else rank - 1
                if 0 <= forward_rank <= 7:
                    forward_sq = chess.square(file_idx, forward_rank)
                    enemy_color = not color
                    if board.is_attacked_by(enemy_color, forward_sq):
                        is_backward = True

            if is_backward:
                score -= sign * 15

    return score


def _king_attack_weakness(board: chess.Board) -> int:
    """Evaluate direct pressure near kings from White's perspective."""
    score = 0
    weights = {
        chess.PAWN: 4,
        chess.KNIGHT: 14,
        chess.BISHOP: 12,
        chess.ROOK: 18,
        chess.QUEEN: 26,
        chess.KING: 0,
    }

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        king_sq = board.king(color)
        if king_sq is None:
            continue

        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)

        pressure = 0
        used_attackers: set[int] = set()
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                f = king_file + df
                r = king_rank + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    sq = chess.square(f, r)
                    for attacker_sq in board.attackers(not color, sq):
                        if attacker_sq in used_attackers:
                            continue
                        attacker = board.piece_at(attacker_sq)
                        if attacker is None:
                            continue
                        used_attackers.add(attacker_sq)
                        pressure += weights[attacker.piece_type]

        score -= sign * pressure

    return score


def _hanging_pieces(board: chess.Board) -> int:
    """Detect attacked and undefended pieces from White's perspective."""
    score = 0

    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue

        sign = 1 if piece.color == chess.WHITE else -1
        enemy_color = not piece.color

        if board.is_attacked_by(enemy_color, square):
            defenders = _defender_values(board, piece.color, square)
            if not defenders:
                score -= sign * min(180, max(35, PIECE_VALUES[piece.piece_type] // 2))

    return score


_ORTHOGONAL_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _slider_directions(piece_type: int) -> tuple[tuple[int, int], ...]:
    if piece_type == chess.BISHOP:
        return _DIAGONAL_DIRECTIONS
    if piece_type == chess.ROOK:
        return _ORTHOGONAL_DIRECTIONS
    if piece_type == chess.QUEEN:
        return _ORTHOGONAL_DIRECTIONS + _DIAGONAL_DIRECTIONS
    return ()


def _ray_squares(square: int, df: int, dr: int) -> list[int]:
    file_idx = chess.square_file(square) + df
    rank = chess.square_rank(square) + dr
    squares = []
    while 0 <= file_idx <= 7 and 0 <= rank <= 7:
        squares.append(chess.square(file_idx, rank))
        file_idx += df
        rank += dr
    return squares


def _discovered_attacks(board: chess.Board) -> int:
    """Detect simple discovered attack potential from White's perspective."""
    score = 0

    for square, piece in board.piece_map().items():
        if piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue

        sign = 1 if piece.color == chess.WHITE else -1
        for df, dr in _slider_directions(piece.piece_type):
            blocker = None
            for target_sq in _ray_squares(square, df, dr):
                target = board.piece_at(target_sq)
                if target is None:
                    continue
                if blocker is None:
                    if target.color != piece.color or target.piece_type == chess.KING:
                        break
                    blocker = target
                    continue
                if target.color != piece.color and PIECE_VALUES[target.piece_type] >= PIECE_VALUES[chess.BISHOP]:
                    score += sign * min(45, 12 + PIECE_VALUES[target.piece_type] // 20)
                break

    return score


def _skewer_score(board: chess.Board) -> int:
    """Detect simple line skewers from White's perspective."""
    score = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue

        sign = 1 if piece.color == chess.WHITE else -1
        for df, dr in _slider_directions(piece.piece_type):
            first_enemy = None
            for target_sq in _ray_squares(square, df, dr):
                target = board.piece_at(target_sq)
                if target is None:
                    continue
                if target.color == piece.color:
                    break
                if first_enemy is None:
                    first_enemy = target
                    continue
                first_value = 10_000 if first_enemy.piece_type == chess.KING else PIECE_VALUES[first_enemy.piece_type]
                second_value = PIECE_VALUES[target.piece_type]
                if first_value >= PIECE_VALUES[chess.ROOK] and second_value >= PIECE_VALUES[chess.BISHOP]:
                    score += sign * min(90, 25 + second_value // 12)
                break
    return score


# ---------------------------------------------------------------------------
# Tactical pattern detection
# ---------------------------------------------------------------------------

def _fork_score(board: chess.Board) -> int:
    """Detect forks: a piece attacking two or more enemy pieces of equal/higher value.

    Returns score from White's perspective.
    """
    score = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        if board.is_pinned(piece.color, square):
            continue
        # Get all squares this piece attacks
        attacks = board.attacks(square)
        enemy_valuable = []
        for target_sq in attacks:
            target = board.piece_at(target_sq)
            if target is not None and target.color != piece.color:
                if target.piece_type == chess.KING:
                    # Attacking the king is very strong (check)
                    enemy_valuable.append(1000)
                elif PIECE_VALUES[target.piece_type] >= PIECE_VALUES[piece.piece_type] - 100:
                    enemy_valuable.append(PIECE_VALUES[target.piece_type])

        # Fork: attacking 2+ valuable pieces
        if len(enemy_valuable) >= 2:
            # Sort by value, take top 2
            enemy_valuable.sort(reverse=True)
            fork_value = enemy_valuable[0] + enemy_valuable[1]
            # Knights and pawns are especially dangerous forking pieces
            if piece.piece_type == chess.KNIGHT:
                fork_value = int(fork_value * 1.3)
            elif piece.piece_type == chess.PAWN:
                fork_value = int(fork_value * 1.5)

            sign = 1 if piece.color == chess.WHITE else -1
            score += sign * min(fork_value // 4, 150)  # Cap at 150cp

    return score


def _pin_score(board: chess.Board) -> int:
    """Detect absolute pins and reward the pinning side."""
    score = 0
    for pinned_color in (chess.WHITE, chess.BLACK):
        for square, piece in board.piece_map().items():
            if piece.color != pinned_color:
                continue
            if piece.piece_type == chess.KING:
                continue

            if board.is_pinned(pinned_color, square):
                pinner_sign = -1 if pinned_color == chess.WHITE else 1
                pinned_value = PIECE_VALUES[piece.piece_type]
                bonus = 35
                if pinned_value >= PIECE_VALUES[chess.QUEEN]:
                    bonus += 60
                elif pinned_value >= PIECE_VALUES[chess.ROOK]:
                    bonus += 30
                elif pinned_value >= PIECE_VALUES[chess.BISHOP]:
                    bonus += 15
                score += pinner_sign * bonus

    return score


def _is_pinned_to(board: chess.Board, pinned_sq: int, king_sq: int, attacker_sq: int, attacker: chess.Piece) -> bool:
    """Check if attacker at attacker_sq pins pinned_sq to king_sq."""
    # Check if all three squares are on the same line
    attacker_file = chess.square_file(attacker_sq)
    attacker_rank = chess.square_rank(attacker_sq)
    pinned_file = chess.square_file(pinned_sq)
    pinned_rank = chess.square_rank(pinned_sq)
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    # Check if they're on the same rank
    if attacker_rank == pinned_rank == king_rank:
        # Check if pinned is between attacker and king on this rank
        files = sorted([attacker_file, pinned_file, king_file])
        if files[1] == pinned_file:
            # Pinned is in the middle - verify it's actually a pin
            return True

    # Check if they're on the same file
    if attacker_file == pinned_file == king_file:
        ranks = sorted([attacker_rank, pinned_rank, king_rank])
        if ranks[1] == pinned_rank:
            return True

    # Check diagonals
    def on_same_diagonal(s1, s2):
        f1, r1 = chess.square_file(s1), chess.square_rank(s1)
        f2, r2 = chess.square_file(s2), chess.square_rank(s2)
        return abs(f1 - f2) == abs(r1 - r2)

    if on_same_diagonal(attacker_sq, pinned_sq) and on_same_diagonal(pinned_sq, king_sq):
        # Verify the attacker can actually move along this diagonal
        if attacker.piece_type == chess.QUEEN or attacker.piece_type == chess.BISHOP:
            # Check if pinned is between attacker and king
            # Use a simple check: the three squares should be on the same diagonal line
            af, ar = chess.square_file(attacker_sq), chess.square_rank(attacker_sq)
            pf, pr = chess.square_file(pinned_sq), chess.square_rank(pinned_sq)
            kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
            # Direction from attacker to king
            df = kf - af
            dr = kr - ar
            if df == 0 and dr == 0:
                return False
            # Check if pinned is between them
            t_pinned = (pf - af) / df if df != 0 else (pr - ar) / dr
            return 0 < t_pinned < 1

    return False


# ---------------------------------------------------------------------------
# Full enhanced evaluation
# ---------------------------------------------------------------------------

def _phase(board: chess.Board) -> float:
    """Return game phase 1.0 (opening) → 0.0 (endgame)."""
    total = 0
    for piece in board.piece_map().values():
        if piece.piece_type == chess.KNIGHT:
            total += 1
        elif piece.piece_type == chess.BISHOP:
            total += 1
        elif piece.piece_type == chess.ROOK:
            total += 2
        elif piece.piece_type == chess.QUEEN:
            total += 4
    return min(1.0, total / 24.0)


def evaluate_position(board: chess.Board, color: bool) -> int:
    """Comprehensive evaluation from *color*'s perspective (centipawns)."""
    if board.is_checkmate():
        return -1_000_000 if board.turn == color else 1_000_000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = _phase(board)
    material_white = 0
    pst_white = 0

    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        material_white += sign * PIECE_VALUES[piece.piece_type]
        pst_white += sign * _pst_value(piece.piece_type, square, piece.color)
        if piece.piece_type == chess.KING:
            pst_white += sign * _king_pst_value(square, piece.color, phase)

    tactics = _fork_score(board) + _pin_score(board) + _skewer_score(board)
    piece_safety = _piece_safety_score(board) + _hanging_pieces(board)
    components = {
        "pst": pst_white,
        "pawn_structure": _pawn_structure_score(board),
        "pawn_chain": _pawn_chain_strength(board),
        "backward_pawns": _backward_pawns(board),
        "space": _space_advantage(board),
        "mobility": _mobility_score(board),
        "rook_activity": _rook_activity(board),
        "bishop_pair": _bishop_pair_bonus(board),
        "piece_coordination": _piece_coordination(board),
        "piece_protection": _piece_protection_score(board),
        "piece_safety": piece_safety,
        "king_safety": _king_safety_score(board),
        "king_attack": _king_attack_weakness(board),
        "tactics": tactics,
        "capture_opportunity": _capture_opportunity_score(board),
        "discovered": _discovered_attacks(board),
    }

    white_total = int(round(material_white * EVAL_WEIGHTS["material"]))
    for name, value in components.items():
        white_total += _weighted_component(name, value)

    tempo_white = TEMPO_BONUS if board.turn == chess.WHITE else -TEMPO_BONUS
    white_total += int(round(tempo_white * EVAL_WEIGHTS["tempo"]))

    return white_total if color == chess.WHITE else -white_total


def evaluate_white_cp(board: chess.Board) -> int:
    """Evaluate from White's perspective in centipawns."""
    return evaluate_position(board, chess.WHITE)


# ---------------------------------------------------------------------------
# AI classes
# ---------------------------------------------------------------------------

class RandomAI:
    """AI baseline that chooses one legal move uniformly at random."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return self.rng.choice(legal_moves)


class BasicAI(RandomAI):
    """Strong rule-based chess AI with enhanced evaluation and search.

    Features:
    - Piece-square tables for positional understanding
    - Pawn structure analysis (doubled, isolated, passed pawns)
    - King safety (pawn shield, open files, pawn storms)
    - Mobility evaluation
    - Rook activity (open/semi-open files)
    - Bishop pair bonus
    - Quiescence search to avoid horizon effects
    - Null-move pruning for faster search
    - Late-move reduction for deeper search
    - Transposition table
    """

    def __init__(
        self,
        seed: int | None = None,
        search_depth: int = 3,
        opening_book: OpeningBook | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.search_depth = search_depth
        self.opening_book = opening_book or build_default_opening_book(seed=seed)
        self.last_book_name: str | None = None

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        self.last_book_name = None
        book_move = self.opening_book.choose(board)
        if book_move is not None:
            self.last_book_name = book_move.name
            return book_move.move

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        color = board.turn
        best_move = legal_moves[0]
        best_score = -1_000_000_000

        ordered = self.order_moves(board, legal_moves)
        for move in ordered:
            if _is_immediate_checkmate(board, move):
                return move

        for idx, move in enumerate(ordered):
            board.push(move)
            try:
                score = self.search(
                    board,
                    depth=self.search_depth - 1,
                    alpha=-1_000_000_000,
                    beta=1_000_000_000,
                    ai_color=color,
                    move_index=idx,
                )
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # ------------------------------------------------------------------
    # Alpha-beta with null-move pruning and late-move reduction
    # ------------------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ai_color: bool,
        move_index: int = 0,
    ) -> int:
        if depth <= 0 or board.is_game_over(claim_draw=True):
            return self.quiescence_search(board, ai_color, alpha, beta, depth=4)

        # Check extension
        in_check = board.is_check()
        if in_check:
            depth += 1

        static_eval = evaluate_position(board, ai_color)

        # Null-move pruning (skip when in check or material is very unbalanced)
        if (
            not in_check
            and depth >= 3
            and self._has_pieces(board, board.turn)
        ):
            reduction = 3 + depth // 6
            if board.turn == ai_color and static_eval >= beta + 75:
                board.push(chess.Move.null())
                try:
                    null_score = self.search(board, depth - 1 - reduction, beta - 1, beta, ai_color)
                finally:
                    board.pop()
                if null_score >= beta:
                    return beta
            elif board.turn != ai_color and static_eval <= alpha - 75:
                board.push(chess.Move.null())
                try:
                    null_score = self.search(board, depth - 1 - reduction, alpha, alpha + 1, ai_color)
                finally:
                    board.pop()
                if null_score <= alpha:
                    return alpha

        legal_moves = list(board.legal_moves)
        ordered = self.order_moves(board, legal_moves)

        if board.turn == ai_color:
            best = -1_000_000_000
            for idx, move in enumerate(ordered):
                is_capture = board.is_capture(move)
                board.push(move)
                try:
                    # Late-move reduction
                    reduction = 0
                    if (
                        depth >= 3
                        and idx >= 4
                        and not is_capture
                        and not move.promotion
                        and not board.is_check()
                    ):
                        reduction = self._lmr_reduction(depth, idx, move, board)

                    score = self.search(
                        board,
                        depth - 1 - reduction,
                        alpha,
                        beta,
                        ai_color,
                        move_index=idx,
                    )
                finally:
                    board.pop()
                if reduction and score > alpha:
                    board.push(move)
                    try:
                        score = self.search(
                            board,
                            depth - 1,
                            alpha,
                            beta,
                            ai_color,
                            move_index=idx,
                        )
                    finally:
                        board.pop()
                best = max(best, score)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best

        best = 1_000_000_000
        for idx, move in enumerate(ordered):
            is_capture = board.is_capture(move)
            board.push(move)
            try:
                reduction = 0
                if (
                    depth >= 3
                    and idx >= 4
                    and not is_capture
                    and not move.promotion
                    and not board.is_check()
                ):
                    reduction = self._lmr_reduction(depth, idx, move, board)

                score = self.search(
                    board,
                    depth - 1 - reduction,
                    alpha,
                    beta,
                    ai_color,
                    move_index=idx,
                )
            finally:
                board.pop()
            if reduction and score < beta:
                board.push(move)
                try:
                    score = self.search(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        ai_color,
                        move_index=idx,
                    )
                finally:
                    board.pop()
            best = min(best, score)
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    # ------------------------------------------------------------------
    # Quiescence search — resolve captures and checks before evaluating
    # ------------------------------------------------------------------

    def quiescence_search(
        self,
        board: chess.Board,
        ai_color: bool,
        alpha: int,
        beta: int,
        depth: int,
    ) -> int:
        stand_pat = evaluate_position(board, ai_color)

        if depth <= 0 or board.is_game_over(claim_draw=True):
            return stand_pat

        if board.turn == ai_color:
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat
            for move in self._noisy_moves(board):
                board.push(move)
                try:
                    score = self.quiescence_search(board, ai_color, alpha, beta, depth - 1)
                finally:
                    board.pop()
                if score > best:
                    best = score
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
            return best

        if stand_pat <= alpha:
            return stand_pat
        if stand_pat < beta:
            beta = stand_pat
        best = stand_pat
        for move in self._noisy_moves(board):
            board.push(move)
            try:
                score = self.quiescence_search(board, ai_color, alpha, beta, depth - 1)
            finally:
                board.pop()
            if score < best:
                best = score
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best

    # ------------------------------------------------------------------
    # Move ordering — captures first (MVV-LVA), then killers / quiet
    # ------------------------------------------------------------------

    def order_moves(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        return sorted(moves, key=lambda move: self.move_hint(board, move), reverse=True)

    def move_hint(self, board: chess.Board, move: chess.Move) -> int:
        score = 0

        moving_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)

        if board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)

        if _is_immediate_checkmate(board, move):
            score += 2_000_000

        score += _urgent_safety_move_score(board, move)

        # MVV-LVA for captures
        if captured_piece is not None:
            victim = PIECE_VALUES[captured_piece.piece_type]
            attacker = PIECE_VALUES[moving_piece.piece_type] if moving_piece else 0
            score += 10 * victim - attacker
            score += 3 * _capture_gain(board, move)

        if move.promotion:
            score += PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]

        # PST bonus for the target square
        if moving_piece is not None:
            score += _pst_value(moving_piece.piece_type, move.to_square, moving_piece.color) // 4
            score += _move_tactical_bonus(board, move)
            if moving_piece.piece_type == chess.KNIGHT and chess.square_file(move.to_square) in (0, 7):
                score -= RIM_KNIGHT_PENALTY
            if moving_piece.piece_type in MINOR_PIECES and captured_piece == chess.Piece(chess.PAWN, not moving_piece.color):
                score -= UNSAFE_MINOR_PAWN_CAPTURE_PENALTY // 2

        # Center control
        if move.to_square in CENTER_SQUARES:
            score += 35
        elif move.to_square in NEAR_CENTER_SQUARES:
            score += 15

        # Check bonus
        board.push(move)
        try:
            if board.is_check():
                score += 80
        finally:
            board.pop()

        score -= HANGING_MOVE_PENALTY_SCALE * _move_hanging_penalty(board, move)

        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _noisy_moves(self, board: chess.Board) -> list[chess.Move]:
        if board.is_check():
            return self.order_moves(board, list(board.legal_moves))

        moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion:
                moves.append(move)
                continue
            board.push(move)
            try:
                if board.is_check():
                    moves.append(move)
            finally:
                board.pop()
        # Sort captures by MVV-LVA
        moves.sort(key=lambda m: self.move_hint(board, m), reverse=True)
        return moves

    @staticmethod
    def _has_pieces(board: chess.Board, color: bool) -> bool:
        """True if *color* has pieces other than pawns and king."""
        for piece in board.piece_map().values():
            if piece.color == color and piece.piece_type not in (chess.PAWN, chess.KING):
                return True
        return False

    @staticmethod
    def _lmr_reduction(depth: int, move_index: int, move: chess.Move, board: chess.Board) -> int:
        """Compute late-move reduction (1-3 ply)."""
        r = 1
        if move_index >= 8:
            r = 2
        if move_index >= 16 and depth >= 5:
            r = 3
        # Don't reduce as much for moves that look tactically interesting
        if board.is_capture(move) or move.promotion:
            r = max(0, r - 1)
        return r
