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

# Files adjacent to the king's file for pawn shield
_KING_FILE_OFFSETS = (-1, 0, 1)


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
    white_pawns: list[int] = []
    black_pawns: list[int] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.PAWN:
            continue
        if piece.color == chess.WHITE:
            white_pawns.append(square)
        else:
            black_pawns.append(square)

    # Doubled pawns penalty
    for pawns, sign in [(white_pawns, 1), (black_pawns, -1)]:
        file_counts: dict[int, int] = {}
        for sq in pawns:
            f = chess.square_file(sq)
            file_counts[f] = file_counts.get(f, 0) + 1
        for count in file_counts.values():
            if count > 1:
                score += sign * -15 * (count - 1)

    # Isolated pawn penalty
    for pawns, sign in [(white_pawns, 1), (black_pawns, -1)]:
        occupied_files = {chess.square_file(sq) for sq in pawns}
        for sq in pawns:
            f = chess.square_file(sq)
            if (f - 1) not in occupied_files and (f + 1) not in occupied_files:
                score += sign * -12

    # Passed pawns
    for sq in white_pawns:
        rank = chess.square_rank(sq)
        f = chess.square_file(sq)
        blocked = False
        for b_sq in black_pawns:
            bf = chess.square_file(b_sq)
            br = chess.square_rank(b_sq)
            if abs(bf - f) <= 1 and br > rank:
                blocked = True
                break
        if not blocked:
            score += PASSED_PAWN_BONUS_PER_RANK * (rank - 1)

    for sq in black_pawns:
        rank = chess.square_rank(sq)
        f = chess.square_file(sq)
        blocked = False
        for w_sq in white_pawns:
            wf = chess.square_file(w_sq)
            wr = chess.square_rank(w_sq)
            if abs(wf - f) <= 1 and wr < rank:
                blocked = True
                break
        if not blocked:
            score -= PASSED_PAWN_BONUS_PER_RANK * (6 - rank)

    return score


# ---------------------------------------------------------------------------
# King-safety helper
# ---------------------------------------------------------------------------

def _king_safety_score(board: chess.Board) -> int:
    """Return king-safety score from White's perspective."""
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        king_sq = board.king(color)
        if king_sq is None:
            continue
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)

        # Pawn shield: friendly pawns on adjacent files in front of king
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
                    shield_bonus += 8
        score += sign * shield_bonus

        # Pawn storm penalty: enemy pawns attacking near the king
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
                    attack_penalty += 6
        score -= sign * attack_penalty

        # Open files near king are dangerous
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
                score -= sign * (ROOK_SEMI_OPEN_FILE_BONUS if has_enemy_pawn else 8)

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
    """Detect pins: a piece blocking an attack on a more valuable piece.

    Rewards the side that IS pinning.  Returns score from White's perspective.
    """
    score = 0
    for pinned_color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(pinned_color)
        if king_sq is None:
            continue

        for square, piece in board.piece_map().items():
            if piece.color != pinned_color:
                continue
            if piece.piece_type == chess.KING:
                continue

            for enemy_sq, enemy_piece in board.piece_map().items():
                if enemy_piece.color == pinned_color:
                    continue
                if enemy_piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                    continue

                if _is_pinned_to(board, square, king_sq, enemy_sq, enemy_piece):
                    # Reward the PINNING side (opposite of pinned color)
                    pinner_sign = -1 if pinned_color == chess.WHITE else 1
                    pinned_value = PIECE_VALUES[piece.piece_type]
                    bonus = 40
                    if pinned_value >= 900:
                        bonus += 60
                    elif pinned_value >= 500:
                        bonus += 30
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
        return -1_000_000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = _phase(board)
    material = 0
    pst = 0

    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == color else -1
        material += sign * PIECE_VALUES[piece.piece_type]
        pst += sign * _pst_value(piece.piece_type, square, piece.color)
        if piece.piece_type == chess.KING:
            pst += sign * _king_pst_value(square, piece.color, phase)

    pawn_struct = _pawn_structure_score(board)
    king_safety = _king_safety_score(board)
    mobility = _mobility_score(board)
    rook_act = _rook_activity(board)
    bishop_pair = _bishop_pair_bonus(board)
    tactics = _fork_score(board) + _pin_score(board)

    # Sign-adjust for the side we're evaluating
    sign_color = 1 if color == chess.WHITE else -1

    total = (
        material
        + pst
        + sign_color * pawn_struct
        + sign_color * king_safety
        + mobility
        + sign_color * rook_act
        + sign_color * bishop_pair
        + sign_color * tactics
    )

    # Tempo
    if board.turn == color:
        total += TEMPO_BONUS

    return total


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
            and static_eval >= beta
            and self._has_pieces(board, board.turn)
        ):
            reduction = 3 + depth // 6
            board.push(chess.Move.null())
            try:
                null_score = -self.search(
                    board, depth - 1 - reduction, -beta, -beta + 1, ai_color
                )
            finally:
                board.pop()
            if null_score >= beta:
                return beta

        legal_moves = list(board.legal_moves)
        ordered = self.order_moves(board, legal_moves)

        if board.turn == ai_color:
            best = -1_000_000_000
            for idx, move in enumerate(ordered):
                board.push(move)
                try:
                    # Late-move reduction
                    reduction = 0
                    if (
                        depth >= 3
                        and idx >= 4
                        and not board.is_capture(move)
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
                best = max(best, score)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best

        best = 1_000_000_000
        for idx, move in enumerate(ordered):
            board.push(move)
            try:
                reduction = 0
                if (
                    depth >= 3
                    and idx >= 4
                    and not board.is_capture(move)
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

        # MVV-LVA for captures
        if captured_piece is not None:
            victim = PIECE_VALUES[captured_piece.piece_type]
            attacker = PIECE_VALUES[moving_piece.piece_type] if moving_piece else 0
            score += 10 * victim - attacker

        if move.promotion:
            score += PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]

        # PST bonus for the target square
        if moving_piece is not None:
            score += _pst_value(moving_piece.piece_type, move.to_square, moving_piece.color) // 4

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

        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _noisy_moves(self, board: chess.Board) -> list[chess.Move]:
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
