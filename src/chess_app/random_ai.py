from __future__ import annotations

import random

import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = {
    chess.D4,
    chess.E4,
    chess.D5,
    chess.E5,
}

NEAR_CENTER_SQUARES = {
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.F4,
    chess.C5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
}


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
    """Simple rule-based chess AI.

    This is intentionally lightweight: it uses a shallow minimax search with
    alpha-beta pruning and evaluates material, checks, mobility, and center
    control.
    """

    def __init__(self, seed: int | None = None, search_depth: int = 2) -> None:
        super().__init__(seed=seed)
        self.search_depth = search_depth

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        color = board.turn
        scored_moves = []
        for move in self.order_moves(board, legal_moves):
            board.push(move)
            try:
                score = self.search(
                    board,
                    depth=self.search_depth - 1,
                    alpha=-1_000_000_000,
                    beta=1_000_000_000,
                    ai_color=color,
                )
            finally:
                board.pop()
            scored_moves.append((score + self.move_hint(board, move), move))

        best_score = max(score for score, _ in scored_moves)
        best_moves = [move for score, move in scored_moves if score == best_score]
        return self.rng.choice(best_moves)

    def search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ai_color: bool,
    ) -> int:
        if depth <= 0 or board.is_game_over(claim_draw=True):
            return self.evaluate_board(board, ai_color)

        legal_moves = list(board.legal_moves)
        if board.turn == ai_color:
            best = -1_000_000_000
            for move in self.order_moves(board, legal_moves):
                board.push(move)
                try:
                    score = self.search(board, depth - 1, alpha, beta, ai_color)
                finally:
                    board.pop()
                best = max(best, score)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best

        best = 1_000_000_000
        for move in self.order_moves(board, legal_moves):
            board.push(move)
            try:
                score = self.search(board, depth - 1, alpha, beta, ai_color)
            finally:
                board.pop()
            best = min(best, score)
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    def order_moves(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        return sorted(moves, key=lambda move: self.move_hint(board, move), reverse=True)

    def move_hint(self, board: chess.Board, move: chess.Move) -> int:
        score = 0

        moving_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)

        if board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)

        if captured_piece is not None:
            victim = PIECE_VALUES[captured_piece.piece_type]
            attacker = PIECE_VALUES[moving_piece.piece_type] if moving_piece else 0
            score += 10 * victim - attacker

        if move.promotion:
            score += PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]

        if move.to_square in CENTER_SQUARES:
            score += 35
        elif move.to_square in NEAR_CENTER_SQUARES:
            score += 15

        board.push(move)
        try:
            if board.is_check():
                score += 80
        finally:
            board.pop()

        return score

    def evaluate_board(self, board: chess.Board, color: bool) -> int:
        if board.is_checkmate():
            return -1_000_000 if board.turn == color else 1_000_000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        material = 0
        center = 0

        for square, piece in board.piece_map().items():
            sign = 1 if piece.color == color else -1
            material += sign * PIECE_VALUES[piece.piece_type]
            if square in CENTER_SQUARES:
                center += sign * 20
            elif square in NEAR_CENTER_SQUARES:
                center += sign * 8

        own_mobility = self.count_legal_moves_for(board, color)
        enemy_mobility = self.count_legal_moves_for(board, not color)

        return material + center + 2 * (own_mobility - enemy_mobility)

    def count_legal_moves_for(self, board: chess.Board, color: bool) -> int:
        original_turn = board.turn
        board.turn = color
        try:
            return board.legal_moves.count()
        finally:
            board.turn = original_turn
