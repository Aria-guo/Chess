from __future__ import annotations

import random

import chess


class RandomAI:
    """AI baseline that chooses one legal move uniformly at random."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return self.rng.choice(legal_moves)

