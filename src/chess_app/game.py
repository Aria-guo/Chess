from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import chess


class PlayerColor(str, Enum):
    WHITE = "white"
    BLACK = "black"

    @property
    def chess_color(self) -> bool:
        return chess.WHITE if self is PlayerColor.WHITE else chess.BLACK

    @property
    def opposite(self) -> "PlayerColor":
        return PlayerColor.BLACK if self is PlayerColor.WHITE else PlayerColor.WHITE


@dataclass(slots=True)
class MoveResult:
    ok: bool
    message: str
    move: chess.Move | None = None


class ChessGame:
    """Small wrapper around python-chess for app-level behavior."""

    def __init__(self, human_color: PlayerColor = PlayerColor.WHITE) -> None:
        self.board = chess.Board()
        self.human_color = human_color

    @property
    def ai_color(self) -> PlayerColor:
        return self.human_color.opposite

    @property
    def is_human_turn(self) -> bool:
        return self.board.turn == self.human_color.chess_color

    @property
    def is_ai_turn(self) -> bool:
        return self.board.turn == self.ai_color.chess_color

    def reset(self) -> None:
        self.board.reset()

    def legal_moves_uci(self) -> list[str]:
        return [move.uci() for move in self.board.legal_moves]

    def parse_move(self, text: str) -> MoveResult:
        raw = text.strip()
        if not raw:
            return MoveResult(False, "Enter a move, for example e2e4 or Nf3.")

        normalized = raw.replace("-", "").replace(" ", "")

        try:
            move = self.board.parse_san(raw)
        except ValueError:
            try:
                move = chess.Move.from_uci(normalized.lower())
            except ValueError:
                return MoveResult(False, f"Could not understand move: {raw}")

        if move not in self.board.legal_moves:
            return MoveResult(False, f"Illegal move in this position: {raw}", move)

        return MoveResult(True, "Move accepted.", move)

    def push_human_move(self, text: str) -> MoveResult:
        result = self.parse_move(text)
        if result.ok and result.move is not None:
            san = self.board.san(result.move)
            self.board.push(result.move)
            return MoveResult(True, san, result.move)
        return result

    def push_ai_move(self, move: chess.Move) -> str:
        san = self.board.san(move)
        self.board.push(move)
        return san

    def status(self) -> str:
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"Checkmate. {winner} wins."
        if self.board.is_stalemate():
            return "Draw by stalemate."
        if self.board.is_insufficient_material():
            return "Draw by insufficient material."
        if self.board.can_claim_threefold_repetition():
            return "Threefold repetition can be claimed."
        if self.board.can_claim_fifty_moves():
            return "Fifty-move draw can be claimed."
        if self.board.is_check():
            return "Check."
        return "Game in progress."

    def result(self) -> str | None:
        outcome = self.board.outcome(claim_draw=True)
        return outcome.result() if outcome else None

