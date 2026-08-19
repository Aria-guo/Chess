from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import chess


STARTING_KEY: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BookMove:
    move: chess.Move
    name: str
    weight: int = 1


class OpeningBook:
    """Small repertoire book for first-move personality choices."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self._moves: dict[tuple[str, ...], list[BookMove]] = defaultdict(list)

    def add_line(
        self,
        name: str,
        san_moves: str,
        weight: int = 1,
        repertoire_color: bool | None = None,
    ) -> None:
        board = chess.Board()
        for san in san_moves.split():
            key = self.key(board)
            move_color = board.turn
            move = board.parse_san(san)
            if repertoire_color is None or move_color == repertoire_color:
                self._add_move(key, BookMove(move=move, name=name, weight=weight))
            board.push(move)

    def choose(self, board: chess.Board) -> BookMove | None:
        entries = [
            entry
            for entry in self._moves.get(self.key(board), [])
            if entry.move in board.legal_moves
        ]
        if not entries:
            return None
        return self.rng.choices(entries, weights=[entry.weight for entry in entries], k=1)[0]

    def _add_move(self, key: tuple[str, ...], entry: BookMove) -> None:
        existing = self._moves[key]
        for index, current in enumerate(existing):
            if current.move == entry.move and current.name == entry.name:
                existing[index] = BookMove(entry.move, entry.name, current.weight + entry.weight)
                return
        existing.append(entry)

    @staticmethod
    def key(board: chess.Board) -> tuple[str, ...]:
        return tuple(move.uci() for move in board.move_stack)


def build_default_opening_book(seed: int | None = None) -> OpeningBook:
    book = OpeningBook(seed=seed)

    book.add_line("Queen's Pawn Opening", "d4", weight=1, repertoire_color=chess.WHITE)
    book.add_line("Sicilian Defense", "e4 c5", weight=1, repertoire_color=chess.BLACK)
    book.add_line("Dutch Defense", "d4 f5", weight=1, repertoire_color=chess.BLACK)

    return book
