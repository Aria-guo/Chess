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
    """Small repertoire book built from curated PGN-style move lines."""

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

    queen_gambit_lines = [
        ("Queen's Gambit Declined: Orthodox", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6"),
        ("Queen's Gambit Declined: Exchange", "d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O Bd3"),
        ("Queen's Gambit Declined: Tarrasch", "d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6 Bg2 Be7"),
        ("Queen's Gambit Accepted", "d4 d5 c4 dxc4 e3 Nf6 Bxc4 e6 Nf3 c5 O-O a6"),
        ("Slav Defense", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4"),
        ("Semi-Slav Defense", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4"),
        ("Albin Countergambit", "d4 d5 c4 e5 dxe5 d4 Nf3 Nc6 g3 Be6 Bg2 Qd7"),
        ("Chigorin Defense", "d4 d5 c4 Nc6 Nf3 Bg4 cxd5 Bxf3 gxf3 Qxd5 e3 e5"),
    ]

    sicilian_lines = [
        ("Sicilian Defense: Najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6"),
        ("Sicilian Defense: Classical", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6"),
        ("Sicilian Defense: Dragon", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7"),
        ("Sicilian Defense: Accelerated Dragon", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 Nc3 Bg7"),
        ("Sicilian Defense: Kan", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6 O-O Qc7"),
        ("Sicilian Defense: Alapin", "e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 e6 Be2"),
        ("Sicilian Defense: Closed", "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 f4 e6"),
    ]

    dutch_lines = [
        ("Dutch Defense: Classical", "d4 f5 c4 Nf6 g3 e6 Bg2 Be7 Nf3 O-O O-O d6 Nc3 Qe8"),
        ("Dutch Defense: Leningrad", "d4 f5 c4 Nf6 g3 g6 Bg2 Bg7 Nf3 O-O O-O d6"),
        ("Dutch Defense vs London", "d4 f5 Nf3 Nf6 Bf4 e6 e3 Be7 h3 O-O Bd3 d6"),
        ("Dutch Defense vs Colle", "d4 f5 Nf3 Nf6 e3 e6 Bd3 Be7 O-O O-O c4 d6"),
        ("Dutch Defense vs English", "c4 f5 Nc3 Nf6 g3 e6 Bg2 Be7 Nf3 O-O O-O d6"),
        ("Dutch Defense vs Zukertort", "Nf3 f5 d4 Nf6 g3 e6 Bg2 Be7 O-O O-O c4 d6"),
    ]

    for name, line in queen_gambit_lines:
        book.add_line(name, line, weight=4, repertoire_color=chess.WHITE)
    for name, line in sicilian_lines:
        book.add_line(name, line, weight=5, repertoire_color=chess.BLACK)
    for name, line in dutch_lines:
        book.add_line(name, line, weight=5, repertoire_color=chess.BLACK)

    return book
