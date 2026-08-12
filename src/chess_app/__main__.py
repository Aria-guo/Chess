from __future__ import annotations

import argparse

from chess_app.game import PlayerColor
from chess_app.terminal import run_terminal
from chess_app.textual_app import run_textual


def parse_color(value: str) -> PlayerColor:
    normalized = value.lower()
    if normalized in {"white", "w"}:
        return PlayerColor.WHITE
    if normalized in {"black", "b"}:
        return PlayerColor.BLACK
    raise argparse.ArgumentTypeError("color must be white or black")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play terminal chess against a random legal AI.")
    parser.add_argument("--tui", action="store_true", help="Use mouse/touch-friendly Textual UI.")
    parser.add_argument("--color", type=parse_color, default=PlayerColor.WHITE, help="Human color: white or black.")
    args = parser.parse_args()

    if args.tui:
        run_textual(args.color)
    else:
        run_terminal(args.color)


if __name__ == "__main__":
    main()

