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
    parser.add_argument("--gui", action="store_true", help="Use graphical mouse UI with large centered pieces.")
    parser.add_argument("--web", action="store_true", help="Run the local browser-based chess app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for --web mode.")
    parser.add_argument("--port", type=int, default=8765, help="Port for --web mode.")
    parser.add_argument("--color", type=parse_color, default=PlayerColor.WHITE, help="Human color: white or black.")
    args = parser.parse_args()

    if args.web:
        from chess_app.web_app import run_web

        run_web(host=args.host, port=args.port, human_color=args.color)
    elif args.gui:
        from chess_app.pygame_app import run_pygame

        run_pygame(args.color)
    elif args.tui:
        run_textual(args.color)
    else:
        run_terminal(args.color)


if __name__ == "__main__":
    main()
