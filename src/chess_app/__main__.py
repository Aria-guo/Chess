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
    parser.add_argument("--lichess-bot", action="store_true", help="Connect this engine to a Lichess Bot account.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for --web mode.")
    parser.add_argument("--port", type=int, default=8765, help="Port for --web mode.")
    parser.add_argument("--color", type=parse_color, default=PlayerColor.WHITE, help="Human color: white or black.")
    parser.add_argument("--lichess-username", default=None, help="Bot username for --lichess-bot.")
    parser.add_argument("--lichess-move-time", type=float, default=None, help="Seconds to think per Lichess move.")
    parser.add_argument("--lichess-max-depth", type=int, default=None, help="Maximum Lichess search depth.")
    parser.add_argument("--lichess-accept-rated", action="store_true", help="Allow rated Lichess games. This is now the default.")
    parser.add_argument("--lichess-casual-only", action="store_true", help="Decline rated Lichess games.")
    parser.add_argument("--lichess-stockfish-sweep", action="store_true", help="Challenge Lichess AI levels 1-8 and train.")
    parser.add_argument("--lichess-games-per-level", type=int, default=1, help="AI games to play at each level.")
    parser.add_argument("--lichess-review-rounds", type=int, default=3, help="Training rounds after each AI game.")
    parser.add_argument("--lichess-ai-color", choices=["random", "white", "black"], default="random", help="Color against Lichess AI.")
    parser.add_argument("--lichess-clock-limit", type=int, default=300, help="Lichess AI clock initial seconds.")
    parser.add_argument("--lichess-clock-increment", type=int, default=3, help="Lichess AI clock increment seconds.")
    args = parser.parse_args()

    if args.lichess_stockfish_sweep:
        from chess_app.lichess_bot import run_lichess_stockfish_sweep
        from chess_app.neural_trainer import ENGINE_MAX_DEPTH, ENGINE_MOVE_TIME_SECONDS

        run_lichess_stockfish_sweep(
            username=args.lichess_username,
            move_time_seconds=args.lichess_move_time or ENGINE_MOVE_TIME_SECONDS,
            max_depth=args.lichess_max_depth or ENGINE_MAX_DEPTH,
            games_per_level=args.lichess_games_per_level,
            review_rounds=args.lichess_review_rounds,
            color=args.lichess_ai_color,
            clock_limit_seconds=args.lichess_clock_limit,
            clock_increment_seconds=args.lichess_clock_increment,
        )
    elif args.lichess_bot:
        from chess_app.lichess_bot import run_lichess_bot
        from chess_app.neural_trainer import ENGINE_MAX_DEPTH, ENGINE_MOVE_TIME_SECONDS

        run_lichess_bot(
            username=args.lichess_username,
            move_time_seconds=args.lichess_move_time or ENGINE_MOVE_TIME_SECONDS,
            max_depth=args.lichess_max_depth or ENGINE_MAX_DEPTH,
            accept_rated=not args.lichess_casual_only,
        )
    elif args.web:
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
