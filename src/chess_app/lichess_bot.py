from __future__ import annotations

import argparse
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator

import chess
import certifi

from chess_app.neural_trainer import ENGINE_MAX_DEPTH, ENGINE_MOVE_TIME_SECONDS, NeuralSelfTrainer
from chess_app.random_ai import BasicAI


LICHESS_BASE_URL = "https://lichess.org"
LICHESS_STANDARD_SPEEDS = {"ultraBullet", "ultrabullet", "bullet", "blitz", "rapid", "classical"}
LICHESS_AI_LEVELS = tuple(range(1, 9))


class LichessApiError(RuntimeError):
    pass


class LichessClient:
    def __init__(self, token: str, base_url: str = LICHESS_BASE_URL) -> None:
        self.token = token.strip()
        self.base_url = base_url.rstrip("/")
        if not self.token:
            raise ValueError("LICHESS_TOKEN is empty.")
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        timeout: float | None = 20,
    ) -> Any:
        body = None
        headers = self.headers()
        if data is not None:
            body = urllib.parse.urlencode(data).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=self.ssl_context) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise LichessApiError(f"Lichess API {method} {path} failed: {exc.code} {detail}") from exc
        if not raw:
            return None
        text = raw.decode(errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def stream(self, path: str) -> Iterator[dict[str, Any]]:
        request = urllib.request.Request(self.base_url + path, headers=self.headers(), method="GET")
        with urllib.request.urlopen(request, timeout=None, context=self.ssl_context) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line:
                    continue
                yield json.loads(line)

    def account(self) -> dict[str, Any]:
        return self.request("GET", "/api/account")

    def stream_events(self) -> Iterator[dict[str, Any]]:
        yield from self.stream("/api/stream/event")

    def stream_game(self, game_id: str) -> Iterator[dict[str, Any]]:
        yield from self.stream(f"/api/bot/game/stream/{game_id}")

    def accept_challenge(self, challenge_id: str) -> None:
        self.request("POST", f"/api/challenge/{challenge_id}/accept", data={})

    def decline_challenge(self, challenge_id: str, reason: str = "generic") -> None:
        self.request("POST", f"/api/challenge/{challenge_id}/decline", data={"reason": reason})

    def make_move(self, game_id: str, move_uci: str) -> None:
        self.request("POST", f"/api/bot/game/{game_id}/move/{move_uci}", data={})

    def challenge_ai(
        self,
        level: int,
        color: str = "random",
        clock_limit_seconds: int = 300,
        clock_increment_seconds: int = 3,
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            "/api/challenge/ai",
            data={
                "level": int(level),
                "color": color,
                "variant": "standard",
                "clock.limit": int(clock_limit_seconds),
                "clock.increment": int(clock_increment_seconds),
            },
        )

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/x-ndjson, application/json",
            "User-Agent": "Resazurin local chess bot",
        }


@dataclass(frozen=True, slots=True)
class LichessBotConfig:
    username: str
    accept_rated: bool = True
    move_time_seconds: float = ENGINE_MOVE_TIME_SECONDS
    max_depth: int = ENGINE_MAX_DEPTH
    allowed_variants: set[str] = field(default_factory=lambda: {"standard"})
    allowed_speeds: set[str] = field(default_factory=lambda: set(LICHESS_STANDARD_SPEEDS))


@dataclass(frozen=True, slots=True)
class LichessGameRecord:
    game_id: str
    moves: list[chess.Move]
    result: str
    status: str


class LichessBot:
    def __init__(
        self,
        client: LichessClient,
        config: LichessBotConfig,
        trainer: NeuralSelfTrainer | None = None,
        ai: BasicAI | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.trainer = trainer or NeuralSelfTrainer()
        self.ai = ai or BasicAI()
        self.game_threads: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

    def run_forever(self) -> None:
        print(f"Connected as {self.config.username}. Waiting for Lichess challenges...")
        while True:
            try:
                for event in self.client.stream_events():
                    self.handle_event(event)
            except Exception as exc:
                print(f"Event stream disconnected: {exc}. Reconnecting in 5 seconds...")
                time.sleep(5)

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "challenge":
            self.handle_challenge(event.get("challenge", {}))
        elif event_type == "gameStart":
            game_id = str(event.get("game", {}).get("id", ""))
            if game_id:
                self.start_game_thread(game_id)

    def handle_challenge(self, challenge: dict[str, Any]) -> None:
        challenge_id = str(challenge.get("id", ""))
        if not challenge_id:
            return

        variant = str(challenge.get("variant", {}).get("key", ""))
        speed = str(challenge.get("speed", ""))
        rated = bool(challenge.get("rated", False))
        challenger = challenge.get("challenger", {}).get("name", "unknown")

        reason = self.challenge_decline_reason(variant=variant, speed=speed, rated=rated)
        if reason is not None:
            print(f"Declining challenge {challenge_id} from {challenger}: {reason}.")
            try:
                self.client.decline_challenge(challenge_id, reason="generic")
            except LichessApiError as exc:
                print(exc)
            return

        print(f"Accepting challenge {challenge_id} from {challenger}: {speed}, rated={rated}.")
        self.client.accept_challenge(challenge_id)

    def challenge_decline_reason(self, variant: str, speed: str, rated: bool) -> str | None:
        if variant not in self.config.allowed_variants:
            return "only standard chess is enabled"
        if speed not in self.config.allowed_speeds:
            return "time control is too fast or unsupported"
        if rated and not self.config.accept_rated:
            return "rated games are disabled"
        return None

    def start_game_thread(self, game_id: str) -> None:
        with self.lock:
            thread = self.game_threads.get(game_id)
            if thread is not None and thread.is_alive():
                return
            thread = threading.Thread(target=self.play_game, args=(game_id,), daemon=True)
            self.game_threads[game_id] = thread
            thread.start()

    def play_game(self, game_id: str) -> LichessGameRecord | None:
        print(f"Starting game {game_id}.")
        initial_fen = chess.STARTING_FEN
        bot_color: bool | None = None
        move_time_seconds = self.config.move_time_seconds
        last_move_request_ply = -1

        try:
            for event in self.client.stream_game(game_id):
                event_type = event.get("type")
                if event_type == "gameFull":
                    initial_fen = parse_initial_fen(event.get("initialFen"))
                    bot_color = self.bot_color_from_game_full(event)
                    move_time_seconds = self.move_time_for_game(event)
                    state = event.get("state", {})
                elif event_type == "gameState":
                    state = event
                else:
                    continue

                status = str(state.get("status", "started"))
                if status != "started":
                    print(f"Game {game_id} ended: {status}.")
                    board = board_from_lichess_state(initial_fen, str(state.get("moves", "")))
                    return LichessGameRecord(
                        game_id=game_id,
                        moves=list(board.move_stack),
                        result=result_from_lichess_state(board, state),
                        status=status,
                    )

                if bot_color is None:
                    continue

                board = board_from_lichess_state(initial_fen, str(state.get("moves", "")))
                current_ply = len(board.move_stack)
                if board.turn != bot_color or current_ply == last_move_request_ply:
                    continue

                move = self.choose_move(board, move_time_seconds=move_time_seconds)
                if move is None:
                    print(f"Game {game_id}: no legal move available.")
                    return None

                san = board.san(move)
                print(f"Game {game_id}: playing {san} ({move.uci()}).")
                self.client.make_move(game_id, move.uci())
                last_move_request_ply = current_ply
        except Exception as exc:
            print(f"Game {game_id} stopped because of an error: {exc}")
        return None

    def play_stockfish_sweep(
        self,
        games_per_level: int = 1,
        review_rounds: int = 3,
        color: str = "random",
        clock_limit_seconds: int = 300,
        clock_increment_seconds: int = 3,
    ) -> None:
        games_per_level = max(1, games_per_level)
        review_rounds = max(1, review_rounds)
        for round_index in range(games_per_level):
            for level in LICHESS_AI_LEVELS:
                print(f"Challenging Lichess AI level {level}, sweep round {round_index + 1}/{games_per_level}.")
                response = self.client.challenge_ai(
                    level=level,
                    color=color,
                    clock_limit_seconds=clock_limit_seconds,
                    clock_increment_seconds=clock_increment_seconds,
                )
                game_id = game_id_from_challenge_response(response)
                if not game_id:
                    print(f"Could not find game id in challenge response: {response}")
                    continue

                record = self.play_game(game_id)
                if record is None:
                    continue
                print(
                    f"Finished AI level {level} game {record.game_id}: "
                    f"{record.result}, {len(record.moves)} plies."
                )
                self.trainer.train_finished_game_moves(
                    record.moves,
                    record.result,
                    review_rounds=review_rounds,
                    label=f"Lichess AI level {level} review",
                )

    def bot_color_from_game_full(self, event: dict[str, Any]) -> bool | None:
        username = self.config.username.lower()
        white = user_identity(event.get("white", {}))
        black = user_identity(event.get("black", {}))
        if username in white:
            return chess.WHITE
        if username in black:
            return chess.BLACK
        print(f"Could not determine bot color. white={white}, black={black}")
        return None

    def move_time_for_game(self, event: dict[str, Any]) -> float:
        clock = event.get("clock", {})
        speed = str(event.get("speed", ""))
        configured = max(0.05, self.config.move_time_seconds)
        if speed in {"ultraBullet", "ultrabullet"}:
            return min(configured, 0.12)
        if speed == "bullet":
            return min(configured, 0.35)
        if speed == "blitz":
            return min(configured, 1.5)
        if isinstance(clock, dict):
            initial = float(clock.get("initial", 0) or 0)
            increment = float(clock.get("increment", 0) or 0)
            if initial > 0:
                budget = initial / 80.0 + increment * 0.6
                return max(0.08, min(configured, budget))
        return configured

    def choose_move(self, board: chess.Board, move_time_seconds: float | None = None) -> chess.Move | None:
        self.ai.last_book_name = None
        book_move = self.ai.opening_book.choose(board)
        if book_move is not None:
            self.ai.last_book_name = book_move.name
            return book_move.move
        return self.trainer.choose_engine_move(
            board,
            self.ai,
            time_limit_seconds=move_time_seconds or self.config.move_time_seconds,
            max_depth=self.config.max_depth,
        )


def user_identity(player: dict[str, Any]) -> set[str]:
    user = player.get("user", {}) if isinstance(player, dict) else {}
    names = {
        str(player.get("id", "")).lower() if isinstance(player, dict) else "",
        str(player.get("name", "")).lower() if isinstance(player, dict) else "",
        str(user.get("id", "")).lower() if isinstance(user, dict) else "",
        str(user.get("name", "")).lower() if isinstance(user, dict) else "",
        str(user.get("username", "")).lower() if isinstance(user, dict) else "",
    }
    return {name for name in names if name}


def parse_initial_fen(value: Any) -> str:
    if value in {None, "", "startpos"}:
        return chess.STARTING_FEN
    return str(value)


def board_from_lichess_state(initial_fen: str, moves: str) -> chess.Board:
    board = chess.Board(initial_fen)
    for move_uci in moves.split():
        board.push(chess.Move.from_uci(move_uci))
    return board


def result_from_lichess_state(board: chess.Board, state: dict[str, Any]) -> str:
    winner = str(state.get("winner", "")).lower()
    if winner == "white":
        return "1-0"
    if winner == "black":
        return "0-1"

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return outcome.result()

    status = str(state.get("status", "")).lower()
    if status in {"draw", "stalemate", "insufficientmaterial"}:
        return "1/2-1/2"
    return "*"


def game_id_from_challenge_response(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    for key in ("id", "gameId"):
        value = response.get(key)
        if value:
            return str(value)
    for key in ("game", "challenge"):
        nested = response.get(key)
        if isinstance(nested, dict):
            value = nested.get("id") or nested.get("gameId")
            if value:
                return str(value)
    return None


def username_from_env_or_account(client: LichessClient) -> str:
    env_username = os.environ.get("LICHESS_BOT_USERNAME", "").strip()
    if env_username:
        return env_username
    account = client.account()
    username = str(account.get("username") or account.get("id") or "").strip()
    if not username:
        raise LichessApiError("Could not read bot username. Set LICHESS_BOT_USERNAME manually.")
    return username


def run_lichess_bot(
    token: str | None = None,
    username: str | None = None,
    move_time_seconds: float = ENGINE_MOVE_TIME_SECONDS,
    max_depth: int = ENGINE_MAX_DEPTH,
    accept_rated: bool = True,
) -> None:
    token = token or os.environ.get("LICHESS_TOKEN", "")
    client = LichessClient(token)
    username = username or username_from_env_or_account(client)
    config = LichessBotConfig(
        username=username,
        move_time_seconds=move_time_seconds,
        max_depth=max_depth,
        accept_rated=accept_rated,
    )
    bot = LichessBot(client=client, config=config)
    bot.run_forever()


def run_lichess_stockfish_sweep(
    token: str | None = None,
    username: str | None = None,
    move_time_seconds: float = ENGINE_MOVE_TIME_SECONDS,
    max_depth: int = ENGINE_MAX_DEPTH,
    games_per_level: int = 1,
    review_rounds: int = 3,
    color: str = "random",
    clock_limit_seconds: int = 300,
    clock_increment_seconds: int = 3,
) -> None:
    token = token or os.environ.get("LICHESS_TOKEN", "")
    client = LichessClient(token)
    username = username or username_from_env_or_account(client)
    config = LichessBotConfig(
        username=username,
        move_time_seconds=move_time_seconds,
        max_depth=max_depth,
    )
    bot = LichessBot(client=client, config=config)
    bot.play_stockfish_sweep(
        games_per_level=games_per_level,
        review_rounds=review_rounds,
        color=color,
        clock_limit_seconds=clock_limit_seconds,
        clock_increment_seconds=clock_increment_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Resazurin engine as a Lichess Bot.")
    parser.add_argument("--token", default=None, help="Lichess API token. Prefer LICHESS_TOKEN instead.")
    parser.add_argument("--username", default=None, help="Bot username. Prefer LICHESS_BOT_USERNAME instead.")
    parser.add_argument("--move-time", type=float, default=ENGINE_MOVE_TIME_SECONDS, help="Seconds to think per move.")
    parser.add_argument("--max-depth", type=int, default=ENGINE_MAX_DEPTH, help="Maximum search depth.")
    parser.add_argument("--accept-rated", action="store_true", help="Accept rated games. This is now the default.")
    parser.add_argument("--casual-only", action="store_true", help="Decline rated games and accept casual games only.")
    parser.add_argument("--stockfish-sweep", action="store_true", help="Challenge Lichess AI levels 1-8 and train after each game.")
    parser.add_argument("--games-per-level", type=int, default=1, help="Lichess AI games to play at each level.")
    parser.add_argument("--review-rounds", type=int, default=3, help="Training review rounds after each Lichess AI game.")
    parser.add_argument("--color", default="random", choices=["random", "white", "black"], help="Color against Lichess AI.")
    parser.add_argument("--clock-limit", type=int, default=300, help="Lichess AI clock initial seconds.")
    parser.add_argument("--clock-increment", type=int, default=3, help="Lichess AI clock increment seconds.")
    args = parser.parse_args()
    if args.stockfish_sweep:
        run_lichess_stockfish_sweep(
            token=args.token,
            username=args.username,
            move_time_seconds=args.move_time,
            max_depth=args.max_depth,
            games_per_level=args.games_per_level,
            review_rounds=args.review_rounds,
            color=args.color,
            clock_limit_seconds=args.clock_limit,
            clock_increment_seconds=args.clock_increment,
        )
        return
    run_lichess_bot(
        token=args.token,
        username=args.username,
        move_time_seconds=args.move_time,
        max_depth=args.max_depth,
        accept_rated=not args.casual_only,
    )


if __name__ == "__main__":
    main()
