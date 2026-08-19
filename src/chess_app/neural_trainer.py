from __future__ import annotations

import json
import threading
from io import StringIO
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import time

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from chess_app.random_ai import (
    BasicAI,
    CENTER_SQUARES,
    NEAR_CENTER_SQUARES,
    evaluate_position,
    evaluate_white_cp,
)


PROMOTION_TO_INDEX = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
MOVE_POLICY_SIZE = 64 * 64 * len(PROMOTION_TO_INDEX)
PGN_REVIEW_ROUNDS = 5
PGN_LEARNING_RATE = 0.0005
POLICY_LOSS_WEIGHT = 0.35
SELF_PLAY_VALUE_LOSS_WEIGHT = 1.0
PGN_VALUE_LOSS_WEIGHT = 0.0
VALUE_TRUST_AFTER_SELF_PLAY_GAMES = 300
NEURAL_EVAL_WEIGHT = 0.35
PGN_CHUNK_POSITIONS = 50_000
PGN_BATCH_SIZE = 2048
ENGINE_MOVE_TIME_SECONDS = 5.0
ENGINE_MAX_DEPTH = 14
ENGINE_ROOT_CANDIDATES = 16
ENGINE_POLICY_WEIGHT = 400.0
ENGINE_VALUE_WEIGHT = 350.0
QUIESCENCE_MAX_DEPTH = 6
INPUT_CHANNELS = 22
MODEL_CHANNELS = 128
MODEL_BLOCKS = 8
NEURAL_SEARCH_WEIGHT = 0.65
HEURISTIC_SEARCH_WEIGHT = 0.35


PIECE_TO_CHANNEL = {
    chess.Piece(chess.PAWN, chess.WHITE): 0,
    chess.Piece(chess.KNIGHT, chess.WHITE): 1,
    chess.Piece(chess.BISHOP, chess.WHITE): 2,
    chess.Piece(chess.ROOK, chess.WHITE): 3,
    chess.Piece(chess.QUEEN, chess.WHITE): 4,
    chess.Piece(chess.KING, chess.WHITE): 5,
    chess.Piece(chess.PAWN, chess.BLACK): 6,
    chess.Piece(chess.KNIGHT, chess.BLACK): 7,
    chess.Piece(chess.BISHOP, chess.BLACK): 8,
    chess.Piece(chess.ROOK, chess.BLACK): 9,
    chess.Piece(chess.QUEEN, chess.BLACK): 10,
    chess.Piece(chess.KING, chess.BLACK): 11,
}


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)


class ResNetPolicyValueNet(nn.Module):
    """Chess ResNet with policy and value heads, inspired by AlphaZero / Stockfish NNUE."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        channels: int = MODEL_CHANNELS,
        blocks: int = MODEL_BLOCKS,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.blocks_count = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, MOVE_POLICY_SIZE),
        )

    def forward(self, x: torch.Tensor, policy_only: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)
        if policy_only:
            return None, self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        policy_logits = self.policy_head(x)
        return value, policy_logits


ResNetValueNet = ResNetPolicyValueNet


@dataclass
class TrainingStats:
    total_self_play_games: int = 0
    total_review_rounds: int = 0
    total_positions: int = 0
    total_pgn_games: int = 0
    total_pgn_positions: int = 0
    last_loss: float | None = None
    last_value_loss: float | None = None
    last_policy_loss: float | None = None
    running: bool = False
    message: str = "Neural trainer is idle."
    model_path: str = "models/resnet_policy_value.pt"
    learning_rate: float = 0.001
    recent_losses: list[float] = field(default_factory=list)
    active_task: str = "idle"
    active_games: int = 0
    active_positions: int = 0
    active_review_round: int = 0
    active_review_rounds: int = 0
    live_self_play_fen: str | None = None
    live_self_play_last_move: str | None = None
    live_self_play_ply: int = 0
    live_self_play_result: str | None = None
    live_self_play_moves: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return {
            "total_self_play_games": self.total_self_play_games,
            "total_review_rounds": self.total_review_rounds,
            "total_positions": self.total_positions,
            "total_pgn_games": self.total_pgn_games,
            "total_pgn_positions": self.total_pgn_positions,
            "last_loss": self.last_loss,
            "last_value_loss": self.last_value_loss,
            "last_policy_loss": self.last_policy_loss,
            "running": self.running,
            "message": self.message,
            "model_path": self.model_path,
            "learning_rate": self.learning_rate,
            "recent_losses": self.recent_losses[-20:],
            "active_task": self.active_task,
            "active_games": self.active_games,
            "active_positions": self.active_positions,
            "active_review_round": self.active_review_round,
            "active_review_rounds": self.active_review_rounds,
            "live_self_play": {
                "fen": self.live_self_play_fen,
                "last_move": self.live_self_play_last_move,
                "ply": self.live_self_play_ply,
                "result": self.live_self_play_result,
                "moves": self.live_self_play_moves[-16:],
            },
        }


class NeuralSelfTrainer:
    def __init__(self, model_path: str = "models/resnet_policy_value.pt", stats_path: str | None = None) -> None:
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = ResNetPolicyValueNet().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.stats = TrainingStats(model_path=model_path)
        self.lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.model_path = Path(model_path)
        self.stats_path = Path(stats_path) if stats_path is not None else self.model_path.with_name("training_stats.json")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model_incompatible = False
        self.load_if_available()
        self.load_stats_if_available()

    def load_if_available(self) -> None:
        if not self.model_path.exists():
            return
        checkpoint = torch.load(self.model_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except RuntimeError:
            self.stats.message = "Model architecture changed; starting fresh with larger model (8 blocks, 128ch)."
            self._model_incompatible = True
            # Reset trust counters so random weights are not used for evaluation
            self.stats.total_self_play_games = 0
            self.stats.total_positions = 0
            return
        self.stats.total_self_play_games = checkpoint.get("total_self_play_games", 0)
        self.stats.total_review_rounds = checkpoint.get("total_review_rounds", 0)
        self.stats.total_positions = checkpoint.get("total_positions", 0)
        self.stats.total_pgn_games = checkpoint.get("total_pgn_games", 0)
        self.stats.total_pgn_positions = checkpoint.get("total_pgn_positions", 0)
        self.stats.last_loss = checkpoint.get("last_loss")
        self.stats.last_value_loss = checkpoint.get("last_value_loss")
        self.stats.last_policy_loss = checkpoint.get("last_policy_loss")
        self.stats.learning_rate = checkpoint.get("learning_rate", self.stats.learning_rate)
        self.stats.recent_losses = checkpoint.get("recent_losses", [])
        self.stats.message = "Loaded existing ResNet policy/value model."
        self.set_learning_rate(self.stats.learning_rate)

    def load_stats_if_available(self) -> None:
        if not self.stats_path.exists():
            return
        try:
            data = json.loads(self.stats_path.read_text())
        except (OSError, json.JSONDecodeError):
            self.stats.message = "Stats file could not be read; using model checkpoint stats."
            return

        if self._model_incompatible:
            # Model was incompatible; only restore non-trust stats
            self.stats.running = False
            self.stats.active_task = "idle"
            self.stats.message = "New model started (architecture changed). Training stats reset."
            return

        self.stats.total_self_play_games = int(data.get("total_self_play_games", self.stats.total_self_play_games))
        self.stats.total_review_rounds = int(data.get("total_review_rounds", self.stats.total_review_rounds))
        self.stats.total_positions = int(data.get("total_positions", self.stats.total_positions))
        self.stats.total_pgn_games = int(data.get("total_pgn_games", self.stats.total_pgn_games))
        self.stats.total_pgn_positions = int(data.get("total_pgn_positions", self.stats.total_pgn_positions))
        self.stats.last_loss = data.get("last_loss", self.stats.last_loss)
        self.stats.last_value_loss = data.get("last_value_loss", self.stats.last_value_loss)
        self.stats.last_policy_loss = data.get("last_policy_loss", self.stats.last_policy_loss)
        self.stats.learning_rate = float(data.get("learning_rate", self.stats.learning_rate))
        self.stats.recent_losses = list(data.get("recent_losses", self.stats.recent_losses))[-20:]
        self.stats.running = False
        self.stats.active_task = "idle"
        self.stats.active_games = int(data.get("active_games", 0))
        self.stats.active_positions = int(data.get("active_positions", 0))
        self.stats.active_review_round = int(data.get("active_review_round", 0))
        self.stats.active_review_rounds = int(data.get("active_review_rounds", 0))
        self.stats.message = "Loaded training statistics."
        self.set_learning_rate(self.stats.learning_rate)

    def save_stats(self) -> None:
        data = self.stats.payload()
        data["running"] = False
        temp_path = self.stats_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        temp_path.replace(self.stats_path)

    def save(self) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_self_play_games": self.stats.total_self_play_games,
                "total_review_rounds": self.stats.total_review_rounds,
                "total_positions": self.stats.total_positions,
                "total_pgn_games": self.stats.total_pgn_games,
                "total_pgn_positions": self.stats.total_pgn_positions,
                "last_loss": self.stats.last_loss,
                "last_value_loss": self.stats.last_value_loss,
                "last_policy_loss": self.stats.last_policy_loss,
                "learning_rate": self.stats.learning_rate,
                "recent_losses": self.stats.recent_losses,
            },
            self.model_path,
        )
        self.save_stats()

    def set_learning_rate(self, learning_rate: float) -> None:
        learning_rate = max(0.00001, min(float(learning_rate), 0.1))
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.stats.learning_rate = learning_rate

    def start_background_training(self, games: int, review_rounds: int, learning_rate: float | None = None) -> bool:
        with self.lock:
            if self.stats.running:
                return False
            if learning_rate is not None:
                learning_rate = max(0.00001, min(float(learning_rate), 0.1))
                for group in self.optimizer.param_groups:
                    group["lr"] = learning_rate
                self.stats.learning_rate = learning_rate
            self.stats.running = True
            self.stats.active_task = "self-play"
            self.stats.active_games = 0
            self.stats.active_positions = 0
            self.stats.active_review_round = 0
            self.stats.active_review_rounds = max(1, min(int(review_rounds), 200))
            self.stats.live_self_play_fen = None
            self.stats.live_self_play_last_move = None
            self.stats.live_self_play_ply = 0
            self.stats.live_self_play_result = None
            self.stats.live_self_play_moves = []
            self.stats.message = (
                f"Starting self-play: {games} games, {review_rounds} review rounds, "
                f"lr {self.stats.learning_rate:g}."
            )

        thread = threading.Thread(
            target=self.train_self_play,
            args=(games, review_rounds),
            daemon=True,
        )
        thread.start()
        return True

    def start_background_pgn_training(
        self,
        pgn_text: str,
        review_rounds: int = PGN_REVIEW_ROUNDS,
        learning_rate: float | None = PGN_LEARNING_RATE,
    ) -> bool:
        review_rounds = PGN_REVIEW_ROUNDS
        learning_rate = PGN_LEARNING_RATE
        with self.lock:
            if self.stats.running:
                return False
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            self.stats.learning_rate = learning_rate
            self.stats.running = True
            self.stats.active_task = "pgn"
            self.stats.active_games = 0
            self.stats.active_positions = 0
            self.stats.active_review_round = 0
            self.stats.active_review_rounds = review_rounds
            self.stats.message = (
                f"Starting PGN review: {review_rounds} rounds, "
                f"lr {self.stats.learning_rate:g}."
            )

        thread = threading.Thread(
            target=self.train_pgn_text,
            args=(pgn_text, review_rounds),
            daemon=True,
        )
        thread.start()
        return True

    def train_self_play(self, games: int, review_rounds: int) -> None:
        try:
            games = max(1, min(games, 1000))
            review_rounds = max(1, min(review_rounds, 200))
            samples = self.generate_self_play_samples(games)
            if not samples:
                raise RuntimeError("Self-play produced no samples.")

            self.train_samples(
                samples,
                review_rounds,
                "Self-play review",
                value_loss_weight=SELF_PLAY_VALUE_LOSS_WEIGHT,
            )

            with self.lock:
                self.stats.message = f"Finished {games} self-play games and {review_rounds} review rounds."
            self.save()
        except Exception as exc:
            with self.lock:
                self.stats.message = f"Training failed: {exc}"
        finally:
            with self.lock:
                self.stats.running = False
                self.stats.active_task = "idle"

    def train_pgn_text(self, pgn_text: str, review_rounds: int) -> None:
        try:
            review_rounds = max(1, min(review_rounds, 200))
            stream = StringIO(pgn_text)
            samples: list[tuple[torch.Tensor, float, int]] = []
            game_count = 0
            position_count = 0
            chunk_count = 0

            while True:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                game_samples = self.samples_from_pgn_game(game)
                if not game_samples:
                    continue

                game_count += 1
                position_count += len(game_samples)
                samples.extend(game_samples)
                with self.lock:
                    self.stats.total_pgn_games += 1
                    self.stats.total_pgn_positions += len(game_samples)
                    self.stats.total_positions += len(game_samples)
                    self.stats.active_games += 1
                    self.stats.active_positions += len(game_samples)
                    self.stats.message = (
                        f"Loaded {self.stats.active_games} PGN games, "
                        f"{self.stats.active_positions} positions. Training in chunks."
                    )
                    should_save_stats = self.stats.active_games % 50 == 0
                if should_save_stats:
                    self.save_stats()

                if len(samples) >= PGN_CHUNK_POSITIONS:
                    chunk_count += 1
                    self.train_samples(
                        samples,
                        review_rounds,
                        f"PGN policy chunk {chunk_count}",
                        value_loss_weight=PGN_VALUE_LOSS_WEIGHT,
                        batch_size=PGN_BATCH_SIZE,
                    )
                    samples = []

            if samples:
                chunk_count += 1
                self.train_samples(
                    samples,
                    review_rounds,
                    f"PGN policy chunk {chunk_count}",
                    value_loss_weight=PGN_VALUE_LOSS_WEIGHT,
                    batch_size=PGN_BATCH_SIZE,
                )

            if game_count == 0:
                raise RuntimeError("No finished PGN games were found.")

            with self.lock:
                self.stats.message = (
                    f"Finished PGN review: {game_count} games, "
                    f"{position_count} positions, {chunk_count} chunks."
                )
            self.save()
        except Exception as exc:
            with self.lock:
                self.stats.message = f"PGN training failed: {exc}"
        finally:
            with self.lock:
                self.stats.running = False
                self.stats.active_task = "idle"

    def train_finished_game_moves(
        self,
        moves: list[chess.Move],
        result: str,
        review_rounds: int = PGN_REVIEW_ROUNDS,
        learning_rate: float = PGN_LEARNING_RATE,
        label: str = "Lichess AI review",
    ) -> None:
        review_rounds = max(1, min(review_rounds, 200))
        samples = self.samples_from_moves(moves, result)
        if not samples:
            return

        with self.lock:
            if self.stats.running:
                self.stats.message = f"Skipped {label}: trainer is already busy."
                return
            self.set_learning_rate(learning_rate)
            self.stats.running = True
            self.stats.active_task = "lichess-ai"
            self.stats.active_games = 1
            self.stats.active_positions = len(samples)
            self.stats.active_review_round = 0
            self.stats.active_review_rounds = review_rounds
            self.stats.total_positions += len(samples)
            self.stats.message = f"Starting {label}: {len(samples)} positions, result {result}."

        try:
            self.train_samples(
                samples,
                review_rounds,
                label,
                value_loss_weight=PGN_VALUE_LOSS_WEIGHT,
                batch_size=PGN_BATCH_SIZE,
            )
            with self.lock:
                self.stats.message = f"Finished {label}: {len(samples)} positions, result {result}."
            self.save()
        except Exception as exc:
            with self.lock:
                self.stats.message = f"{label} failed: {exc}"
        finally:
            with self.lock:
                self.stats.running = False
                self.stats.active_task = "idle"

    def train_samples(
        self,
        samples: list[tuple[torch.Tensor, float, int]],
        review_rounds: int,
        label: str,
        value_loss_weight: float = SELF_PLAY_VALUE_LOSS_WEIGHT,
        batch_size: int = 64,
    ) -> None:
        inputs = torch.stack([sample[0] for sample in samples])
        value_targets = torch.tensor([sample[1] for sample in samples], dtype=torch.float32)
        policy_targets = torch.tensor([sample[2] for sample in samples], dtype=torch.long)
        dataset = TensorDataset(inputs, value_targets, policy_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for round_index in range(review_rounds):
            total_loss = 0.0
            total_value_loss = 0.0
            total_policy_loss = 0.0
            batches = 0
            for batch_inputs, batch_value_targets, batch_policy_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_policy_targets = batch_policy_targets.to(self.device)
                with self.model_lock:
                    self.optimizer.zero_grad()
                    if value_loss_weight > 0.0:
                        batch_value_targets = batch_value_targets.to(self.device)
                        value_predictions, policy_logits = self.model(batch_inputs)
                        if value_predictions is None:
                            raise RuntimeError("Value head was skipped during value training.")
                        value_loss = F.mse_loss(value_predictions, batch_value_targets)
                    else:
                        _, policy_logits = self.model(batch_inputs, policy_only=True)
                        value_loss = batch_inputs.new_tensor(0.0)
                    policy_loss = F.cross_entropy(policy_logits, batch_policy_targets)
                    loss = value_loss_weight * value_loss + POLICY_LOSS_WEIGHT * policy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total_loss += float(loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_policy_loss += float(policy_loss.detach().cpu())
                batches += 1

            avg_loss = total_loss / max(1, batches)
            avg_value_loss = total_value_loss / max(1, batches)
            avg_policy_loss = total_policy_loss / max(1, batches)
            with self.lock:
                self.stats.last_loss = avg_loss
                self.stats.last_value_loss = avg_value_loss
                self.stats.last_policy_loss = avg_policy_loss
                self.stats.recent_losses.append(avg_loss)
                self.stats.active_review_round = round_index + 1
                self.stats.total_review_rounds += 1
                self.stats.message = (
                    f"{label} {round_index + 1}/{review_rounds}, "
                    f"loss {avg_loss:.4f}, value {avg_value_loss:.4f}, policy {avg_policy_loss:.4f}."
                )
            self.save_stats()

    def generate_self_play_samples(self, games: int) -> list[tuple[torch.Tensor, float, int]]:
        samples: list[tuple[torch.Tensor, float, int]] = []
        for game_index in range(games):
            board = chess.Board()
            ai = BasicAI(seed=game_index, search_depth=2)
            history: list[tuple[chess.Board, bool, chess.Move]] = []
            live_moves: list[str] = []
            with self.lock:
                self.stats.live_self_play_fen = board.fen()
                self.stats.live_self_play_last_move = None
                self.stats.live_self_play_ply = 0
                self.stats.live_self_play_result = None
                self.stats.live_self_play_moves = []

            for _ in range(600):
                if board.is_game_over(claim_draw=True):
                    break
                position = board.copy(stack=False)
                move = self.choose_self_play_move(board, ai)
                if move is None:
                    break
                san = board.san(move)
                history.append((position, board.turn, move))
                board.push(move)
                live_moves.append(san)
                with self.lock:
                    self.stats.live_self_play_fen = board.fen()
                    self.stats.live_self_play_last_move = san
                    self.stats.live_self_play_ply = len(history)
                    self.stats.live_self_play_moves = live_moves[-16:]
                    self.stats.message = (
                        f"Self-play game {game_index + 1}/{games}, "
                        f"move {len(history)}: {san}."
                    )

            result = board.result(claim_draw=True)
            if result == "*":
                result = adjudicated_result(board)
            for position, turn, move in history:
                samples.append((encode_board(position), outcome_value(result, turn), move_to_policy_index(move)))
            with self.lock:
                self.stats.total_self_play_games += 1
                self.stats.total_positions += len(history)
                self.stats.active_games += 1
                self.stats.active_positions += len(history)
                self.stats.live_self_play_fen = board.fen()
                self.stats.live_self_play_last_move = live_moves[-1] if live_moves else None
                self.stats.live_self_play_ply = len(history)
                self.stats.live_self_play_result = result
                self.stats.live_self_play_moves = live_moves[-16:]
                self.stats.message = (
                    f"Self-play generated {self.stats.active_games}/{games} games, "
                    f"{self.stats.active_positions} positions. Last result: {result}."
                )
                should_save_stats = self.stats.active_games % 10 == 0 or self.stats.active_games == games
            if should_save_stats:
                self.save_stats()
        return samples

    def choose_self_play_move(self, board: chess.Board, fallback_ai: BasicAI) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        if self.stats.total_positions < 1 or random.random() < 0.10:
            return fallback_ai.choose_move(board)

        self.model.eval()
        with torch.no_grad():
            tensor = encode_board(board).unsqueeze(0).to(self.device)
            with self.model_lock:
                _, logits = self.model(tensor)
            scores = logits[0].detach().cpu()

        # Also get value-based scores for each legal move
        value_scores: dict[chess.Move, float] = {}
        with self.lock:
            has_enough_games = self.stats.total_self_play_games >= VALUE_TRUST_AFTER_SELF_PLAY_GAMES
        if has_enough_games:
            try:
                encoded_after = []
                for move in legal_moves:
                    board.push(move)
                    encoded_after.append(encode_board(board))
                    board.pop()
                batch = torch.stack(encoded_after).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    with self.model_lock:
                        values, _ = self.model(batch)
                    if values is not None:
                        raw = values.detach().cpu().tolist()
                        for move, side_val in zip(legal_moves, raw):
                            board.push(move)
                            white_val = side_val if board.turn == chess.WHITE else -side_val
                            board.pop()
                            value_scores[move] = white_val if board.turn == chess.WHITE else -white_val
            except Exception:
                value_scores = {}

        best_move = legal_moves[0]
        best_score = float("-inf")
        for move in legal_moves:
            policy_score = float(scores[move_to_policy_index(move)])
            heuristic_score = fallback_ai.move_hint(board, move) / 2000.0
            value_score = value_scores.get(move, 0.0)
            # Blend policy + value + heuristic for self-play move selection
            score = 0.5 * policy_score + 0.3 * value_score + 0.2 * heuristic_score
            # Add temperature-style noise for exploration
            score += random.gauss(0, 0.05)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def choose_engine_move(
        self,
        board: chess.Board,
        fallback_ai: BasicAI,
        time_limit_seconds: float = ENGINE_MOVE_TIME_SECONDS,
        max_depth: int = ENGINE_MAX_DEPTH,
    ) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        deadline = time.monotonic() + max(0.05, time_limit_seconds)
        root_color = board.turn
        policy_scores = self.policy_scores_for_moves(board, legal_moves)
        ordered_moves = self.order_engine_moves(board, legal_moves, fallback_ai, policy_scores)
        root_candidates = ordered_moves[: min(len(ordered_moves), ENGINE_ROOT_CANDIDATES)]
        root_value_scores = self.root_value_scores_for_moves(board, root_candidates, root_color)
        transposition: dict[tuple[str, int, bool], float] = {}
        best_move = root_candidates[0]
        best_score = float("-inf")

        for depth in range(1, max(1, max_depth) + 1):
            if time.monotonic() >= deadline:
                break
            depth_best_move = best_move
            depth_best_score = float("-inf")
            completed_depth = True
            for move in root_candidates:
                if time.monotonic() >= deadline:
                    completed_depth = False
                    break
                board.push(move)
                try:
                    score = self.search_engine_position(
                        board,
                        depth - 1,
                        -1_000_000_000.0,
                        1_000_000_000.0,
                        root_color,
                        fallback_ai,
                        deadline,
                        transposition,
                    )
                finally:
                    board.pop()
                score += self.root_move_bonus(board, move, fallback_ai, policy_scores, root_value_scores)
                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move = move
            if completed_depth:
                best_move = depth_best_move
                best_score = depth_best_score

        return best_move if best_score != float("-inf") else root_candidates[0]

    def search_engine_position(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        root_color: bool,
        fallback_ai: BasicAI,
        deadline: float,
        transposition: dict[tuple[str, int, bool], float],
    ) -> float:
        if time.monotonic() >= deadline:
            return self.static_engine_score(board, root_color)
        if depth <= 0 or board.is_game_over(claim_draw=True):
            return self.quiescence_search(board, root_color, alpha, beta, QUIESCENCE_MAX_DEPTH, deadline)

        key = (board.transposition_key() if hasattr(board, "transposition_key") else board.board_fen(), depth, board.turn)
        cached = transposition.get(key)
        if cached is not None:
            return cached

        legal_moves = list(board.legal_moves)
        ordered_moves = fallback_ai.order_moves(board, legal_moves)
        if board.turn == root_color:
            best = -1_000_000_000.0
            for move in ordered_moves:
                board.push(move)
                try:
                    score = self.search_engine_position(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        root_color,
                        fallback_ai,
                        deadline,
                        transposition,
                    )
                finally:
                    board.pop()
                best = max(best, score)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            transposition[key] = best
            return best

        best = 1_000_000_000.0
        for move in ordered_moves:
            board.push(move)
            try:
                score = self.search_engine_position(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    root_color,
                    fallback_ai,
                    deadline,
                    transposition,
                )
            finally:
                board.pop()
            best = min(best, score)
            beta = min(beta, best)
            if alpha >= beta:
                break
        transposition[key] = best
        return best

    def quiescence_search(
        self,
        board: chess.Board,
        root_color: bool,
        alpha: float,
        beta: float,
        depth: int,
        deadline: float,
    ) -> float:
        stand_pat = self.static_engine_score(board, root_color)
        if depth <= 0 or board.is_game_over(claim_draw=True) or time.monotonic() >= deadline:
            return stand_pat

        if board.turn == root_color:
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            best = stand_pat
            for move in self.noisy_moves(board):
                board.push(move)
                try:
                    score = self.quiescence_search(board, root_color, alpha, beta, depth - 1, deadline)
                finally:
                    board.pop()
                best = max(best, score)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best

        if stand_pat <= alpha:
            return stand_pat
        beta = min(beta, stand_pat)
        best = stand_pat
        for move in self.noisy_moves(board):
            board.push(move)
            try:
                score = self.quiescence_search(board, root_color, alpha, beta, depth - 1, deadline)
            finally:
                board.pop()
            best = min(best, score)
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    def noisy_moves(self, board: chess.Board) -> list[chess.Move]:
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
        return sorted(moves, key=lambda move: board.is_capture(move), reverse=True)

    def static_engine_score(self, board: chess.Board, root_color: bool) -> float:
        if board.is_checkmate():
            return -1_000_000.0 if board.turn == root_color else 1_000_000.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        # Blend enhanced heuristic with neural evaluation when trusted
        heuristic_cp = float(evaluate_white_cp(board))
        with self.lock:
            trusted = self.stats.total_self_play_games >= VALUE_TRUST_AFTER_SELF_PLAY_GAMES

        if trusted:
            try:
                neural_val = self.evaluate_white(board)
                neural_cp = neural_val * 1000.0
                blended = NEURAL_SEARCH_WEIGHT * neural_cp + HEURISTIC_SEARCH_WEIGHT * heuristic_cp
                return blended if root_color == chess.WHITE else -blended
            except Exception:
                pass

        return heuristic_cp if root_color == chess.WHITE else -heuristic_cp

    def order_engine_moves(
        self,
        board: chess.Board,
        legal_moves: list[chess.Move],
        fallback_ai: BasicAI,
        policy_scores: dict[chess.Move, float],
    ) -> list[chess.Move]:
        return sorted(
            legal_moves,
            key=lambda move: self.root_move_bonus(board, move, fallback_ai, policy_scores),
            reverse=True,
        )

    def root_move_bonus(
        self,
        board: chess.Board,
        move: chess.Move,
        fallback_ai: BasicAI,
        policy_scores: dict[chess.Move, float],
        root_value_scores: dict[chess.Move, float] | None = None,
    ) -> float:
        return (
            ENGINE_POLICY_WEIGHT * policy_scores.get(move, 0.0)
            + ENGINE_VALUE_WEIGHT * (root_value_scores or {}).get(move, 0.0)
            + float(fallback_ai.move_hint(board, move))
            + float(opening_principle_score(board, move))
        )

    def root_value_scores_for_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        root_color: bool,
    ) -> dict[chess.Move, float]:
        with self.lock:
            trusted = self.stats.total_self_play_games >= VALUE_TRUST_AFTER_SELF_PLAY_GAMES
        if not trusted or not moves:
            return {}

        encoded_positions = []
        for move in moves:
            board.push(move)
            try:
                encoded_positions.append(encode_board(board))
            finally:
                board.pop()

        self.model.eval()
        with torch.no_grad():
            tensor = torch.stack(encoded_positions).to(self.device)
            with self.model_lock:
                values, _ = self.model(tensor)
            if values is None:
                return {}
            raw_values = values.detach().cpu().tolist()

        scores = {}
        for move, side_to_move_value in zip(moves, raw_values):
            board.push(move)
            try:
                white_value = side_to_move_value if board.turn == chess.WHITE else -side_to_move_value
            finally:
                board.pop()
            root_value = white_value if root_color == chess.WHITE else -white_value
            scores[move] = clamp_value(float(root_value))
        return scores

    def policy_scores_for_moves(self, board: chess.Board, legal_moves: list[chess.Move]) -> dict[chess.Move, float]:
        self.model.eval()
        with torch.no_grad():
            tensor = encode_board(board).unsqueeze(0).to(self.device)
            with self.model_lock:
                _, logits = self.model(tensor, policy_only=True)
            logits = logits[0].detach().cpu()

        raw_scores = [float(logits[move_to_policy_index(move)]) for move in legal_moves]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        if math.isclose(min_score, max_score):
            return {move: 0.0 for move in legal_moves}
        return {
            move: (score - min_score) / (max_score - min_score)
            for move, score in zip(legal_moves, raw_scores)
        }

    def generate_pgn_samples(self, pgn_text: str) -> tuple[list[tuple[torch.Tensor, float, int]], int]:
        samples: list[tuple[torch.Tensor, float, int]] = []
        game_count = 0
        stream = StringIO(pgn_text)

        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            game_samples = self.samples_from_pgn_game(game)
            if not game_samples:
                continue
            game_count += 1
            samples.extend(game_samples)
            with self.lock:
                self.stats.total_pgn_games += 1
                self.stats.total_pgn_positions += len(game_samples)
                self.stats.total_positions += len(game_samples)
                self.stats.active_games += 1
                self.stats.active_positions += len(game_samples)
                self.stats.message = (
                    f"Loaded {self.stats.active_games} PGN games, "
                    f"{self.stats.active_positions} positions. Review will start after parsing."
                )
                should_save_stats = self.stats.active_games % 50 == 0
            if should_save_stats:
                self.save_stats()

        return samples, game_count

    def samples_from_moves(self, moves: list[chess.Move], result: str) -> list[tuple[torch.Tensor, float, int]]:
        if result not in {"1-0", "0-1", "1/2-1/2"}:
            return []

        board = chess.Board()
        samples: list[tuple[torch.Tensor, float, int]] = []
        for move in moves:
            if move not in board.legal_moves:
                return []
            samples.append((encode_board(board), outcome_value(result, board.turn), move_to_policy_index(move)))
            board.push(move)
        return samples

    def samples_from_pgn_game(self, game: chess.pgn.Game) -> list[tuple[torch.Tensor, float, int]]:
        result = game.headers.get("Result", "*")
        if result not in {"1-0", "0-1", "1/2-1/2"}:
            return []

        board = game.board()
        samples: list[tuple[torch.Tensor, float, int]] = []
        for move in game.mainline_moves():
            samples.append((encode_board(board), outcome_value(result, board.turn), move_to_policy_index(move)))
            board.push(move)
        return samples

    def evaluate(self, board: chess.Board) -> float:
        self.model.eval()
        with torch.no_grad():
            tensor = encode_board(board).unsqueeze(0).to(self.device)
            with self.model_lock:
                value, _ = self.model(tensor)
                return float(value.detach().cpu()[0])

    def policy_payload(self, board: chess.Board, limit: int = 5) -> list[dict]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        self.model.eval()
        with torch.no_grad():
            tensor = encode_board(board).unsqueeze(0).to(self.device)
            with self.model_lock:
                _, logits = self.model(tensor)
            logits = logits[0].detach().cpu()

        ranked = sorted(
            legal_moves,
            key=lambda move: float(logits[move_to_policy_index(move)]),
            reverse=True,
        )
        payload = []
        for move in ranked[:limit]:
            payload.append(
                {
                    "uci": move.uci(),
                    "from": chess.square_name(move.from_square),
                    "to": chess.square_name(move.to_square),
                    "logit": float(logits[move_to_policy_index(move)]),
                }
            )
        return payload

    def evaluate_white(self, board: chess.Board) -> float:
        side_to_move_value = self.evaluate(board)
        return side_to_move_value if board.turn == chess.WHITE else -side_to_move_value

    def evaluation_payload(self, board: chess.Board) -> dict:
        neural_white_value = self.evaluate_white(board)
        heuristic_value = heuristic_white_value(board)
        with self.lock:
            self_play_games = self.stats.total_self_play_games

        neural_weight = NEURAL_EVAL_WEIGHT if self_play_games >= VALUE_TRUST_AFTER_SELF_PLAY_GAMES else 0.0
        if abs(neural_white_value) > 0.95 and abs(heuristic_value) < 0.25:
            neural_weight = 0.0

        white_value = clamp_value(heuristic_value * (1.0 - neural_weight) + neural_white_value * neural_weight)
        white_percent = max(0.0, min(100.0, (white_value + 1.0) * 50.0))
        return {
            "white_value": white_value,
            "neural_white_value": neural_white_value,
            "heuristic_white_value": heuristic_value,
            "white_percent": white_percent,
            "black_percent": 100.0 - white_percent,
            "label": f"{white_value:+.2f}",
        }

    def payload(self) -> dict:
        with self.lock:
            payload = self.stats.payload()
        payload["device"] = str(self.device)
        payload["architecture"] = f"ResNet {MODEL_BLOCKS}×{MODEL_CHANNELS}ch + policy/value heads"
        payload["pgn_review_rounds"] = PGN_REVIEW_ROUNDS
        payload["pgn_learning_rate"] = PGN_LEARNING_RATE
        return payload


def encode_board(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((INPUT_CHANNELS, 8, 8), dtype=torch.float32)
    for square, piece in board.piece_map().items():
        channel = PIECE_TO_CHANNEL[piece]
        rank = chess.square_rank(square)
        file_index = chess.square_file(square)
        row = 7 - rank
        tensor[channel, row, file_index] = 1.0

    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
    tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    tensor[17, :, :] = min(board.fullmove_number, 120) / 120.0

    # Channel 18: en passant target square
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        tensor[18, 7 - ep_rank, ep_file] = 1.0

    # Channel 19: halfmove clock (normalized, capped at 100)
    tensor[19, :, :] = min(board.halfmove_clock, 100) / 100.0

    # Channel 20-21: material balance (normalised)
    _piece_material = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    white_mat = 0
    black_mat = 0
    for piece in board.piece_map().values():
        v = _piece_material[piece.piece_type]
        if piece.color == chess.WHITE:
            white_mat += v
        else:
            black_mat += v
    max_mat = 39.0  # starting material per side (excluding king)
    tensor[20, :, :] = white_mat / max_mat
    tensor[21, :, :] = black_mat / max_mat

    return tensor


def outcome_value(result: str, side_to_move: bool) -> float:
    if result == "1/2-1/2" or result == "*":
        return 0.0
    white_won = result == "1-0"
    if side_to_move == chess.WHITE:
        return 1.0 if white_won else -1.0
    return -1.0 if white_won else 1.0


def clamp_value(value: float) -> float:
    return max(-1.0, min(1.0, value))


def heuristic_white_value(board: chess.Board) -> float:
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    cp = evaluate_white_cp(board)
    return clamp_value(math.tanh(cp / 900.0))


def opening_principle_score(board: chess.Board, move: chess.Move) -> int:
    if board.fullmove_number > 12:
        return 0

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        return 0

    score = 0
    gives_check = False
    board.push(move)
    try:
        gives_check = board.is_check()
    finally:
        board.pop()

    is_capture = board.is_capture(move)
    if board.is_castling(move):
        score += 120

    if moving_piece.piece_type == chess.QUEEN and not is_capture and not gives_check:
        score -= 260

    if moving_piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
        from_rank = chess.square_rank(move.from_square)
        home_rank = 0 if moving_piece.color == chess.WHITE else 7
        if from_rank == home_rank:
            score += 70
        if move.to_square in CENTER_SQUARES:
            score += 35
        elif move.to_square in NEAR_CENTER_SQUARES:
            score += 20

    if moving_piece.piece_type == chess.PAWN and move.to_square in CENTER_SQUARES:
        score += 45

    return score


def move_to_policy_index(move: chess.Move) -> int:
    promotion_index = PROMOTION_TO_INDEX.get(move.promotion, 0)
    return ((move.from_square * 64) + move.to_square) * len(PROMOTION_TO_INDEX) + promotion_index


def adjudicated_result(board: chess.Board) -> str:
    material = 0
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    for piece in board.piece_map().values():
        sign = 1 if piece.color == chess.WHITE else -1
        material += sign * piece_values[piece.piece_type]
    if material > 2:
        return "1-0"
    if material < -2:
        return "0-1"
    return "1/2-1/2"
