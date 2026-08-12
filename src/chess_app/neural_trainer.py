from __future__ import annotations

import json
import threading
from io import StringIO
from dataclasses import dataclass, field
import math
from pathlib import Path
import random

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from chess_app.random_ai import BasicAI


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
VALUE_TRUST_AFTER_POSITIONS = 10_000
NEURAL_EVAL_WEIGHT = 0.25
PGN_CHUNK_POSITIONS = 50_000
PGN_BATCH_SIZE = 1024


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
    """Small chess ResNet with policy and value heads."""

    def __init__(self, input_channels: int = 18, channels: int = 48, blocks: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, MOVE_POLICY_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
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
            self.stats.message = "Existing value-only model is incompatible; started policy/value model."
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
                batch_value_targets = batch_value_targets.to(self.device)
                batch_policy_targets = batch_policy_targets.to(self.device)
                with self.model_lock:
                    self.optimizer.zero_grad()
                    value_predictions, policy_logits = self.model(batch_inputs)
                    value_loss = F.mse_loss(value_predictions, batch_value_targets)
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
            ai = BasicAI(seed=game_index, search_depth=1)
            history: list[tuple[chess.Board, bool, chess.Move]] = []

            for _ in range(600):
                if board.is_game_over(claim_draw=True):
                    break
                position = board.copy(stack=False)
                move = self.choose_self_play_move(board, ai)
                if move is None:
                    break
                history.append((position, board.turn, move))
                board.push(move)

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
                self.stats.message = (
                    f"Self-play generated {self.stats.active_games}/{games} games, "
                    f"{self.stats.active_positions} positions."
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

        best_move = legal_moves[0]
        best_score = float("-inf")
        for move in legal_moves:
            score = float(scores[move_to_policy_index(move)]) + fallback_ai.move_hint(board, move) / 2000.0
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

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
            trained_positions = self.stats.total_positions

        neural_weight = NEURAL_EVAL_WEIGHT if trained_positions >= VALUE_TRUST_AFTER_POSITIONS else 0.0
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
        payload["architecture"] = "ResNet CNN + policy head + value head"
        payload["pgn_review_rounds"] = PGN_REVIEW_ROUNDS
        payload["pgn_learning_rate"] = PGN_LEARNING_RATE
        return payload


def encode_board(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((18, 8, 8), dtype=torch.float32)
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
    tensor[17, :, :] = min(board.fullmove_number / 100.0, 1.0)
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

    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }
    center_squares = {chess.D4, chess.E4, chess.D5, chess.E5}
    near_center_squares = {chess.C3, chess.D3, chess.E3, chess.F3, chess.C4, chess.F4, chess.C5, chess.F5, chess.C6, chess.D6, chess.E6, chess.F6}

    score = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        score += sign * piece_values[piece.piece_type]
        if square in center_squares:
            score += sign * 20
        elif square in near_center_squares:
            score += sign * 8

    # Convert centipawn-like scores into a stable [-1, 1] bar value.
    return clamp_value(math.tanh(score / 900.0))


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
