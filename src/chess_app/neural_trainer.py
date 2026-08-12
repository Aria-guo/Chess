from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from chess_app.random_ai import BasicAI


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


class ResNetValueNet(nn.Module):
    """Small chess ResNet with a value head only."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.value_head(x).squeeze(-1)


@dataclass
class TrainingStats:
    total_self_play_games: int = 0
    total_review_rounds: int = 0
    total_positions: int = 0
    last_loss: float | None = None
    running: bool = False
    message: str = "Neural trainer is idle."
    model_path: str = "models/resnet_value.pt"
    recent_losses: list[float] = field(default_factory=list)

    def payload(self) -> dict:
        return {
            "total_self_play_games": self.total_self_play_games,
            "total_review_rounds": self.total_review_rounds,
            "total_positions": self.total_positions,
            "last_loss": self.last_loss,
            "running": self.running,
            "message": self.message,
            "model_path": self.model_path,
            "recent_losses": self.recent_losses[-20:],
        }


class NeuralSelfTrainer:
    def __init__(self, model_path: str = "models/resnet_value.pt") -> None:
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = ResNetValueNet().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.stats = TrainingStats(model_path=model_path)
        self.lock = threading.Lock()
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.load_if_available()

    def load_if_available(self) -> None:
        if not self.model_path.exists():
            return
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.stats.total_self_play_games = checkpoint.get("total_self_play_games", 0)
        self.stats.total_review_rounds = checkpoint.get("total_review_rounds", 0)
        self.stats.total_positions = checkpoint.get("total_positions", 0)
        self.stats.last_loss = checkpoint.get("last_loss")
        self.stats.recent_losses = checkpoint.get("recent_losses", [])
        self.stats.message = "Loaded existing ResNet value model."

    def save(self) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_self_play_games": self.stats.total_self_play_games,
                "total_review_rounds": self.stats.total_review_rounds,
                "total_positions": self.stats.total_positions,
                "last_loss": self.stats.last_loss,
                "recent_losses": self.stats.recent_losses,
            },
            self.model_path,
        )

    def start_background_training(self, games: int, review_rounds: int) -> bool:
        with self.lock:
            if self.stats.running:
                return False
            self.stats.running = True
            self.stats.message = f"Starting self-play: {games} games, {review_rounds} review rounds."

        thread = threading.Thread(
            target=self.train_self_play,
            args=(games, review_rounds),
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

            inputs = torch.stack([sample[0] for sample in samples])
            targets = torch.tensor([sample[1] for sample in samples], dtype=torch.float32)
            dataset = TensorDataset(inputs, targets)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            self.model.train()
            for round_index in range(review_rounds):
                total_loss = 0.0
                batches = 0
                for batch_inputs, batch_targets in loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    self.optimizer.zero_grad()
                    predictions = self.model(batch_inputs)
                    loss = F.mse_loss(predictions, batch_targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    batches += 1

                avg_loss = total_loss / max(1, batches)
                with self.lock:
                    self.stats.last_loss = avg_loss
                    self.stats.recent_losses.append(avg_loss)
                    self.stats.message = f"Review round {round_index + 1}/{review_rounds}, loss {avg_loss:.4f}."

            with self.lock:
                self.stats.total_self_play_games += games
                self.stats.total_review_rounds += review_rounds
                self.stats.total_positions += len(samples)
                self.stats.message = f"Finished {games} self-play games and {review_rounds} review rounds."
            self.save()
        except Exception as exc:
            with self.lock:
                self.stats.message = f"Training failed: {exc}"
        finally:
            with self.lock:
                self.stats.running = False

    def generate_self_play_samples(self, games: int) -> list[tuple[torch.Tensor, float]]:
        samples: list[tuple[torch.Tensor, float]] = []
        for game_index in range(games):
            board = chess.Board()
            ai = BasicAI(seed=game_index, search_depth=1)
            history: list[tuple[chess.Board, bool]] = []

            for _ in range(100):
                if board.is_game_over(claim_draw=True):
                    break
                history.append((board.copy(stack=False), board.turn))
                move = ai.choose_move(board)
                if move is None:
                    break
                board.push(move)

            result = board.result(claim_draw=True)
            for position, turn in history:
                samples.append((encode_board(position), outcome_value(result, turn)))
        return samples

    def evaluate(self, board: chess.Board) -> float:
        self.model.eval()
        with torch.no_grad():
            tensor = encode_board(board).unsqueeze(0).to(self.device)
            return float(self.model(tensor).detach().cpu()[0])

    def payload(self) -> dict:
        with self.lock:
            payload = self.stats.payload()
        payload["device"] = str(self.device)
        payload["architecture"] = "ResNet CNN + value head"
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

