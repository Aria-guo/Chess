#!/usr/bin/env python3
"""
Neural network training script for chess engine.

Uses a small model with heuristic evaluation as targets to avoid overfitting.
Generates self-play data using the heuristic engine and trains with continuous
value targets (not just +1/-1 from game results).
"""

import sys
import os
import time
import random
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import chess

from chess_app.neural_trainer import (
    ResNetPolicyValueNet,
    encode_board,
    move_to_policy_index,
    SMALL_MODEL_CHANNELS,
    SMALL_MODEL_BLOCKS,
    SMALL_MODEL_DROPOUT,
    INPUT_CHANNELS,
    MOVE_POLICY_SIZE,
)
from chess_app.random_ai import BasicAI, evaluate_position


def generate_self_play_data(num_games: int, seed: int = 42) -> list:
    """Generate self-play data using the heuristic engine.
    
    Returns list of (tensor, value_target, policy_target) tuples.
    Value targets are continuous heuristic evaluations, not just +1/-1.
    """
    samples = []
    rng = random.Random(seed)
    
    for game_idx in range(num_games):
        board = chess.Board()
        ai = BasicAI(seed=seed + game_idx, search_depth=2)
        game_positions = []
        
        while not board.is_game_over(claim_draw=True) and len(game_positions) < 300:
            # Get heuristic evaluation as value target (from side-to-move perspective)
            heuristic_cp = evaluate_position(board, board.turn)
            # Normalize to [-1, 1] range
            value_target = max(-1.0, min(1.0, heuristic_cp / 1000.0))
            
            # Get the move the AI would play
            move = ai.choose_move(board)
            if move is None:
                break
            
            # Encode position
            tensor = encode_board(board)
            policy_idx = move_to_policy_index(move)
            
            game_positions.append((tensor, value_target, policy_idx))
            board.push(move)
        
        samples.extend(game_positions)
        
        if (game_idx + 1) % 10 == 0:
            print(f"  Generated {game_idx + 1}/{num_games} games, {len(samples)} positions", flush=True)
    
    return samples


def train_model(
    samples: list,
    num_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    device: str = "mps",
) -> ResNetPolicyValueNet:
    """Train a small neural network on the samples.
    
    Uses heuristic evaluations as value targets (continuous, not binary).
    """
    print(f"\n=== Training Configuration ===", flush=True)
    print(f"Samples: {len(samples)}", flush=True)
    print(f"Epochs: {num_epochs}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Learning rate: {learning_rate}", flush=True)
    print(f"Device: {device}", flush=True)
    
    # Create small model
    model = ResNetPolicyValueNet(
        input_channels=INPUT_CHANNELS,
        channels=SMALL_MODEL_CHANNELS,
        blocks=SMALL_MODEL_BLOCKS,
        dropout=SMALL_MODEL_DROPOUT,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}", flush=True)
    print(f"Params/sample ratio: {total_params / len(samples):.1f}x", flush=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # Prepare data
    tensors = torch.stack([s[0] for s in samples])
    value_targets = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    policy_targets = torch.tensor([s[2] for s in samples], dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(tensors, value_targets, policy_targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print(f"\n=== Training ===", flush=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0
        
        for batch_tensors, batch_values, batch_policies in loader:
            batch_tensors = batch_tensors.to(device)
            batch_values = batch_values.to(device)
            batch_policies = batch_policies.to(device)
            
            optimizer.zero_grad()
            
            value_pred, policy_logits = model(batch_tensors)
            
            # Value loss: MSE between predicted and heuristic evaluation
            value_loss = F.mse_loss(value_pred.squeeze(), batch_values)
            
            # Policy loss: cross-entropy with AI's chosen move
            policy_loss = F.cross_entropy(policy_logits, batch_policies)
            
            # Combined loss
            loss = value_loss + 0.5 * policy_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: loss={avg_loss:.4f} value={avg_value_loss:.4f} policy={avg_policy_loss:.4f}", flush=True)
    
    return model


def test_model(model: ResNetPolicyValueNet, device: str = "mps"):
    """Test the trained model on various positions."""
    print(f"\n=== Model Testing ===", flush=True)
    
    positions = [
        ("Starting", chess.Board()),
        ("After e4", chess.Board()),
        ("Italian Game", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3")),
        ("White+Q", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 1")),
        ("Black+R", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 3")),
    ]
    
    # Fix the positions that need moves
    positions[1][1].push_san("e4")
    
    model.eval()
    with torch.no_grad():
        for name, board in positions:
            tensor = encode_board(board).unsqueeze(0).to(device)
            value, _ = model(tensor)
            print(f"{name:20s}: {value.item():+.4f}", flush=True)
    
    # Test relative ranking
    print(f"\n=== Relative Ranking Test ===", flush=True)
    test_fens = [
        ("White down Q", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 1"),
        ("White down R", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5R2/PPPP1PPP/RNBQK1NR b KQkq - 0 1"),
        ("Equal", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3"),
        ("White up R", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 3"),
        ("White up Q", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 3"),
    ]
    
    results = []
    with torch.no_grad():
        for name, fen in test_fens:
            board = chess.Board(fen)
            tensor = encode_board(board).unsqueeze(0).to(device)
            value, _ = model(tensor)
            results.append((name, value.item()))
    
    results.sort(key=lambda x: x[1])
    for name, val in results:
        print(f"{val:+.4f} | {name}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("Chess Neural Network Training", flush=True)
    print("=" * 60, flush=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    # Configuration
    num_games = 200  # Self-play games to generate
    num_epochs = 50  # Training epochs
    batch_size = 128
    learning_rate = 0.001
    
    # Step 1: Generate self-play data
    print(f"\n=== Step 1: Generating {num_games} self-play games ===", flush=True)
    start_time = time.time()
    
    # Check if we have cached samples
    cache_path = "/tmp/selfplay_heuristic_samples.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached samples from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            samples = pickle.load(f)
        print(f"Loaded {len(samples)} samples", flush=True)
    else:
        samples = generate_self_play_data(num_games, seed=42)
        print(f"Generated {len(samples)} samples in {time.time() - start_time:.1f}s", flush=True)
        
        # Cache for future use
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"Cached to {cache_path}", flush=True)
    
    # Step 2: Train the model
    print(f"\n=== Step 2: Training model ===", flush=True)
    start_time = time.time()
    model = train_model(
        samples,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    print(f"Training completed in {time.time() - start_time:.1f}s", flush=True)
    
    # Step 3: Test the model
    test_model(model, device)
    
    # Step 4: Save the model
    model_path = "models/resnet_policy_value_small.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n=== Model saved to {model_path} ===", flush=True)
    
    # Also save as the main model for immediate use
    main_model_path = "models/resnet_policy_value.pt"
    torch.save(model.state_dict(), main_model_path)
    print(f"Also saved as {main_model_path}", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("Training complete!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
