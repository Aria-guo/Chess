import time

import chess

from chess_app.game import ChessGame, PlayerColor
from chess_app.neural_trainer import (
    MOVE_POLICY_SIZE,
    PGN_LEARNING_RATE,
    PGN_REVIEW_ROUNDS,
    NeuralSelfTrainer,
    ResNetValueNet,
    encode_board,
    heuristic_white_value,
    move_to_policy_index,
)
from chess_app.random_ai import BasicAI, RandomAI
from chess_app.web_app import create_app


def test_human_move_accepts_uci():
    game = ChessGame()
    result = game.push_human_move("e2e4")

    assert result.ok
    assert game.board.peek() == chess.Move.from_uci("e2e4")


def test_human_move_rejects_illegal_move():
    game = ChessGame()
    result = game.push_human_move("e2e5")

    assert not result.ok
    assert len(game.board.move_stack) == 0


def test_random_ai_only_returns_legal_moves():
    board = chess.Board()
    ai = RandomAI(seed=7)

    for _ in range(80):
        move = ai.choose_move(board)
        if move is None:
            break
        assert move in board.legal_moves
        board.push(move)


def test_basic_ai_prefers_winning_material():
    board = chess.Board("4k3/8/8/3q4/4B3/8/8/4K3 w - - 0 1")
    ai = BasicAI(seed=7)

    assert ai.choose_move(board) == chess.Move.from_uci("e4d5")


def test_opening_book_starts_queen_gambit_as_white():
    board = chess.Board()
    ai = BasicAI(seed=7)

    assert ai.choose_move(board) == chess.Move.from_uci("d2d4")


def test_opening_book_plays_sicilian_against_king_pawn():
    board = chess.Board()
    board.push_san("e4")
    ai = BasicAI(seed=7)

    assert ai.choose_move(board) == chess.Move.from_uci("c7c5")


def test_opening_book_plays_dutch_against_queen_pawn():
    board = chess.Board()
    board.push_san("d4")
    ai = BasicAI(seed=7)

    assert ai.choose_move(board) == chess.Move.from_uci("f7f5")


def test_black_human_means_ai_starts():
    game = ChessGame(human_color=PlayerColor.BLACK)

    assert game.is_ai_turn
    assert not game.is_human_turn


def test_web_app_can_play_a_move_and_get_ai_reply():
    app = create_app()
    client = app.test_client()

    response = client.post("/api/move", json={"from": "d2", "to": "d4"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["turn"] == "white"
    assert payload["moves"][0].startswith("You:")
    assert payload["moves"][1].startswith("AI:")


def test_web_app_black_choice_gets_ai_opening_move():
    app = create_app()
    client = app.test_client()

    response = client.post("/api/new", json={"color": "black"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["human_color"] == "black"
    assert payload["moves"][0].startswith("AI:")


def test_resnet_value_head_outputs_one_value():
    game = ChessGame()
    tensor = encode_board(game.board).unsqueeze(0)
    model = ResNetValueNet(channels=16, blocks=1)

    value, policy = model(tensor)

    assert tensor.shape == (1, 18, 8, 8)
    assert value.shape == (1,)
    assert policy.shape == (1, MOVE_POLICY_SIZE)


def test_web_training_status_available():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/training")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["training"]["architecture"] == "ResNet CNN + policy head + value head"
    assert "total_self_play_games" in payload["training"]
    assert "active_games" in payload["training"]
    assert "active_review_round" in payload["training"]
    assert payload["training"]["pgn_review_rounds"] == PGN_REVIEW_ROUNDS
    assert payload["training"]["pgn_learning_rate"] == PGN_LEARNING_RATE
    assert 0 <= payload["evaluation"]["white_percent"] <= 100


def test_web_home_includes_pgn_file_import():
    app = create_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="pgn-file"' in response.get_data(as_text=True)


def test_web_home_includes_training_stats_panel():
    app = create_app()
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Training Stats" in html
    assert 'id="stats-self-play-games"' in html
    assert 'id="stats-total-trained-games"' in html
    assert 'id="stats-pgn-games"' in html


def test_web_state_includes_evaluation_bar_data():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/state")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["evaluation"]["label"].startswith(("+", "-"))
    assert payload["evaluation"]["black_percent"] == 100.0 - payload["evaluation"]["white_percent"]
    assert len(payload["policy"]) > 0


def test_initial_position_evaluation_is_near_even(tmp_path):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"))
    board = chess.Board()

    payload = trainer.evaluation_payload(board)

    assert heuristic_white_value(board) == 0.0
    assert 35.0 <= payload["white_percent"] <= 65.0


def test_learning_rate_can_be_adjusted(tmp_path):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"))

    trainer.set_learning_rate(0.0005)

    assert trainer.payload()["learning_rate"] == 0.0005


def test_training_stats_persist_independently_from_model(tmp_path):
    model_path = tmp_path / "model.pt"
    stats_path = tmp_path / "training_stats.json"
    trainer = NeuralSelfTrainer(model_path=str(model_path), stats_path=str(stats_path))
    trainer.stats.total_pgn_games = 12
    trainer.stats.total_self_play_games = 3
    trainer.stats.total_review_rounds = 7

    trainer.save_stats()
    restored = NeuralSelfTrainer(model_path=str(tmp_path / "fresh_model.pt"), stats_path=str(stats_path))

    assert restored.payload()["total_pgn_games"] == 12
    assert restored.payload()["total_self_play_games"] == 3
    assert restored.payload()["total_review_rounds"] == 7


def test_pgn_games_can_be_converted_to_value_samples(tmp_path):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"))
    pgn = """
[Event "Miniature"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
"""

    samples, game_count = trainer.generate_pgn_samples(pgn)

    assert game_count == 1
    assert len(samples) == 4
    assert samples[0][0].shape == (18, 8, 8)
    assert isinstance(samples[0][2], int)


def test_pgn_parsing_updates_progress_stats(tmp_path):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"))
    pgn = """
[Event "Short"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
"""

    samples, game_count = trainer.generate_pgn_samples(pgn)
    payload = trainer.payload()

    assert game_count == 1
    assert len(samples) == 4
    assert payload["total_pgn_games"] == 1
    assert payload["total_pgn_positions"] == 4
    assert payload["active_games"] == 1
    assert payload["active_positions"] == 4


def test_move_policy_index_handles_promotions():
    move = chess.Move.from_uci("a7a8q")

    assert 0 <= move_to_policy_index(move) < MOVE_POLICY_SIZE


def test_web_pgn_training_requires_pgn_text(tmp_path):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"), stats_path=str(tmp_path / "stats.json"))
    app = create_app(trainer=trainer)
    client = app.test_client()

    response = client.post("/api/train-pgn", json={"pgn": ""})

    assert response.status_code == 400


def test_web_pgn_training_can_start(tmp_path):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"), stats_path=str(tmp_path / "stats.json"))
    app = create_app(trainer=trainer)
    client = app.test_client()
    pgn = """
[Event "Short"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
"""

    response = client.post("/api/train-pgn", json={"pgn": pgn, "review_rounds": 1})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["running"] is True


def test_web_pgn_training_uses_fixed_preset(tmp_path):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"), stats_path=str(tmp_path / "stats.json"))
    app = create_app(trainer=trainer)
    client = app.test_client()
    pgn = """
[Event "Short"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
"""

    response = client.post(
        "/api/train-pgn",
        json={"pgn": pgn, "review_rounds": 99, "learning_rate": 0.02},
    )
    payload = response.get_json()

    try:
        assert response.status_code == 200
        assert payload["active_review_rounds"] == PGN_REVIEW_ROUNDS
        assert payload["learning_rate"] == PGN_LEARNING_RATE
    finally:
        deadline = time.time() + 10
        while payload["running"] and time.time() < deadline:
            time.sleep(0.05)
            payload = client.get("/api/training").get_json()["training"]
