import time

import chess
import pytest

from chess_app.game import ChessGame, PlayerColor
from chess_app.lichess_bot import (
    LichessBot,
    LichessBotConfig,
    board_from_lichess_state,
    game_id_from_challenge_response,
    result_from_lichess_state,
)
from chess_app.neural_trainer import (
    INPUT_CHANNELS,
    MOVE_POLICY_SIZE,
    PGN_LEARNING_RATE,
    PGN_REVIEW_ROUNDS,
    NeuralSelfTrainer,
    ResNetValueNet,
    encode_board,
    heuristic_white_value,
    move_to_policy_index,
    opening_principle_score,
)
from chess_app.random_ai import BasicAI, RandomAI
from chess_app.web_app import WebSession, create_app


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


def test_opening_book_stops_after_first_repertoire_move():
    board = chess.Board()
    ai = BasicAI(seed=7)
    board.push(ai.choose_move(board))
    board.push_san("d5")

    assert ai.choose_move(board) is not None
    assert ai.last_book_name is None


def test_black_opening_book_stops_after_sicilian_first_move():
    board = chess.Board()
    board.push_san("e4")
    ai = BasicAI(seed=7)
    board.push(ai.choose_move(board))
    board.push_san("Nf3")

    assert ai.choose_move(board) is not None
    assert ai.last_book_name is None


def test_opening_principles_penalize_early_queen_development():
    board = chess.Board()
    board.push_san("d4")
    board.push_san("d5")

    queen_move = chess.Move.from_uci("d1d3")
    knight_move = chess.Move.from_uci("b1c3")

    assert opening_principle_score(board, queen_move) < 0
    assert opening_principle_score(board, knight_move) > 0


def test_neural_engine_uses_policy_after_opening(tmp_path, monkeypatch):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"), stats_path=str(tmp_path / "stats.json"))
    board = chess.Board()
    board.push_san("d4")
    board.push_san("d5")
    ai = BasicAI(seed=7)

    # The improved AI now prefers Nf3 over Qd3 in this position
    assert ai.choose_move(board) == chess.Move.from_uci("g1f3")

    preferred_move = chess.Move.from_uci("b1c3")

    def fake_policy_scores(board: chess.Board, legal_moves: list[chess.Move]) -> dict[chess.Move, float]:
        return {move: 1.0 if move == preferred_move else 0.0 for move in legal_moves}

    monkeypatch.setattr(trainer, "policy_scores_for_moves", fake_policy_scores)

    assert trainer.choose_engine_move(board, ai, time_limit_seconds=0.2, max_depth=1) == preferred_move


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
    assert payload["turn"] == "black"
    assert payload["is_ai_turn"]
    assert len(payload["moves"]) == 1
    assert payload["moves"][0].startswith("You:")

    ai_response = client.post("/api/ai-move", json={})
    ai_payload = ai_response.get_json()

    assert ai_response.status_code == 200
    assert ai_payload["turn"] == "white"
    assert len(ai_payload["moves"]) == 2
    assert ai_payload["moves"][1].startswith("AI:")


def test_web_app_black_choice_gets_ai_opening_move():
    app = create_app()
    client = app.test_client()

    response = client.post("/api/new", json={"color": "black"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["human_color"] == "black"
    assert payload["moves"][0].startswith("AI:")


def test_web_ai_uses_neural_engine_after_opening():
    class FakeTrainer:
        def choose_engine_move(self, board: chess.Board, ai: BasicAI) -> chess.Move:
            return chess.Move.from_uci("b1c3")

    session = WebSession(trainer=FakeTrainer())
    session.game = ChessGame(human_color=PlayerColor.BLACK)
    session.game.board = chess.Board()
    session.game.board.push_san("d4")
    session.game.board.push_san("d5")

    session.play_ai_if_needed()

    assert session.game.board.peek() == chess.Move.from_uci("b1c3")
    assert "neural policy search" in session.message


def test_web_click_promotion_requires_a_piece_choice():
    session = WebSession()
    session.game.board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

    with pytest.raises(ValueError, match="Choose a piece"):
        session.parse_click_move("a7", "a8")


def test_web_click_promotion_can_choose_knight():
    session = WebSession()
    session.game.board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

    move = session.parse_click_move("a7", "a8", "n")
    promotions = {
        legal["promotion"]
        for legal in session.state()["legal_moves"]
        if legal["from"] == "a7" and legal["to"] == "a8"
    }

    assert move == chess.Move.from_uci("a7a8n")
    assert promotions == {"q", "r", "b", "n"}


def test_lichess_state_rebuilds_board_from_uci_moves():
    board = board_from_lichess_state(chess.STARTING_FEN, "d2d4 f7f5 c2c4")

    assert board.turn == chess.BLACK
    assert board.peek() == chess.Move.from_uci("c2c4")


def test_lichess_challenge_response_finds_game_id():
    assert game_id_from_challenge_response({"id": "abc123"}) == "abc123"
    assert game_id_from_challenge_response({"game": {"id": "def456"}}) == "def456"


def test_lichess_result_uses_winner_or_board_outcome():
    board = chess.Board()

    assert result_from_lichess_state(board, {"winner": "white"}) == "1-0"
    assert result_from_lichess_state(board, {"winner": "black"}) == "0-1"
    assert result_from_lichess_state(board, {"status": "draw"}) == "1/2-1/2"


def test_trainer_can_build_samples_from_finished_move_stack():
    trainer = NeuralSelfTrainer()
    moves = [chess.Move.from_uci("d2d4"), chess.Move.from_uci("d7d5")]

    samples = trainer.samples_from_moves(moves, "1-0")

    assert len(samples) == 2


def test_lichess_bot_declines_unsafe_challenge_types():
    bot = LichessBot(
        client=object(),
        config=LichessBotConfig(username="ResazurinAI"),
        trainer=object(),
        ai=BasicAI(),
    )

    assert bot.challenge_decline_reason("standard", "rapid", rated=False) is None
    assert bot.challenge_decline_reason("standard", "bullet", rated=False) is None
    assert bot.challenge_decline_reason("standard", "ultraBullet", rated=False) is None
    assert bot.challenge_decline_reason("standard", "rapid", rated=True) is None
    assert bot.challenge_decline_reason("chess960", "rapid", rated=False) == "only standard chess is enabled"
    assert bot.challenge_decline_reason("standard", "correspondence", rated=False) == "time control is too fast or unsupported"


def test_lichess_bot_can_be_set_back_to_casual_only():
    bot = LichessBot(
        client=object(),
        config=LichessBotConfig(username="ResazurinAI", accept_rated=False),
        trainer=object(),
        ai=BasicAI(),
    )

    assert bot.challenge_decline_reason("standard", "rapid", rated=True) == "rated games are disabled"


def test_lichess_bot_uses_shorter_move_time_for_fast_chess():
    bot = LichessBot(
        client=object(),
        config=LichessBotConfig(username="ResazurinAI", move_time_seconds=5.0),
        trainer=object(),
        ai=BasicAI(),
    )

    assert bot.move_time_for_game({"speed": "ultraBullet"}) == pytest.approx(0.12)
    assert bot.move_time_for_game({"speed": "bullet"}) == pytest.approx(0.35)
    assert bot.move_time_for_game({"speed": "blitz"}) == pytest.approx(1.5)
    assert bot.move_time_for_game({"speed": "rapid", "clock": {"initial": 600, "increment": 0}}) == pytest.approx(5.0)


def test_lichess_bot_detects_own_color_from_game_full_event():
    bot = LichessBot(
        client=object(),
        config=LichessBotConfig(username="ResazurinAI"),
        trainer=object(),
        ai=BasicAI(),
    )

    color = bot.bot_color_from_game_full(
        {
            "white": {"user": {"id": "someone-else", "name": "SomeoneElse"}},
            "black": {"user": {"id": "resazurinai", "name": "ResazurinAI"}},
        }
    )

    assert color == chess.BLACK


def test_resnet_value_head_outputs_one_value():
    game = ChessGame()
    tensor = encode_board(game.board).unsqueeze(0)
    model = ResNetValueNet(input_channels=INPUT_CHANNELS, channels=16, blocks=1)

    value, policy = model(tensor)

    assert tensor.shape == (1, INPUT_CHANNELS, 8, 8)
    assert value.shape == (1,)
    assert policy.shape == (1, MOVE_POLICY_SIZE)


def test_web_training_status_available():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/training")
    payload = response.get_json()

    assert response.status_code == 200
    assert "ResNet" in payload["training"]["architecture"]
    assert "policy" in payload["training"]["architecture"]
    assert "value" in payload["training"]["architecture"]
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


def test_web_home_includes_self_play_watch_controls():
    app = create_app()
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="watch-self-play"' in html
    assert 'id="live-self-play-board"' in html
    assert 'id="train-games" type="number" min="1" max="1000" value="200"' in html
    assert 'id="train-rounds" type="number" min="1" max="200" value="20"' in html
    assert 'id="learning-rate" type="number" min="0.00001" max="0.1" step="0.0001" value="0.0005"' in html


def test_web_home_includes_promotion_picker():
    app = create_app()
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="promotion-modal"' in html
    assert "Choose promotion" in html


def test_web_home_includes_eval_side_labels():
    app = create_app()
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="eval-top-label"' in html
    assert 'id="eval-bottom-label"' in html


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

    assert abs(heuristic_white_value(board)) < 0.08
    assert 35.0 <= payload["white_percent"] <= 65.0


def test_pgn_positions_do_not_make_value_head_trusted(tmp_path, monkeypatch):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"))
    trainer.stats.total_positions = 1_000_000
    trainer.stats.total_pgn_positions = 1_000_000
    trainer.stats.total_self_play_games = 0
    monkeypatch.setattr(trainer, "evaluate_white", lambda board: -1.0)

    payload = trainer.evaluation_payload(chess.Board())

    # Value head is not trusted (0 self-play games), so neural eval is ignored.
    # The heuristic gives a small tempo bonus to the side to move.
    assert 45.0 <= payload["white_percent"] <= 55.0


def test_learning_rate_can_be_adjusted(tmp_path):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"))

    trainer.set_learning_rate(0.0005)

    assert trainer.payload()["learning_rate"] == 0.0005


def test_self_play_payload_includes_live_board(tmp_path):
    trainer = NeuralSelfTrainer(model_path=str(tmp_path / "model.pt"), stats_path=str(tmp_path / "stats.json"))
    trainer.generate_self_play_samples(1)
    live = trainer.payload()["live_self_play"]

    assert live["fen"]
    assert live["ply"] > 0
    assert live["result"] in {"1-0", "0-1", "1/2-1/2"}


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
    assert samples[0][0].shape == (INPUT_CHANNELS, 8, 8)
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
