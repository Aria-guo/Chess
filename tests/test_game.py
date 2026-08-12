import chess

from chess_app.game import ChessGame, PlayerColor
from chess_app.random_ai import BasicAI, RandomAI


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


def test_black_human_means_ai_starts():
    game = ChessGame(human_color=PlayerColor.BLACK)

    assert game.is_ai_turn
    assert not game.is_human_turn
