from __future__ import annotations

import chess
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Footer, Header, Input, Static

from chess_app.game import ChessGame, PlayerColor
from chess_app.random_ai import BasicAI
from chess_app.terminal import piece_symbol


SQUARE_WIDTH = 4
SQUARE_HEIGHT = 2


class BoardView(Static):
    class SquareSelected(Message):
        def __init__(self, square: chess.Square) -> None:
            super().__init__()
            self.square = square

    DEFAULT_CSS = """
    BoardView {
        width: 42;
        height: 22;
        border: solid cyan;
        content-align: center middle;
    }
    """

    def __init__(self, game: ChessGame) -> None:
        super().__init__()
        self.game = game
        self.selected: chess.Square | None = None

    def render(self) -> str:
        board = self.game.board
        lines = ["    " + "".join(file_name.center(SQUARE_WIDTH) for file_name in "abcdefgh")]
        for rank in range(7, -1, -1):
            blank_row = ["    "]
            piece_row = [f" {rank + 1}  "]
            for file_index in range(8):
                square = chess.square(file_index, rank)
                piece = piece_symbol(board.piece_at(square))
                if square == self.selected:
                    piece_row.append(f"[{piece}] ")
                else:
                    piece_row.append(f" {piece}  ")
                blank_row.append(" " * SQUARE_WIDTH)
            piece_row.append(f" {rank + 1}")
            lines.append("".join(blank_row))
            lines.append("".join(piece_row))
        lines.append("    " + "".join(file_name.center(SQUARE_WIDTH) for file_name in "abcdefgh"))
        return "\n".join(lines)

    def on_click(self, event) -> None:
        offset = event.get_content_offset(self)
        if offset is None:
            return
        x, y = offset
        rank_line = (y - 1) // SQUARE_HEIGHT
        file_col = (x - 4) // SQUARE_WIDTH
        if not (0 <= rank_line < 8 and 0 <= file_col < 8):
            return
        rank = 7 - rank_line
        square = chess.square(file_col, rank)
        self.post_message(self.SquareSelected(square))


class ChessTui(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    #side {
        width: 38;
        padding: 1;
    }
    Input {
        margin-top: 1;
    }
    Button {
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reset", "Reset"),
    ]

    def __init__(self, human_color: PlayerColor = PlayerColor.WHITE) -> None:
        super().__init__()
        self.game = ChessGame(human_color=human_color)
        self.ai = BasicAI()
        self.board_view = BoardView(self.game)
        self.info = Static()
        self.input = Input(placeholder="e2e4, Nf3, O-O")
        self.selected: chess.Square | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield self.board_view
            with Vertical(id="side"):
                yield self.info
                yield self.input
                yield Button("New Game", id="new-game")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_all()

    def action_reset(self) -> None:
        self.game.reset()
        self.selected = None
        self.refresh_all()

    def refresh_all(self, message: str | None = None) -> None:
        status = [
            f"You: {self.game.human_color.value}",
            f"Turn: {'white' if self.game.board.turn == chess.WHITE else 'black'}",
            f"Status: {self.game.status()}",
            "Click source and target squares, or type a move.",
        ]
        if message:
            status.insert(0, message)
        self.info.update("\n".join(status))
        self.board_view.selected = self.selected
        self.board_view.refresh()

        if self.game.is_ai_turn and not self.game.board.is_game_over(claim_draw=True):
            self.set_timer(0.25, self.play_ai_move)

    def play_ai_move(self) -> None:
        move = self.ai.choose_move(self.game.board)
        if move is not None:
            san = self.game.push_ai_move(move)
            self.refresh_all(f"AI played {san}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        result = self.game.push_human_move(event.value)
        event.input.value = ""
        self.refresh_all(result.message)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-game":
            self.action_reset()

    def on_board_view_square_selected(self, event: BoardView.SquareSelected) -> None:
        if self.selected is None:
            self.selected = event.square
            self.refresh_all(chess.square_name(event.square))
            return

        move_text = chess.square_name(self.selected) + chess.square_name(event.square)
        self.selected = None
        result = self.game.push_human_move(move_text)
        self.refresh_all(result.message)


def run_textual(human_color: PlayerColor = PlayerColor.WHITE) -> None:
    ChessTui(human_color=human_color).run()
