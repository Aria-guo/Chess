from __future__ import annotations

import chess
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Footer, Header, Input, Static

from chess_app.game import ChessGame, PlayerColor
from chess_app.random_ai import BasicAI
from chess_app.terminal import (
    DARK_SQUARE,
    LIGHT_SQUARE,
    SQUARE_HEIGHT,
    SQUARE_WIDTH,
    piece_symbol,
)


class BoardView(Static):
    class SquareSelected(Message):
        def __init__(self, square: chess.Square) -> None:
            super().__init__()
            self.square = square

    DEFAULT_CSS = """
    BoardView {
        width: 92;
        height: 48;
        border: solid cyan;
        content-align: center middle;
    }
    """

    def __init__(self, game: ChessGame) -> None:
        super().__init__()
        self.game = game
        self.selected: chess.Square | None = None

    def render(self) -> Text:
        board = self.game.board
        files = range(8) if self.game.human_color is PlayerColor.WHITE else range(7, -1, -1)
        ranks = range(7, -1, -1) if self.game.human_color is PlayerColor.WHITE else range(8)
        output = Text()
        output.append(
            "    "
            + "".join(chess.FILE_NAMES[file_index].center(SQUARE_WIDTH) for file_index in files)
            + "\n"
        )

        for rank in ranks:
            square_lines = [Text(f" {rank + 1}  " if line == 1 else "    ") for line in range(SQUARE_HEIGHT)]
            for file_index in files:
                square = chess.square(file_index, rank)
                is_light = (rank + file_index) % 2 == 0
                background = LIGHT_SQUARE if is_light else DARK_SQUARE
                if square == self.selected:
                    background = "#d7c14d"
                square_style = f"on {background}"
                piece = board.piece_at(square)

                for line_index, line in enumerate(square_lines):
                    if piece is not None and line_index == SQUARE_HEIGHT // 2:
                        fg = "white" if piece.color == chess.WHITE else "black"
                        symbol = piece_symbol(piece)
                        left = (SQUARE_WIDTH - 1) // 2
                        right = SQUARE_WIDTH - left - 1
                        line.append(" " * left, style=square_style)
                        line.append(symbol, style=f"bold {fg} on {background}")
                        line.append(" " * right, style=square_style)
                    else:
                        line.append(" " * SQUARE_WIDTH, style=square_style)

            for line_index, line in enumerate(square_lines):
                line.append(f" {rank + 1}" if line_index == 1 else "")
                line.append("\n")
                output.append_text(line)

        output.append(
            "    "
            + "".join(chess.FILE_NAMES[file_index].center(SQUARE_WIDTH) for file_index in files)
        )
        return output

    def on_click(self, event) -> None:
        offset = event.get_content_offset(self)
        if offset is None:
            return
        x, y = offset
        display_rank = (y - 1) // SQUARE_HEIGHT
        display_file = (x - 4) // SQUARE_WIDTH
        if not (0 <= display_rank < 8 and 0 <= display_file < 8):
            return
        if self.game.human_color is PlayerColor.WHITE:
            rank = 7 - display_rank
            file_index = display_file
        else:
            rank = display_rank
            file_index = 7 - display_file
        square = chess.square(file_index, rank)
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
        width: 44;
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
                yield Button("Play White", id="play-white")
                yield Button("Play Black", id="play-black")
                yield Button("New Game", id="new-game")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_all()

    def action_reset(self) -> None:
        self.game.reset()
        self.selected = None
        self.refresh_all()

    def set_human_color(self, color: PlayerColor) -> None:
        self.game = ChessGame(human_color=color)
        self.board_view.game = self.game
        self.ai = BasicAI()
        self.selected = None
        self.refresh_all(f"You play {color.value}.")

    def refresh_all(self, message: str | None = None) -> None:
        status = [
            f"You: {self.game.human_color.value}",
            f"Turn: {'white' if self.game.board.turn == chess.WHITE else 'black'}",
            f"Status: {self.game.status()}",
            f"Book: {self.ai.last_book_name or 'search'}",
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
            book_name = self.ai.last_book_name
            san = self.game.push_ai_move(move)
            suffix = f" ({book_name})" if book_name else ""
            self.refresh_all(f"AI played {san}{suffix}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        result = self.game.push_human_move(event.value)
        event.input.value = ""
        self.refresh_all(result.message)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-game":
            self.action_reset()
        elif event.button.id == "play-white":
            self.set_human_color(PlayerColor.WHITE)
        elif event.button.id == "play-black":
            self.set_human_color(PlayerColor.BLACK)

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
