from __future__ import annotations

import chess
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from chess_app.game import ChessGame, PlayerColor
from chess_app.random_ai import RandomAI


PIECES = {
    chess.PAWN: ("♙", "♟"),
    chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"),
    chess.ROOK: ("♖", "♜"),
    chess.QUEEN: ("♕", "♛"),
    chess.KING: ("♔", "♚"),
}


def piece_symbol(piece: chess.Piece | None) -> str:
    if piece is None:
        return " "
    white, black = PIECES[piece.piece_type]
    return white if piece.color == chess.WHITE else black


def render_board(board: chess.Board, perspective: PlayerColor) -> Table:
    ranks = range(7, -1, -1) if perspective is PlayerColor.WHITE else range(8)
    files = range(8) if perspective is PlayerColor.WHITE else range(7, -1, -1)

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right")
    for _ in files:
        table.add_column(justify="center", width=3)
    table.add_column()

    file_labels = [chess.FILE_NAMES[file_index] for file_index in files]
    table.add_row("", *file_labels, "")

    for rank_index in ranks:
        cells: list[Text] = []
        for file_index in files:
            square = chess.square(file_index, rank_index)
            symbol = piece_symbol(board.piece_at(square))
            is_light = (rank_index + file_index) % 2 == 0
            style = "black on #d8c99b" if is_light else "white on #6d7f8f"
            cells.append(Text(f" {symbol} ", style=style))
        rank_label = str(rank_index + 1)
        table.add_row(rank_label, *cells, rank_label)

    table.add_row("", *file_labels, "")
    return table


def render_info(game: ChessGame, last_ai_move: str | None) -> Panel:
    turn = "White" if game.board.turn == chess.WHITE else "Black"
    human = "White" if game.human_color is PlayerColor.WHITE else "Black"
    legal_count = game.board.legal_moves.count()
    lines = [
        f"Turn: {turn}",
        f"You: {human}",
        f"Legal moves: {legal_count}",
        f"Status: {game.status()}",
    ]
    if last_ai_move:
        lines.append(f"AI last move: {last_ai_move}")
    lines.append("Input: e2e4, Nf3, O-O, resign, quit")
    return Panel("\n".join(lines), title="Chess", border_style="cyan")


def run_terminal(human_color: PlayerColor = PlayerColor.WHITE) -> None:
    console = Console()
    game = ChessGame(human_color=human_color)
    ai = RandomAI()
    last_ai_move: str | None = None

    while not game.board.is_game_over(claim_draw=True):
        console.clear()
        console.print(render_board(game.board, game.human_color))
        console.print(render_info(game, last_ai_move))

        if game.is_ai_turn:
            move = ai.choose_move(game.board)
            if move is None:
                break
            last_ai_move = game.push_ai_move(move)
            continue

        move_text = console.input("[bold green]Your move> [/]")
        command = move_text.strip().lower()
        if command in {"quit", "exit", "q"}:
            console.print("Bye.")
            return
        if command in {"resign", "ff"}:
            console.print("You resigned.")
            return

        result = game.push_human_move(move_text)
        if not result.ok:
            console.print(f"[red]{result.message}[/]")
            console.input("Press Enter to continue...")

    console.clear()
    console.print(render_board(game.board, game.human_color))
    console.print(render_info(game, last_ai_move))
    console.print(f"[bold]Result: {game.result() or game.status()}[/]")

