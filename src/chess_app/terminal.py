from __future__ import annotations

import chess
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from chess_app.game import ChessGame, PlayerColor
from chess_app.random_ai import BasicAI


PIECES = {
    chess.PAWN: ("♙", "♟"),
    chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"),
    chess.ROOK: ("♖", "♜"),
    chess.QUEEN: ("♕", "♛"),
    chess.KING: ("♔", "♚"),
}

SQUARE_WIDTH = 8
SQUARE_HEIGHT = 5
PIECE_BLOCK_WIDTH = 3
PIECE_BLOCK_HEIGHT = 3
LIGHT_SQUARE = "#f0d9b5"
DARK_SQUARE = "#b58863"


def piece_symbol(piece: chess.Piece | None) -> str:
    if piece is None:
        return " "
    white, black = PIECES[piece.piece_type]
    return white if piece.color == chess.WHITE else black


def piece_block_line(piece: chess.Piece | None, line_index: int) -> str | None:
    if piece is None:
        return None
    block_start = (SQUARE_HEIGHT - PIECE_BLOCK_HEIGHT) // 2
    if block_start <= line_index < block_start + PIECE_BLOCK_HEIGHT:
        return piece_symbol(piece) * PIECE_BLOCK_WIDTH
    return None


def render_board(board: chess.Board, perspective: PlayerColor) -> Panel:
    ranks = range(7, -1, -1) if perspective is PlayerColor.WHITE else range(8)
    files = range(8) if perspective is PlayerColor.WHITE else range(7, -1, -1)

    board_text = Text()
    board_text.append("    ")
    for file_index in files:
        board_text.append(chess.FILE_NAMES[file_index].center(SQUARE_WIDTH))
    board_text.append("\n")

    for rank_index in ranks:
        square_lines = [Text(f" {rank_index + 1}  " if line == 1 else "    ") for line in range(SQUARE_HEIGHT)]
        for file_index in files:
            square = chess.square(file_index, rank_index)
            is_light = (rank_index + file_index) % 2 == 0
            background = LIGHT_SQUARE if is_light else DARK_SQUARE
            square_style = f"on {background}"
            piece = board.piece_at(square)

            for line_index, line in enumerate(square_lines):
                piece_text = piece_block_line(piece, line_index)
                if piece_text is not None:
                    fg = "white" if piece.color == chess.WHITE else "black"
                    left = (SQUARE_WIDTH - PIECE_BLOCK_WIDTH) // 2
                    right = SQUARE_WIDTH - left - PIECE_BLOCK_WIDTH
                    line.append(" " * left, style=square_style)
                    line.append(piece_text, style=f"bold {fg} on {background}")
                    line.append(" " * right, style=square_style)
                else:
                    line.append(" " * SQUARE_WIDTH, style=square_style)

        for line_index, line in enumerate(square_lines):
            line.append(f"  {rank_index + 1}" if line_index == 1 else "")
            line.append("\n")
            board_text.append_text(line)

    board_text.append("    ")
    for file_index in files:
        board_text.append(chess.FILE_NAMES[file_index].center(SQUARE_WIDTH))

    return Panel(board_text, title="Board", border_style="cyan", expand=False)


def render_info(game: ChessGame, last_ai_move: str | None, book_name: str | None = None) -> Panel:
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
    lines.append(f"Book: {book_name or 'search'}")
    lines.append("Input: e2e4, Nf3, O-O, resign, quit")
    return Panel("\n".join(lines), title="Chess", border_style="cyan")


def run_terminal(human_color: PlayerColor = PlayerColor.WHITE) -> None:
    console = Console()
    game = ChessGame(human_color=human_color)
    ai = BasicAI()
    last_ai_move: str | None = None
    last_book_name: str | None = None

    while not game.board.is_game_over(claim_draw=True):
        console.clear()
        console.print(render_board(game.board, game.human_color))
        console.print(render_info(game, last_ai_move, last_book_name))

        if game.is_ai_turn:
            move = ai.choose_move(game.board)
            if move is None:
                break
            last_book_name = ai.last_book_name
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
    console.print(render_info(game, last_ai_move, last_book_name))
    console.print(f"[bold]Result: {game.result() or game.status()}[/]")
