from __future__ import annotations

import chess
import pygame

from chess_app.game import ChessGame, PlayerColor
from chess_app.random_ai import BasicAI
from chess_app.terminal import piece_symbol


BOARD_SIZE = 704
SQUARE_SIZE = BOARD_SIZE // 8
SIDE_WIDTH = 300
WINDOW_SIZE = (BOARD_SIZE + SIDE_WIDTH, BOARD_SIZE)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
SELECTED_SQUARE = (220, 194, 75)
WHITE_PIECE = (248, 248, 248)
BLACK_PIECE = (18, 18, 18)
TEXT_COLOR = (32, 34, 36)
PANEL_BG = (236, 238, 240)
BUTTON_BG = (56, 117, 162)
BUTTON_HOVER = (73, 143, 196)


class PygameChessApp:
    def __init__(self, human_color: PlayerColor = PlayerColor.WHITE) -> None:
        pygame.init()
        pygame.display.set_caption("Chess")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.game = ChessGame(human_color=human_color)
        self.ai = BasicAI()
        self.selected_square: chess.Square | None = None
        self.message = "Click a piece, then a target square."
        self.piece_font = self.load_piece_font()
        self.text_font = pygame.font.SysFont("Arial", 22)
        self.small_font = pygame.font.SysFont("Arial", 18)
        self.buttons = {
            "white": pygame.Rect(BOARD_SIZE + 32, 130, 236, 44),
            "black": pygame.Rect(BOARD_SIZE + 32, 186, 236, 44),
            "new": pygame.Rect(BOARD_SIZE + 32, 252, 236, 44),
        }

    def load_piece_font(self) -> pygame.font.Font:
        font_names = [
            "Arial Unicode MS",
            "DejaVu Sans",
            "Noto Sans Symbols 2",
            "Apple Symbols",
            "Symbola",
        ]
        for name in font_names:
            path = pygame.font.match_font(name)
            if path:
                return pygame.font.Font(path, 76)
        return pygame.font.Font(None, 84)

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            if self.game.is_ai_turn and not self.game.board.is_game_over(claim_draw=True):
                self.play_ai_move()

            self.draw()
            self.clock.tick(60)

        pygame.quit()

    def handle_click(self, pos: tuple[int, int]) -> None:
        x, y = pos
        if x >= BOARD_SIZE:
            self.handle_button_click(pos)
            return

        if not self.game.is_human_turn:
            return

        square = self.square_at_position(x, y)
        if square is None:
            return

        if self.selected_square is None:
            piece = self.game.board.piece_at(square)
            if piece is None or piece.color != self.game.human_color.chess_color:
                self.message = "Choose one of your pieces."
                return
            self.selected_square = square
            self.message = f"Selected {chess.square_name(square)}."
            return

        move_text = chess.square_name(self.selected_square) + chess.square_name(square)
        self.selected_square = None
        result = self.game.push_human_move(move_text)
        self.message = result.message

    def handle_button_click(self, pos: tuple[int, int]) -> None:
        if self.buttons["white"].collidepoint(pos):
            self.set_color(PlayerColor.WHITE)
        elif self.buttons["black"].collidepoint(pos):
            self.set_color(PlayerColor.BLACK)
        elif self.buttons["new"].collidepoint(pos):
            self.reset()

    def set_color(self, color: PlayerColor) -> None:
        self.game = ChessGame(human_color=color)
        self.ai = BasicAI()
        self.selected_square = None
        self.message = f"You play {color.value}."

    def reset(self) -> None:
        self.game.reset()
        self.ai = BasicAI()
        self.selected_square = None
        self.message = "New game."

    def play_ai_move(self) -> None:
        move = self.ai.choose_move(self.game.board)
        if move is None:
            return
        book_name = self.ai.last_book_name
        san = self.game.push_ai_move(move)
        self.message = f"AI played {san}" + (f" ({book_name})" if book_name else "")

    def square_at_position(self, x: int, y: int) -> chess.Square | None:
        file_display = x // SQUARE_SIZE
        rank_display = y // SQUARE_SIZE
        if not (0 <= file_display < 8 and 0 <= rank_display < 8):
            return None
        if self.game.human_color is PlayerColor.WHITE:
            file_index = file_display
            rank = 7 - rank_display
        else:
            file_index = 7 - file_display
            rank = rank_display
        return chess.square(file_index, rank)

    def square_rect(self, square: chess.Square) -> pygame.Rect:
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)
        if self.game.human_color is PlayerColor.WHITE:
            display_file = file_index
            display_rank = 7 - rank
        else:
            display_file = 7 - file_index
            display_rank = rank
        return pygame.Rect(
            display_file * SQUARE_SIZE,
            display_rank * SQUARE_SIZE,
            SQUARE_SIZE,
            SQUARE_SIZE,
        )

    def draw(self) -> None:
        self.draw_board()
        self.draw_side_panel()
        pygame.display.flip()

    def draw_board(self) -> None:
        for square in chess.SQUARES:
            rect = self.square_rect(square)
            is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 0
            color = LIGHT_SQUARE if is_light else DARK_SQUARE
            if square == self.selected_square:
                color = SELECTED_SQUARE
            pygame.draw.rect(self.screen, color, rect)

            piece = self.game.board.piece_at(square)
            if piece is not None:
                self.draw_piece(piece, rect)

    def draw_piece(self, piece: chess.Piece, rect: pygame.Rect) -> None:
        color = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
        shadow_color = (70, 70, 70) if piece.color == chess.WHITE else (220, 220, 220)
        symbol = piece_symbol(piece)
        shadow = self.piece_font.render(symbol, True, shadow_color)
        glyph = self.piece_font.render(symbol, True, color)
        shadow_rect = shadow.get_rect(center=(rect.centerx + 2, rect.centery + 3))
        glyph_rect = glyph.get_rect(center=rect.center)
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(glyph, glyph_rect)

    def draw_side_panel(self) -> None:
        panel = pygame.Rect(BOARD_SIZE, 0, SIDE_WIDTH, BOARD_SIZE)
        pygame.draw.rect(self.screen, PANEL_BG, panel)
        self.draw_text("Chess", BOARD_SIZE + 32, 34, self.text_font)
        self.draw_text(f"You: {self.game.human_color.value}", BOARD_SIZE + 32, 76, self.small_font)
        turn = "white" if self.game.board.turn == chess.WHITE else "black"
        self.draw_text(f"Turn: {turn}", BOARD_SIZE + 32, 102, self.small_font)

        mouse = pygame.mouse.get_pos()
        self.draw_button("Play White", self.buttons["white"], self.buttons["white"].collidepoint(mouse))
        self.draw_button("Play Black", self.buttons["black"], self.buttons["black"].collidepoint(mouse))
        self.draw_button("New Game", self.buttons["new"], self.buttons["new"].collidepoint(mouse))

        self.draw_wrapped(self.message, BOARD_SIZE + 32, 330, 236)
        self.draw_wrapped(self.game.status(), BOARD_SIZE + 32, 420, 236)
        book = self.ai.last_book_name or "search"
        self.draw_wrapped(f"Book: {book}", BOARD_SIZE + 32, 500, 236)

    def draw_button(self, label: str, rect: pygame.Rect, hovered: bool) -> None:
        pygame.draw.rect(self.screen, BUTTON_HOVER if hovered else BUTTON_BG, rect, border_radius=6)
        text = self.small_font.render(label, True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=rect.center))

    def draw_text(self, text: str, x: int, y: int, font: pygame.font.Font) -> None:
        surface = font.render(text, True, TEXT_COLOR)
        self.screen.blit(surface, (x, y))

    def draw_wrapped(self, text: str, x: int, y: int, width: int) -> None:
        words = text.split()
        line = ""
        line_y = y
        for word in words:
            candidate = f"{line} {word}".strip()
            if self.small_font.size(candidate)[0] <= width:
                line = candidate
            else:
                self.draw_text(line, x, line_y, self.small_font)
                line = word
                line_y += 24
        if line:
            self.draw_text(line, x, line_y, self.small_font)


def run_pygame(human_color: PlayerColor = PlayerColor.WHITE) -> None:
    PygameChessApp(human_color=human_color).run()

