from __future__ import annotations

from dataclasses import dataclass, field

import chess
from flask import Flask, jsonify, request

from chess_app.game import ChessGame, PlayerColor
from chess_app.random_ai import BasicAI
from chess_app.terminal import piece_symbol


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chess</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef1f3;
      --surface: #ffffff;
      --ink: #1d2329;
      --muted: #66717c;
      --line: #cfd7df;
      --light-square: #f0d9b5;
      --dark-square: #b58863;
      --selected: #f1d44f;
      --legal: rgba(39, 117, 82, 0.35);
      --accent: #2f6f9f;
      --accent-hover: #255d86;
      --danger: #9f3434;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(360px, 76vmin) 340px;
      gap: 24px;
      align-items: start;
      justify-content: center;
      padding: 28px;
    }

    .board-wrap {
      width: min(76vmin, calc(100vw - 420px));
      min-width: 360px;
    }

    .files {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      padding: 0 0 8px;
      color: var(--muted);
      font-weight: 700;
      text-align: center;
      user-select: none;
    }

    .board {
      aspect-ratio: 1 / 1;
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      grid-template-rows: repeat(8, 1fr);
      border: 2px solid #27313b;
      box-shadow: 0 18px 45px rgba(30, 39, 48, 0.18);
      background: #27313b;
    }

    .square {
      position: relative;
      border: 0;
      margin: 0;
      padding: 0;
      display: grid;
      place-items: center;
      font-size: clamp(38px, 7.6vmin, 86px);
      line-height: 1;
      cursor: pointer;
      font-family: "Arial Unicode MS", "Noto Sans Symbols 2", "DejaVu Sans", "Apple Symbols", serif;
    }

    .square.light {
      background: var(--light-square);
    }

    .square.dark {
      background: var(--dark-square);
    }

    .square.selected {
      background: var(--selected);
    }

    .square.legal::after {
      content: "";
      width: 28%;
      height: 28%;
      border-radius: 50%;
      background: var(--legal);
      position: absolute;
    }

    .square.capture.legal::after {
      width: 82%;
      height: 82%;
      background: transparent;
      border: 5px solid var(--legal);
    }

    .piece {
      transform: translateY(-1.5%);
      text-align: center;
      text-shadow: 0 2px 2px rgba(0, 0, 0, 0.28);
      pointer-events: none;
    }

    .piece.white {
      color: #fafafa;
      -webkit-text-stroke: 1px rgba(30, 30, 30, 0.34);
    }

    .piece.black {
      color: #111;
      -webkit-text-stroke: 1px rgba(255, 255, 255, 0.22);
    }

    .rank-label,
    .file-label {
      position: absolute;
      font-size: 13px;
      font-family: inherit;
      font-weight: 800;
      opacity: 0.72;
      pointer-events: none;
    }

    .rank-label {
      top: 6px;
      left: 7px;
    }

    .file-label {
      right: 7px;
      bottom: 4px;
    }

    .side {
      background: var(--surface);
      border: 1px solid var(--line);
      padding: 20px;
      min-height: 520px;
      box-shadow: 0 12px 28px rgba(30, 39, 48, 0.08);
    }

    h1 {
      margin: 0 0 18px;
      font-size: 28px;
      letter-spacing: 0;
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 16px;
    }

    button.control {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      min-height: 42px;
      font-weight: 750;
      cursor: pointer;
    }

    button.control:hover {
      background: var(--accent-hover);
    }

    button.control.secondary {
      grid-column: 1 / -1;
      background: #ffffff;
      color: var(--accent);
    }

    .status {
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 14px 0;
      display: grid;
      gap: 10px;
      font-size: 15px;
    }

    .row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }

    .row span:first-child {
      color: var(--muted);
    }

    .message {
      margin: 16px 0 0;
      min-height: 48px;
      color: var(--ink);
      font-weight: 650;
    }

    .move-list {
      margin-top: 16px;
      border: 1px solid var(--line);
      min-height: 120px;
      max-height: 210px;
      overflow: auto;
      padding: 10px 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 14px;
      background: #f8fafb;
    }

    .hint {
      margin-top: 14px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }

    @media (max-width: 900px) {
      .app {
        grid-template-columns: 1fr;
        padding: 16px;
      }

      .board-wrap {
        width: 100%;
        min-width: 0;
      }

      .side {
        min-height: 0;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="board-wrap" aria-label="Chess board">
      <div class="files" id="files"></div>
      <div class="board" id="board"></div>
    </section>
    <aside class="side">
      <h1>Chess</h1>
      <div class="controls">
        <button class="control" type="button" data-new="white">Play White</button>
        <button class="control" type="button" data-new="black">Play Black</button>
        <button class="control secondary" type="button" id="new-game">New Game</button>
      </div>
      <div class="status">
        <div class="row"><span>You</span><strong id="human-color">white</strong></div>
        <div class="row"><span>Turn</span><strong id="turn">white</strong></div>
        <div class="row"><span>Status</span><strong id="status">Loading</strong></div>
        <div class="row"><span>Book</span><strong id="book">search</strong></div>
      </div>
      <p class="message" id="message">Loading game...</p>
      <div class="move-list" id="moves"></div>
      <p class="hint">Click one of your pieces, then click a target square. Pawn promotion by click defaults to queen.</p>
    </aside>
  </main>
  <script>
    const boardEl = document.getElementById("board");
    const filesEl = document.getElementById("files");
    const messageEl = document.getElementById("message");
    let state = null;
    let selected = null;

    async function api(path, body) {
      const options = body === undefined ? {} : {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      };
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Request failed");
      return payload;
    }

    async function loadState() {
      state = await api("/api/state");
      selected = null;
      render();
    }

    function legalTargets(from) {
      return new Set(state.legal_moves.filter(m => m.from === from).map(m => m.to));
    }

    function pieceOn(squareName) {
      return state.squares.find(s => s.square === squareName)?.piece || null;
    }

    function renderFiles() {
      filesEl.innerHTML = "";
      state.files.forEach(file => {
        const cell = document.createElement("div");
        cell.textContent = file;
        filesEl.appendChild(cell);
      });
    }

    function render() {
      renderFiles();
      boardEl.innerHTML = "";
      const targets = selected ? legalTargets(selected) : new Set();
      const targetPieces = new Map(state.squares.map(s => [s.square, s.piece]));

      state.squares.forEach(sq => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `square ${sq.light ? "light" : "dark"}`;
        button.dataset.square = sq.square;
        button.setAttribute("aria-label", sq.square);
        if (sq.square === selected) button.classList.add("selected");
        if (targets.has(sq.square)) {
          button.classList.add("legal");
          if (targetPieces.get(sq.square)) button.classList.add("capture");
        }

        if (sq.showRank) {
          const rank = document.createElement("span");
          rank.className = "rank-label";
          rank.textContent = sq.rank;
          button.appendChild(rank);
        }
        if (sq.showFile) {
          const file = document.createElement("span");
          file.className = "file-label";
          file.textContent = sq.file;
          button.appendChild(file);
        }
        if (sq.piece) {
          const piece = document.createElement("span");
          piece.className = `piece ${sq.piece.color}`;
          piece.textContent = sq.piece.symbol;
          button.appendChild(piece);
        }
        button.addEventListener("click", () => clickSquare(sq.square));
        boardEl.appendChild(button);
      });

      document.getElementById("human-color").textContent = state.human_color;
      document.getElementById("turn").textContent = state.turn;
      document.getElementById("status").textContent = state.status;
      document.getElementById("book").textContent = state.book || "search";
      messageEl.textContent = state.message;
      document.getElementById("moves").textContent = state.moves.length ? state.moves.join("\n") : "No moves yet.";
    }

    async function clickSquare(square) {
      if (!state.is_human_turn || state.game_over) return;
      if (!selected) {
        const piece = pieceOn(square);
        if (!piece || piece.color !== state.human_color) {
          messageEl.textContent = "Choose one of your pieces.";
          return;
        }
        selected = square;
        render();
        return;
      }

      const from = selected;
      selected = null;
      try {
        state = await api("/api/move", {from, to: square});
      } catch (error) {
        messageEl.textContent = error.message;
        render();
        return;
      }
      render();
    }

    async function newGame(color) {
      state = await api("/api/new", {color});
      selected = null;
      render();
    }

    document.querySelectorAll("[data-new]").forEach(button => {
      button.addEventListener("click", () => newGame(button.dataset.new));
    });
    document.getElementById("new-game").addEventListener("click", () => newGame(state?.human_color || "white"));

    loadState().catch(error => {
      messageEl.textContent = error.message;
    });
  </script>
</body>
</html>
"""


@dataclass
class WebSession:
    game: ChessGame = field(default_factory=ChessGame)
    ai: BasicAI = field(default_factory=BasicAI)
    message: str = "Click one of your pieces, then a target square."
    moves: list[str] = field(default_factory=list)

    def reset(self, color: PlayerColor) -> None:
        self.game = ChessGame(human_color=color)
        self.ai = BasicAI()
        self.message = f"You play {color.value}."
        self.moves = []
        self.play_ai_if_needed()

    def play_human_move(self, from_square: str, to_square: str) -> None:
        if not self.game.is_human_turn:
            raise ValueError("It is not your turn.")

        move = self.parse_click_move(from_square, to_square)
        if move not in self.game.board.legal_moves:
            raise ValueError(f"Illegal move: {from_square}{to_square}")

        san = self.game.board.san(move)
        self.game.board.push(move)
        self.moves.append(f"You: {san}")
        self.message = f"You played {san}."
        self.play_ai_if_needed()

    def parse_click_move(self, from_square: str, to_square: str) -> chess.Move:
        try:
            from_index = chess.parse_square(from_square)
            to_index = chess.parse_square(to_square)
        except ValueError as exc:
            raise ValueError("Invalid square.") from exc

        move = chess.Move(from_index, to_index)
        piece = self.game.board.piece_at(from_index)
        if (
            piece is not None
            and piece.piece_type == chess.PAWN
            and chess.square_rank(to_index) in {0, 7}
        ):
            promoted = chess.Move(from_index, to_index, promotion=chess.QUEEN)
            if promoted in self.game.board.legal_moves:
                return promoted
        return move

    def play_ai_if_needed(self) -> None:
        if not self.game.is_ai_turn or self.game.board.is_game_over(claim_draw=True):
            return
        move = self.ai.choose_move(self.game.board)
        if move is None:
            return
        book_name = self.ai.last_book_name
        san = self.game.push_ai_move(move)
        self.moves.append(f"AI: {san}")
        self.message = f"AI played {san}" + (f" ({book_name})." if book_name else ".")

    def state(self) -> dict:
        board = self.game.board
        files = list(range(8)) if self.game.human_color is PlayerColor.WHITE else list(range(7, -1, -1))
        ranks = list(range(7, -1, -1)) if self.game.human_color is PlayerColor.WHITE else list(range(8))
        squares = []

        for rank in ranks:
            for file_index in files:
                square = chess.square(file_index, rank)
                piece = board.piece_at(square)
                square_name = chess.square_name(square)
                squares.append(
                    {
                        "square": square_name,
                        "file": chess.FILE_NAMES[file_index],
                        "rank": str(rank + 1),
                        "showFile": rank == ranks[-1],
                        "showRank": file_index == files[0],
                        "light": (rank + file_index) % 2 == 0,
                        "piece": self.piece_payload(piece),
                    }
                )

        return {
            "human_color": self.game.human_color.value,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "is_human_turn": self.game.is_human_turn,
            "game_over": board.is_game_over(claim_draw=True),
            "status": self.game.status(),
            "book": self.ai.last_book_name,
            "message": self.message,
            "files": [chess.FILE_NAMES[file_index] for file_index in files],
            "squares": squares,
            "legal_moves": [
                {"from": chess.square_name(move.from_square), "to": chess.square_name(move.to_square)}
                for move in board.legal_moves
            ],
            "moves": self.moves[-24:],
        }

    @staticmethod
    def piece_payload(piece: chess.Piece | None) -> dict | None:
        if piece is None:
            return None
        return {
            "symbol": piece_symbol(piece),
            "color": "white" if piece.color == chess.WHITE else "black",
            "type": chess.piece_name(piece.piece_type),
        }


def parse_color(value: str) -> PlayerColor:
    return PlayerColor.BLACK if value == "black" else PlayerColor.WHITE


def create_app(human_color: PlayerColor = PlayerColor.WHITE) -> Flask:
    app = Flask(__name__)
    session = WebSession()
    session.reset(human_color)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.get("/api/state")
    def state():
        return jsonify(session.state())

    @app.post("/api/new")
    def new_game():
        payload = request.get_json(silent=True) or {}
        session.reset(parse_color(payload.get("color", "white")))
        return jsonify(session.state())

    @app.post("/api/move")
    def move():
        payload = request.get_json(silent=True) or {}
        try:
            session.play_human_move(str(payload.get("from", "")), str(payload.get("to", "")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(session.state())

    return app


def run_web(host: str = "127.0.0.1", port: int = 8765, human_color: PlayerColor = PlayerColor.WHITE) -> None:
    app = create_app(human_color=human_color)
    app.run(host=host, port=port, debug=False)

