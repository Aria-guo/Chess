from __future__ import annotations

from dataclasses import dataclass, field

import chess
from flask import Flask, jsonify, request

from chess_app.game import ChessGame, PlayerColor
from chess_app.neural_trainer import NeuralSelfTrainer
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
      grid-template-columns: minmax(390px, calc(76vmin + 42px)) 340px;
      gap: 24px;
      align-items: start;
      justify-content: center;
      padding: 28px;
    }

    .board-wrap {
      width: min(calc(76vmin + 42px), calc(100vw - 420px));
      min-width: 390px;
    }

    .board-shell {
      display: grid;
      grid-template-columns: 24px 1fr;
      gap: 14px;
      align-items: stretch;
    }

    .board-column {
      min-width: 0;
    }

    .eval-bar {
      position: relative;
      min-height: 360px;
      border: 2px solid #27313b;
      background: #111;
      overflow: hidden;
      box-shadow: 0 18px 45px rgba(30, 39, 48, 0.14);
    }

    .eval-white {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 50%;
      background: #f7f7f7;
      transition: height 260ms ease;
    }

    .eval-mid {
      position: absolute;
      left: 0;
      right: 0;
      top: 50%;
      height: 2px;
      background: rgba(203, 62, 62, 0.76);
      transform: translateY(-1px);
    }

    .eval-label {
      position: absolute;
      left: 50%;
      bottom: 8px;
      transform: translateX(-50%) rotate(-90deg);
      transform-origin: center;
      font-size: 12px;
      font-weight: 850;
      color: #111;
      white-space: nowrap;
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

    .trainer {
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 16px;
      display: grid;
      gap: 12px;
    }

    .trainer h2 {
      margin: 0;
      font-size: 18px;
    }

    .train-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    input[type="number"],
    input[type="file"],
    textarea {
      width: 100%;
      border: 1px solid var(--line);
      padding: 8px 10px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }

    input[type="number"],
    input[type="file"] {
      min-height: 40px;
    }

    input[type="file"] {
      padding: 7px 10px;
      cursor: pointer;
    }

    textarea {
      grid-column: 1 / -1;
      min-height: 128px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
      line-height: 1.4;
    }

    .train-button {
      grid-column: 1 / -1;
      border: 1px solid #2c6d55;
      background: #2c6d55;
      color: #fff;
      min-height: 42px;
      font-weight: 800;
      cursor: pointer;
    }

    .train-button.alt {
      background: #6b4c9a;
      border-color: #6b4c9a;
    }

    .train-button:disabled {
      cursor: wait;
      opacity: 0.55;
    }

    .training-status {
      display: grid;
      gap: 8px;
      font-size: 14px;
      padding: 10px 0;
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

      .board-shell {
        grid-template-columns: 18px 1fr;
        gap: 10px;
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
      <div class="board-shell">
        <div class="eval-bar" aria-label="Position evaluation">
          <div class="eval-white" id="eval-white"></div>
          <div class="eval-mid"></div>
          <div class="eval-label" id="eval-label">+0.00</div>
        </div>
        <div class="board-column">
          <div class="files" id="files"></div>
          <div class="board" id="board"></div>
        </div>
      </div>
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
          <div class="row"><span>Eval</span><strong id="eval">+0.00</strong></div>
          <div class="row"><span>Policy</span><strong id="policy-best">-</strong></div>
      </div>
      <p class="message" id="message">Loading game...</p>
      <div class="move-list" id="moves"></div>
      <p class="hint">Click one of your pieces, then click a target square. Pawn promotion by click defaults to queen.</p>
      <section class="trainer">
        <h2>Self Training</h2>
        <div class="train-grid">
          <label>Games
            <input id="train-games" type="number" min="1" max="1000" value="100">
          </label>
          <label>Review rounds
            <input id="train-rounds" type="number" min="1" max="200" value="30">
          </label>
          <label>Learning rate
            <input id="learning-rate" type="number" min="0.00001" max="0.1" step="0.0001" value="0.001">
          </label>
          <button class="train-button" type="button" id="train-button">Train model</button>
          <label>PGN file
            <input id="pgn-file" type="file" accept=".pgn,.txt,text/plain,application/x-chess-pgn">
          </label>
          <label>Master PGN
            <textarea id="pgn-input" spellcheck="false" placeholder='Paste PGN here, for example:&#10;[Event "Model game"]&#10;[Result "1-0"]&#10;&#10;1. d4 d5 2. c4 e6 3. Nc3 Nf6 1-0'></textarea>
          </label>
          <button class="train-button alt" type="button" id="pgn-train-button">Train from PGN</button>
        </div>
        <div class="training-status">
          <div class="row"><span>Architecture</span><strong id="train-arch">ResNet CNN + policy/value heads</strong></div>
          <div class="row"><span>Device</span><strong id="train-device">-</strong></div>
          <div class="row"><span>Total games</span><strong id="train-total-games">0</strong></div>
          <div class="row"><span>Total rounds</span><strong id="train-total-rounds">0</strong></div>
          <div class="row"><span>Positions</span><strong id="train-positions">0</strong></div>
          <div class="row"><span>PGN games</span><strong id="train-pgn-games">0</strong></div>
          <div class="row"><span>PGN positions</span><strong id="train-pgn-positions">0</strong></div>
          <div class="row"><span>Last loss</span><strong id="train-loss">-</strong></div>
          <div class="row"><span>Value loss</span><strong id="train-value-loss">-</strong></div>
          <div class="row"><span>Policy loss</span><strong id="train-policy-loss">-</strong></div>
          <div class="row"><span>Learning rate</span><strong id="train-lr">0.001</strong></div>
        </div>
        <p class="hint" id="train-message">Neural trainer is idle.</p>
      </section>
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
      document.getElementById("eval").textContent = state.evaluation.label;
      document.getElementById("policy-best").textContent = state.policy.length ? state.policy[0].uci : "-";
      messageEl.textContent = state.message;
      document.getElementById("moves").textContent = state.moves.length ? state.moves.join("\n") : "No moves yet.";
      renderEvaluation(state.evaluation);
      renderTraining(state.training);
    }

    function renderEvaluation(evaluation) {
      document.getElementById("eval-white").style.height = `${evaluation.white_percent}%`;
      document.getElementById("eval-label").textContent = evaluation.label;
      document.getElementById("eval-label").style.color = evaluation.white_percent > 35 ? "#111" : "#f7f7f7";
    }

    function renderTraining(training) {
      document.getElementById("train-arch").textContent = training.architecture;
      document.getElementById("train-device").textContent = training.device;
      document.getElementById("train-total-games").textContent = training.total_self_play_games;
      document.getElementById("train-total-rounds").textContent = training.total_review_rounds;
      document.getElementById("train-positions").textContent = training.total_positions;
      document.getElementById("train-pgn-games").textContent = training.total_pgn_games;
      document.getElementById("train-pgn-positions").textContent = training.total_pgn_positions;
      document.getElementById("train-loss").textContent = training.last_loss === null ? "-" : training.last_loss.toFixed(4);
      document.getElementById("train-value-loss").textContent = training.last_value_loss === null ? "-" : training.last_value_loss.toFixed(4);
      document.getElementById("train-policy-loss").textContent = training.last_policy_loss === null ? "-" : training.last_policy_loss.toFixed(4);
      document.getElementById("train-lr").textContent = Number(training.learning_rate).toPrecision(3);
      const lrInput = document.getElementById("learning-rate");
      if (document.activeElement !== lrInput) {
        lrInput.value = training.learning_rate;
      }
      document.getElementById("train-message").textContent = training.message;
      document.getElementById("train-button").disabled = training.running;
      document.getElementById("pgn-train-button").disabled = training.running;
      document.getElementById("pgn-file").disabled = training.running;
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

    async function refreshTraining() {
      const payload = await api("/api/training");
      if (state) {
        state.training = payload.training;
        state.evaluation = payload.evaluation;
        state.policy = payload.policy;
        renderTraining(payload.training);
        renderEvaluation(payload.evaluation);
        document.getElementById("eval").textContent = payload.evaluation.label;
        document.getElementById("policy-best").textContent = payload.policy.length ? payload.policy[0].uci : "-";
      }
    }

    async function startTraining() {
      const games = Number(document.getElementById("train-games").value || 1);
      const reviewRounds = Number(document.getElementById("train-rounds").value || 1);
      const learningRate = Number(document.getElementById("learning-rate").value || 0.001);
      const training = await api("/api/train", {games, review_rounds: reviewRounds, learning_rate: learningRate});
      if (state) {
        state.training = training;
        renderTraining(training);
      }
    }

    async function startPgnTraining() {
      const pgn = document.getElementById("pgn-input").value;
      const reviewRounds = Number(document.getElementById("train-rounds").value || 1);
      const learningRate = Number(document.getElementById("learning-rate").value || 0.001);
      const training = await api("/api/train-pgn", {pgn, review_rounds: reviewRounds, learning_rate: learningRate});
      if (state) {
        state.training = training;
        renderTraining(training);
      }
    }

    async function loadPgnFile(file) {
      if (!file) return;
      const text = await file.text();
      document.getElementById("pgn-input").value = text;
      document.getElementById("train-message").textContent = `Loaded ${file.name} (${text.length.toLocaleString()} characters).`;
    }

    document.querySelectorAll("[data-new]").forEach(button => {
      button.addEventListener("click", () => newGame(button.dataset.new));
    });
    document.getElementById("new-game").addEventListener("click", () => newGame(state?.human_color || "white"));
    document.getElementById("train-button").addEventListener("click", () => {
      startTraining().catch(error => {
        document.getElementById("train-message").textContent = error.message;
      });
    });
    document.getElementById("pgn-train-button").addEventListener("click", () => {
      startPgnTraining().catch(error => {
        document.getElementById("train-message").textContent = error.message;
      });
    });
    document.getElementById("pgn-file").addEventListener("change", event => {
      loadPgnFile(event.target.files[0]).catch(error => {
        document.getElementById("train-message").textContent = error.message;
      });
    });
    setInterval(() => refreshTraining().catch(() => {}), 1500);

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

    def state(self, evaluation: dict | None = None, policy: list[dict] | None = None) -> dict:
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
            "evaluation": evaluation or {"white_value": 0.0, "white_percent": 50.0, "black_percent": 50.0, "label": "+0.00"},
            "policy": policy or [],
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
    trainer = NeuralSelfTrainer()
    session.reset(human_color)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.get("/api/state")
    def state():
        payload = session.state(
            evaluation=trainer.evaluation_payload(session.game.board),
            policy=trainer.policy_payload(session.game.board),
        )
        payload["training"] = trainer.payload()
        return jsonify(payload)

    @app.post("/api/new")
    def new_game():
        payload = request.get_json(silent=True) or {}
        session.reset(parse_color(payload.get("color", "white")))
        response = session.state(
            evaluation=trainer.evaluation_payload(session.game.board),
            policy=trainer.policy_payload(session.game.board),
        )
        response["training"] = trainer.payload()
        return jsonify(response)

    @app.post("/api/move")
    def move():
        payload = request.get_json(silent=True) or {}
        try:
            session.play_human_move(str(payload.get("from", "")), str(payload.get("to", "")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        response = session.state(
            evaluation=trainer.evaluation_payload(session.game.board),
            policy=trainer.policy_payload(session.game.board),
        )
        response["training"] = trainer.payload()
        return jsonify(response)

    @app.get("/api/training")
    def training():
        return jsonify(
            {
                "training": trainer.payload(),
                "evaluation": trainer.evaluation_payload(session.game.board),
                "policy": trainer.policy_payload(session.game.board),
            }
        )

    @app.post("/api/train")
    def train():
        payload = request.get_json(silent=True) or {}
        games = int(payload.get("games", 100))
        review_rounds = int(payload.get("review_rounds", 30))
        learning_rate = float(payload.get("learning_rate", 0.001))
        started = trainer.start_background_training(
            games=games,
            review_rounds=review_rounds,
            learning_rate=learning_rate,
        )
        response = trainer.payload()
        if not started:
            response["message"] = "Training is already running."
            return jsonify(response), 409
        return jsonify(response)

    @app.post("/api/train-pgn")
    def train_pgn():
        payload = request.get_json(silent=True) or {}
        pgn_text = str(payload.get("pgn", ""))
        review_rounds = int(payload.get("review_rounds", 30))
        learning_rate = float(payload.get("learning_rate", 0.001))
        if not pgn_text.strip():
            return jsonify({"error": "Paste at least one PGN game first."}), 400
        started = trainer.start_background_pgn_training(
            pgn_text=pgn_text,
            review_rounds=review_rounds,
            learning_rate=learning_rate,
        )
        response = trainer.payload()
        if not started:
            response["message"] = "Training is already running."
            return jsonify(response), 409
        return jsonify(response)

    return app


def run_web(host: str = "127.0.0.1", port: int = 8765, human_color: PlayerColor = PlayerColor.WHITE) -> None:
    app = create_app(human_color=human_color)
    app.run(host=host, port=port, debug=False, threaded=True)
