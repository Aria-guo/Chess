from __future__ import annotations

from dataclasses import dataclass, field

import chess
from flask import Flask, jsonify, request

from chess_app.game import ChessGame, PlayerColor
from chess_app.neural_trainer import PGN_LEARNING_RATE, PGN_REVIEW_ROUNDS, NeuralSelfTrainer
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
      top: 50%;
      bottom: auto;
      z-index: 2;
      padding: 3px 6px;
      background: rgba(120, 132, 145, 0.22);
      border-radius: 3px;
      transform: translate(-50%, -50%) rotate(-90deg);
      transform-origin: center;
      font-size: 12px;
      font-weight: 850;
      color: #111;
      white-space: nowrap;
    }

    .eval-side-label {
      position: absolute;
      left: 50%;
      z-index: 2;
      transform: translateX(-50%) rotate(-90deg);
      transform-origin: center;
      font-size: 11px;
      font-weight: 900;
      letter-spacing: 0;
      white-space: nowrap;
      pointer-events: none;
    }

    .eval-side-label.top {
      top: 18px;
    }

    .eval-side-label.bottom {
      bottom: 8px;
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

    .promotion-modal {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      background: rgba(18, 24, 31, 0.38);
      z-index: 20;
    }

    .promotion-modal.open {
      display: flex;
    }

    .promotion-dialog {
      width: min(360px, 100%);
      background: var(--surface);
      border: 1px solid var(--line);
      box-shadow: 0 22px 60px rgba(17, 24, 31, 0.28);
      padding: 18px;
    }

    .promotion-dialog h2 {
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: 0;
    }

    .promotion-options {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }

    .promotion-choice {
      border: 1px solid var(--line);
      background: #f8fafb;
      min-height: 74px;
      display: grid;
      place-items: center;
      font-size: 44px;
      line-height: 1;
      cursor: pointer;
      font-family: "Arial Unicode MS", "Noto Sans Symbols 2", "DejaVu Sans", "Apple Symbols", serif;
    }

    .promotion-choice:hover {
      border-color: var(--accent);
      background: #eef6fa;
    }

    .promotion-cancel {
      margin-top: 12px;
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      min-height: 38px;
      font-weight: 750;
      cursor: pointer;
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

    .watch-row {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      background: #f8fafb;
      padding: 10px 12px;
    }

    .watch-row input {
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }

    .live-self-play {
      display: none;
      gap: 10px;
      border: 1px solid var(--line);
      background: #f8fafb;
      padding: 12px;
    }

    .live-self-play.open {
      display: grid;
    }

    .live-board {
      width: min(100%, 260px);
      aspect-ratio: 1 / 1;
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      grid-template-rows: repeat(8, 1fr);
      border: 2px solid #27313b;
      background: #27313b;
    }

    .live-square {
      display: grid;
      place-items: center;
      font-family: "Arial Unicode MS", "Noto Sans Symbols 2", "DejaVu Sans", "Apple Symbols", serif;
      font-size: clamp(18px, 4.5vmin, 32px);
      line-height: 1;
    }

    .live-square.light {
      background: var(--light-square);
    }

    .live-square.dark {
      background: var(--dark-square);
    }

    .live-piece.white {
      color: #fafafa;
      -webkit-text-stroke: 0.7px rgba(30, 30, 30, 0.34);
      text-shadow: 0 1px 1px rgba(0, 0, 0, 0.22);
    }

    .live-piece.black {
      color: #111;
      -webkit-text-stroke: 0.7px rgba(255, 255, 255, 0.2);
    }

    .live-moves {
      min-height: 58px;
      max-height: 92px;
      overflow: auto;
      border: 1px solid var(--line);
      background: #fff;
      padding: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
    }

    .stats-panel {
      grid-column: 1 / -1;
      background: var(--surface);
      border: 1px solid var(--line);
      padding: 18px 20px 20px;
      box-shadow: 0 12px 28px rgba(30, 39, 48, 0.08);
    }

    .stats-panel h2 {
      margin: 0 0 14px;
      font-size: 20px;
      letter-spacing: 0;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }

    .stat {
      border: 1px solid var(--line);
      background: #f8fafb;
      padding: 14px;
      min-height: 86px;
      display: grid;
      align-content: space-between;
      gap: 8px;
    }

    .stat span {
      color: var(--muted);
      font-size: 13px;
      font-weight: 750;
    }

    .stat strong {
      font-size: 28px;
      line-height: 1;
      letter-spacing: 0;
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

      .stats-grid {
        grid-template-columns: 1fr 1fr;
      }
    }

    @media (max-width: 520px) {
      .stats-grid {
        grid-template-columns: 1fr;
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
          <div class="eval-side-label top" id="eval-top-label">B 50%</div>
          <div class="eval-side-label bottom" id="eval-bottom-label">W 50%</div>
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
      <p class="hint">Click one of your pieces, then click a target square. Pawns promote on the final rank.</p>
      <section class="trainer">
        <h2>Self Training</h2>
        <div class="train-grid">
          <label>Games
            <input id="train-games" type="number" min="1" max="1000" value="200">
          </label>
          <label>Review rounds
            <input id="train-rounds" type="number" min="1" max="200" value="20">
          </label>
          <label>Learning rate
            <input id="learning-rate" type="number" min="0.00001" max="0.1" step="0.0001" value="0.0005">
          </label>
          <label class="watch-row">Watch self-play
            <input id="watch-self-play" type="checkbox" checked>
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
          <div class="row"><span>PGN preset</span><strong id="train-pgn-preset">5 rounds / 0.0005</strong></div>
        </div>
        <div class="live-self-play" id="live-self-play">
          <div class="row"><span>Live game</span><strong id="live-self-play-meta">-</strong></div>
          <div class="live-board" id="live-self-play-board"></div>
          <div class="live-moves" id="live-self-play-moves">-</div>
        </div>
        <p class="hint" id="train-message">Neural trainer is idle.</p>
      </section>
    </aside>
    <section class="stats-panel" aria-label="Training statistics">
      <h2>Training Stats</h2>
      <div class="stats-grid">
        <div class="stat">
          <span>Self-play games</span>
          <strong id="stats-self-play-games">0</strong>
        </div>
        <div class="stat">
          <span>Total trained games</span>
          <strong id="stats-total-trained-games">0</strong>
        </div>
        <div class="stat">
          <span>PGN master games</span>
          <strong id="stats-pgn-games">0</strong>
        </div>
        <div class="stat">
          <span>Review rounds</span>
          <strong id="stats-review-rounds">0</strong>
        </div>
        <div class="stat">
          <span>Current task</span>
          <strong id="stats-active-task">idle</strong>
        </div>
        <div class="stat">
          <span>Current games</span>
          <strong id="stats-active-games">0</strong>
        </div>
        <div class="stat">
          <span>Current positions</span>
          <strong id="stats-active-positions">0</strong>
        </div>
        <div class="stat">
          <span>Review progress</span>
          <strong id="stats-review-progress">0/0</strong>
        </div>
        <div class="stat">
          <span>Total positions</span>
          <strong id="stats-total-positions">0</strong>
        </div>
        <div class="stat">
          <span>PGN positions</span>
          <strong id="stats-pgn-positions">0</strong>
        </div>
        <div class="stat">
          <span>Value loss</span>
          <strong id="stats-value-loss">-</strong>
        </div>
        <div class="stat">
          <span>Policy loss</span>
          <strong id="stats-policy-loss">-</strong>
        </div>
      </div>
    </section>
  </main>
  <div class="promotion-modal" id="promotion-modal" role="dialog" aria-modal="true" aria-labelledby="promotion-title">
    <div class="promotion-dialog">
      <h2 id="promotion-title">Choose promotion</h2>
      <div class="promotion-options" id="promotion-options"></div>
      <button class="promotion-cancel" type="button" id="promotion-cancel">Cancel</button>
    </div>
  </div>
  <script>
    const boardEl = document.getElementById("board");
    const filesEl = document.getElementById("files");
    const messageEl = document.getElementById("message");
    const promotionModal = document.getElementById("promotion-modal");
    const promotionOptionsEl = document.getElementById("promotion-options");
    const promotionPieceSymbols = {
      white: {q: "♕", r: "♖", b: "♗", n: "♘"},
      black: {q: "♛", r: "♜", b: "♝", n: "♞"}
    };
    const promotionNames = {q: "Queen", r: "Rook", b: "Bishop", n: "Knight"};
    const fenPieceSymbols = {
      P: ["♙", "white"], N: ["♘", "white"], B: ["♗", "white"], R: ["♖", "white"], Q: ["♕", "white"], K: ["♔", "white"],
      p: ["♟", "black"], n: ["♞", "black"], b: ["♝", "black"], r: ["♜", "black"], q: ["♛", "black"], k: ["♚", "black"]
    };
    let state = null;
    let selected = null;
    let pendingPromotion = null;
    let aiThinking = false;

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

    function promotionOptions(from, to) {
      return state.legal_moves
        .filter(m => m.from === from && m.to === to && m.promotion)
        .map(m => m.promotion);
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
      if (aiThinking) messageEl.textContent = "AI is thinking...";
      renderEvaluation(state.evaluation);
      renderTraining(state.training);
    }

    function renderEvaluation(evaluation) {
      const humanIsWhite = state?.human_color !== "black";
      const bar = document.querySelector(".eval-bar");
      const fill = document.getElementById("eval-white");

      // The fill div is always at bottom:0, growing upward.
      // When human is white: fill = white (light), height = white_percent
      // When human is black: flip the bar so black is at the bottom (matching the board).
      //   After scaleY(-1), the fill moves to the TOP. So we set fill = white (light)
      //   with height = white_percent. After flipping, white is at top, black at bottom.
      if (humanIsWhite) {
        bar.style.transform = "none";
        fill.style.height = `${evaluation.white_percent}%`;
        fill.style.background = "#f7f7f7";
        bar.style.background = "#111";
      } else {
        bar.style.transform = "scaleY(-1)";
        fill.style.height = `${evaluation.white_percent}%`;
        fill.style.background = "#f7f7f7";
        bar.style.background = "#111";
      }

      // Show eval from human's perspective
      const humanEvalValue = humanIsWhite ? evaluation.white_value : -evaluation.white_value;
      const humanEvalLabel = (humanEvalValue >= 0 ? "+" : "") + humanEvalValue.toFixed(2);
      // Flip the label text back so it's readable when bar is flipped
      const labelEl = document.getElementById("eval-label");
      labelEl.textContent = humanEvalLabel;
      labelEl.style.transform = humanIsWhite ? "translate(-50%, -50%) rotate(-90deg)" : "translate(-50%, -50%) rotate(90deg)";
      document.getElementById("eval").textContent = humanEvalLabel;

      // Side labels
      const whitePct = Math.round(evaluation.white_percent);
      const blackPct = 100 - whitePct;
      if (humanIsWhite) {
        document.getElementById("eval-top-label").textContent = `B ${blackPct}%`;
        document.getElementById("eval-top-label").style.color = "#f7f7f7";
        document.getElementById("eval-bottom-label").textContent = `W ${whitePct}%`;
        document.getElementById("eval-bottom-label").style.color = "#111";
      } else {
        // When flipped, top/bottom are swapped visually
        document.getElementById("eval-top-label").textContent = `W ${whitePct}%`;
        document.getElementById("eval-top-label").style.color = "#111";
        document.getElementById("eval-bottom-label").textContent = `B ${blackPct}%`;
        document.getElementById("eval-bottom-label").style.color = "#f7f7f7";
      }
    }

    function formatInteger(value) {
      return Number(value || 0).toLocaleString();
    }

    function formatLoss(value) {
      return value === null ? "-" : Number(value).toFixed(4);
    }

    function formatTask(value) {
      if (value === "pgn") return "PGN";
      if (value === "self-play") return "Self-play";
      return "Idle";
    }

    function renderTraining(training) {
      const totalTrainedGames = Number(training.total_self_play_games || 0) + Number(training.total_pgn_games || 0);
      const reviewProgress = `${formatInteger(training.active_review_round)}/${formatInteger(training.active_review_rounds)}`;
      document.getElementById("train-arch").textContent = training.architecture;
      document.getElementById("train-device").textContent = training.device;
      document.getElementById("train-total-games").textContent = training.total_self_play_games;
      document.getElementById("train-total-rounds").textContent = training.total_review_rounds;
      document.getElementById("train-positions").textContent = training.total_positions;
      document.getElementById("train-pgn-games").textContent = training.total_pgn_games;
      document.getElementById("train-pgn-positions").textContent = training.total_pgn_positions;
      document.getElementById("train-loss").textContent = formatLoss(training.last_loss);
      document.getElementById("train-value-loss").textContent = formatLoss(training.last_value_loss);
      document.getElementById("train-policy-loss").textContent = formatLoss(training.last_policy_loss);
      document.getElementById("train-pgn-preset").textContent = `${training.pgn_review_rounds} rounds / ${training.pgn_learning_rate}`;
      document.getElementById("stats-self-play-games").textContent = formatInteger(training.total_self_play_games);
      document.getElementById("stats-total-trained-games").textContent = formatInteger(totalTrainedGames);
      document.getElementById("stats-pgn-games").textContent = formatInteger(training.total_pgn_games);
      document.getElementById("stats-review-rounds").textContent = formatInteger(training.total_review_rounds);
      document.getElementById("stats-active-task").textContent = formatTask(training.active_task);
      document.getElementById("stats-active-games").textContent = formatInteger(training.active_games);
      document.getElementById("stats-active-positions").textContent = formatInteger(training.active_positions);
      document.getElementById("stats-review-progress").textContent = reviewProgress;
      document.getElementById("stats-total-positions").textContent = formatInteger(training.total_positions);
      document.getElementById("stats-pgn-positions").textContent = formatInteger(training.total_pgn_positions);
      document.getElementById("stats-value-loss").textContent = formatLoss(training.last_value_loss);
      document.getElementById("stats-policy-loss").textContent = formatLoss(training.last_policy_loss);
      document.getElementById("train-lr").textContent = Number(training.learning_rate).toPrecision(3);
      const lrInput = document.getElementById("learning-rate");
      if (document.activeElement !== lrInput) {
        lrInput.value = training.learning_rate;
      }
      document.getElementById("train-message").textContent = training.message;
      document.getElementById("train-button").disabled = training.running;
      document.getElementById("pgn-train-button").disabled = training.running;
      document.getElementById("pgn-file").disabled = training.running;
      renderLiveSelfPlay(training);
    }

    function renderLiveSelfPlay(training) {
      const panel = document.getElementById("live-self-play");
      const watch = document.getElementById("watch-self-play").checked;
      const live = training.live_self_play || {};
      if (!watch || !live.fen) {
        panel.classList.remove("open");
        return;
      }
      panel.classList.add("open");
      const gameLabel = training.active_task === "self-play" ? `${formatInteger(training.active_games + 1)} playing` : "last game";
      const result = live.result ? ` · ${live.result}` : "";
      document.getElementById("live-self-play-meta").textContent = `${gameLabel} · ${formatInteger(live.ply)} ply${result}`;
      document.getElementById("live-self-play-moves").textContent = live.moves?.length ? live.moves.join(" ") : "-";
      renderFenBoard(live.fen);
    }

    function renderFenBoard(fen) {
      const board = document.getElementById("live-self-play-board");
      board.innerHTML = "";
      const rows = fen.split(" ")[0].split("/");
      rows.forEach((row, rankIndex) => {
        let fileIndex = 0;
        [...row].forEach(token => {
          const empty = Number(token);
          if (Number.isInteger(empty) && empty > 0) {
            for (let index = 0; index < empty; index += 1) {
              appendLiveSquare(board, rankIndex, fileIndex, null);
              fileIndex += 1;
            }
            return;
          }
          appendLiveSquare(board, rankIndex, fileIndex, token);
          fileIndex += 1;
        });
      });
    }

    function appendLiveSquare(board, rankIndex, fileIndex, token) {
      const square = document.createElement("div");
      square.className = `live-square ${(rankIndex + fileIndex) % 2 === 0 ? "light" : "dark"}`;
      if (token && fenPieceSymbols[token]) {
        const piece = document.createElement("span");
        const [symbol, color] = fenPieceSymbols[token];
        piece.className = `live-piece ${color}`;
        piece.textContent = symbol;
        square.appendChild(piece);
      }
      board.appendChild(square);
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
      const promotions = promotionOptions(from, square);
      if (promotions.length) {
        pendingPromotion = {from, to: square, options: promotions};
        render();
        showPromotionPicker();
        return;
      }
      await submitMove(from, square);
    }

    async function submitMove(from, to, promotion = null) {
      try {
        state = await api("/api/move", {from, to, promotion});
      } catch (error) {
        messageEl.textContent = error.message;
        render();
        return;
      }
      render();
      if (state.is_ai_turn && !state.game_over) {
        await requestAiMove();
      }
    }

    async function requestAiMove() {
      aiThinking = true;
      messageEl.textContent = "AI is thinking...";
      try {
        state = await api("/api/ai-move", {});
      } catch (error) {
        messageEl.textContent = error.message;
      } finally {
        aiThinking = false;
      }
      render();
    }

    function showPromotionPicker() {
      if (!pendingPromotion) return;
      promotionOptionsEl.innerHTML = "";
      pendingPromotion.options.forEach(piece => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "promotion-choice";
        button.textContent = promotionPieceSymbols[state.human_color][piece];
        button.setAttribute("aria-label", promotionNames[piece]);
        button.addEventListener("click", () => choosePromotion(piece));
        promotionOptionsEl.appendChild(button);
      });
      promotionModal.classList.add("open");
    }

    function hidePromotionPicker() {
      promotionModal.classList.remove("open");
      promotionOptionsEl.innerHTML = "";
      pendingPromotion = null;
    }

    function choosePromotion(piece) {
      if (!pendingPromotion) return;
      const move = pendingPromotion;
      hidePromotionPicker();
      submitMove(move.from, move.to, piece);
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
      const training = await api("/api/train-pgn", {pgn});
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
    document.getElementById("watch-self-play").addEventListener("change", () => {
      if (state?.training) renderLiveSelfPlay(state.training);
    });
    document.getElementById("promotion-cancel").addEventListener("click", hidePromotionPicker);
    promotionModal.addEventListener("click", event => {
      if (event.target === promotionModal) hidePromotionPicker();
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") hidePromotionPicker();
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
    trainer: NeuralSelfTrainer | None = None
    message: str = "Click one of your pieces, then a target square."
    moves: list[str] = field(default_factory=list)
    ai_thinking: bool = False

    def reset(self, color: PlayerColor) -> None:
        self.game = ChessGame(human_color=color)
        self.ai = BasicAI()
        self.message = f"You play {color.value}."
        self.moves = []
        self.play_ai_if_needed()

    def play_human_move(self, from_square: str, to_square: str, promotion: str | None = None, play_ai: bool = False) -> None:
        if not self.game.is_human_turn:
            raise ValueError("It is not your turn.")

        move = self.parse_click_move(from_square, to_square, promotion)
        if move not in self.game.board.legal_moves:
            raise ValueError(f"Illegal move: {from_square}{to_square}")

        san = self.game.board.san(move)
        self.game.board.push(move)
        self.moves.append(f"You: {san}")
        self.message = f"You played {san}."
        if play_ai:
            self.play_ai_if_needed()
        elif self.game.is_ai_turn and not self.game.board.is_game_over(claim_draw=True):
            self.message = f"You played {san}. AI is thinking..."

    def parse_click_move(self, from_square: str, to_square: str, promotion: str | None = None) -> chess.Move:
        try:
            from_index = chess.parse_square(from_square)
            to_index = chess.parse_square(to_square)
        except ValueError as exc:
            raise ValueError("Invalid square.") from exc

        promotion_piece = parse_promotion_piece(promotion)
        move = chess.Move(from_index, to_index, promotion=promotion_piece)
        piece = self.game.board.piece_at(from_index)
        if (
            piece is not None
            and piece.piece_type == chess.PAWN
            and chess.square_rank(to_index) in {0, 7}
            and promotion_piece is None
        ):
            raise ValueError("Choose a piece for pawn promotion.")
        return move

    def play_ai_if_needed(self) -> None:
        if not self.game.is_ai_turn or self.game.board.is_game_over(claim_draw=True):
            return
        self.ai_thinking = True
        self.ai.last_book_name = None
        try:
            book_move = self.ai.opening_book.choose(self.game.board)
            if book_move is not None:
                self.ai.last_book_name = book_move.name
                move = book_move.move
                engine_name = book_move.name
            elif self.trainer is not None:
                move = self.trainer.choose_engine_move(self.game.board, self.ai)
                engine_name = "neural policy search"
            else:
                move = self.ai.choose_move(self.game.board)
                engine_name = self.ai.last_book_name
            if move is None:
                return
            book_name = self.ai.last_book_name
            san = self.game.push_ai_move(move)
            self.moves.append(f"AI: {san}")
            self.message = f"AI played {san}" + (f" ({book_name or engine_name})." if book_name or engine_name else ".")
        finally:
            self.ai_thinking = False

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
            "is_ai_turn": self.game.is_ai_turn,
            "ai_thinking": self.ai_thinking,
            "game_over": board.is_game_over(claim_draw=True),
            "status": self.game.status(),
            "book": self.ai.last_book_name,
            "evaluation": evaluation or {"white_value": 0.0, "white_percent": 50.0, "black_percent": 50.0, "label": "+0.00"},
            "policy": policy or [],
            "message": self.message,
            "files": [chess.FILE_NAMES[file_index] for file_index in files],
            "squares": squares,
            "legal_moves": [
                {
                    "from": chess.square_name(move.from_square),
                    "to": chess.square_name(move.to_square),
                    "promotion": promotion_piece_symbol(move.promotion),
                    "uci": move.uci(),
                }
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


def parse_promotion_piece(value: str | None) -> chess.PieceType | None:
    if value in {None, "", "none"}:
        return None
    pieces = {
        "q": chess.QUEEN,
        "queen": chess.QUEEN,
        "r": chess.ROOK,
        "rook": chess.ROOK,
        "b": chess.BISHOP,
        "bishop": chess.BISHOP,
        "n": chess.KNIGHT,
        "knight": chess.KNIGHT,
    }
    piece = pieces.get(str(value).lower())
    if piece is None:
        raise ValueError("Promotion must be queen, rook, bishop, or knight.")
    return piece


def promotion_piece_symbol(piece: chess.PieceType | None) -> str | None:
    if piece == chess.QUEEN:
        return "q"
    if piece == chess.ROOK:
        return "r"
    if piece == chess.BISHOP:
        return "b"
    if piece == chess.KNIGHT:
        return "n"
    return None


def create_app(human_color: PlayerColor = PlayerColor.WHITE, trainer: NeuralSelfTrainer | None = None) -> Flask:
    app = Flask(__name__)
    trainer = trainer or NeuralSelfTrainer()
    session = WebSession(trainer=trainer)
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
            session.play_human_move(
                str(payload.get("from", "")),
                str(payload.get("to", "")),
                payload.get("promotion"),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        response = session.state(
            evaluation=trainer.evaluation_payload(session.game.board),
            policy=trainer.policy_payload(session.game.board),
        )
        response["training"] = trainer.payload()
        return jsonify(response)

    @app.post("/api/ai-move")
    def ai_move():
        session.play_ai_if_needed()
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
        if not pgn_text.strip():
            return jsonify({"error": "Paste at least one PGN game first."}), 400
        started = trainer.start_background_pgn_training(
            pgn_text=pgn_text,
            review_rounds=PGN_REVIEW_ROUNDS,
            learning_rate=PGN_LEARNING_RATE,
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
