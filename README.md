# Chess

A clean terminal chess project. The first version is deliberately simple:

- Standard chess rules via `python-chess`
- Human vs random legal-move AI
- Pretty terminal board with Unicode pieces
- Text input such as `e2e4`, `Nf3`, `O-O`, `quit`
- Optional Textual UI with mouse/touch-friendly square selection

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

If your shell does not like extras yet, install normally:

```bash
python -m pip install -e .
```

## Play

Simple terminal mode:

```bash
python -m chess_app
```

Mouse/touch-friendly Textual mode:

```bash
python -m chess_app --tui
```

## Move Input

You can enter moves in UCI or SAN:

```text
e2e4
g1f3
Nf3
O-O
resign
quit
```

## Roadmap

1. Add stronger rule-based scoring.
2. Add alpha-beta search.
3. Add PGN import for master games.
4. Add self-play data generation.
5. Add neural network policy/value model.

