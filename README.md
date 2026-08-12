# Chess

A clean terminal chess project. The first version is deliberately simple:

- Standard chess rules via `python-chess`
- Human vs basic rule-based AI with shallow alpha-beta search
- Square terminal board with black and white Unicode pieces
- Text input such as `e2e4`, `Nf3`, `O-O`, `quit`
- Optional Textual UI with mouse/touch-friendly square selection
- Built-in repertoire opening book

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

The Textual mode supports clicking the source square and target square. It also
has buttons to choose whether you play White or Black.

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

1. Deepen alpha-beta search with transposition tables.
2. Add PGN import for master games.
3. Add opening-book support.
4. Add self-play data generation.
5. Add neural network policy/value model.

## Opening Repertoire

The built-in book gives the AI a clear style before search takes over:

- As White: Queen's Gambit lines, including declined, exchange, accepted, Slav,
  Semi-Slav, Tarrasch, Albin, and Chigorin structures.
- As Black against `1.e4`: Sicilian Defense lines, including Najdorf,
  Classical, Dragon, Accelerated Dragon, Kan, Alapin, and Closed Sicilian.
- As Black against `1.d4`, `1.c4`, or `1.Nf3`: Dutch Defense and related
  Classical/Leningrad setups.

This is a curated starter repertoire, not a claim of theoretical perfection.
The next step is importing large PGN files and using engine/database statistics
to expand move weights automatically.
