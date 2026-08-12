# Chess

A clean terminal chess project. The first version is deliberately simple:

- Standard chess rules via `python-chess`
- Human vs basic rule-based AI with shallow alpha-beta search
- Square terminal board with black and white Unicode pieces
- Text input such as `e2e4`, `Nf3`, `O-O`, `quit`
- Optional Textual UI with mouse/touch-friendly square selection
- Optional graphical mouse UI with large centered pieces
- Local browser app with a responsive board and mouse controls
- Browser self-training panel backed by a ResNet CNN policy/value model
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

Graphical mode with large pieces:

```bash
python -m chess_app --gui
```

The graphical mode renders pieces with a large font in the exact center of each
square and supports mouse-only play plus White/Black selection buttons.

Local browser mode:

```bash
python -m chess_app --web
```

Then open `http://127.0.0.1:8765`. The web app uses the same Python rules,
opening book, and AI search as the terminal app.

The web app also includes a self-training panel. Enter a number of self-play
games and review rounds, then start training. The trainer plays complete
self-play games between AI agents, records every position and selected move,
labels each position from the side-to-move perspective using the final result,
and trains a small ResNet CNN with both a policy head and a value head.
Cumulative training games, review rounds, positions, device, value loss, policy
loss, and latest combined loss are shown in the page. A larger statistics panel
below the board summarizes self-play games, total trained games, PGN master
games, review rounds, and trained positions.

You can also paste PGN master games into the training panel or load a `.pgn`
file directly with the PGN file picker. PGN training uses the game result as
the value target and the actual game move as the policy target, so it can
imitate master move choices instead of only learning from self-play outcomes.

The board includes an evaluation bar on the left, similar in spirit to common
chess sites. It uses the current ResNet value model to estimate the position
from White's perspective and animates as the game changes. The training panel
also exposes the learning rate; the default is `0.001`, and smaller values such
as `0.0005` make updates more conservative.

The current model is saved locally to:

```text
models/resnet_policy_value.pt
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
