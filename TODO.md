# Final Submission Checklist

This file is kept as a teacher/TA-facing readiness note. The original
step-by-step development plan has been completed or folded into the submitted
scope; remaining items below are defense preparation and optional future work,
not blockers for the course submission.

## Submission Status

The project satisfies the mandatory course requirements:

- Classes and encapsulation across `src/chess`, `src/game`, `src/nn`,
  `src/io`, and `src/viz`.
- Multi-file C++ structure with matching headers and implementation files.
- Real file I/O for configuration, FEN save/load, PGN export support, model
  blobs, and test reference data.
- A complete human-vs-human chess game for Tutorial III.
- Automated tests plus manual UI test documentation.
- Required documents and PDF exports under `docs/`.
- Screen recording and animated tutorial under `docs/`.
- AI usage description under `docs/ai-usage.md`.

## What To Show In The Defense

1. Build and readiness check:
   ```sh
   ./scripts/demo_check.sh
   ```
   This verifies required PDFs, tutorial media, local model files, startup
   config, media metadata, and automated tests.

2. Launch:
   ```sh
   ./scripts/run.sh
   ```
   The submitted default starts windowed with the activation panel OFF.

3. Chess-game workflow:
   - Play legal moves by clicking or dragging pieces.
   - Show move highlighting, check/status text, move history, undo/redo, and
     promotion if useful.
   - Use Load FEN with Kiwipete:
     `r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`
   - Use Save FEN to write `position.fen`.

4. Editor workflow:
   - Click Setup or press `T`.
   - Clear the board, place a legal K+Q vs K position, validate, and apply.
   - Enter Setup again and create an invalid position to show validation.

5. Neural-network visualization:
   - Use the architecture buttons or press `A` to cycle NNUE, CNN, BT4, and
     OFF.
   - Show abstract and detailed views for each architecture.
   - Start Search to show depth, node count, evaluation, and PV updates.

## Current Scope Boundaries

These are intentional limitations documented in the user manual:

- Move input is mouse-based; there is no SAN/UCI move-entry box.
- PGN export support exists, but PGN import is not implemented.
- FEN loading expects complete six-field FEN strings.
- Play mode is human-vs-human; there is no engine-opponent mode.
- BT4 is a native deterministic visualization path, not exact LC0 BT4
  trained-weight protobuf inference.
- Save FEN writes to `position.fen`; there is no file picker.

## Final Pre-Submission Check

Before uploading or demonstrating, run:

```sh
./scripts/demo_check.sh
git status --short
```

Expected result: the readiness check passes and `git status` is clean after
committing the final changes.

## Optional Future Work

These are good extensions after the course submission, but not required for the
current grading scope:

- Engine-vs-human mode using the existing search/evaluation path.
- PGN import and a file picker for FEN/PGN paths.
- Side-by-side architecture comparison.
- Activation difference mode between two positions.
- Exact native LC0 BT4 trained-weight import.
