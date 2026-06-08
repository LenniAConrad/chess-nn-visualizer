# Final Checklist And Future Work

This file is a defense/pre-submission note. It is not a list of blockers.

## Course Requirement Checklist

| Requirement from `Project2026.pdf` | Status | Evidence |
| --- | --- | --- |
| Project proposal and requirements analysis | Done | `docs/project-proposal-requirements.md`, `docs/pdf/project-proposal-requirements.pdf`. |
| Design specification | Done | `docs/design-spec.md`, class diagram, data/member descriptions, architecture notes. |
| User manual | Done | `docs/user-manual.md`, `docs/pdf/user-manual.pdf`. |
| Source code with comments | Done | Multi-file C++17 source under `src/`, public header comments, German/Armenian-style implementation comments where useful. |
| Classes and file I/O | Done | `Position`, `Game`, NN loaders, config/FEN/model/test-data I/O. |
| Multi-file `.h`/`.cpp` structure | Done | Separate modules under `src/chess`, `src/game`, `src/io`, `src/nn`, `src/search`, `src/viz`. |
| Optional scoring items | Done | Inheritance/polymorphism, templates, arrays, linked list, encapsulated module structure. |
| Test cases | Done | `docs/test-cases.md`, `docs/pdf/test-cases.pdf`, automated and manual workflow cases. |
| Demo screen recording | Done | `docs/demo.mp4`, `docs/tutorial.gif`. |
| Screenshot evidence | Done | `docs/screenshots/*.png`, included in the test-case report. |
| Summary report | Done | `docs/summary-report.md`, `docs/pdf/summary-report.pdf`. |
| AI usage description | Done | `docs/ai-usage.md`, summary-report section 9, explicit statement that the app was not made by AI. |
| Defense data/package | Done | Runtime models, Windows package, and final submission zip under `dist/`. |

## Current Submission Status

The project currently includes:

- legal human-vs-human chess play,
- FEN load/save and setup-editor validation,
- linked-list move history with undo/redo,
- NNUE, CNN, BT4, Classical, and Off activation choices,
- Classical evaluator heatmaps by piece type,
- BT4 attention boards that show the current pieces,
- active-color Search button state,
- PUCT MCTS tree workbench,
- Linux build scripts,
- Windows x64 package under `dist/`,
- Markdown documents and PDF exports under `docs/`,
- tutorial GIF/MP4 media,
- screenshot evidence in `docs/screenshots/`,
- automated test suite passing on 2026-06-08 with 151 tests.

## Before A Live Demo

Run:

```sh
./scripts/test.sh
./build/tests/cnnv_tests
./scripts/demo_check.sh
```

Use this archive for a complete file upload:

```text
dist/cpp-nn-visualizer-submission.zip
```

The archive includes ignored local runtime artifacts such as `models/*.bin` and
the Windows package, so it is safer for submission than a plain Git export.

Launch:

```sh
./scripts/run.sh
```

The checked-in config starts windowed with `startup.arch=cnn`. If a quieter
opening is preferred for a recorded walkthrough, set `startup.arch=off` before
recording.

## Demo Flow

1. Build and launch.
2. Play a few legal moves and show highlights, move list, status, undo, and
   redo.
3. Load Kiwipete:
   ```text
   r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
   ```
4. Save FEN to `position.fen`.
5. Enter Setup, create a legal K+Q vs K position, validate, and apply.
6. Create an invalid setup position and show validation blocking Apply.
7. Cycle NNUE, CNN, BT4, Classical, and Off.
8. In BT4 Trace/Atlas, point out that attention boards include pieces.
9. In Classical detailed mode, show per-piece PST heatmaps.
10. Start and stop Search; confirm the button color changes while active.
11. Open Tree, run MCTS, use Fit/zoom/pan, branch/depth filters, Follow, and
    the scrubber.
12. Return to the board.

## Current Scope Boundaries

- Move input is mouse-based.
- FEN load/save is supported; PGN import is not.
- Save FEN writes to `position.fen`.
- There is no engine-opponent play mode yet.
- The bundled CNN and BT4 model files are compact untrained visualization
  models. They run real native forward paths but are not strength models.
- Official LC0 protobuf import is not implemented.

## Optional Future Work

- Engine-vs-human mode using the existing search/evaluation path.
- PGN import.
- File pickers for FEN/PGN/model paths.
- Side-by-side architecture comparison.
- Difference view between two positions.
- Official LC0 protobuf model import.
- More automated graphical UI tests.
