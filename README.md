# cpp-nn-visualizer

`cpp-nn-visualizer` is a native C++17/raylib chess application for exploring
chess evaluation models. It combines a legal human-vs-human chess board with
activation views for NNUE, an LC0-style CNN, an LC0 BT4-style transformer, and
a handcrafted Classical evaluator. The app also includes a full-screen PUCT
search-tree workbench with mini-board nodes, principal-variation highlighting,
and pan/zoom navigation.

The runtime is self-contained C++. It does not call external chess engines,
Java programs, Python scripts, web services, or AI APIs while running. Model
files are local binary blobs loaded from `models/`.

The app was not made by AI. AI assistants were used only for minor support work
such as documentation wording, debugging explanations, and review checklists;
project design, implementation, integration, and testing remained student work.
See [docs/ai-usage.md](docs/ai-usage.md).

## Team

| Name | Student number |
| --- | --- |
| Lennart Axel Conrad | 2025080264 |
| Erik Mkrtchyan | 2025080273 |

Course requirements are archived in [Project2026.pdf](Project2026.pdf).

Related prior chess work:

- [LenniAConrad/chess-rtk](https://github.com/LenniAConrad/chess-rtk):
  Java chess research toolkit used as reference material for chess-domain
  design and selected algorithm translation.
- [LenniAConrad/chess-web](https://github.com/LenniAConrad/chess-web):
  separate TypeScript chess interface used as UI and gameplay background
  reference.

## Current Features

- Legal chess play with mouse click/drag input, legal target highlighting,
  promotion selection, undo/redo, move history, FEN load/save, and setup
  validation.
- NNUE HalfKP visualization: active features, accumulators, clipped activations,
  output contributions, learned-weight atlas, and architecture diagram.
- LC0 CNN visualization: 112 input planes, residual stack, policy logits,
  WDL/value head, board heatmaps, atlas, and diagram modes.
- LC0 BT4 visualization: compact real `BT4J` v2 transformer weights, 64 square
  tokens, multi-head attention, policy logits, WDL/value head, and attention
  boards with live pieces.
- Classical evaluator visualization: material/PST/activity/threat/etc.
  breakdown, WDL triplet, game phase, and per-piece piece-square heatmaps.
- PUCT MCTS workbench: live tree growth, top-branch filtering, transposition
  merge display, batched leaf blobs, growth scrubber, and follow/trace mode.
- Optional chess clock and borderless fullscreen toggle.

The checked-in `config.ini` starts windowed at 1280x800 with `startup.arch=cnn`
so a model-backed panel is visible immediately. Set `startup.arch=off` for a
quiet chess-board-only launch.

## Build On Linux

Requirements: a C++17 compiler, CMake 3.20 or newer, and the X11/OpenGL
libraries required by raylib. On Debian/Ubuntu:

```bash
sudo apt install build-essential cmake libgl1-mesa-dev libx11-dev \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

Build and run:

```bash
./scripts/build.sh
./scripts/run.sh
```

Manual equivalent:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCNNV_ENABLE_CUDA=OFF
cmake --build build -j
./build/cnnv
```

`scripts/build.sh` uses `~/.local/src/raylib` when that checkout exists. To use
another local raylib source, set `RAYLIB_SOURCE_DIR=/path/to/raylib`. Without a
local source, CMake fetches raylib 5.0 through `FetchContent`.

CUDA tensor kernels are optional. The helper scripts default to CPU mode; set
`CNNV_ENABLE_CUDA=ON` when configuring if an `nvcc` toolchain is available.

## Windows Package

A ready-to-run Windows x64 package is staged at:

```text
dist/cpp-nn-visualizer-windows-x64/
```

Run `run-cnnv.bat` or `run-cnnv.ps1` from that directory. The package includes
`cnnv.exe`, `config.ini`, assets, compact runtime models, and the MinGW runtime
DLL required by the executable.

## Model Files

The default runtime model paths are:

```text
models/nnue-halfkp-demo.bin
models/lc0-cnn-small-112p-4x32-policy4672-wdl3.bin
models/lc0-bt4-tiny-96x4x4h.bin
```

Populate or refresh them with:

```bash
./scripts/import_models.sh
```

The CNN and BT4 defaults are compact generated models. They are intentionally
small enough for a teaching visualizer, but they still exercise the real native
forward paths. The BT4 default is a `BT4J` v2 file with 96-dimensional tokens,
4 encoder blocks, 4 attention heads, an attention policy head, and a WDL value
head. The full official LC0 protobuf format is not a runtime dependency.

More details are in [models/README.md](models/README.md).

## Tests

```bash
./scripts/test.sh
```

Current local verification on 2026-06-08:

```text
ctest: 1/1 passed
cnnv_tests: 151 passed, 0 failed
```

The broader demo/readiness check is:

```bash
./scripts/demo_check.sh
```

It checks required documents, PDF exports, tutorial media, model files,
selected config values, media metadata when `ffprobe` is available, and the
automated test suite.

## API Documentation

Public headers use Doxygen/Javadoc-style comments. If Doxygen is installed,
generate browsable API docs with:

```bash
cmake --build build --target docs
```

or:

```bash
doxygen Doxyfile
```

HTML output is written under `build/docs/doxygen/html`.

## Repository Layout

```text
src/chess/       legal-move chess core, FEN/SAN, perft, hashing
src/chess/eval/  handcrafted Classical evaluator and PST heatmaps
src/game/        game state, move history, PGN export
src/io/          config, FEN files, and binary weight loading
src/nn/          tensors, ops, activation snapshots, network interface
src/nn/nnue/     NNUE HalfKP encoder, accumulator, and forward pass
src/nn/lc0_cnn/  LC0J CNN loader, encoder, policy map, and forward pass
src/nn/lc0_bt4/  BT4J loader, real forward pass, and synthetic fallback
src/search/      PUCT MCTS engine used by the tree workbench
src/viz/         raylib app, board, controls, views, editor, tree UI
tests/           unit, regression, numerical, and smoke tests
docs/            project documents, tutorial media, PDF exports
models/          local model blobs and model-format notes
assets/          piece sprites and other UI assets
dist/            packaged Windows build artifacts
```

## Project Documents

- [docs/project-proposal-requirements.md](docs/project-proposal-requirements.md)
- [docs/design-spec.md](docs/design-spec.md)
- [docs/user-manual.md](docs/user-manual.md)
- [docs/test-cases.md](docs/test-cases.md)
- [docs/summary-report.md](docs/summary-report.md)
- [docs/ai-usage.md](docs/ai-usage.md)
- [docs/tutorial.gif](docs/tutorial.gif)
- [docs/demo.mp4](docs/demo.mp4)

Regenerate PDF exports with:

```bash
./scripts/export_docs.sh
```

[TODO.md](TODO.md) is kept as the final defense checklist and future-work note.
