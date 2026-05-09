# cpp-nn-visualizer — Step-by-step TODO

A native C++ raylib visualizer for chess neural-net activations across three
architectures: NNUE (half-KP), LC0 CNN ResNet, LC0 BT4 transformer. Same binary
also serves as Tutorial III deliverable (May 27): a complete human-vs-human
chess game.

This file is the master execution list. Work the steps in order unless a
dependency-free parallel branch is explicitly marked. Each step states:

- **Goal** — what to produce
- **Files** — where it lives
- **Done when** — verifiable acceptance check

When a step is finished, prefix it with `[x]`.

---

## Conventions (read once, apply throughout)

- **Languages / libs.** C++17. Only external libs: `raylib` (graphics) and the
  C++ standard library. No Eigen, no protobuf, no ONNX runtime, no crtk
  subprocess. If a model file format is hard to parse, pre-convert it with
  crtk to a flat binary (one-time dev step).
- **Build.** CMake. Target Linux first; keep platform-specific code behind
  `#ifdef`. No Windows-specific code in core modules.
- **Naming.** Classes `PascalCase`, methods `camelCase`, member fields `m_…`,
  constants `kPascalCase`, files match the primary class (`Position.h` /
  `Position.cpp`). One primary class per `.h`/`.cpp` pair.
- **Headers.** Include guards via `#pragma once`. Headers expose only what
  callers need; everything else `private`. Forward-declare where possible.
- **Const correctness.** Methods that don't mutate state are `const`. Pass
  large objects by `const&`.
- **Comments.** Default to none. Add a comment only when WHY is non-obvious
  (a bug workaround, a non-trivial invariant, a quirk of crtk's encoding).
  Never narrate WHAT well-named code already says.
- **Reference source.** When a step says "transpile from crtk," the canonical
  source is at `/home/lennart/Code/chess-rtk/src/chess/...`. Use it as reading
  reference only: project code, tests, and runtime paths stay native C++ and do
  not shell out to crtk/Java.
- **Tests.** Each phase has its own test step. Add tests as you go; do not
  defer until the end. Tests live under `tests/` and run via `ctest`.
- **Commits.** Small, focused, one logical change each. Don't mix transpile
  + UI + NN in one commit.

### Defaults baked in (override before starting if needed)

- Build target: Linux only.
- ML lib: none.
- Partner split: no enforced ownership; tasks within a phase are mostly
  independent and either partner can pick up the next unblocked one. Tags
  `[chess]`, `[ui]`, `[nn]`, `[docs]` mark the stream.
- Models: weight files copied into `cpp-nn-visualizer/models/` (not
  symlinked) so the project is self-contained for the grader.

---

## Phase 0 — Repo bootstrap & build system

[x] **Step 1.** [bootstrap] Initialise git in `cpp-nn-visualizer/`. Add a
`.gitignore` covering `build/`, `out/`, `*.o`, `*.so`, `models/*.bin`,
`.cache/`, `.vscode/`.
- Files: `.gitignore`
- Done when: `git status` is clean and ignored paths don't show up.

[x] **Step 2.** [bootstrap] Write top-level `CMakeLists.txt` that:
- Sets C++17, warnings-as-errors (`-Wall -Wextra -Wpedantic -Werror`).
- Fetches raylib via `FetchContent` pinned to a stable tag (e.g. `5.0`).
- Defines a static lib `cnnv_core` (chess + nn + io) and a static lib
  `cnnv_viz` (raylib-dependent code).
- Defines an executable `cnnv` linking both.
- Enables `ctest` and adds a test executable target `cnnv_tests` linking
  `cnnv_core`.
- Files: `CMakeLists.txt`
- Done when: `cmake -B build && cmake --build build` succeeds with no
  sources yet (uses placeholder empty `main.cpp` from Step 3).

[x] **Step 3.** [bootstrap] Create `src/main.cpp` that opens a 1280x800 raylib
window titled "cpp-nn-visualizer", clears to a dark background each frame,
exits on ESC. No other behaviour.
- Files: `src/main.cpp`
- Done when: `./build/cnnv` shows the window and closes cleanly.

[x] **Step 4.** [bootstrap] Create the directory skeleton with empty `.gitkeep`
placeholders: `src/chess/`, `src/game/`, `src/nn/`, `src/nn/ops/`,
`src/nn/nnue/`, `src/nn/lc0_cnn/`, `src/nn/lc0_bt4/`, `src/viz/`, `src/io/`,
`tests/`, `assets/`, `models/`, `docs/`.
- Done when: tree matches the layout in `README.md` (Step 5).

[x] **Step 5.** [docs] Write a short `README.md` with: one-paragraph project
description, build steps (`cmake -B build && cmake --build build -j &&
./build/cnnv`), run instructions, and a "where things live" pointer to the
source layout. No marketing fluff.
- Files: `README.md`

[x] **Step 6.** [bootstrap] Add a minimal test harness. Either pull `doctest` as
a single-header dep into `tests/doctest.h` or write a hand-rolled
`tests/TestMain.cpp` with assert macros and a registry. Either way: one test
file `tests/SmokeTest.cpp` containing `TEST(smoke_runs) { CHECK(1+1 == 2); }`.
- Files: `tests/doctest.h` (or `tests/TestMain.cpp`), `tests/SmokeTest.cpp`,
  `tests/CMakeLists.txt`
- Done when: `ctest --test-dir build` passes one test.

[x] **Step 7.** [bootstrap] Wire a CI-friendly `scripts/build.sh` and
`scripts/test.sh` that run cmake configure + build + ctest. Make them
exit-non-zero on failure.
- Files: `scripts/build.sh`, `scripts/test.sh`
- Done when: both scripts succeed locally.

[x] **Step 8.** [docs] Add `docs/design-spec.md`, `docs/user-manual.md`,
`docs/test-cases.md`, `docs/summary-report.md`, `docs/ai-usage.md` as
placeholders with section headings only. We fill them in incrementally as
features land — not at the end.
- Done when: all five files exist with section skeletons.

---

## Phase 1 — Chess core (transpiled subset of crtk)

Reference: `/home/lennart/Code/chess-rtk/src/chess/core/`. We need exactly
enough to play legal chess and feed positions into NNs: bitboards, piece
type, move type, position state, legal move generation, FEN, SAN.

[x] **Step 9.** [chess] Implement `Bitboard.h` with: `using Bitboard = uint64_t`,
`constexpr` square indices A1..H8 (0..63), file/rank masks, `popcount`,
`lsb`, `pop_lsb`, `set_bit`, `clear_bit`, `test_bit`. Header-only.
- Reference: `chess-rtk/src/chess/core/Bits.java`
- Files: `src/chess/Bitboard.h`
- Done when: a unit test verifies popcount on `0xFFFF`, lsb on `0x80`, etc.

[x] **Step 10.** [chess] Implement `Piece.{h,cpp}` with `enum class Color {
White, Black }`, `enum class PieceType { Pawn, Knight, Bishop, Rook, Queen,
King, None }`, and a `Piece` struct combining both. Add helpers:
`fromFenChar(char)`, `toFenChar()`.
- Reference: `chess-rtk/src/chess/core/Piece.java`
- Files: `src/chess/Piece.{h,cpp}`
- Done when: round-trip test `'K' -> Piece -> 'K'` passes for all 12 pieces.

[x] **Step 11.** [chess] Implement `Move.{h,cpp}` as a 16-bit packed move: 6 bits
from-square, 6 bits to-square, 4 bits flags (promo type / castling /
en-passant / capture). Provide `from()`, `to()`, `promotion()`, `isCastle()`,
`isEnPassant()`, `isCapture()`, `==`, `toUci()` returning e.g. "e2e4" or
"e7e8q".
- Reference: `chess-rtk/src/chess/core/Move.java`
- Files: `src/chess/Move.{h,cpp}`
- Done when: UCI round-trip test passes for normal, capture, castle, promo,
  en-passant moves.

[x] **Step 12.** [chess] Implement `MoveList.{h,cpp}` as a fixed-capacity array
(`std::array<Move, 256>` + size). Methods: `push`, `size`, `operator[]`,
`begin/end`, `clear`. No heap allocation in the move-gen path.
- Files: `src/chess/MoveList.{h,cpp}`
- Done when: a test pushes 218 moves and iterates them.

[x] **Step 13.** [chess] Implement `SlidingAttacks.{h,cpp}` providing
ray-attack tables for bishop/rook/queen. Use simple "classical" approach
(loop along ray with blocker mask) — no magic bitboards needed for a
visualizer.
- Reference: `chess-rtk/src/chess/core/SlidingAttacks.java`
- Files: `src/chess/SlidingAttacks.{h,cpp}`
- Done when: bishop attacks from D4 with a blocker on F6 are correct
  (verified in a test).

[x] **Step 14.** [chess] Implement `Position.{h,cpp}` core state: piece bitboards
per color×type (12 boards), occupancy bitboards (white/black/all), side to
move, castling rights, en-passant square, halfmove clock, fullmove number.
Methods: `pieceAt(Square)`, `colorAt(Square)`, `setPiece`, `clearPiece`,
`sideToMove`, accessors only — make/unmake comes in Step 16.
- Reference: `chess-rtk/src/chess/core/Position.java` (just the state +
  accessors part — Position.java is 2.5kloc, we won't transpile all of it)
- Files: `src/chess/Position.{h,cpp}`

[x] **Step 15.** [chess] Implement `Fen.{h,cpp}` with `parse(string) ->
Position` and `format(const Position&) -> string`. Handle all six fields
including Chess960-style castling (KQkq letters; you can defer Shredder-FEN
to a later step if pressed).
- Reference: `chess-rtk/src/chess/core/Fen.java`
- Files: `src/chess/Fen.{h,cpp}`
- Done when: round-trip test for startpos and 5 hand-picked FENs (kiwipete,
  endgame, en-passant, promotion-ready, castling-rights-partial) preserves
  the FEN exactly. Compare to `crtk fen normalize --fen "<FEN>"` output.

[x] **Step 16.** [chess] Add `make(Move)` and `unmake()` to `Position`. Use a
`StateInfo` struct with previous castling rights, ep square, captured
piece, halfmove clock — push to a stack inside `Position` so `unmake` is
O(1).
- Done when: a test plays 4 moves and unmakes them, asserting the FEN
  equals startpos.

[x] **Step 17.** [chess] Implement `MoveGenerator.{h,cpp}` with
`generateLegal(const Position&, MoveList&)`. Generate pseudo-legal moves
for all piece types (including castling and en-passant), then filter by
making/unmaking and checking own-king-not-in-check. Use
`SlidingAttacks` for bishop/rook/queen.
- Reference: `chess-rtk/src/chess/core/MoveGenerator.java`
- Files: `src/chess/MoveGenerator.{h,cpp}`
- Done when: legal move count from startpos equals 20.

[x] **Step 18.** [chess] Implement `perft(Position&, int depth) -> uint64_t`.
Recursive: for each legal move, make, recurse, unmake.
- Files: `src/chess/Perft.{h,cpp}`

[x] **Step 19.** [chess][test] Add perft tests against the standard reference
counts:
- startpos depth 1..5: 20 / 400 / 8902 / 197281 / 4865609
- kiwipete depth 1..4
- position 3 (`8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -`) depth 1..4
- Files: `tests/PerftTest.cpp`
- Done when: all counts match.

[x] **Step 20.** [chess] Implement `San.{h,cpp}` with `toSan(const Position&,
Move) -> string` and `parseSan(const Position&, string) -> Move`. Handle
disambiguation (file/rank/both), checks (`+`), checkmates (`#`), captures,
castling (`O-O`/`O-O-O`), promotions.
- Reference: `chess-rtk/src/chess/core/SAN.java`
- Files: `src/chess/San.{h,cpp}`
- Done when: round-trip test for 20 positions × all legal moves produces
  the same `Move` after parsing back. Compare a sample to
  `crtk move list --fen "<FEN>" --format both`.

[x] **Step 21.** [chess] Add helpers to `Position`: `inCheck() -> bool`,
`isCheckmate() -> bool`, `isStalemate() -> bool`,
`isInsufficientMaterial() -> bool`, `isFiftyMoveDraw() -> bool`. These are
needed for the game loop in Phase 3.
- Done when: tests for fool's mate (checkmate), stalemate-in-1, and KvK
  (insufficient material) return true.

[x] **Step 22.** [chess] Add Zobrist hashing to `Position` (`uint64_t hash()
const`). Used for threefold-repetition detection later. Initialize keys
once with a deterministic PRNG seed.
- Files: `src/chess/Zobrist.{h,cpp}`, modify `Position`
- Done when: making and unmaking a move returns the position to the same
  hash; making the same move from two transposition orders yields the same
  hash.

[x] **Step 23.** [chess] Add threefold-repetition detection to `Position` via a
small ring-buffer of recent hashes (since the last irreversible move).
Method `isThreefoldRepetition() const -> bool`.

[x] **Step 24.** [chess][docs] In `docs/design-spec.md`, fill in the "Chess
core class diagram" and "Data member descriptions" for the classes from
Steps 9–22. Use a Mermaid class diagram block.

[x] **Step 25.** [chess][test] Final chess-core sanity test: load 50 random
FENs from `chess-rtk` test data (`/home/lennart/Code/chess-rtk/testdata/`),
run perft depth 2, compare against `crtk engine perft --fen "<FEN>"
--depth 2`. All must match.

---

## Phase 2 — Single-board UI + file I/O

[x] **Step 26.** [ui] Copy chess piece sprites from
`/home/lennart/Code/chess-rtk/assets/` (or wiki) into
`cpp-nn-visualizer/assets/pieces/` as 12 PNGs (`wK.png`, `wQ.png`, …,
`bP.png`). Pick a single consistent set.
- Done when: all 12 files exist and load without raylib errors.

[x] **Step 27.** [ui] Implement `viz/Theme.{h,cpp}` with named colour constants
(light square, dark square, highlight, last-move, check-warning, panel
backgrounds, text). One central place to tweak look-and-feel.

[x] **Step 28.** [ui] Implement `viz/BoardView.{h,cpp}`. Constructor takes a
`const Position&` reference plus a `Rectangle` bounds. Methods:
- `draw()` — squares + pieces.
- `setFlipped(bool)` — toggle perspective.
- `squareAtPixel(Vector2) -> std::optional<Square>` — for click handling.
- `setHighlights(Bitboard squares, Color)` — generic per-square overlay
  that NN views can use later.
- Files: `src/viz/BoardView.{h,cpp}`
- Done when: `main.cpp` shows the start position rendered correctly.

[x] **Step 29.** [ui] Add click-to-move to `BoardView`: first click selects a
piece (highlight legal targets via `MoveGenerator`), second click attempts
the move. If illegal, deselect; if a promotion is needed, queue a small
modal (Step 32). Emits `onMoveAttempt(Move)` callback.

[x] **Step 30.** [io] Implement `io/FenIo.{h,cpp}`:
- `loadFromFile(path) -> Position`
- `saveToFile(path, const Position&)`
- Files: `src/io/FenIo.{h,cpp}`
- Done when: a test writes a FEN, reads it back, asserts equality.

[x] **Step 31.** [io] Implement `io/ConfigIo.{h,cpp}` reading a simple
`config.ini`-style file from the working dir. Keys we'll need: window size,
default FEN, model paths, theme variant. Write a default config if absent
(this is one piece of demonstrable file-I/O for the grader).
- Files: `src/io/ConfigIo.{h,cpp}`, default `config.ini` in repo root
- Done when: changing `window.width` in `config.ini` changes the launched
  window size.

[x] **Step 32.** [ui] Implement `viz/Controls.{h,cpp}`: a vertical button strip
on the side. Buttons: Reset, Flip, Load FEN (opens a text input),
Save FEN, Undo, Redo. (Architecture switch button is deferred to Phase 8
when networks land.) Wire each to a callback.

[x] **Step 33.** [ui] Implement a promotion-piece picker modal: when a pawn
reaches the back rank, pop a 4-square overlay (Q/R/B/N), clicking one
finalises the move.

[x] **Step 34.** [ui] Add a "FEN load" text input dialog that validates input
via `Fen::parse` and only commits on success. On failure, show an inline
error string.

[x] **Step 35.** [ui][test] Manual smoke test, recorded in
`docs/test-cases.md`: launch app, play 10 moves with mouse, undo to
startpos, load a kiwipete FEN, flip the board, save FEN to file, and include
the flow in the tutorial GIF/video.

---

## Phase 2b — Board editor

A "setup mode" that lets the user freely place pieces, set side-to-move,
castling rights, and en-passant target, then validate and commit the edited
position back into the live `Game`. Useful both for general play setup and
for crafting interesting positions to compare across the three NN
architectures.

[x] **Step 35b1.** [chess] Add a `Position::Builder` helper (or free functions
in `src/chess/PositionEditor.{h,cpp}`) that exposes a *mutable* view of
position state for editor use only: `placePiece(Square, Piece)`,
`removePiece(Square)`, `clearBoard()`, `setSideToMove(Color)`,
`setCastlingRight(Color, Side, bool)`, `setEnPassantSquare(std::optional<Square>)`,
`setHalfmoveClock(int)`, `setFullmoveNumber(int)`, `build() -> Position`.
Keep this *separate* from the normal `Position` API — the play loop
should never see these mutators.
- Files: `src/chess/PositionEditor.{h,cpp}`
- Done when: a test builds startpos via the editor and asserts equality
  to `Fen::parse("rnbqkbnr/...")`.

[x] **Step 35b2.** [chess] Add `validateForEditor(const Position&) ->
EditorValidation` returning a small struct with: `bool legal`,
`std::vector<std::string> issues`. Checks: exactly one king per side, no
pawns on rank 1 or 8, side-not-to-move is not currently in check,
castling rights only set when king + corresponding rook are on their
home squares, en-passant square (if any) is on the correct rank for the
side to move and the supposedly-just-moved pawn exists.
- Reference: equivalent checks in `chess-rtk/src/chess/core/Setup.java`.
- Files: `src/chess/PositionEditor.{h,cpp}` (extend), `tests/EditorValidationTest.cpp`
- Done when: tests cover each rejection reason at least once, plus a
  legal hand-built position that passes.

[x] **Step 35b3.** [ui] Implement `viz/EditorMode.{h,cpp}`: an editor-state
class owned by `App` and toggled with a "Setup" button in `Controls`. While
active it intercepts `BoardView` clicks instead of the play-mode
click-to-move handler. Holds the in-progress `PositionEditor` and the
currently-selected palette piece (or eraser).

[x] **Step 35b4.** [ui] Implement `viz/EditorPalette.{h,cpp}` rendered next
to the board in editor mode: 12 piece icons (6 white + 6 black), an
eraser tool, a "clear board" button, and a "reset to startpos" button.
Selecting a piece sets it as the active brush; left-click on the board
places it, right-click always erases (regardless of brush).

[x] **Step 35b5.** [ui] Implement `viz/EditorPanel.{h,cpp}`: a side panel
shown only in editor mode with controls for:
- Side to move (radio: White / Black).
- Castling rights (4 checkboxes).
- En-passant target square (cycle button: None + 16 valid squares — the 8
  squares on rank 3 and 8 on rank 6).
- Halfmove clock (-/+ buttons around an integer ≥ 0).
- Fullmove number (-/+ buttons around an integer ≥ 1).
- A live FEN string display (read-only, regenerated each frame from the
  editor state).
- Buttons: "Validate", "Apply", "Cancel".

[x] **Step 35b6.** [ui] Wire the Validate / Apply / Cancel actions:
- Validate: runs `validateForEditor`, displays the issues list under the
  panel; disables Apply if not legal.
- Apply: commits the built `Position` into the live `Game`, clears the
  move history (it's a fresh setup), exits editor mode.
- Cancel: discards edits, exits editor mode without touching the `Game`.
- A separate "Edit FEN" control opens the FEN dialog and routes the
  parsed position into the editor instead of the live game.

[x] **Step 35b7.** [ui][test] Manual scripted test recorded in
`docs/test-cases.md`: enter editor, clear board, place K+Q vs K, set
white to move, validate (should pass), apply, observe game continues
from that position; then enter editor again, place two white kings,
validate (should fail with clear error).

---

## Phase 3 — Game model + play loop (Tutorial III deliverable)

Goal of this phase: a complete human-vs-human chess game. Due May 27.

[x] **Step 36.** [game] Implement `game/MoveHistory.{h,cpp}` as a **doubly
linked list** of `MoveRecord` nodes (`Move`, FEN-before, FEN-after, SAN
string, UTC timestamp). Methods: `pushMove`, `undo`, `redo`, `truncateAt`,
`begin/end` iterators. This is the ticked-box for the "Linked Lists"
optional grading item — keep it as a real hand-rolled linked list, not
`std::list`.
- Files: `src/game/MoveHistory.{h,cpp}`
- Done when: a test records 10 moves, undoes 5, redoes 3, prunes a branch.

[x] **Step 37.** [game] Implement `game/Game.{h,cpp}` owning a `Position` and a
`MoveHistory`. Methods: `tryMove(Move)` (validates, applies, records),
`undo()`, `redo()`, `reset()`, `loadFen(string)`,
`status() -> GameStatus { Ongoing, WhiteWins, BlackWins, Draw_*}` driven
by `Position` helpers from Step 21 + repetition + 50-move. Wired into
`App` so play-loop, undo/redo, and status banner all flow through `Game`.

[x] **Step 38.** [game] Implement `game/Pgn.{h,cpp}`: export the move history
of a `Game` to PGN string and to file. Skip parsing PGN input — only
needed for Tutorial III is export.
- Done when: exported PGN re-imported in any chess GUI plays back the same
  moves.

[x] **Step 39.** [ui] Add a move-list panel on the right of the board showing
SAN moves in two columns (white/black). Clicking a row jumps the game to
that ply.

[x] **Step 40.** [ui] Add a status banner above the board: "White to move",
"Black to move", "Checkmate — White wins", "Stalemate — Draw", etc. Pulls
from `Game::status()`.

[x] **Step 41.** [ui] Add a clock area (optional for Tutorial III): two
countdown timers per side, configurable initial time + increment. If clock
hits zero, status flips to time-loss. Disable by default in `config.ini`.

[x] **Step 42.** [ui] Wire keyboard shortcuts: arrow keys for navigation
through history, `R` reset, `F` flip, `S` save FEN, `Ctrl+Z`/`Ctrl+Y`
undo/redo, `Esc` deselect.

[x] **Step 43.** [game][test] Add `tests/GameLoopTest.cpp`: scripted sequences
that play out fool's mate, scholar's mate, stalemate, threefold draw,
50-move draw. Each asserts the correct `GameStatus`.

[x] **Step 44.** [docs] Update `docs/user-manual.md` Section "Playing a game"
with the tutorial GIF/video reference, all keyboard shortcuts, and known input
restrictions (mouse only for moves; no PGN import).

**Step 45.** [milestone] **Tutorial III demo cut.** Tag the repo
`tutorial-iii` and record a 2-minute screen recording of a full game from
start to checkmate. This is the May 27 deliverable.

---

## Phase 4 — NN scaffolding (architecture-agnostic)

[x] **Step 46.** [nn] Implement `src/nn/Tensor.h` as a header-only template:
`template<typename T, size_t Rank> class Tensor` storing
`std::vector<T>` data + `std::array<size_t, Rank> shape`. Methods: `at(i,
j, k, …)`, `reshape`, `data()`, `size()`, `fill`. Template specialisations
not required — keep one generic class.
- Files: `src/nn/Tensor.h`
- Done when: a test creates a `Tensor<float, 3>` of shape (2,3,4), fills
  it, reads back.

[x] **Step 47.** [nn] Implement `src/nn/ops/MatMul.{h,cpp}`:
`matmul(const float* A, const float* B, float* C, M, K, N)` — naive triple
loop, row-major. No SIMD yet.
- Done when: a 4×3 × 3×5 test matches a hand-computed result.

[x] **Step 48.** [nn] Implement `src/nn/ops/Conv2d.{h,cpp}`:
`conv2d(input[B,C,H,W], weight[Co,Ci,Kh,Kw], bias[Co], output, stride,
pad)`. Implementation: im2col + matmul or direct nested loops — direct is
fine for visualizer perf.
- Done when: a 1×1 conv with identity weight passes the input through; a
  3×3 conv matches a hand-computed sample on a small input.

[x] **Step 49.** [nn] Implement `src/nn/ops/Activations.{h,cpp}`: `relu`,
`clipped_relu(int8 / int16 variants for NNUE)`, `sigmoid`, `tanh`, `gelu`
(approx for BT4), `softmax(float*, n)`.

[x] **Step 50.** [nn] Implement `src/nn/ops/LayerNorm.{h,cpp}` and
`src/nn/ops/BatchNorm.{h,cpp}` (BatchNorm in inference mode = affine using
folded mean/var/scale/bias).

[x] **Step 51.** [nn] Implement `src/nn/ops/Attention.{h,cpp}`: scaled dot-
product multi-head attention. Inputs Q/K/V each `[Tokens, Heads, HeadDim]`.
Output `[Tokens, Heads, HeadDim]`. Returns the attention weight matrix
`[Heads, Tokens, Tokens]` separately so the BT4 view can render heads.

[x] **Step 52.** [nn] Implement `src/nn/ActivationSnapshot.{h,cpp}`: a
named-slot container. Each slot holds a `Tensor<float, N>` keyed by a
string label (e.g. "block3.relu", "head7.attention"). Networks write into
it during `forward`; views read from it.
- Files: `src/nn/ActivationSnapshot.{h,cpp}`

[x] **Step 53.** [nn] Implement `src/nn/INetwork.h` abstract base:
```cpp
class INetwork {
public:
    virtual ~INetwork() = default;
    virtual void load(const std::string& path) = 0;
    virtual void evaluate(const Position& pos, ActivationSnapshot& out) const = 0;
    virtual std::string name() const = 0;
};
```
- This is the polymorphism backbone: `App` holds
  `std::vector<std::unique_ptr<INetwork>>`, iterates through them.

[x] **Step 54.** [nn] Implement `src/io/WeightFileReader.{h,cpp}`: a small
helper that reads little-endian `int8`, `int16`, `int32`, `float32` blocks
from a binary file and validates a magic header + version. Used by all
three loaders.

[x] **Step 55.** [nn][test] Add `tests/TensorOpsTest.cpp` covering matmul,
conv2d, layernorm, softmax, attention against hand-computed expected
values on tiny inputs.

---

## Phase 5 — NNUE (half-KP)

Reference: `/home/lennart/Code/chess-rtk/src/chess/nn/nnue/`. Stockfish big
NNUE source format, converted locally to the project `.bin` convention.

[x] **Step 56.** [nn] Implement `src/nn/nnue/FeatureEncoder.{h,cpp}`: given a
`Position`, produce the active half-KP feature indices for white and black
perspectives. Each feature index = `64 * (10*king_sq + piece_index) +
piece_sq`.
- Reference: `chess-rtk/.../nnue/FeatureEncoder.java`
- Done when: a startpos test produces the expected 30 active features per
  side, validated by native C++ tests and static reference fixtures where
  needed.

[x] **Step 57.** [nn] Implement `src/nn/nnue/Accumulator.{h,cpp}`: an
`int16_t[256]` (per side) holding the current sum of active feature
weights. Methods: `refresh(Position)`, `update(Move, MoveDelta)`. For the
visualizer the simple `refresh` path is enough — incremental updates are
optional perf.

[x] **Step 58.** [nn] Implement `src/nn/nnue/NnueLoader.{h,cpp}`: parses
Stockfish big NNUE binary header (magic + version + 32-byte description),
feature transformer weights/biases, then four affine layers
(8x16x32x1) with clipped-ReLU between. Reference numerical layout in
crtk's `Network.java`.
- Done when: loading `models/nnue-halfkp-demo.bin` produces correct sizes.

[x] **Step 59.** [nn] Implement `src/nn/nnue/NnueNetwork.{h,cpp}` implementing
`INetwork`. `evaluate`: refresh accumulator, run feature transformer,
apply quantised affine layers, output centipawn score. Capture into the
snapshot:
- `nnue.accumulator.white` (256 floats from int16)
- `nnue.accumulator.black`
- `nnue.fc1.relu` (32 floats)
- `nnue.fc2.relu` (32 floats)
- `nnue.value` (1 float, centipawns)
- `nnue.feature_active.<color>` — bitboard-like representation of which
  half-KP features fire (for piece-highlight overlay)

[x] **Step 60.** [nn][test] Numerical match test: dump
20 CRTK static NNUE reference predictions, load the same FEN in the native
C++ NNUE, compare value within ±2 cp tolerance. Store the reference dump under
`tests/data/nnue_ref.jsonl`; the test itself has no CRTK subprocess/runtime
dependency.

**Step 61.** [nn][test] Feature-encoder match test: for the same 20 FENs,
log active feature indices and compare to a Java-side dump
(small one-off `crtk` command added if needed).

[x] **Step 62.** [bootstrap] Generate/copy NNUE weights into
`cpp-nn-visualizer/models/nnue-halfkp-demo.bin`. Update `.gitignore` to
exclude weight blobs but keep `models/README.md` describing what to download
and how to convert.

**Step 63.** [docs] Document the NNUE class layout in
`docs/design-spec.md`: `FeatureEncoder`, `Accumulator`, `NnueLoader`,
`NnueNetwork`. Include a Mermaid sequence diagram of `evaluate()`.

**Step 64.** [nn] Optional: implement the incremental accumulator path
(update on `make`/`unmake` instead of `refresh`). Skip if Phase 5
testing already confirms correctness — visualizer perf is fine with full
refresh on each move.

**Step 65.** [milestone] **NNUE end-to-end works.** App can load NNUE,
evaluate any position, and the snapshot is populated. UI integration is
Phase 8.

---

## Phase 6 — LC0 CNN ResNet

Reference: `/home/lennart/Code/chess-rtk/src/chess/nn/lc0/cnn/`. CRTK
flat `.bin` format (well-defined, parseable in C++).

[x] **Step 66.** [nn] Implement `src/nn/lc0_cnn/Lc0CnnEncoder.{h,cpp}`: build
the 112-plane input tensor for a `Position` — current position planes,
last-7-positions planes (zeroed for visualizer mode), castling rights,
side-to-move, fifty-move counter, fill planes. Mirror crtk's
`Encoder.java` byte-for-byte.

[x] **Step 67.** [nn] Implement `src/nn/lc0_cnn/Lc0CnnLoader.{h,cpp}` reading
crtk's flat `.bin` format. Cross-check the byte layout against
`chess-rtk/.../lc0/cnn/Network.java` (search for "BinLoader" or similar).
Parse: magic, header, conv-stem weights, N residual blocks (each: conv1,
bn1, conv2, bn2), policy head conv, policy fc, value head conv, value fc.

[x] **Step 68.** [nn] Implement `src/nn/lc0_cnn/Lc0CnnNetwork.{h,cpp}`:
implements `INetwork`. `evaluate`: encode → conv-stem → N residual blocks
→ policy head + value head. Snapshot writes:
- `cnn.input.planes` (only the live ones)
- `cnn.stem.relu`
- `cnn.block<i>.relu` for each block
- `cnn.policy.logits` (1858 or 4672 depending on policy head)
- `cnn.value.wdl` (3 floats)

**Step 69.** [nn][test] Numerical match test: 10 FENs through `crtk
engine static --fen "<FEN>" --network lc0-cnn --policy --value --json`,
match policy top-5 indices and value within tolerance.

**Step 70.** [nn] Use the smaller
`lc0-cnn-112p-10x128-policy4672-wdl3.bin` network as the default (faster,
smaller — good for live demo). Keep the 30-block weights as an opt-in
alternate.

**Step 71.** [nn] Performance pass: profile `evaluate()` on a single
position. If > 200 ms with the 10-block net, optimise the inner conv loop
(loop reordering, blocking). Stop optimising once you can hit 5 fps in the
viewer.

**Step 72.** [bootstrap] Copy/import the 10-block LC0 CNN as
`cpp-nn-visualizer/models/lc0-cnn-112p-10x128-policy4672-wdl3.bin`.

[x] **Step 73.** [nn] Add a per-block activation hook: while running residual
blocks, write the post-ReLU feature maps into the snapshot. This is what
`CnnView` will render as a grid of small heatmaps.

**Step 74.** [docs] Document CNN classes in `docs/design-spec.md`
including a high-level diagram of the residual block.

**Step 75.** [milestone] **CNN end-to-end works.** App can switch
architecture to CNN and snapshot fills with per-block activations.

---

## Phase 7 — LC0 BT4 transformer

Reference: `/home/lennart/Code/chess-rtk/src/chess/nn/lc0/bt4/` and
`ARCHITECTURE.md` in that dir. Highest-risk phase.

> **Status (2026-05-08):** a native visual BT4 backend now feeds the app:
> it builds 64 board tokens from the 112-plane encoder, runs a lightweight
> deterministic 15-block token transformer, and snapshots attention, FFN,
> policy, and WDL tensors for inspection. Exact LC0 BT4 trained-weight loading
> remains gated on a future native `.bin` import path and numerical parity
> tests.

**Step 76.** [nn] **One-time weight conversion.** BT4 runtime consumes a
flat binary `lc0-bt4-1024x15x32h-visual.bin` with a documented header. Do not add
Java tooling or protobuf/gzip runtime dependencies to this project. If a
converter is needed in this repo, implement it as a native C++ development
tool; otherwise copy in a preconverted `.bin`. Place the converter/import
command in `models/README.md`. Document the binary layout (header magic, embed
dim, num blocks, num heads, then per-block: q/k/v proj, out proj, ffn
weights/biases, layernorm gamma/beta).
- Done when: `models/lc0-bt4-1024x15x32h-visual.bin` exists in the cpp project.

**Step 77.** [nn] Implement `src/nn/lc0_bt4/Bt4Encoder.{h,cpp}`: build the
64-token input embedding from a `Position` matching crtk's
`InputFormat.java`. Each square = one token; piece type + side encoded.

**Step 78.** [nn] Implement `src/nn/lc0_bt4/Bt4Loader.{h,cpp}` reading the
flat `.bin` from Step 76.

**Step 79.** [nn] Implement `src/nn/lc0_bt4/Bt4EncoderBlock.{h,cpp}`:
single transformer encoder block (LN → MHA → residual → LN → FFN →
residual). Use `ops::Attention`, `ops::LayerNorm`, `ops::MatMul`, GELU.

**Step 80.** [nn] Implement `src/nn/lc0_bt4/Bt4Network.{h,cpp}`:
implements `INetwork`. `evaluate`: encode → 15 encoder blocks → policy
head → value head. Snapshot writes:
- `bt4.tokens.embedding` (64 × 1024)
- `bt4.block<i>.attention.heads` (32 × 64 × 64) — attention weight matrix
  per head
- `bt4.block<i>.ffn.relu` (64 × 1024)
- `bt4.policy.logits`
- `bt4.value.wdl`

**Step 81.** [nn][test] Numerical match test: 10 FENs through `crtk` BT4
inference, compare value WDL within tolerance and top-5 policy indices.
Tighter tolerance acceptable for value (1e-3).

**Step 82.** [nn] Performance pass. BT4 is the slowest net (15 × 32-head
attention on 64 tokens). Acceptable target: < 1.5 s per position. If
slower, profile and optimise hot matmuls.

**Step 83.** [nn] Add per-head attention extraction: store the post-
softmax attention weights so `Bt4View` can render heatmap grids.

**Step 84.** [docs] Document BT4 classes including a diagram of one
encoder block.

**Step 85.** [milestone] **All three architectures evaluate any
`Position`.** This is the integration cliff edge for the whole project.

---

## Phase 8 — Activation views

[x] **Step 86.** [ui] Implement `viz/IActivationView.{h,cpp}` abstract:
```cpp
class IActivationView {
public:
    virtual ~IActivationView() = default;
    virtual void update(const ActivationSnapshot& snap) = 0;
    virtual void draw(Rectangle bounds) = 0;
    virtual std::string name() const = 0;
};
```

[x] **Step 87.** [ui] Implement `viz/NnueView.{h,cpp}`. Layout:
- Top: 256-bin bar chart of accumulator activations per side (toggle
  white/black/sum).
- Middle: small heatmaps for each FC layer's post-ReLU activations.
- Bottom: centipawn value with a horizontal bar.
- Overlay on `BoardView`: highlight squares whose half-KP features are
  active, intensity = absolute weight contribution.

**Step 88.** [ui] Implement `viz/CnnView.{h,cpp}`. Layout:
- Per-block grid of 8 representative feature maps (8×8 squares each),
  selectable up to all channels.
- Policy head: top-10 move probabilities listed as SAN with bar lengths.
- Value head: WDL pie / bar.
- Overlay on `BoardView`: per-square activation intensity from a
  user-selected feature map (default: average across channels of the
  final residual block).

**Step 89.** [ui] Implement `viz/Bt4View.{h,cpp}`. Layout:
- Block selector (1..15) + head selector (1..32).
- Main panel: 64×64 attention-weight heatmap with row/col labels = squares.
- Side panel: token embedding magnitudes per square (8×8 heatmap).
- Policy + value displays as in CNN view.
- Overlay on `BoardView`: when a square is hovered, highlight the squares
  it most attends to (selected head).

**Step 90.** [ui] Multi-board mode: `App` lays out three `BoardView`s
side-by-side (or stacked) when "All" is selected. A single `Game` drives
all three; each board renders the same `Position`. Each board has its own
`IActivationView` panel below or beside it. Moves on any board propagate.

[x] **Step 91.** [ui] Architecture switcher in `Controls`: NNUE / CNN / BT4 /
All. In single-architecture mode, the right panel devotes more space to
that architecture's view. (Partial: NNUE↔CNN cycle button + `A` shortcut
implemented; per-arch dedicated panel for CNN still pending — see Step 88.)

**Step 92.** [ui][test] Manual test recorded in `docs/test-cases.md`:
load 5 FENs (startpos, mating attack, fortress, endgame, tactical
puzzle), record each architecture mode in the tutorial/demo evidence, and
verify activations visibly differ.

---

## Phase 9 — Documentation, tests, polish, demo

**Step 93.** [docs] Finalise `docs/design-spec.md`: complete class
diagram (Mermaid) covering all classes; data-member tables for each;
function-prototype tables; one section per technical difficulty (NNUE
quantised affine, CNN conv perf, BT4 attention) and how it was solved.
Target ~30 grading points.

**Step 94.** [docs] Finalise `docs/user-manual.md`: install, run, all
mouse / keyboard controls, all menu options, FEN format restrictions,
list of supported networks and where to obtain weights, troubleshooting.
Target ~20 grading points.

**Step 95.** [docs] Finalise `docs/test-cases.md`: enumerate every test
in `tests/`, each with input, expected output, actual result. Plus the
manual UI flows from Steps 35 and 92 with tutorial GIF/video evidence.

**Step 96.** [docs] Finalise `docs/summary-report.md`: project
experience, biggest challenges (likely BT4), achievements, what we'd do
differently. Required deliverable.

**Step 97.** [docs] Finalise `docs/ai-usage.md`: which files were
AI-assisted, what we hand-reviewed and rewrote, where AI was wrong, what
we fully wrote by hand. Be honest — the rubric explicitly penalises
unmodified AI output (1–15 points).

**Step 98.** [test] Full automated test sweep: `ctest` green on the
machine that will demo. All numerical-match tests for NNUE/CNN/BT4 pass.
Perft passes. Game-loop tests pass.

**Step 99.** [demo] Record the screen-recording deliverable. ~5 minutes
covering: build & launch, loading a FEN, playing a complete game,
switching architectures, examining each activation view, undo/redo,
saving FEN/PGN. Save as `docs/demo.mp4` (or upload + link in
`docs/test-cases.md`).

**Step 100.** [milestone] **Final submission package.** Verify the
submission contains: source tree, `docs/` folder, `models/README.md` with
download/conversion instructions, build scripts, demo recording, all
five required documents. Tag the repo `final-submission` and zip for
upload.

---

## Stretch goals (only after Step 100)

- S1. Engine-vs-human play mode using a tiny C++ alpha-beta search on
  top of the NNUE eval (reuses your existing C++ NNUE). Becomes a
  bonus demo angle.
- S2. Side-by-side architecture comparison panel: top-5 moves per net
  with disagreement highlighting.
- S3. Activation difference mode: subtract activations between two
  positions to see what changed when a move was played.
- S4. Save activation snapshots to `.bin` for offline analysis (more
  file I/O, cheap win for the "File I/O" rubric item).
- S5. Move-tree branching in `MoveHistory` (true tree, not just linear
  undo/redo) — bigger linked-list data structure showcase.
