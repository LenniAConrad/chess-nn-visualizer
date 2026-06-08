# User Manual

## 1. Install And Run

Install a C++17 compiler, CMake 3.20 or newer, and the system libraries needed
by raylib. On Debian/Ubuntu:

```sh
sudo apt install build-essential cmake libgl1-mesa-dev libx11-dev \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

Build and launch from the repository root:

```sh
./scripts/build.sh
./scripts/run.sh
```

The checked-in configuration starts windowed at 1280x800 with the CNN
activation panel selected. Edit `config.ini` and restart to change startup
size, fullscreen mode, default FEN, board palette, model paths, startup
architecture, MCTS settings, fonts, or clock settings.

Before a model demo, make sure the runtime model files exist:

```sh
./scripts/import_models.sh
```

On Windows, use the packaged build in `dist/cpp-nn-visualizer-windows-x64/` and
run either `run-cnnv.bat` or `run-cnnv.ps1`.

## 2. Basic Chess Play

Move input is mouse based. Click a piece and then click a legal destination, or
drag a piece to a legal destination. The board highlights the selected piece,
legal targets, and the last move. Promotion opens a picker; choose the promoted
piece with the mouse or press `Esc` to cancel the pending selection.

The status area reports the game state. The move list records SAN moves and can
be clicked to jump to a previous ply. Undo/redo keeps the linked-list move
history intact, so rewinding and replaying does not lose future moves unless a
new move is made from the rewound position.

## 3. Command Buttons

| Button | Action |
| --- | --- |
| Reset | Restores the configured start position and clears transient UI state. |
| Flip | Reverses board orientation without changing the position. |
| Undo | Steps one ply backward through move history. |
| Redo | Steps one ply forward after an undo. |
| Random | Plays one random legal move from the current position. |
| Search | Starts or stops the live search preview; the button uses its active color while search is running. |
| Load FEN | Opens a text dialog for loading a complete six-field FEN. |
| Save FEN | Writes the current position to `position.fen`. |
| Setup | Opens the board editor. |
| Edit FEN | Opens the current position as editable FEN text and then enters setup mode. |

## 4. FEN Loading And Saving

Click Load FEN or press `L` to open the FEN dialog. Enter a complete six-field
FEN string and press Enter. Invalid input stays in the dialog and shows an
inline error.

Click Save FEN or press `S` to write the current position to `position.fen`.
There is no file picker yet.

## 5. Setup Editor

Click Setup or press `T` to enter editor mode. Select a piece from the palette,
left-click a square to place it, or right-click a square to erase it. The
palette also includes Eraser, Clear, and Startpos controls.

The editor panel controls side to move, castling rights, en-passant target,
halfmove clock, and fullmove number. Click Validate before Apply. Apply is
enabled only when the edited position is legal. Invalid positions stay in the
editor and show the first validation issues.

## 6. Activation Views

The right-side architecture selector has NNUE, CNN, BT4, Classical, Off, and a
Tree tab. Press `A` to cycle NNUE -> CNN -> BT4 -> Classical -> Off -> NNUE.
Changing the architecture does not change the chess position.

The activation panel can be zoomed and panned:

| Input | Action |
| --- | --- |
| Mouse wheel over panel | Zoom toward the cursor. |
| Right or middle drag | Pan the panel. |
| `+` / `-` | Zoom about the panel center. |
| `0` | Reset panel camera. |

### NNUE

NNUE shows HalfKP-style active features, raw accumulators, clipped activations,
output weights, centipawn value, and learned-weight atlases. Its five internal
modes are Overview, Trace, All, Atlas, and Diagram.

### CNN

CNN uses the compact `LC0J` runtime file. It encodes the board into 112 planes,
runs the residual trunk, policy head, and WDL/value head, and publishes
heatmaps and tensor statistics. Its modes are Overview, Trace, All, Atlas, and
Diagram.

### BT4

BT4 loads the compact `BT4J` v2 runtime file when the view is first selected.
The default file is a real native transformer with 64 square tokens,
96-dimensional embeddings, 4 encoder blocks, 4 attention heads, policy logits,
and a WDL value head. The view derives block and head counts from the loaded
tensors, so it is not tied to one hardcoded transformer size.

BT4 modes are Overview, Trace, All, Atlas, and Diagram. The attention-board
views show the current pieces on the board so attention movement can be read in
chess terms rather than only as matrix cells.

### Classical

Classical is model-free. It evaluates the current position with a handcrafted
centipawn evaluator and shows the term breakdown, WDL triplet, phase, and
piece-square-table heatmaps. The detailed mode shows the heatmaps by piece
type.

### Off

Off hides the activation panel and leaves the chess board playable.

## 7. Search Preview

Press Search or `E` to run a live engine-style preview on the current board.
Search stops automatically if the position changes. `Esc` also stops the
search. The selected architecture determines the evaluation route when
available:

- CNN uses its real WDL head and policy logits for value/priors.
- BT4 uses its WDL head and fallback priors.
- NNUE uses its centipawn value and fallback priors.
- Classical and Off use the handcrafted evaluator.

## 8. MCTS Tree Workbench

Press `G` or click Tree to open the full-window search-tree workbench. Click
Start to run PUCT search from the current position. Click Stop to stop it, Fit
to frame the tree, and Board or `Esc`/`G` to return to the main board.

Tree controls:

| Control | Action |
| --- | --- |
| visits +/- | Change the playout budget. |
| cpuct +/- | Change the PUCT exploration constant. |
| Follow | Trace the currently explored leaf into the activation views. |
| branches +/- | Limit children shown per node; 0 shows all. |
| depth +/- | Limit displayed tree depth. |
| Batch | Collapse sibling leaf groups into summary blobs. |
| Guides | Toggle per-ply guide lines. |
| Merge | Merge repeated position signatures for transposition display. |

Tree navigation:

| Input | Action |
| --- | --- |
| Mouse wheel | Zoom. |
| Left drag | Pan. |
| Click node | Select that node and trace its FEN into the network views. |
| Space | Play/pause recorded growth frames. |
| Left / Right | Step the growth scrubber. |
| Home / End | Jump to first growth frame / live tree. |

## 9. Keyboard Shortcuts

| Shortcut | Action |
| --- | --- |
| `F` | Flip board. |
| `R` | Reset to configured start position. |
| `L` | Open Load FEN. |
| `S` | Save current position to `position.fen`. |
| `T` | Enter setup editor. |
| `A` | Cycle activation architecture. |
| `E` | Start/stop search preview. |
| `G` | Enter/leave tree workbench. |
| `F11` | Toggle borderless fullscreen. |
| `Ctrl+Z` | Undo one ply. |
| `Ctrl+Y` | Redo one ply. |
| `Left` / `Right` | Step backward / forward one ply in board mode; scrub tree frames in tree mode. |
| `Home` / `End` | Jump to start/end of move history in board mode; first/live frame in tree mode. |
| `Esc` | Deselect, close modal, stop search, leave editor/tree, or cancel transient input. |

## 10. Configuration

Important `config.ini` keys:

```ini
window.width = 1280
window.height = 800
window.fullscreen = false
default.fen = rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
assets.pieces = assets/pieces
board.palette = red
startup.arch = cnn
startup.tree = false
mcts.visits = 20000
mcts.cpuct = 2.8
model.nnue = models/nnue-halfkp-demo.bin
model.lc0_cnn = models/lc0-cnn-small-112p-4x32-policy4672-wdl3.bin
model.lc0_bt4 = models/lc0-bt4-tiny-96x4x4h.bin
clock.enabled = false
```

`startup.arch` accepts `nnue`, `cnn`, `bt4`, `classical`, or `off`.
`board.palette` accepts `red`, `classic`, `warm`, `neural`, `analysis`,
`heatmap`, `blue`, `cool`, `green`, or `forest`.

## 11. Restrictions

- Move input is mouse-based; there is no SAN/UCI move-entry box.
- FEN loading expects a complete standard six-field FEN string.
- PGN export exists in the codebase, but PGN import is not implemented.
- There is no engine-opponent play mode yet.
- The bundled CNN and BT4 models are compact untrained visualization models.
  They demonstrate the real native architecture paths, not upstream engine
  strength.
- The app reads local compact binary model formats (`NNUE`, `LC0J`, `BT4J` and
  legacy `BT4V` fallback metadata), not official LC0 protobuf files.
- Save FEN writes to `position.fen`; there is no file picker yet.

## 12. Troubleshooting

- If pieces are missing, check `assets.pieces` and run from the repository root
  or package directory.
- If model panels report missing weights, check `model.*` paths and rerun
  `./scripts/import_models.sh`.
- If text input in the FEN dialog is wrong, switch the active keyboard/input
  method to plain Latin text.
- If the window opens too large or too small, edit `window.width`,
  `window.height`, or `window.fullscreen` in `config.ini`.
