# Test Cases

## 1. Automated tests

### 1.1 Chess core (perft, FEN round-trip, SAN round-trip)

Command:

```sh
./scripts/test.sh
```

Latest automated run:

```text
Date: 2026-05-09
Result: PASS
ctest: 1/1 test passed, 0 failed
Total test time: 3.20 seconds
```

Coverage:

- `PerftTest.cpp` checks legal move generation against known perft counts.
- `FenTest.cpp` and `FenIoTest.cpp` check parsing, formatting, and text-file
  save/load.
- `SanTest.cpp` checks SAN conversion and disambiguation.
- `PositionTest.cpp`, `MoveTest.cpp`, `PieceTest.cpp`, `BitboardTest.cpp`, and
  `SlidingAttacksTest.cpp` check board state, encoding, attacks, and helpers.

### 1.2 Game-loop tests

`GameLoopTest.cpp`, `MoveHistoryTest.cpp`, `MoveListTest.cpp`, and
`EditorValidationTest.cpp` cover legal move application, undo/redo behavior,
linked-list history, move-list storage, and setup-editor validation.

### 1.3 Tensor-op tests

`TensorOpsTest.cpp` covers the reusable tensor and operation layer used by the
network implementations.

### 1.4 NNUE numerical-match tests

`NnueTest.cpp`, `NnueReferenceTest.cpp`, and `tests/data/nnue_ref.jsonl` verify
NNUE feature encoding, accumulator/output behavior, and reference numerical
matches.

### 1.5 LC0 CNN numerical-match tests

`Lc0CnnTest.cpp` checks board-plane encoding, convolutional network loading,
and CNN forward-pass behavior.

### 1.6 LC0 BT4 numerical-match tests

`Bt4Test.cpp` checks the BT4-style token transformer path and snapshot keys.

## 2. Manual UI tests

### 2.1 Single-board play smoke test

Date: 2026-05-08

Environment: Linux/X11, default fullscreen application window.

Kiwipete FEN used:

```text
r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
```

Result: PASS

| Step | Action | Expected result | Evidence |
| --- | --- | --- | --- |
| 1 | Launch the app. | Start position appears with White to move. | Covered in [`tutorial.gif`](tutorial.gif) and [`demo.mp4`](demo.mp4). |
| 2 | Play 10 mouse moves: `1. e4 d5 2. exd5 Qxd5 3. Nc3 Qd8 4. Nf3 Nf6 5. Bc4 e6`. | Board, status, and move history update after each legal move. | Covered in the gameplay segment of the tutorial. |
| 3 | Press Undo and Redo. | Board rewinds and replays through the linked-list move history. | Covered in the Undo/Redo segment of the tutorial. |
| 4 | Open Load FEN and type Kiwipete. | Dialog accepts the full FEN text. | Covered in the Load FEN segment of the tutorial. |
| 5 | Press Enter to load Kiwipete. | Kiwipete appears on the board with White to move. | Covered in the Kiwipete segment of the tutorial. |
| 6 | Press Flip. | Board orientation reverses while the position stays unchanged. | Covered in the Flip segment of the tutorial. |
| 7 | Press Save FEN. | `position.fen` is written with the Kiwipete FEN. | Covered in the Save FEN segment of the tutorial. |
| 8 | Press Random. | One legal random move is played and the activation view updates. | Covered in the Random segment of the tutorial. |
| 9 | Press Search. | Native engine-style search starts, shows depth/nodes/evaluation/PV, and can be stopped. | Covered in the Search segment of the tutorial. |

Saved file verification:

```text
r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
```

### 2.2 Board-editor flow

Date: 2026-05-08

Environment: Linux/X11, default fullscreen application window.

Result: PASS

| Step | Action | Expected result | Evidence |
| --- | --- | --- | --- |
| 1 | Enter Setup, then click Clear. | Board is empty and validation reports the missing kings. | Covered in the Setup editor segment of the tutorial. |
| 2 | Place White king on e1, White queen on d1, Black king on h8, with White to move, then Validate. | Editor reports `Position is legal.` and Apply is enabled. | Covered in the Validate/Apply segment of the tutorial. |
| 3 | Click Apply. | App returns to play mode from the edited K+Q vs. K position. | Covered in the Apply segment of the tutorial. |
| 4 | Enter Setup again, place a second White king on a2, then Validate. | Editor rejects the position with a clear two-kings error and disables Apply. | Covered in the invalid-editor-validation segment of the tutorial. |

### 2.3 Multi-architecture demonstration

Date: 2026-05-09

Environment: Linux/X11, default fullscreen application window.

Result: PASS when all three architecture buttons can be selected without
changing the legal chess position.

| Step | Action | Expected result |
| --- | --- | --- |
| 1 | Start the app and play at least one legal move. | Board, move history, and status update. |
| 2 | Launch with `startup.arch=off`, then select NNUE and use abstract mode. | NNUE panel shows the HalfKP feature-to-accumulator-to-value flow. |
| 3 | Switch NNUE to detailed mode. | NNUE panel shows node/layer details, accumulator ranges, clipped activations, and centipawn value. |
| 4 | Select CNN and use abstract mode. | CNN panel shows input planes, residual trunk, policy head, and WDL/value head as a high-level flow. |
| 5 | Switch CNN to detailed mode. | CNN panel shows heatmaps, selected tensor statistics, policy/value outputs, or a clear missing-weights status. |
| 6 | Select BT4 and use abstract mode. | BT4 panel shows the token, embedding, transformer, policy, and value flow. |
| 7 | Switch BT4 to detailed mode. | BT4 panel shows square tokens, attention-style maps, policy logits, and WDL/value summaries. |
| 8 | Press Search with the active architecture selected. | The native search preview uses the selected architecture's evaluation path when possible and shows live depth, nodes, evaluation, and principal variation. |
| 9 | Select OFF. | Activation panel is disabled while the chess board remains playable. |

## 3. Demo screen recording

Animated tutorial: [`docs/tutorial.gif`](tutorial.gif), generated 2026-05-09.
Video source: [`docs/demo.mp4`](demo.mp4), duration 102.97 seconds at
2560x1550 fullscreen. The GIF is 1280x775 at 12 FPS with 1236 frames. It was
regenerated from a fresh live capture of the current app window, not from saved
still-image files.

The demo walkthrough shows:

1. Fullscreen launch in OFF mode, then Reset and Flip.
2. NNUE abstract and detailed views, legal chess play, undo/redo, and move
   history.
3. Random, Load FEN, Save FEN, and native search.
4. Native engine-style search with live depth, node count, evaluation, and PV.
5. CNN abstract and detailed views.
6. BT4 abstract and detailed views.
7. OFF mode with the chess board still playable.
8. Setup editor buttons: Clear, Startpos, palette placement, Eraser, side to
   move, castling rights, en-passant, halfmove/fullmove counters, Validate,
   Apply, and Cancel.
9. Edit FEN flow with validation and applying a custom legal position.
