# Models

Weight blobs are local artifacts and are ignored by git. Runtime model files
use a `.bin` suffix so the project has one consistent binary-model convention.
Populate the directory from local `chess-models` / `chess-rtk` checkouts with:

```sh
./scripts/import_models.sh
```

The importer searches these directories first:

```text
$MODEL_SOURCE_DIR
~/Code/chess-models/models
~/Code/chess-rtk/models
```

It downloads only when a known source URL is available and no local file was
found. Set `MODEL_IMPORT_MODE=symlink` to link local files instead of copying
them, or `FORCE=1` to replace existing files.

## Runtime files

These three files are required for the complete visual defense demo:

- `nnue-halfkp-demo.bin` - compact deterministic HalfKP/NNUE float model used
  by the C++ visualizer. This is the file loaded by `model.nnue`.
- `lc0-cnn-112p-10x128-policy4672-wdl3.bin` - small LC0-style CNN ResNet file
  copied from `chess-models` or `chess-rtk`. This is loaded by
  `model.lc0_cnn`.
- `lc0-bt4-1024x15x32h-visual.bin` - BT4 visual-model metadata consumed by
  the native deterministic token-transformer path. This is loaded by
  `model.lc0_bt4` and validates the BT4 dimensions used by this build.

Run `./scripts/demo_check.sh` before the final demo to confirm that the local
ignored model files and tutorial evidence are present on the presentation
machine.

## Optional reference blobs

The importer may also install upstream reference blobs using `.bin` filenames:

- `stockfish-nnue-f68ec79f0fe3-reference.bin` - official Stockfish NNUE
  archive retained only as reference material. It is not loaded by the C++
  runtime because the runtime expects the compact visualizer NNUE layout below.
- `lc0-bt4-1024x15x32h-policytune-reference.bin` - official LCZero BT4
  protobuf/gzip archive retained only as reference material. The runtime uses
  `lc0-bt4-1024x15x32h-visual.bin` instead of parsing protobuf/gzip.

## NNUE visualizer binary layout

Little-endian. All tensors are `float32`.

```text
"NNUE"                          (4 bytes)
uint32  version                 == 1
uint32  featureCount            == 64 * 10 * 64 = 40960
uint32  hiddenSize              H
float32 outputScale
uint32  N0  + N0 * float32      featureBias     (N0 == H)
uint32  N1  + N1 * float32      featureWeights  (N1 == 40960 * H, feature-major)
uint32  N2  + N2 * float32      outputWeights   (N2 == 2 * H; first H = us, second H = them)
float32 outputBias
```

## BT4 visualizer binary layout

Little-endian metadata used to validate the native BT4 visualizer dimensions:

```text
"BT4V"                          (4 bytes)
uint32  version                 == 1
uint32  inputChannels           == 112
uint32  tokens                  == 64
uint32  tokenWidth              == 176
uint32  modelDim                == 64
uint32  blocks                  == 15
uint32  heads                   == 8
uint32  policySize              == 1858
uint32  seed                    reserved deterministic visual seed
```
