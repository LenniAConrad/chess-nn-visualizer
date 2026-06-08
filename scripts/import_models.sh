#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

model_dir="models"
mkdir -p "$model_dir"

copy_mode="${MODEL_IMPORT_MODE:-copy}" # copy or symlink
force="${FORCE:-0}"

sources=()
if [[ -n "${MODEL_SOURCE_DIR:-}" ]]; then
  sources+=("$MODEL_SOURCE_DIR")
fi
sources+=(
  "$HOME/Code/chess-models/models"
  "$HOME/Code/chess-rtk/models"
)

find_local() {
  local name src
  for name in "$@"; do
    for src in "${sources[@]}"; do
      [[ -d "$src" ]] || continue
      if [[ -f "$src/$name" || -L "$src/$name" ]]; then
        printf '%s\n' "$src/$name"
        return 0
      fi
    done
  done
  return 1
}

install_file() {
  local label="$1"
  local dest="$2"
  local url="$3"
  shift 3
  local source_names=("$@")

  if [[ -e "$dest" && "$force" != "1" ]]; then
    printf '[skip] %s already exists: %s\n' "$label" "$dest"
    return 0
  fi

  local local_path=""
  if local_path="$(find_local "${source_names[@]}")"; then
    rm -f "$dest"
    if [[ "$copy_mode" == "symlink" ]]; then
      ln -s "$local_path" "$dest"
      printf '[link] %s <- %s\n' "$dest" "$local_path"
    else
      cp -L "$local_path" "$dest"
      printf '[copy] %s <- %s\n' "$dest" "$local_path"
    fi
    return 0
  fi

  if [[ -n "$url" ]]; then
    printf '[download] %s -> %s\n' "$url" "$dest"
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail --retry 3 --output "$dest" "$url"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$dest" "$url"
    else
      printf '[error] neither curl nor wget is available for %s\n' "$label" >&2
      return 1
    fi
    return 0
  fi

  printf '[warn] missing %s; searched local model dirs\n' "$label" >&2
  return 1
}

generate_demo_nnue() {
  local dest="$1"
  if [[ -e "$dest" && "$force" != "1" ]]; then
    printf '[skip] compatible CRTK NNUE already exists: %s\n' "$dest"
    return 0
  fi

  python3 - "$dest" <<'PY'
import array
import pathlib
import struct
import sys

dest = pathlib.Path(sys.argv[1])
feature_count = 64 * 10 * 64
hidden = 16

feature_bias = array.array("f", [((i % 5) - 2) * 0.01 for i in range(hidden)])
feature_weights = array.array("f")
for feature in range(feature_count):
    square = feature % 64
    plane = (feature // 64) % 10
    king = feature // (10 * 64)
    sign = -1.0 if plane >= 5 else 1.0
    for i in range(hidden):
        lane = 0.035 if i == (plane % hidden) else 0.0
        jitter = (((square * 13 + king * 7 + i * 3) % 17) - 8) * 0.00075
        feature_weights.append(sign * (lane + jitter))

output_weights = array.array("f")
for i in range(hidden):
    output_weights.append(18.0 / (1.0 + i))
for i in range(hidden):
    output_weights.append(-14.0 / (1.0 + i))

def write_array(f, values):
    f.write(struct.pack("<I", len(values)))
    out = array.array("f", values)
    if sys.byteorder != "little":
        out.byteswap()
    out.tofile(f)

with dest.open("wb") as f:
    f.write(b"NNUE")
    f.write(struct.pack("<III", 1, feature_count, hidden))
    f.write(struct.pack("<f", 1.0))
    write_array(f, feature_bias)
    write_array(f, feature_weights)
    write_array(f, output_weights)
    f.write(struct.pack("<f", 0.0))

print(f"[generate] {dest} ({dest.stat().st_size} bytes)")
PY
}

generate_small_bt4() {
  # Generates a compact, real-architecture BT4J transformer that the
  # native BT4 forward pass runs. We deliberately do NOT ship chess-rtk's full
  # 706 MB 1024x15x32h net — it is far too large for a visualizer.
  local dest="$1"
  if [[ -e "$dest" && "$force" != "1" ]]; then
    printf '[skip] small BT4 already exists: %s\n' "$dest"
    return 0
  fi
  python3 scripts/generate_small_bt4.py "$dest"
}

generate_tiny_bt4() {
  local dest="$1"
  if [[ -e "$dest" && "$force" != "1" ]]; then
    printf '[skip] tiny BT4 already exists: %s\n' "$dest"
    return 0
  fi
  BT4_NAME=bt4-tiny-96x4x4h \
  BT4_D=96 \
  BT4_BLOCKS=4 \
  BT4_HEADS=4 \
  BT4_FFN=192 \
  BT4_POLICY_EMB=48 \
  BT4_POLICY_DMODEL=48 \
  BT4_VALUE_EMB=24 \
  BT4_VALUE_HIDDEN=96 \
  BT4_SEED=20260608 \
    python3 scripts/generate_small_bt4.py "$dest"
}

generate_small_cnn() {
  local dest="$1"
  if [[ -e "$dest" && "$force" != "1" ]]; then
    printf '[skip] small CNN already exists: %s\n' "$dest"
    return 0
  fi
  python3 scripts/generate_small_cnn.py "$dest"
}

generate_demo_nnue "$model_dir/nnue-halfkp-demo.bin"
generate_small_cnn "$model_dir/lc0-cnn-small-112p-4x32-policy4672-wdl3.bin"
generate_tiny_bt4 "$model_dir/lc0-bt4-tiny-96x4x4h.bin"

printf '\nImported model files:\n'
find "$model_dir" -maxdepth 1 \( -type f -o -type l \) | sort | while read -r path; do
  [[ "$path" == "$model_dir/.gitkeep" || "$path" == "$model_dir/README.md" ]] && continue
  if [[ -L "$path" ]]; then
    printf '  %s -> %s\n' "$path" "$(readlink "$path")"
  else
    printf '  %s (%s bytes)\n' "$path" "$(stat -c '%s' "$path")"
  fi
done
