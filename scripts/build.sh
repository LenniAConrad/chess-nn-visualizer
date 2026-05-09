#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cmake_args=(
    -B build
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
    -DCNNV_ENABLE_CUDA="${CNNV_ENABLE_CUDA:-OFF}"
)

raylib_source="${RAYLIB_SOURCE_DIR:-}"
if [[ -z "$raylib_source" && -n "${HOME:-}" && -f "$HOME/.local/src/raylib/CMakeLists.txt" ]]; then
    raylib_source="$HOME/.local/src/raylib"
fi
if [[ -n "$raylib_source" ]]; then
    if [[ ! -f "$raylib_source/CMakeLists.txt" ]]; then
        echo "RAYLIB_SOURCE_DIR does not contain CMakeLists.txt: $raylib_source" >&2
        exit 1
    fi
    cmake_args+=("-DFETCHCONTENT_SOURCE_DIR_RAYLIB=$raylib_source")
fi

cmake "${cmake_args[@]}" "$@"
cmake --build build -j
