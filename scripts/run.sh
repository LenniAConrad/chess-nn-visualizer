#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ./build/cnnv ]]; then
    ./scripts/build.sh
else
    cmake --build build -j
fi

exec ./build/cnnv "$@"
