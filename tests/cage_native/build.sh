#!/usr/bin/env bash
# Build tests/cage_native (host clang++, -O2) into build/native-cage.
#
#   tests/cage_native/build.sh           then: build/native-cage/cage_native checks | g1 ... | g3 ... | cn
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$ROOT/build/native-cage}"
NINJA="$(command -v ninja || echo "$HOME/.pixi/bin/ninja.exe")"
m() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
cmake -S "$(m "$HERE")" -B "$(m "$OUT")" -G Ninja -DCMAKE_MAKE_PROGRAM="$(m "$NINJA")" \
	-DCMAKE_C_COMPILER="${CC:-clang}" -DCMAKE_CXX_COMPILER="${CXX:-clang++}" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$OUT" --target cage_native
ls -la "$OUT/cage_native"*
