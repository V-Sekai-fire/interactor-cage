#!/bin/sh
# Build and run the cage-fit oracle (Gate 9, G3): host-native, clang++ -O2 -std=c++17.
# Writes gates/9-cage/oracle/{cage.obj,body.obj,phi.txt,psi.txt,problem.txt,solution.txt,gen.log}.
#
#   tests/cage_oracle/build.sh               clone deps if missing, build, generate
#   OUT=<dir> tests/cage_oracle/build.sh     generate somewhere else (e.g. to diff)
#
# LBFGSpp 0.3.0 and Eigen 3.4.90 come from V-Sekai-fire/interactor-aria-lbfgspp @ 10086b6b,
# the pin tests/lbfgsb_oracle uses; BHC.h and point3.h from
# V-Sekai-fire/interactor-tool-godot-cage-deformer @ f0b27162 (src/). All three are cloned
# into .deps/ (gitignored) and reach no guest ELF. CXX overrides the compiler.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEPS="$HERE/.deps"
ARIA="$DEPS/interactor-aria-lbfgspp"
CAGE="$DEPS/interactor-tool-godot-cage-deformer"
LBFGSPP_REV=10086b6b2022802d7942ba16842c43063b3cff91
CAGE_REV=f0b271620a4db16390101cc8ac2375c16680a9ab
mkdir -p "$DEPS"
if [ ! -d "$ARIA/.git" ]; then
  git clone -q --no-checkout https://github.com/V-Sekai-fire/interactor-aria-lbfgspp.git "$ARIA"
fi
git -C "$ARIA" config core.longpaths true
git -C "$ARIA" checkout -q -f "$LBFGSPP_REV"
if [ ! -d "$CAGE/.git" ]; then
  git clone -q --no-checkout https://github.com/V-Sekai-fire/interactor-tool-godot-cage-deformer.git "$CAGE"
fi
git -C "$CAGE" checkout -q -f "$CAGE_REV" -- src/BHC.h src/point3.h
EIGEN=${EIGEN:-$ARIA/thirdparty/eigen}
CXX=${CXX:-clang++}
"$CXX" -O2 -std=c++17 -I"$ARIA/thirdparty/LBFGSpp/include" -I"$EIGEN" -I"$CAGE/src" \
  "$HERE/gen.cpp" -o "$DEPS/gen"
OUT=${OUT:-$ROOT/gates/9-cage/oracle}
mkdir -p "$OUT"
"$DEPS/gen" "$OUT" | tee "$OUT/gen.log"
