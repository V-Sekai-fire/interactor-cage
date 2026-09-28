#!/usr/bin/env bash
# cage.elf's kernels (RFD 2277 Phase A), from Lean to both targets, with no
# Python anywhere (RFD 2239): the other stages' gen.sh call embed_spv.py,
# gen_avbd_kernel_table.py and tools/inline_prelude.py; here the Lean
# emitter writes the Slang, the pins and the binding table, CMake embeds the
# SPIR-V, and the shell inlines the slangc prelude.
#
#   kernels/cage/gen.sh               # emit from Lean (and check the pins), then cpp + spirv + table + embed
#   kernels/cage/gen.sh --pin         # the same, rewriting stale pins in lean/Cage/SlangCodegen first
#   kernels/cage/gen.sh --no-emit     # use the committed slang/ (no lake): cpp + spirv + embed
#
#   Lean (lean/, `lake exe emit_cage slang`)   ->  slang/<k>.slang                   (committed)
#     `lake exe emit_cage table`               ->  CageKernelTable.inc               (committed)
#     slangc -target cpp                       ->  cpp/<k>_emit.cpp                  (committed)
#     slangc -target spirv -fp-mode precise    ->  <build>/spv-cage/<k>.spv          (build artefact)
#       spirv-val --target-env vulkan1.2 (when on PATH)
#       embed_spv.cmake                        ->  <build>/cage_kernels.inc          (build artefact)
#
# Every kernel is compiled for both targets (none shares group memory); the
# bind kernels are double precision (SPIR-V Float64), the fit's float.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEAN="${CLOTH_LEAN:-$ROOT/lean}"
BUILD="${BUILD_DIR:-$ROOT/build}"
SPV="$BUILD/spv-cage"
SLANGC="${SLANGC:-slangc}"
command -v "$SLANGC" >/dev/null 2>&1 || SLANGC="$HOME/scoop/apps/vulkan/current/Bin/slangc"
CMAKE="${CMAKE:-cmake}"

MODE=emit
case "${1:-}" in
	--no-emit) MODE=none ;;
	--pin) MODE=pin ;;
	"") ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
esac

KERNELS=$(grep -v '^#' "$HERE/kernels.txt" | awk 'NF {print $1}' | tr '\n' ' ')

if [ "$MODE" != none ]; then
	command -v lake >/dev/null 2>&1 || { echo "error: lake not on PATH (or pass --no-emit)" >&2; exit 1; }
	if [ "$MODE" = pin ]; then
		# The emitter imports the modules, pins included: empty every pin block
		# first so a stale one cannot stop it from building.
		for f in "$LEAN"/Cage/SlangCodegen/*.lean; do
			grep -q "^-- BEGIN PIN$" "$f" || continue
			awk '/^-- BEGIN PIN$/ { print; skip = 1; next } /^-- END PIN$/ { skip = 0 } !skip' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
		done
	fi
	( cd "$LEAN" && lake build emit_cage >/dev/null )
	if [ "$MODE" = pin ]; then
		( cd "$LEAN" && lake exe emit_cage pin "$LEAN/Cage/SlangCodegen" )
		( cd "$LEAN" && lake build Cage >/dev/null )
	fi
	( cd "$LEAN" && lake exe emit_cage check-pins "$LEAN/Cage/SlangCodegen" )
	FROM="$(mktemp -d)"
	echo "== emitting Slang from Lean at $LEAN =="
	( cd "$LEAN" && lake exe emit_cage slang "$FROM" >/dev/null )
	# The kernel list must be the Lean list, in the same order.
	LISTED=$(cd "$LEAN" && lake exe emit_cage list | tr '\n' ' ')
	if [ "$LISTED" != "$KERNELS" ]; then
		echo "error: kernels.txt ($KERNELS) differs from Cage.Kernels.all ($LISTED)" >&2
		exit 1
	fi
	mkdir -p "$HERE/slang"
	for k in $KERNELS; do cp "$FROM/$k.slang" "$HERE/slang/$k.slang"; done
	rm -rf "$FROM"
	( cd "$LEAN" && lake exe emit_cage table "$HERE/CageKernelTable.inc" >/dev/null )
	echo "== $(echo $KERNELS | wc -w) kernels into slang/, CageKernelTable.inc =="
fi

# A Linux slangc writes its prelude as an absolute #include; the committed
# emits carry it inline (tools/inline_prelude.py's form). The inline block is
# taken from the first emit under kernels/*/cpp that carries it (as
# tools/inline_prelude.py does): its lines up to the one before
# `#ifdef SLANG_PRELUDE_NAMESPACE` + `using namespace`. It is copied first,
# since this run may rewrite that very file.
REF=""
for f in "$ROOT"/kernels/*/cpp/*_emit.cpp; do
	if head -1 "$f" | grep -q '^#ifndef SLANG_CPP_PRELUDE_H'; then REF="$f"; break; fi
done
[ -n "$REF" ] || { echo "error: no emit under kernels/*/cpp carries the inline prelude" >&2; exit 1; }
PRE="$(mktemp)"
trap 'rm -f "$PRE"' EXIT
PRELUDE_END=$(awk '/^#ifdef SLANG_PRELUDE_NAMESPACE$/ { n = NR } n && NR == n + 1 && /^using namespace/ { print n - 2; exit }' "$REF")
[ -n "$PRELUDE_END" ] || { echo "error: no inline prelude in $REF" >&2; exit 1; }
head -n "$PRELUDE_END" "$REF" > "$PRE"

mkdir -p "$HERE/cpp" "$SPV"
echo "== slangc -target cpp ($(echo $KERNELS | wc -w)) =="
for k in $KERNELS; do
	# Relative paths: slangc writes the input path into #line directives.
	( cd "$HERE" && "$SLANGC" -target cpp -stage compute -entry main -o "cpp/${k}_emit.cpp" "slang/$k.slang" 2>&1 |
		grep -v "has been renamed to 'main_0'" || true )
	f="$HERE/cpp/${k}_emit.cpp"
	if head -1 "$f" | grep -q '^#include ".*slang-cpp-prelude.h"$'; then
		{ cat "$PRE"; tail -n +2 "$f"; } > "$f.tmp" && mv "$f.tmp" "$f"
	elif ! head -1 "$f" | grep -q '^#ifndef SLANG_CPP_PRELUDE_H'; then
		echo "error: unexpected first line in $f" >&2; exit 1
	fi
done

# -fp-mode precise: no FMA contraction on the GPU, as the guest builds the cpp
# emits -ffp-contract=off (the drape kernels' reason, gates/5-drape/lbfgsb).
echo "== slangc -target spirv ($(echo $KERNELS | wc -w)) =="
rm -f "$SPV"/*.spv
VAL=$(command -v spirv-val || true)
for k in $KERNELS; do
	"$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main -fp-mode precise \
		-o "$SPV/$k.spv" "$HERE/slang/$k.slang"
	if [ -n "$VAL" ]; then "$VAL" --target-env vulkan1.2 "$SPV/$k.spv"; fi
done
[ -n "$VAL" ] && echo "spirv-val: $(echo $KERNELS | wc -w) kernels valid (vulkan1.2)"
echo "== embedding SPIR-V =="
"$CMAKE" -DSPV_DIR="$SPV" -DOUT="$BUILD/cage_kernels.inc" -DNAMESPACE=cage_kernels -P "$HERE/embed_spv.cmake"
