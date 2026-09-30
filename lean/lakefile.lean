import Lake
open Lake DSL

package Cage where

-- Every dependency is pinned in a V-Sekai-fire repo or fork.
require LeanSlang from git
  "https://github.com/V-Sekai-fire/contract-lean-slang.git" @ "60532aef8ed70cc669ecab481182d0636c9e1ac3"

-- A sibling checkout in the manifest layout (contract-manifest-taskweft).
require Drape from "../../../2-contract/lbfgsb/lean"

-- cage.elf's kernels (RFD 2277 Phase A): the (1,3) biharmonic bind of
-- Thiery, Michel and Chen (SIGGRAPH 2024) in double, and the in-motion
-- fit's forward and backward kernels. A default target, so a bare
-- `lake build` checks their native_decide pins. `lake exe emit_cage`
-- also writes kernels/cage's binding table (kernels/cage/embed_spv.cmake
-- embeds the SPIR-V; no Python, RFD 2239).
@[default_target] lean_lib Cage

lean_exe emit_cage where
  root := `EmitCage
