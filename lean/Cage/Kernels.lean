import Cage.SlangCodegen.BindSamples
import Cage.SlangCodegen.BhcCoords
import Cage.SlangCodegen.BindGram
import Cage.SlangCodegen.DenseLu
import Cage.SlangCodegen.DenseLuSolve
import Cage.SlangCodegen.BindBlend
import Cage.SlangCodegen.Normals
import Cage.SlangCodegen.NormalVjp
import Cage.SlangCodegen.Laplacian
import Cage.SlangCodegen.SumSq
import Cage.SlangCodegen.BoneBlend
import Cage.SlangCodegen.Lbs
import Cage.SlangCodegen.LbsT
import Cage.SlangCodegen.BodyNormals
import Cage.SlangCodegen.BodyVnormals
import Cage.SlangCodegen.Contact
import Cage.SlangCodegen.Winding

/-!
# `Cage.Kernels` — the kernel list `lake exe emit_cage` works from

Each entry: the kernel's name (its `.slang`, `_emit.cpp` and `.spv`
basename), the Lean module file under `lean/Cage/SlangCodegen/` that
holds its `-- BEGIN PIN` block, and its shader. Every one is compiled
for both targets (none shares group memory).
-/

namespace Cage.Kernels

open LeanSlang

structure Kernel where
  name : String
  module : String
  shader : SlangShaderModule

def all : List Kernel :=
  [ ⟨"cage_bind_samples",   "BindSamples",   Cage.SlangCodegen.BindSamples.shader⟩
  , ⟨"cage_bhc_coords",     "BhcCoords",     Cage.SlangCodegen.BhcCoords.shader⟩
  , ⟨"cage_bind_gram",      "BindGram",      Cage.SlangCodegen.BindGram.shader⟩
  , ⟨"cage_dense_lu",       "DenseLu",       Cage.SlangCodegen.DenseLu.shader⟩
  , ⟨"cage_dense_lu_solve", "DenseLuSolve",  Cage.SlangCodegen.DenseLuSolve.shader⟩
  , ⟨"cage_bind_blend",     "BindBlend",     Cage.SlangCodegen.BindBlend.shader⟩
  , ⟨"cage_normals",        "Normals",       Cage.SlangCodegen.Normals.shader⟩
  , ⟨"cage_normal_vjp",     "NormalVjp",     Cage.SlangCodegen.NormalVjp.shader⟩
  , ⟨"cage_laplacian",      "Laplacian",     Cage.SlangCodegen.Laplacian.shader⟩
  , ⟨"cage_sumsq",          "SumSq",         Cage.SlangCodegen.SumSq.shader⟩
  , ⟨"cage_bone_blend",     "BoneBlend",     Cage.SlangCodegen.BoneBlend.shader⟩
  , ⟨"cage_lbs",            "Lbs",           Cage.SlangCodegen.Lbs.shader⟩
  , ⟨"cage_lbs_t",          "LbsT",          Cage.SlangCodegen.LbsT.shader⟩
  , ⟨"cage_body_normals",   "BodyNormals",   Cage.SlangCodegen.BodyNormals.shader⟩
  , ⟨"cage_body_vnormals",  "BodyVnormals",  Cage.SlangCodegen.BodyVnormals.shader⟩
  , ⟨"cage_contact",        "Contact",       Cage.SlangCodegen.Contact.shader⟩
  , ⟨"cage_winding",        "Winding",       Cage.SlangCodegen.Winding.shader⟩
  ]

end Cage.Kernels
