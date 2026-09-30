import Cage.SlangCodegen.Common
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
import Cage.Kernels

/-!
# `Cage` — cage.elf's kernels (RFD 2277 Phase A)

**The bind** (`bhc13`, double precision): the (1,3) biharmonic
coordinates of Thiery, Michel and Chen, *Biharmonic Coordinates and their
Derivatives for Triangular 3D Cages* (SIGGRAPH 2024), restated from `BHC.h`
of V-Sekai-fire/interactor-tool-godot-cage-deformer (MIT), and their
constraint solve with a dense LU:

    cage_bind_samples     cage, bary          -> constraint samples (64 per triangle)
    cage_bhc_coords       samples             -> rows [B | A]
    cage_bind_gram        [B | A]             -> G = AᵀA + BᵀB, RHS = −AᵀB
    cage_dense_lu         G                   -> L\U, pivots
    cage_dense_lu_solve   RHS                 -> X = G⁻¹ RHS = [[C11, C21], [C12, C22]]
    cage_bhc_coords       mesh points         -> rows [h | bh]
    cage_bind_blend       rows, X             -> (Φ | Ψ)

**The fit** (float32, sums in df32 where they are reductions), in bind
space, `y = Φ c + Ψ n(c)`, `c = c0 + u`:

    cage_normals          c                   -> n, |N|
    anny_csr_gemv3 ×2     Φ c, += Ψ n         -> y           (lean/Anny, reused)
    cage_lbs              blended bones, y    -> z_p = LBS_p(y)
    cage_contact          z_p, body_p         -> hinge h, ∂/∂z_p
    cage_lbs_t            ∂/∂z_p              -> ∂/∂y
    anny_csr_gemv3 ×2     Φᵀ g, Ψᵀ g          -> g_c, g_n
    cage_normal_vjp       g_n                 -> per-corner (I − nnᵀ) g_n / |N| terms
    anny_csr_gemv3        corner gather       -> += g_c
    cage_laplacian ×2     u                   -> L u, += 2 w_L Lᵀ L u
    cage_sumsq            h, L u, u           -> Σ h², ‖Lu‖², ‖u‖² (df32)

per frame, once: `cage_bone_blend` (Σ_b w_ib M_{p,b}), `cage_body_normals`
and `cage_body_vnormals` (the skinned body's face normals and
angle-weighted vertex pseudonormals). The curvenet cage checks containment with
`cage_winding` (the generalized winding number, double). `Cage.Kernels` lists them for
`lake exe emit_cage`.
-/
