import Cage.SlangCodegen.Common
import Cage.SlangCodegen.BindSamples

/-!
# `Cage.SlangCodegen.BindGram` — the normal equations of the (1,3) constraints

From the sample rows `[B | A]` that `cage_bhc_coords` leaves (S rows of
`stride = 2K`), one thread per entry (i, j) of the K×K system, entry
`ij = ij0 + id` for `id < count` (a slice of the K² entries per host tick,
AGENTS.md rule 4):

    G[i,j]   = Σ_s A[s,i]·A[s,j]  +  Σ_s B[s,i]·B[s,j]      (AᵀA + BᵀB)
    RHS[i,j] = −Σ_s A[s,i]·B[s,j]                          (Aᵀ R, R = −B)

so that `X = G⁻¹ RHS` is BHC.h's `C · (M − H_Φ | −H_Ψ)` stacked:
rows `0..nV` are `C_L`, rows `nV..K` `C_D`; columns `0..nV` come from
`M − H_Φ`, the rest from `−H_Ψ`, i.e. `X = [[C11, C21], [C12, C22]]` with `gamma_D = 1` (the
reference's default and the only value this bind takes). Each sum runs
over s in order, in double; the two Gram terms are summed apart and then
added, as Eigen forms `AᵀA + BᵀB`.

Bindings (set 0):

  0  ConstantBuffer<CageGramParams> { uint S; uint K; uint stride; uint ij0; uint count; }
  1  StructuredBuffer<double>   rows   (S stride)
  2  RWStructuredBuffer<double> G      (K K)
  3  RWStructuredBuffer<double> rhs    (K K)
-/

namespace Cage.SlangCodegen.BindGram

open LeanSlang
open Drape.SlangCodegen.Dsl
open Cage.SlangCodegen.Common
open Cage.SlangCodegen.BindSamples (roD rwD)

def shader : SlangShaderModule :=
  { structs := [ { name := "CageGramParams", fields := [fld "S" uT, fld "K" uT, fld "stride" uT, fld "ij0" uT, fld "count" uT] } ]
  , globals := [ paramsCB "CageGramParams", roD "rows" 1, rwD "G" 2, rwD "rhs" 3 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "id" (.member (v "tid") "x")
          , if_ (ge (v "id") (p "count")) [ ret ]
          , let_ uT "ij" (p "ij0" + v "id")
          , if_ (ge (v "ij") (p "K" * p "K")) [ ret ]
          , let_ uT "i" (v "ij" / p "K")
          , let_ uT "j" (v "ij" % p "K")
          , let_ dT "ga" (dl 0.0)
          , let_ dT "gb" (dl 0.0)
          , let_ dT "r" (dl 0.0)
          , for_ "s" (u 0) (p "S")
              [ let_ uT "o" (v "s" * p "stride")
              , let_ dT "ai" (at_ "rows" (v "o" + p "K" + v "i"))
              , let_ dT "aj" (at_ "rows" (v "o" + p "K" + v "j"))
              , let_ dT "bi" (at_ "rows" (v "o" + v "i"))
              , let_ dT "bj" (at_ "rows" (v "o" + v "j"))
              , setv "ga" (v "ga" + v "ai" * v "aj")
              , setv "gb" (v "gb" + v "bi" * v "bj")
              , setv "r" (v "r" - v "ai" * v "bj") ]
          , setAt "G" (v "ij") (v "ga" + v "gb")
          , setAt "rhs" (v "ij") (v "r") ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageGramParams {
  uint S;
  uint K;
  uint stride;
  uint ij0;
  uint count;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageGramParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> rows;
[[vk::binding(2, 0)]]
RWStructuredBuffer<double> G;
[[vk::binding(3, 0)]]
RWStructuredBuffer<double> rhs;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= params.count)) {
    return;
  }
  uint ij = (params.ij0 + id);
  if ((ij >= (params.K * params.K))) {
    return;
  }
  uint i = (ij / params.K);
  uint j = (ij % params.K);
  double ga = 0.0L;
  double gb = 0.0L;
  double r = 0.0L;
  for (uint s = 0u; s < params.S; ++s) {
    uint o = (s * params.stride);
    double ai = rows[((o + params.K) + i)];
    double aj = rows[((o + params.K) + j)];
    double bi = rows[(o + i)];
    double bj = rows[(o + j)];
    ga = (ga + (ai * aj));
    gb = (gb + (bi * bj));
    r = (r - (ai * bj));
  }
  G[ij] = (ga + gb);
  rhs[ij] = r;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.BindGram
