import Cage.SlangCodegen.Common
import Cage.SlangCodegen.BindSamples

/-!
# `Cage.SlangCodegen.BindBlend` — the (1,3) weights of a mesh point

`compute_13_blending_from_unconstrained_biharmonics`, one thread per
mesh point, from its `cage_bhc_coords` row `[h | bh]` (h = (h_φ, h_ψ),
bh = (bh_φ, bh_ψ), K each) and `X` of `cage_dense_lu_solve`:

    w[k] = h[k] + Σ_{m<K} X[m, k] · bh[m]          k < K

`w = (Φ | Ψ)`: the first nV are the vertex weights Φ, the next nT the
normal weights Ψ. The sum runs over m in order, as BHC.h's does (the C11
/ C21 rows, then C12 / C22). Written in double (`wd`) and rounded to
float (`wf`, what the deform reads).

Bindings (set 0):

  0  ConstantBuffer<CageBlendParams> { uint P; uint K; uint stride; }
  1  StructuredBuffer<double>   rows   (P stride)
  2  StructuredBuffer<double>   X      (K K)
  3  RWStructuredBuffer<double> wd     (P K)
  4  RWStructuredBuffer<float>  wf     (P K)
-/

namespace Cage.SlangCodegen.BindBlend

open LeanSlang
open Drape.SlangCodegen.Dsl
open Cage.SlangCodegen.Common
open Cage.SlangCodegen.BindSamples (roD rwD)

def shader : SlangShaderModule :=
  { structs := [ { name := "CageBlendParams", fields := [fld "P" uT, fld "K" uT, fld "stride" uT] } ]
  , globals := [ paramsCB "CageBlendParams", roD "rows" 1, roD "X" 2, rwD "wd" 3, rwF "wf" 4 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "P")) [ ret ]
          , let_ uT "o" (v "i" * p "stride")
          , for_ "k" (u 0) (p "K")
              [ let_ dT "w" (at_ "rows" (v "o" + v "k"))
              , for_ "m" (u 0) (p "K")
                  [ setv "w" (v "w" + at_ "X" (v "m" * p "K" + v "k") * at_ "rows" (v "o" + p "K" + v "m")) ]
              , setAt "wd" (v "i" * p "K" + v "k") (v "w")
              , setAt "wf" (v "i" * p "K" + v "k") (.cast fT (v "w")) ] ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageBlendParams {
  uint P;
  uint K;
  uint stride;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageBlendParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> rows;
[[vk::binding(2, 0)]]
StructuredBuffer<double> X;
[[vk::binding(3, 0)]]
RWStructuredBuffer<double> wd;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> wf;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  uint o = (i * params.stride);
  for (uint k = 0u; k < params.K; ++k) {
    double w = rows[(o + k)];
    for (uint m = 0u; m < params.K; ++m) {
      w = (w + (X[((m * params.K) + k)] * rows[((o + params.K) + m)]));
    }
    wd[((i * params.K) + k)] = w;
    wf[((i * params.K) + k)] = float(w);
  }
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.BindBlend
