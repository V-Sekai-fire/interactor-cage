import Cage.SlangCodegen.Common
import Cage.SlangCodegen.BindSamples

/-!
# `Cage.SlangCodegen.DenseLuSolve` — `X ← G⁻¹ X`, one right-hand side per thread

With `cage_dense_lu`'s factor and pivots, thread j solves column j of
`X` (K × ncols, row-major) in place: the row swaps in order, forward
substitution with the unit lower factor, back substitution with U.

Bindings (set 0):

  0  ConstantBuffer<CageLuSolveParams> { uint K; uint ncols; }
  1  StructuredBuffer<double>   G      (K K, the factor)
  2  StructuredBuffer<uint>     piv    (K)
  3  RWStructuredBuffer<double> X      (K ncols)
-/

namespace Cage.SlangCodegen.DenseLuSolve

open LeanSlang
open Drape.SlangCodegen.Dsl
open Cage.SlangCodegen.Common
open Cage.SlangCodegen.BindSamples (roD rwD)

def g (r c : E) : E := at_ "G" (r * p "K" + c)
def x (r : E) : E := at_ "X" (r * p "ncols" + v "j")

def shader : SlangShaderModule :=
  { structs := [ { name := "CageLuSolveParams", fields := [fld "K" uT, fld "ncols" uT] } ]
  , globals := [ paramsCB "CageLuSolveParams", roD "G" 1, roU "piv" 2, rwD "X" 3 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "j" (.member (v "tid") "x")
          , if_ (ge (v "j") (p "ncols")) [ ret ]
          , for_ "k" (u 0) (p "K")
              [ let_ uT "pr" (at_ "piv" (v "k"))
              , if_ (ne (v "pr") (v "k"))
                  [ let_ dT "tmp" (x (v "k"))
                  , set (x (v "k")) (x (v "pr"))
                  , set (x (v "pr")) (v "tmp") ] ]
          , for_ "r" (u 0) (p "K")
              [ let_ dT "s" (x (v "r"))
              , for_ "c" (u 0) (v "r") [ setv "s" (v "s" - g (v "r") (v "c") * x (v "c")) ]
              , set (x (v "r")) (v "s") ]
          , for_ "q" (u 0) (p "K")
              [ let_ uT "r" (p "K" - u 1 - v "q")
              , let_ dT "s" (x (v "r"))
              , for_ "c" (v "r" + u 1) (p "K") [ setv "s" (v "s" - g (v "r") (v "c") * x (v "c")) ]
              , set (x (v "r")) (v "s" / g (v "r") (v "r")) ] ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageLuSolveParams {
  uint K;
  uint ncols;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLuSolveParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> G;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> piv;
[[vk::binding(3, 0)]]
RWStructuredBuffer<double> X;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint j = tid.x;
  if ((j >= params.ncols)) {
    return;
  }
  for (uint k = 0u; k < params.K; ++k) {
    uint pr = piv[k];
    if ((pr != k)) {
      double tmp = X[((k * params.ncols) + j)];
      X[((k * params.ncols) + j)] = X[((pr * params.ncols) + j)];
      X[((pr * params.ncols) + j)] = tmp;
    }
  }
  for (uint r = 0u; r < params.K; ++r) {
    double s = X[((r * params.ncols) + j)];
    for (uint c = 0u; c < r; ++c) {
      s = (s - (G[((r * params.K) + c)] * X[((c * params.ncols) + j)]));
    }
    X[((r * params.ncols) + j)] = s;
  }
  for (uint q = 0u; q < params.K; ++q) {
    uint r = ((params.K - 1u) - q);
    double s = X[((r * params.ncols) + j)];
    for (uint c = (r + 1u); c < params.K; ++c) {
      s = (s - (G[((r * params.K) + c)] * X[((c * params.ncols) + j)]));
    }
    X[((r * params.ncols) + j)] = (s / G[((r * params.K) + r)]);
  }
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.DenseLuSolve
