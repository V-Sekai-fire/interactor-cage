import Cage.SlangCodegen.Common
import Cage.SlangCodegen.BindSamples

/-!
# `Cage.SlangCodegen.DenseLu` — in-place LU with partial pivoting (double)

One thread. `G` (K×K, row-major) becomes `L\U` (L unit lower, below the
diagonal; U on and above it) with the row swaps in `piv` in LAPACK's
convention: at step k, rows k and `piv[k]` were exchanged. The pivot is
the first row of largest |entry| in column k at or below the diagonal. A
zero pivot column clears `status[0]` (1: every pivot non-zero) and skips
its elimination, so the factor stays finite; the bind refuses a singular
system by that flag.

This is the solve BHC.h leaves to Eigen's `inverse()` (a PartialPivLU):
the same pivoting rule, unblocked.

Bindings (set 0):

  0  ConstantBuffer<CageLuParams> { uint K; }
  1  RWStructuredBuffer<double> G       (K K)
  2  RWStructuredBuffer<uint>   piv     (K)
  3  RWStructuredBuffer<uint>   status  (1)
-/

namespace Cage.SlangCodegen.DenseLu

open LeanSlang
open Drape.SlangCodegen.Dsl
open Cage.SlangCodegen.Common
open Cage.SlangCodegen.BindSamples (roD rwD)

def g (r c : E) : E := at_ "G" (r * p "K" + c)

def shader : SlangShaderModule :=
  { structs := [ { name := "CageLuParams", fields := [fld "K" uT] } ]
  , globals := [ paramsCB "CageLuParams", rwD "G" 1, rwU "piv" 2, rwU "status" 3 ]
  , functions :=
      [ entry 1 [dtid]
          [ if_ (ne (.member (v "tid") "x") (u 0)) [ ret ]
          , let_ uT "ok" (u 1)
          , for_ "k" (u 0) (p "K")
              [ let_ uT "pr" (v "k")
              , let_ dT "mx" (fabs (g (v "k") (v "k")))
              , for_ "r" (v "k" + u 1) (p "K")
                  [ let_ dT "a" (fabs (g (v "r") (v "k")))
                  , if_ (gt (v "a") (v "mx")) [ setv "mx" (v "a"), setv "pr" (v "r") ] ]
              , setAt "piv" (v "k") (v "pr")
              , if_ (eq (v "mx") (dl 0.0))
                  [ setv "ok" (u 0) ]
                  [ if_ (ne (v "pr") (v "k"))
                      [ for_ "c" (u 0) (p "K")
                          [ let_ dT "tmp" (g (v "k") (v "c"))
                          , set (g (v "k") (v "c")) (g (v "pr") (v "c"))
                          , set (g (v "pr") (v "c")) (v "tmp") ] ]
                  , let_ dT "dk" (g (v "k") (v "k"))
                  , for_ "r" (v "k" + u 1) (p "K")
                      [ let_ dT "l" (g (v "r") (v "k") / v "dk")
                      , set (g (v "r") (v "k")) (v "l")
                      , for_ "c" (v "k" + u 1) (p "K")
                          [ set (g (v "r") (v "c")) (g (v "r") (v "c") - v "l" * g (v "k") (v "c")) ] ] ] ]
          , setAt "status" (u 0) (v "ok") ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageLuParams {
  uint K;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLuParams> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<double> G;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> piv;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> status;

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  uint ok = 1u;
  for (uint k = 0u; k < params.K; ++k) {
    uint pr = k;
    double mx = abs(G[((k * params.K) + k)]);
    for (uint r = (k + 1u); r < params.K; ++r) {
      double a = abs(G[((r * params.K) + k)]);
      if ((a > mx)) {
        mx = a;
        pr = r;
      }
    }
    piv[k] = pr;
    if ((mx == 0.0L)) {
      ok = 0u;
    } else {
      if ((pr != k)) {
        for (uint c = 0u; c < params.K; ++c) {
          double tmp = G[((k * params.K) + c)];
          G[((k * params.K) + c)] = G[((pr * params.K) + c)];
          G[((pr * params.K) + c)] = tmp;
        }
      }
      double dk = G[((k * params.K) + k)];
      for (uint r = (k + 1u); r < params.K; ++r) {
        double l = (G[((r * params.K) + k)] / dk);
        G[((r * params.K) + k)] = l;
        for (uint c = (k + 1u); c < params.K; ++c) {
          G[((r * params.K) + c)] = (G[((r * params.K) + c)] - (l * G[((k * params.K) + c)]));
        }
      }
    }
  }
  status[0u] = ok;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.DenseLu
