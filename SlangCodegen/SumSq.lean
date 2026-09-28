import Drape.SlangCodegen.Common

/-!
# `Cage.SlangCodegen.SumSq` — `Σ x_k²` in df32, one thread

The loss's three sums (the hinge terms `Σ_p Σ_i w_p h²`, `‖L u‖²`,
`‖u‖²`) are reductions: one thread, in order, with the df32 `df_acc` of
`lean/Drape` (~48 mantissa bits, the `dot_reduce` precision), written as
the pair `(hi, lo)` to `out[slot]`, `out[slot + 1]`.

Bindings (set 0):

  0  ConstantBuffer<CageSumSqParams> { uint n; uint slot; }
  1  StructuredBuffer<float>   x    (n)
  2  RWStructuredBuffer<float> out  (≥ slot + 2)
-/

namespace Cage.SlangCodegen.SumSq

open LeanSlang
open Drape.SlangCodegen.Dsl
open Drape.SlangCodegen.Common (dfHelpers)

def shader : SlangShaderModule :=
  { structs := [ { name := "CageSumSqParams", fields := [fld "n" uT, fld "slot" uT] } ]
  , globals := [ paramsCB "CageSumSqParams", roF "x" 1, rwF "out" 2 ]
  , functions := dfHelpers ++
      [ entry 1 [dtid]
          [ if_ (ne (.member (v "tid") "x") (u 0)) [ ret ]
          , let_ fT "hi" (fl 0.0)
          , let_ fT "lo" (fl 0.0)
          , for_ "k" (u 0) (p "n")
              [ let_ fT "a" (at_ "x" (v "k"))
              , do_ (call "df_acc" [v "hi", v "lo", v "a", v "a"]) ]
          , setAt "out" (p "slot") (v "hi")
          , setAt "out" (p "slot" + u 1) (v "lo") ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageSumSqParams {
  uint n;
  uint slot;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageSumSqParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> x;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> out;

void two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float bb = (h - a);
  float ah = (h - bb);
  float lo_a = (a - ah);
  float lo_b = (b - bb);
  hi = h;
  lo = (lo_a + lo_b);
  return;
}

void quick_two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float t = (h - a);
  hi = h;
  lo = (b - t);
  return;
}

void two_prod(float a, float b, out float hi, out float lo) {
  float h = (a * b);
  hi = h;
  lo = fma(a, b, (-h));
  return;
}

void df_add(float x_hi, float x_lo, float y_hi, float y_lo, out float z_hi, out float z_lo) {
  float sh;
  float sl;
  two_sum(x_hi, y_hi, sh, sl);
  float xy_lo = (x_lo + y_lo);
  float sl2 = (sl + xy_lo);
  quick_two_sum(sh, sl2, z_hi, z_lo);
  return;
}

void df_acc(inout float hi, inout float lo, float a, float b) {
  float p_hi;
  float p_lo;
  two_prod(a, b, p_hi, p_lo);
  float n_hi;
  float n_lo;
  df_add(hi, lo, p_hi, p_lo, n_hi, n_lo);
  hi = n_hi;
  lo = n_lo;
  return;
}

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  float hi = 0.000000;
  float lo = 0.000000;
  for (uint k = 0u; k < params.n; ++k) {
    float a = x[k];
    df_acc(hi, lo, a, a);
  }
  out[params.slot] = hi;
  out[(params.slot + 1u)] = lo;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.SumSq
