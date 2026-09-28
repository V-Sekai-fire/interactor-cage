import Drape.SlangCodegen.Dsl

/-!
# `Cage.SlangCodegen.LbsT` — the skinning's transpose, over every frame

The vector-Jacobian product of `cage_lbs` with respect to the bind-space
position, gathered per vertex so no two threads write one entry:

    g_y_i = (accumulate ? g_y_i : 0) + Σ_p R_{p,i}ᵀ g_z_{p,i}

frames in order. One thread per vertex.

Bindings (set 0):

  0  ConstantBuffer<CageLbsTParams> { uint F; uint P; uint accumulate; }
  1  StructuredBuffer<float>   blend  (F P 12)
  2  StructuredBuffer<float>   gz     (3 F P)
  3  RWStructuredBuffer<float> gy     (3 P)
-/

namespace Cage.SlangCodegen.LbsT

open LeanSlang
open Drape.SlangCodegen.Dsl

def a (k : Nat) : E := at_ "blend" (v "id" * u 12 + u k)
def g (k : Nat) : E := at_ "gz" (v "id" * u 3 + u k)

def shader : SlangShaderModule :=
  { structs := [ { name := "CageLbsTParams", fields := [fld "F" uT, fld "P" uT, fld "accumulate" uT] } ]
  , globals := [ paramsCB "CageLbsTParams", roF "blend" 1, roF "gz" 2, rwF "gy" 3 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "i" (.member (v "tid") "x")
           , if_ (ge (v "i") (p "P")) [ ret ]
           , let_ bT "acc" (ne (p "accumulate") (u 0)) ] ++
           ([0, 1, 2].map fun c =>
             let_ fT ("s" ++ toString c) (sel (v "acc") (at_ "gy" (v "i" * u 3 + u c)) (fl 0.0))) ++
           [ for_ "fp" (u 0) (p "F")
               ([ let_ uT "id" (v "fp" * p "P" + v "i") ] ++
                ([0, 1, 2].map fun c =>
                  setv ("s" ++ toString c)
                    (v ("s" ++ toString c) + (a c * g 0 + a (4 + c) * g 1 + a (8 + c) * g 2)))) ] ++
           ([0, 1, 2].map fun c => setAt "gy" (v "i" * u 3 + u c) (v ("s" ++ toString c)))) ] }

-- BEGIN PIN
def expected : String :=
"struct CageLbsTParams {
  uint F;
  uint P;
  uint accumulate;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLbsTParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> blend;
[[vk::binding(2, 0)]]
StructuredBuffer<float> gz;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> gy;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  bool acc = (params.accumulate != 0u);
  float s0 = (acc ? gy[((i * 3u) + 0u)] : 0.000000);
  float s1 = (acc ? gy[((i * 3u) + 1u)] : 0.000000);
  float s2 = (acc ? gy[((i * 3u) + 2u)] : 0.000000);
  for (uint fp = 0u; fp < params.F; ++fp) {
    uint id = ((fp * params.P) + i);
    s0 = (s0 + (((blend[((id * 12u) + 0u)] * gz[((id * 3u) + 0u)]) + (blend[((id * 12u) + 4u)] * gz[((id * 3u) + 1u)])) + (blend[((id * 12u) + 8u)] * gz[((id * 3u) + 2u)])));
    s1 = (s1 + (((blend[((id * 12u) + 1u)] * gz[((id * 3u) + 0u)]) + (blend[((id * 12u) + 5u)] * gz[((id * 3u) + 1u)])) + (blend[((id * 12u) + 9u)] * gz[((id * 3u) + 2u)])));
    s2 = (s2 + (((blend[((id * 12u) + 2u)] * gz[((id * 3u) + 0u)]) + (blend[((id * 12u) + 6u)] * gz[((id * 3u) + 1u)])) + (blend[((id * 12u) + 10u)] * gz[((id * 3u) + 2u)])));
  }
  gy[((i * 3u) + 0u)] = s0;
  gy[((i * 3u) + 1u)] = s1;
  gy[((i * 3u) + 2u)] = s2;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.LbsT
