import Drape.SlangCodegen.Dsl

/-!
# `Cage.SlangCodegen.Lbs` — skin the bind-space garment into every frame

One thread per (frame p, vertex i), with `cage_bone_blend`'s affine
`A_{p,i} = [R | t]`:

    z_{p,i} = R y_i + t

Bindings (set 0):

  0  ConstantBuffer<CageLbsParams> { uint F; uint P; }
  1  StructuredBuffer<float>   blend  (F P 12)
  2  StructuredBuffer<float>   y      (3 P)
  3  RWStructuredBuffer<float> z      (3 F P)
-/

namespace Cage.SlangCodegen.Lbs

open LeanSlang
open Drape.SlangCodegen.Dsl

def a (k : Nat) : E := at_ "blend" (v "id" * u 12 + u k)

def shader : SlangShaderModule :=
  { structs := [ { name := "CageLbsParams", fields := [fld "F" uT, fld "P" uT] } ]
  , globals := [ paramsCB "CageLbsParams", roF "blend" 1, roF "y" 2, rwF "z" 3 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "id" (.member (v "tid") "x")
           , if_ (ge (v "id") (p "F" * p "P")) [ ret ]
           , let_ uT "i" (v "id" % p "P")
           , let_ fT "y0" (at_ "y" (v "i" * u 3))
           , let_ fT "y1" (at_ "y" (v "i" * u 3 + u 1))
           , let_ fT "y2" (at_ "y" (v "i" * u 3 + u 2)) ] ++
           ([0, 1, 2].map fun r =>
             setAt "z" (v "id" * u 3 + u r)
               (a (4 * r) * v "y0" + a (4 * r + 1) * v "y1" + a (4 * r + 2) * v "y2" + a (4 * r + 3)))) ] }

-- BEGIN PIN
def expected : String :=
"struct CageLbsParams {
  uint F;
  uint P;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLbsParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> blend;
[[vk::binding(2, 0)]]
StructuredBuffer<float> y;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> z;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.P))) {
    return;
  }
  uint i = (id % params.P);
  float y0 = y[(i * 3u)];
  float y1 = y[((i * 3u) + 1u)];
  float y2 = y[((i * 3u) + 2u)];
  z[((id * 3u) + 0u)] = ((((blend[((id * 12u) + 0u)] * y0) + (blend[((id * 12u) + 1u)] * y1)) + (blend[((id * 12u) + 2u)] * y2)) + blend[((id * 12u) + 3u)]);
  z[((id * 3u) + 1u)] = ((((blend[((id * 12u) + 4u)] * y0) + (blend[((id * 12u) + 5u)] * y1)) + (blend[((id * 12u) + 6u)] * y2)) + blend[((id * 12u) + 7u)]);
  z[((id * 3u) + 2u)] = ((((blend[((id * 12u) + 8u)] * y0) + (blend[((id * 12u) + 9u)] * y1)) + (blend[((id * 12u) + 10u)] * y2)) + blend[((id * 12u) + 11u)]);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.Lbs
