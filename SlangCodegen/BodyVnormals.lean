import Drape.SlangCodegen.Dsl

/-!
# `Cage.SlangCodegen.BodyVnormals` — angle-weighted vertex pseudonormals

One thread per (frame p, body vertex v): the sum over v's incident
corners (a CSR of corner ids `3t + k`) of the corner angle times the face
normal (Bærentzen and Aanæs, 2005). Unnormalized: only its sign against
`z − q` is read.

Bindings (set 0):

  0  ConstantBuffer<CageBodyParams> { uint F; uint T; uint BV; }
  1  StructuredBuffer<uint>    rowptr  (BV + 1)
  2  StructuredBuffer<uint>    corner  (3 T)
  3  StructuredBuffer<float>   fn      (3 F T)
  4  StructuredBuffer<float>   ang     (3 F T)
  5  RWStructuredBuffer<float> vn      (3 F BV)
-/

namespace Cage.SlangCodegen.BodyVnormals

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "CageBodyParams", fields := [fld "F" uT, fld "T" uT, fld "BV" uT] } ]
  , globals := [ paramsCB "CageBodyParams", roU "rowptr" 1, roU "corner" 2, roF "fn" 3, roF "ang" 4,
                 rwF "vn" 5 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "id" (.member (v "tid") "x")
           , if_ (ge (v "id") (p "F" * p "BV")) [ ret ]
           , let_ uT "fp" (v "id" / p "BV")
           , let_ uT "vi" (v "id" % p "BV")
           , let_ fT "x" (fl 0.0)
           , let_ fT "y" (fl 0.0)
           , let_ fT "z" (fl 0.0)
           , for_ "k" (at_ "rowptr" (v "vi")) (at_ "rowptr" (v "vi" + u 1))
               [ let_ uT "cn" (at_ "corner" (v "k"))
               , let_ uT "ft" (v "fp" * p "T" + v "cn" / u 3)
               , let_ fT "an" (at_ "ang" (v "fp" * p "T" * u 3 + v "cn"))
               , setv "x" (v "x" + v "an" * at_ "fn" (v "ft" * u 3))
               , setv "y" (v "y" + v "an" * at_ "fn" (v "ft" * u 3 + u 1))
               , setv "z" (v "z" + v "an" * at_ "fn" (v "ft" * u 3 + u 2)) ]
           , setAt "vn" (v "id" * u 3) (v "x")
           , setAt "vn" (v "id" * u 3 + u 1) (v "y")
           , setAt "vn" (v "id" * u 3 + u 2) (v "z") ]) ] }

-- BEGIN PIN
def expected : String :=
"struct CageBodyParams {
  uint F;
  uint T;
  uint BV;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageBodyParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> rowptr;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> corner;
[[vk::binding(3, 0)]]
StructuredBuffer<float> fn;
[[vk::binding(4, 0)]]
StructuredBuffer<float> ang;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> vn;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.BV))) {
    return;
  }
  uint fp = (id / params.BV);
  uint vi = (id % params.BV);
  float x = 0.000000;
  float y = 0.000000;
  float z = 0.000000;
  for (uint k = rowptr[vi]; k < rowptr[(vi + 1u)]; ++k) {
    uint cn = corner[k];
    uint ft = ((fp * params.T) + (cn / 3u));
    float an = ang[(((fp * params.T) * 3u) + cn)];
    x = (x + (an * fn[(ft * 3u)]));
    y = (y + (an * fn[((ft * 3u) + 1u)]));
    z = (z + (an * fn[((ft * 3u) + 2u)]));
  }
  vn[(id * 3u)] = x;
  vn[((id * 3u) + 1u)] = y;
  vn[((id * 3u) + 2u)] = z;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.BodyVnormals
