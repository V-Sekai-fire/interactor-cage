import Drape.SlangCodegen.Dsl

/-!
# `Cage.SlangCodegen.BoneBlend` — the per-vertex bone blend of each frame

LBS is linear in the rest position, so each frame's skinning of a
garment vertex is one affine map, blended once per frame (the bones do
not change while the cage moves):

    A_{p,i} = Σ_{k ∈ row i} w_k · M_{p, col_k}            (12 floats, [R | t])

One thread per (frame p, vertex i), `p = id / P`. The skin weights are a
CSR over the vertex's bones (`rowptr`, `col`, `w`), the bones of frame p
`bones[(p·NB + b)·12 …]` in the Anny layout (row r at `4r`, translation
in column 3).

Bindings (set 0):

  0  ConstantBuffer<CageBoneBlendParams> { uint F; uint P; uint NB; }
  1  StructuredBuffer<uint>    rowptr  (P + 1)
  2  StructuredBuffer<uint>    col     (nnz)
  3  StructuredBuffer<float>   w       (nnz)
  4  StructuredBuffer<float>   bones   (F NB 12)
  5  RWStructuredBuffer<float> blend   (F P 12)
-/

namespace Cage.SlangCodegen.BoneBlend

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "CageBoneBlendParams", fields := [fld "F" uT, fld "P" uT, fld "NB" uT] } ]
  , globals := [ paramsCB "CageBoneBlendParams", roU "rowptr" 1, roU "col" 2, roF "w" 3, roF "bones" 4,
                 rwF "blend" 5 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "id" (.member (v "tid") "x")
           , if_ (ge (v "id") (p "F" * p "P")) [ ret ]
           , let_ uT "fp" (v "id" / p "P")
           , let_ uT "i" (v "id" % p "P") ] ++
           ((List.range 12).map fun k => let_ fT ("a" ++ toString k) (fl 0.0)) ++
           [ for_ "k" (at_ "rowptr" (v "i")) (at_ "rowptr" (v "i" + u 1))
               ([ let_ fT "wk" (at_ "w" (v "k"))
                , let_ uT "b" ((v "fp" * p "NB" + at_ "col" (v "k")) * u 12) ] ++
                ((List.range 12).map fun m =>
                  setv ("a" ++ toString m) (v ("a" ++ toString m) + v "wk" * at_ "bones" (v "b" + u m)))) ] ++
           ((List.range 12).map fun m => setAt "blend" (v "id" * u 12 + u m) (v ("a" ++ toString m)))) ] }

-- BEGIN PIN
def expected : String :=
"struct CageBoneBlendParams {
  uint F;
  uint P;
  uint NB;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageBoneBlendParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> rowptr;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> col;
[[vk::binding(3, 0)]]
StructuredBuffer<float> w;
[[vk::binding(4, 0)]]
StructuredBuffer<float> bones;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> blend;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.P))) {
    return;
  }
  uint fp = (id / params.P);
  uint i = (id % params.P);
  float a0 = 0.000000;
  float a1 = 0.000000;
  float a2 = 0.000000;
  float a3 = 0.000000;
  float a4 = 0.000000;
  float a5 = 0.000000;
  float a6 = 0.000000;
  float a7 = 0.000000;
  float a8 = 0.000000;
  float a9 = 0.000000;
  float a10 = 0.000000;
  float a11 = 0.000000;
  for (uint k = rowptr[i]; k < rowptr[(i + 1u)]; ++k) {
    float wk = w[k];
    uint b = (((fp * params.NB) + col[k]) * 12u);
    a0 = (a0 + (wk * bones[(b + 0u)]));
    a1 = (a1 + (wk * bones[(b + 1u)]));
    a2 = (a2 + (wk * bones[(b + 2u)]));
    a3 = (a3 + (wk * bones[(b + 3u)]));
    a4 = (a4 + (wk * bones[(b + 4u)]));
    a5 = (a5 + (wk * bones[(b + 5u)]));
    a6 = (a6 + (wk * bones[(b + 6u)]));
    a7 = (a7 + (wk * bones[(b + 7u)]));
    a8 = (a8 + (wk * bones[(b + 8u)]));
    a9 = (a9 + (wk * bones[(b + 9u)]));
    a10 = (a10 + (wk * bones[(b + 10u)]));
    a11 = (a11 + (wk * bones[(b + 11u)]));
  }
  blend[((id * 12u) + 0u)] = a0;
  blend[((id * 12u) + 1u)] = a1;
  blend[((id * 12u) + 2u)] = a2;
  blend[((id * 12u) + 3u)] = a3;
  blend[((id * 12u) + 4u)] = a4;
  blend[((id * 12u) + 5u)] = a5;
  blend[((id * 12u) + 6u)] = a6;
  blend[((id * 12u) + 7u)] = a7;
  blend[((id * 12u) + 8u)] = a8;
  blend[((id * 12u) + 9u)] = a9;
  blend[((id * 12u) + 10u)] = a10;
  blend[((id * 12u) + 11u)] = a11;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.BoneBlend
