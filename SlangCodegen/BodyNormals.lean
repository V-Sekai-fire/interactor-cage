import Cage.SlangCodegen.Normals

/-!
# `Cage.SlangCodegen.BodyNormals` — the skinned body's face normals and corner angles

One thread per (frame p, body triangle t), over the body's vertices of
frame p (`bv[(p·BV + v)·3 …]`):

    n_{p,t} = N / max(|N|, 1e-20),   N = (b − a) × (c − a)
    θ_{p,t,k} = atan2(|e1 × e2|, e1·e2)   at corner k (e1, e2 its two edges)

The corner angles weight Bærentzen and Aanæs's angle-weighted vertex
pseudonormals (`cage_body_vnormals`), which sign the distance when the
closest point is a vertex.

Bindings (set 0):

  0  ConstantBuffer<CageBodyParams> { uint F; uint T; uint BV; }
  1  StructuredBuffer<float>   bv    (3 F BV)
  2  StructuredBuffer<uint>    tris  (3 T)
  3  RWStructuredBuffer<float> fn    (3 F T)
  4  RWStructuredBuffer<float> ang   (3 F T)
-/

namespace Cage.SlangCodegen.BodyNormals

open LeanSlang
open Drape.SlangCodegen.Dsl
open Cage.SlangCodegen.Normals (load3 s cross3 sub3 dot3 tiny)

/-- The angle between `e` and `f` (both 3-vector prefixes) into `out`. -/
def angle (out pre e f : String) : List St :=
  cross3 (pre ++ "x") e f ++
  [ let_ fT out (call "atan2" [call "sqrt" [dot3 (pre ++ "x") (pre ++ "x")], dot3 e f]) ]

def shader : SlangShaderModule :=
  { structs := [ { name := "CageBodyParams", fields := [fld "F" uT, fld "T" uT, fld "BV" uT] } ]
  , globals := [ paramsCB "CageBodyParams", roF "bv" 1, roU "tris" 2, rwF "fn" 3, rwF "ang" 4 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "id" (.member (v "tid") "x")
           , if_ (ge (v "id") (p "F" * p "T")) [ ret ]
           , let_ uT "fp" (v "id" / p "T")
           , let_ uT "t" (v "id" % p "T")
           , let_ uT "o" (v "fp" * p "BV") ] ++
           load3 "a" "bv" (v "o" + at_ "tris" (v "t" * u 3)) ++
           load3 "b" "bv" (v "o" + at_ "tris" (v "t" * u 3 + u 1)) ++
           load3 "q" "bv" (v "o" + at_ "tris" (v "t" * u 3 + u 2)) ++
           sub3 "ab" "b" "a" ++ sub3 "aq" "q" "a" ++ sub3 "ba" "a" "b" ++ sub3 "bq" "q" "b" ++
           sub3 "qa" "a" "q" ++ sub3 "qb" "b" "q" ++
           cross3 "N" "ab" "aq" ++
           [ let_ fT "l" (fmax (call "sqrt" [dot3 "N" "N"]) tiny) ] ++
           ([0, 1, 2].map fun k => setAt "fn" (v "id" * u 3 + u k) (s "N" k / v "l")) ++
           angle "t0" "ca" "ab" "aq" ++ angle "t1" "cb" "bq" "ba" ++ angle "t2" "cq" "qa" "qb" ++
           [ setAt "ang" (v "id" * u 3) (v "t0")
           , setAt "ang" (v "id" * u 3 + u 1) (v "t1")
           , setAt "ang" (v "id" * u 3 + u 2) (v "t2") ]) ] }

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
StructuredBuffer<float> bv;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> fn;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> ang;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.T))) {
    return;
  }
  uint fp = (id / params.T);
  uint t = (id % params.T);
  uint o = (fp * params.BV);
  float a0 = bv[(((o + tris[(t * 3u)]) * 3u) + 0u)];
  float a1 = bv[(((o + tris[(t * 3u)]) * 3u) + 1u)];
  float a2 = bv[(((o + tris[(t * 3u)]) * 3u) + 2u)];
  float b0 = bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 0u)];
  float b1 = bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 1u)];
  float b2 = bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 2u)];
  float q0 = bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 0u)];
  float q1 = bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 1u)];
  float q2 = bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 2u)];
  float ab0 = (b0 - a0);
  float ab1 = (b1 - a1);
  float ab2 = (b2 - a2);
  float aq0 = (q0 - a0);
  float aq1 = (q1 - a1);
  float aq2 = (q2 - a2);
  float ba0 = (a0 - b0);
  float ba1 = (a1 - b1);
  float ba2 = (a2 - b2);
  float bq0 = (q0 - b0);
  float bq1 = (q1 - b1);
  float bq2 = (q2 - b2);
  float qa0 = (a0 - q0);
  float qa1 = (a1 - q1);
  float qa2 = (a2 - q2);
  float qb0 = (b0 - q0);
  float qb1 = (b1 - q1);
  float qb2 = (b2 - q2);
  float N0 = ((ab1 * aq2) - (ab2 * aq1));
  float N1 = ((ab2 * aq0) - (ab0 * aq2));
  float N2 = ((ab0 * aq1) - (ab1 * aq0));
  float l = max(sqrt((((N0 * N0) + (N1 * N1)) + (N2 * N2))), 1.0e-20f);
  fn[((id * 3u) + 0u)] = (N0 / l);
  fn[((id * 3u) + 1u)] = (N1 / l);
  fn[((id * 3u) + 2u)] = (N2 / l);
  float cax0 = ((ab1 * aq2) - (ab2 * aq1));
  float cax1 = ((ab2 * aq0) - (ab0 * aq2));
  float cax2 = ((ab0 * aq1) - (ab1 * aq0));
  float t0 = atan2(sqrt((((cax0 * cax0) + (cax1 * cax1)) + (cax2 * cax2))), (((ab0 * aq0) + (ab1 * aq1)) + (ab2 * aq2)));
  float cbx0 = ((bq1 * ba2) - (bq2 * ba1));
  float cbx1 = ((bq2 * ba0) - (bq0 * ba2));
  float cbx2 = ((bq0 * ba1) - (bq1 * ba0));
  float t1 = atan2(sqrt((((cbx0 * cbx0) + (cbx1 * cbx1)) + (cbx2 * cbx2))), (((bq0 * ba0) + (bq1 * ba1)) + (bq2 * ba2)));
  float cqx0 = ((qa1 * qb2) - (qa2 * qb1));
  float cqx1 = ((qa2 * qb0) - (qa0 * qb2));
  float cqx2 = ((qa0 * qb1) - (qa1 * qb0));
  float t2 = atan2(sqrt((((cqx0 * cqx0) + (cqx1 * cqx1)) + (cqx2 * cqx2))), (((qa0 * qb0) + (qa1 * qb1)) + (qa2 * qb2)));
  ang[(id * 3u)] = t0;
  ang[((id * 3u) + 1u)] = t1;
  ang[((id * 3u) + 2u)] = t2;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.BodyNormals
