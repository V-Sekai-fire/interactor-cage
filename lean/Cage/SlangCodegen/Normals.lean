import Drape.SlangCodegen.Dsl

/-!
# `Cage.SlangCodegen.Normals` — the cage's unit face normals

One thread per cage triangle (a, b, c):

    N = (c_b − c_a) × (c_c − c_a),   |N| = max(√(N·N), 1e-20),   n = N / |N|

the `normals` of the deform (`point3d::cross(...).direction()` in the
reference), with the norm guarded (AGENTS.md, Gate 5 G10). `nlen` keeps
|N| for `cage_normal_vjp`.

Bindings (set 0):

  0  ConstantBuffer<CageNormalsParams> { uint nT; }
  1  StructuredBuffer<float>   c     (3 nV)
  2  StructuredBuffer<uint>    tris  (3 nT)
  3  RWStructuredBuffer<float> n     (3 nT)
  4  RWStructuredBuffer<float> nlen  (nT)
-/

namespace Cage.SlangCodegen.Normals

open LeanSlang
open Drape.SlangCodegen.Dsl

def tiny : E := .litFloatExact 1e-20

/-- `pfx{0,1,2}` = `buf[3 idx + k]`. -/
def load3 (pfx buf : String) (idx : E) : List St :=
  [0, 1, 2].map fun k => let_ fT (pfx ++ toString k) (at_ buf (idx * u 3 + u k))

def s (pfx : String) (k : Nat) : E := v (pfx ++ toString k)

def cross3 (c a b : String) : List St :=
  [0, 1, 2].map fun i =>
    let j := (i + 1) % 3
    let k := (i + 2) % 3
    let_ fT (c ++ toString i) (s a j * s b k - s a k * s b j)

def sub3 (c a b : String) : List St :=
  [0, 1, 2].map fun k => let_ fT (c ++ toString k) (s a k - s b k)

def dot3 (a b : String) : E := s a 0 * s b 0 + s a 1 * s b 1 + s a 2 * s b 2

def shader : SlangShaderModule :=
  { structs := [ { name := "CageNormalsParams", fields := [fld "nT" uT] } ]
  , globals := [ paramsCB "CageNormalsParams", roF "c" 1, roU "tris" 2, rwF "n" 3, rwF "nlen" 4 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "t" (.member (v "tid") "x")
           , if_ (ge (v "t") (p "nT")) [ ret ] ] ++
           load3 "a" "c" (at_ "tris" (v "t" * u 3)) ++
           load3 "b" "c" (at_ "tris" (v "t" * u 3 + u 1)) ++
           load3 "q" "c" (at_ "tris" (v "t" * u 3 + u 2)) ++
           sub3 "e" "b" "a" ++ sub3 "f" "q" "a" ++ cross3 "N" "e" "f" ++
           [ let_ fT "l" (fmax (call "sqrt" [dot3 "N" "N"]) tiny) ] ++
           ([0, 1, 2].map fun k => setAt "n" (v "t" * u 3 + u k) (s "N" k / v "l")) ++
           [ setAt "nlen" (v "t") (v "l") ]) ] }

-- BEGIN PIN
def expected : String :=
"struct CageNormalsParams {
  uint nT;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageNormalsParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> c;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> n;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> nlen;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  if ((t >= params.nT)) {
    return;
  }
  float a0 = c[((tris[(t * 3u)] * 3u) + 0u)];
  float a1 = c[((tris[(t * 3u)] * 3u) + 1u)];
  float a2 = c[((tris[(t * 3u)] * 3u) + 2u)];
  float b0 = c[((tris[((t * 3u) + 1u)] * 3u) + 0u)];
  float b1 = c[((tris[((t * 3u) + 1u)] * 3u) + 1u)];
  float b2 = c[((tris[((t * 3u) + 1u)] * 3u) + 2u)];
  float q0 = c[((tris[((t * 3u) + 2u)] * 3u) + 0u)];
  float q1 = c[((tris[((t * 3u) + 2u)] * 3u) + 1u)];
  float q2 = c[((tris[((t * 3u) + 2u)] * 3u) + 2u)];
  float e0 = (b0 - a0);
  float e1 = (b1 - a1);
  float e2 = (b2 - a2);
  float f0 = (q0 - a0);
  float f1 = (q1 - a1);
  float f2 = (q2 - a2);
  float N0 = ((e1 * f2) - (e2 * f1));
  float N1 = ((e2 * f0) - (e0 * f2));
  float N2 = ((e0 * f1) - (e1 * f0));
  float l = max(sqrt((((N0 * N0) + (N1 * N1)) + (N2 * N2))), 1.0e-20f);
  n[((t * 3u) + 0u)] = (N0 / l);
  n[((t * 3u) + 1u)] = (N1 / l);
  n[((t * 3u) + 2u)] = (N2 / l);
  nlen[t] = l;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.Normals
