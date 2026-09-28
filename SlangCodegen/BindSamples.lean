import Cage.SlangCodegen.Common

/-!
# `Cage.SlangCodegen.BindSamples` — the constraint samples on the cage

`computeConstrainedBiharmonicMatrices_13` samples every cage triangle at
the centroids of its 4^s midpoint subdivision (s = 3: 64 per triangle),
the same points for the Laplacian and the Dirichlet constraints. The
barycentric table `bary` (3 doubles a sample, `(v0 + v1 + v2) / 3` of
each sub-triangle) is the host's; this kernel places the samples:

    eta[s]   = γ0·c[tri0] + γ1·c[tri1] + γ2·c[tri2]      (point3 order)
    gamma[s] = γ,  own[s] = t,     s = t·nB + k

One thread per sample.

Bindings (set 0):

  0  ConstantBuffer<CageSamplesParams> { uint nT; uint nB; }
  1  StructuredBuffer<double>   cage   (3 nV)
  2  StructuredBuffer<uint>     tris   (3 nT)
  3  StructuredBuffer<double>   bary   (3 nB)
  4  RWStructuredBuffer<double> pts    (3 nT nB)
  5  RWStructuredBuffer<double> gamma  (3 nT nB)
  6  RWStructuredBuffer<uint>   own    (nT nB)
-/

namespace Cage.SlangCodegen.BindSamples

open LeanSlang
open Drape.SlangCodegen.Dsl
open Cage.SlangCodegen.Common

def roD (n : String) (b : Nat) : SlangBinding := glob n (.roBuf dT) b
def rwD (n : String) (b : Nat) : SlangBinding := glob n (.rwBuf dT) b

def vert (k : Nat) : E :=
  let c := at_ "tris" (v "t" * u 3 + u k) * u 3
  d3 (at_ "cage" c) (at_ "cage" (c + u 1)) (at_ "cage" (c + u 2))

def shader : SlangShaderModule :=
  { structs := [ { name := "CageSamplesParams", fields := [fld "nT" uT, fld "nB" uT] } ]
  , globals :=
      [ paramsCB "CageSamplesParams", roD "cage" 1, roU "tris" 2, roD "bary" 3,
        rwD "pts" 4, rwD "gamma" 5, rwU "own" 6 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "s" (.member (v "tid") "x")
          , if_ (ge (v "s") (p "nT" * p "nB")) [ ret ]
          , let_ uT "t" (v "s" / p "nB")
          , let_ uT "k" (v "s" % p "nB")
          , let_ dT "g0" (at_ "bary" (v "k" * u 3))
          , let_ dT "g1" (at_ "bary" (v "k" * u 3 + u 1))
          , let_ dT "g2" (at_ "bary" (v "k" * u 3 + u 2))
          , let_ d3T "eta" (v "g0" * vert 0 + v "g1" * vert 1 + v "g2" * vert 2)
          , setAt "pts" (v "s" * u 3) (cx (v "eta"))
          , setAt "pts" (v "s" * u 3 + u 1) (cy (v "eta"))
          , setAt "pts" (v "s" * u 3 + u 2) (cz (v "eta"))
          , setAt "gamma" (v "s" * u 3) (v "g0")
          , setAt "gamma" (v "s" * u 3 + u 1) (v "g1")
          , setAt "gamma" (v "s" * u 3 + u 2) (v "g2")
          , setAt "own" (v "s") (v "t") ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageSamplesParams {
  uint nT;
  uint nB;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageSamplesParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> cage;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(3, 0)]]
StructuredBuffer<double> bary;
[[vk::binding(4, 0)]]
RWStructuredBuffer<double> pts;
[[vk::binding(5, 0)]]
RWStructuredBuffer<double> gamma;
[[vk::binding(6, 0)]]
RWStructuredBuffer<uint> own;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint s = tid.x;
  if ((s >= (params.nT * params.nB))) {
    return;
  }
  uint t = (s / params.nB);
  uint k = (s % params.nB);
  double g0 = bary[(k * 3u)];
  double g1 = bary[((k * 3u) + 1u)];
  double g2 = bary[((k * 3u) + 2u)];
  double3 eta = (((g0 * double3(cage[(tris[((t * 3u) + 0u)] * 3u)], cage[((tris[((t * 3u) + 0u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 0u)] * 3u) + 2u)])) + (g1 * double3(cage[(tris[((t * 3u) + 1u)] * 3u)], cage[((tris[((t * 3u) + 1u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 1u)] * 3u) + 2u)]))) + (g2 * double3(cage[(tris[((t * 3u) + 2u)] * 3u)], cage[((tris[((t * 3u) + 2u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 2u)] * 3u) + 2u)])));
  pts[(s * 3u)] = eta.x;
  pts[((s * 3u) + 1u)] = eta.y;
  pts[((s * 3u) + 2u)] = eta.z;
  gamma[(s * 3u)] = g0;
  gamma[((s * 3u) + 1u)] = g1;
  gamma[((s * 3u) + 2u)] = g2;
  own[s] = t;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.BindSamples
