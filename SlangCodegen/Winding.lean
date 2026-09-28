import Cage.SlangCodegen.Common
import Cage.SlangCodegen.BindSamples

/-!
# `Cage.SlangCodegen.Winding` — the generalized winding number of points in a cage

One thread per point p, every cage triangle in order:

    w(p) = Σ_t Ω_t(p) / 4π,   Ω_t = solid_angle(u_a, u_b, u_c),  u_k = (c_k − p)/|c_k − p|

with `Common.solidAngle` (`get_signed_solid_angle`, on unit vectors as
`h_coordinates` calls it). For a closed, outward-oriented cage w is 1
inside and 0 outside; the curvenet cage's containment test (every bound
vertex inside) reads it before any bind.

Bindings (set 0):

  0  ConstantBuffer<CageWindingParams> { uint P; uint nT; }
  1  StructuredBuffer<double>   pts   (3 P)
  2  StructuredBuffer<double>   cage  (3 nV)
  3  StructuredBuffer<uint>     tris  (3 nT)
  4  RWStructuredBuffer<double> w     (P)
-/

namespace Cage.SlangCodegen.Winding

open LeanSlang
open Drape.SlangCodegen.Dsl
open Cage.SlangCodegen.Common
open Cage.SlangCodegen.BindSamples (roD rwD)

def unitTo (k : Nat) : E :=
  let c := at_ "tris" (v "t" * u 3 + u k) * u 3
  d3 (at_ "cage" c) (at_ "cage" (c + u 1)) (at_ "cage" (c + u 2)) - v "pt"

def shader : SlangShaderModule :=
  { structs := [ { name := "CageWindingParams", fields := [fld "P" uT, fld "nT" uT] } ]
  , globals := [ paramsCB "CageWindingParams", roD "pts" 1, roD "cage" 2, roU "tris" 3, rwD "w" 4 ]
  , functions := [ vdot, vcross, vnorm, vdir, datan01, datan2, solidAngle ] ++
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "P")) [ ret ]
          , let_ d3T "pt" (d3 (at_ "pts" (v "i" * u 3)) (at_ "pts" (v "i" * u 3 + u 1)) (at_ "pts" (v "i" * u 3 + u 2)))
          , let_ dT "s" (dl 0.0)
          , for_ "t" (u 0) (p "nT")
              [ setv "s" (v "s" + call "solid_angle"
                  [call "vdir" [unitTo 0], call "vdir" [unitTo 1], call "vdir" [unitTo 2]]) ]
          , setAt "w" (v "i") (v "s" / (dl 4.0 * piD)) ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageWindingParams {
  uint P;
  uint nT;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageWindingParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> pts;
[[vk::binding(2, 0)]]
StructuredBuffer<double> cage;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(4, 0)]]
RWStructuredBuffer<double> w;

double vdot(double3 a, double3 b) {
  return (((a.x * b.x) + (a.y * b.y)) + (a.z * b.z));
}

double3 vcross(double3 a, double3 b) {
  return double3(((a.y * b.z) - (a.z * b.y)), ((a.z * b.x) - (a.x * b.z)), ((a.x * b.y) - (a.y * b.x)));
}

double vnorm(double3 a) {
  return sqrt((((a.x * a.x) + (a.y * a.y)) + (a.z * a.z)));
}

double3 vdir(double3 a) {
  double n = vnorm(a);
  return double3((a.x / n), (a.y / n), (a.z / n));
}

double datan01(double z) {
  double t = (z / (1.0L + sqrt((1.0L + (z * z)))));
  t = (t / (1.0L + sqrt((1.0L + (t * t)))));
  double t2 = (t * t);
  double p = asdouble(0xBDA12F68u, 0xBFA2F684u);
  p = (0.04L + (t2 * p));
  p = (asdouble(0x590B2164u, 0xBFA642C8u) + (t2 * p));
  p = (asdouble(0x18618618u, 0x3FA86186u) + (t2 * p));
  p = ((-0.05263157894736842L) + (t2 * p));
  p = (asdouble(0x1E1E1E1Eu, 0x3FAE1E1Eu) + (t2 * p));
  p = ((-0.06666666666666667L) + (t2 * p));
  p = (0.07692307692307693L + (t2 * p));
  p = ((-0.09090909090909091L) + (t2 * p));
  p = (0.1111111111111111L + (t2 * p));
  p = ((-0.14285714285714285L) + (t2 * p));
  p = (0.2L + (t2 * p));
  p = ((-0.3333333333333333L) + (t2 * p));
  p = (1.0L + (t2 * p));
  return ((4.0L * t) * p);
}

double datan2(double y, double x) {
  double ax = abs(x);
  double ay = abs(y);
  double lo = min(ax, ay);
  double hi = max(ax, ay);
  double r = 0.0L;
  if ((hi > 0.0L)) {
    r = datan01((lo / hi));
  }
  if ((ay > ax)) {
    r = ((3.141592653589793L / 2.0L) - r);
  }
  if ((x < 0.0L)) {
    r = (3.141592653589793L - r);
  }
  if ((y < 0.0L)) {
    r = (-r);
  }
  return r;
}

double solid_angle(double3 a, double3 b, double3 c) {
  double det = vdot(a, vcross(b, c));
  if ((abs(det) < 1.0e-10L)) {
    return 0.0L;
  }
  double al = vnorm(a);
  double bl = vnorm(b);
  double cl = vnorm(c);
  double dv = (((((al * bl) * cl) + (vdot(a, b) * cl)) + (vdot(a, c) * bl)) + (vdot(b, c) * al));
  double at = datan2(abs(det), dv);
  if ((at < 0.0L)) {
    at = (at + 3.141592653589793L);
  }
  double om = (2.0L * at);
  if ((det > 0.0L)) {
    return om;
  }
  return (-om);
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  double3 pt = double3(pts[(i * 3u)], pts[((i * 3u) + 1u)], pts[((i * 3u) + 2u)]);
  double s = 0.0L;
  for (uint t = 0u; t < params.nT; ++t) {
    s = (s + solid_angle(vdir((double3(cage[(tris[((t * 3u) + 0u)] * 3u)], cage[((tris[((t * 3u) + 0u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 0u)] * 3u) + 2u)]) - pt)), vdir((double3(cage[(tris[((t * 3u) + 1u)] * 3u)], cage[((tris[((t * 3u) + 1u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 1u)] * 3u) + 2u)]) - pt)), vdir((double3(cage[(tris[((t * 3u) + 2u)] * 3u)], cage[((tris[((t * 3u) + 2u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 2u)] * 3u) + 2u)]) - pt))));
  }
  w[i] = (s / (4.0L * 3.141592653589793L));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.Winding
