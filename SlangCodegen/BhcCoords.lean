import Cage.SlangCodegen.Common
import Cage.SlangCodegen.BindSamples

/-!
# `Cage.SlangCodegen.BhcCoords` — unconstrained harmonic and biharmonic coordinates

One thread per point `eta`, every cage triangle in order (BHC.h's
`computeCoordinates` and `computeCoordinatesOnCageTriangles`). The row
of point i, `stride = 2K` doubles with `K = nV + nT`:

    [ h_φ (nV) | h_ψ (nT) | bh_φ (nV) | bh_ψ (nT) ]

h_φ and bh_φ accumulate each triangle's three corner terms onto the
corner's vertex, in triangle order then corner order, from 0.

A constraint sample lies on its own triangle `own[i]` (`0xFFFFFFFF`:
none, a mesh point). On that triangle the harmonic part is
`h_coordinates_inside_triangle`, bh_ψ is `bh_psi_inside_triangle` and
bh_φ is 0; and the sample's mass `γ` is subtracted from h_φ at the three
corners once every triangle is in. The row is then `[B | A]` of
`computeConstrainedBiharmonicMatrices_13`: `B = (Φ_L − M | Ψ_L)`,
`A = (bh_φ_V | bh_ψ_V)`. Its right-hand side `R = (M − H_Φ_V | −H_Ψ_V)`
is `−B` exactly, because the Laplacian and the Dirichlet samples are the
same points (both subdivide `triangle_Laplacian_subdiv_count` times) and
their harmonic coordinates the same expressions.

Bindings (set 0):

  0  ConstantBuffer<CageBhcParams> { uint P; uint nV; uint nT; uint stride; }
  1  StructuredBuffer<double>   pts    (3 P)
  2  StructuredBuffer<double>   cage   (3 nV)
  3  StructuredBuffer<uint>     tris   (3 nT)
  4  StructuredBuffer<uint>     own    (P)
  5  StructuredBuffer<double>   gamma  (3 P; read where own != none)
  6  RWStructuredBuffer<double> rows   (P stride)
-/

namespace Cage.SlangCodegen.BhcCoords

open LeanSlang
open Drape.SlangCodegen.Dsl
open Cage.SlangCodegen.Common
open Cage.SlangCodegen.BindSamples (roD rwD)

def none_ : E := u 4294967295

def vertAt (k : Nat) : E :=
  let c := at_ "tris" (v "t" * u 3 + u k) * u 3
  d3 (at_ "cage" c) (at_ "cage" (c + u 1)) (at_ "cage" (c + u 2))

def addAt (i : E) (x : E) : St := setAt "rows" i (at_ "rows" i + x)

def shader : SlangShaderModule :=
  { structs := [ { name := "CageBhcParams", fields := [fld "P" uT, fld "nV" uT, fld "nT" uT, fld "stride" uT] } ]
  , globals :=
      [ paramsCB "CageBhcParams", roD "pts" 1, roD "cage" 2, roU "tris" 3, roU "own" 4, roD "gamma" 5,
        rwD "rows" 6 ]
  , functions := bhcHelpers ++
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "P")) [ ret ]
          , let_ uT "base" (v "i" * p "stride")
          , let_ uT "K" (p "nV" + p "nT")
          , for_ "k" (u 0) (p "stride") [ setAt "rows" (v "base" + v "k") (dl 0.0) ]
          , let_ d3T "eta" (d3 (at_ "pts" (v "i" * u 3)) (at_ "pts" (v "i" * u 3 + u 1)) (at_ "pts" (v "i" * u 3 + u 2)))
          , let_ uT "ot" (at_ "own" (v "i"))
          , for_ "t" (u 0) (p "nT")
              [ let_ uT "a" (at_ "tris" (v "t" * u 3))
              , let_ uT "b" (at_ "tris" (v "t" * u 3 + u 1))
              , let_ uT "c" (at_ "tris" (v "t" * u 3 + u 2))
              , let_ d3T "v0" (vertAt 0)
              , let_ d3T "v1" (vertAt 1)
              , let_ d3T "v2" (vertAt 2)
              , let_ bT "ins" (eq (v "ot") (v "t"))
              , decl dT "hpsi"
              , decl d3T "hphi"
              , do_ (call "h_coords" [v "eta", v "v0", v "v1", v "v2", v "ins", v "hpsi", v "hphi"])
              , let_ dT "bpsi" (call "bh_psi" [v "eta", v "v0", v "v1", v "v2", v "ins"])
              , let_ d3T "bphi" (d3 (dl 0.0) (dl 0.0) (dl 0.0))
              , if_ (not_ (v "ins")) [ do_ (call "bh_phi" [v "eta", v "v0", v "v1", v "v2", v "hpsi", v "bphi"]) ]
              , setAt "rows" (v "base" + p "nV" + v "t") (v "hpsi")
              , addAt (v "base" + v "a") (cx (v "hphi"))
              , addAt (v "base" + v "b") (cy (v "hphi"))
              , addAt (v "base" + v "c") (cz (v "hphi"))
              , setAt "rows" (v "base" + v "K" + p "nV" + v "t") (v "bpsi")
              , addAt (v "base" + v "K" + v "a") (cx (v "bphi"))
              , addAt (v "base" + v "K" + v "b") (cy (v "bphi"))
              , addAt (v "base" + v "K" + v "c") (cz (v "bphi")) ]
          , if_ (ne (v "ot") none_)
              [ let_ uT "oa" (at_ "tris" (v "ot" * u 3))
              , let_ uT "ob" (at_ "tris" (v "ot" * u 3 + u 1))
              , let_ uT "oc" (at_ "tris" (v "ot" * u 3 + u 2))
              , addAt (v "base" + v "oa") (-(at_ "gamma" (v "i" * u 3)))
              , addAt (v "base" + v "ob") (-(at_ "gamma" (v "i" * u 3 + u 1)))
              , addAt (v "base" + v "oc") (-(at_ "gamma" (v "i" * u 3 + u 2))) ] ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageBhcParams {
  uint P;
  uint nV;
  uint nT;
  uint stride;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageBhcParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> pts;
[[vk::binding(2, 0)]]
StructuredBuffer<double> cage;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> own;
[[vk::binding(5, 0)]]
StructuredBuffer<double> gamma;
[[vk::binding(6, 0)]]
RWStructuredBuffer<double> rows;

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

double dlog(double x) {
  int e = 0;
  double m = frexp(x, e);
  if ((m < 0.7071067811865476L)) {
    m = (m * 2.0L);
    e = (e - 1);
  }
  double s = ((m - 1.0L) / (m + 1.0L));
  double s2 = (s * s);
  double p = 0.04L;
  p = (asdouble(0x590B2164u, 0x3FA642C8u) + (s2 * p));
  p = (asdouble(0x18618618u, 0x3FA86186u) + (s2 * p));
  p = (0.05263157894736842L + (s2 * p));
  p = (asdouble(0x1E1E1E1Eu, 0x3FAE1E1Eu) + (s2 * p));
  p = (0.06666666666666667L + (s2 * p));
  p = (0.07692307692307693L + (s2 * p));
  p = (0.09090909090909091L + (s2 * p));
  p = (0.1111111111111111L + (s2 * p));
  p = (0.14285714285714285L + (s2 * p));
  p = (0.2L + (s2 * p));
  p = (0.3333333333333333L + (s2 * p));
  p = (1.0L + (s2 * p));
  return (((2.0L * s) * p) + (double(e) * 0.6931471805599453L));
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

void h_coords(double3 eta, double3 t0, double3 t1, double3 t2, bool inside, out double psi, out double3 phi) {
  double3 Nt = vcross((t1 - t0), (t2 - t0));
  double NtNorm = vnorm(Nt);
  double At = (NtNorm / 2.0L);
  Nt = (Nt / NtNorm);
  double3 e0 = (t0 - eta);
  double3 e1 = (t1 - eta);
  double3 e2 = (t2 - eta);
  double en0 = vnorm(e0);
  double en1 = vnorm(e1);
  double en2 = vnorm(e2);
  double3 u0 = (e0 / en0);
  double3 u1 = (e1 / en1);
  double3 u2 = (e2 / en2);
  double so = 0.5L;
  if ((!inside)) {
    so = (solid_angle(u0, u1, u2) / (4.0L * 3.141592653589793L));
  }
  double sv = (vdot(vcross(e0, e1), e2) / 6.0L);
  double R0 = (en1 + en2);
  double R1 = (en2 + en0);
  double R2 = (en0 + en1);
  double3 d0 = (t1 - t2);
  double3 d1 = (t2 - t0);
  double3 d2 = (t0 - t1);
  double dn0 = vnorm(d0);
  double dn1 = vnorm(d1);
  double dn2 = vnorm(d2);
  double C0 = (dlog(((R0 + dn0) / (R0 - dn0))) / ((4.0L * 3.141592653589793L) * dn0));
  double C1 = (dlog(((R1 + dn1) / (R1 - dn1))) / ((4.0L * 3.141592653589793L) * dn1));
  double C2 = (dlog(((R2 + dn2) / (R2 - dn2))) / ((4.0L * 3.141592653589793L) * dn2));
  double3 Pt = ((-so) * Nt);
  Pt = (Pt + vcross(Nt, (C0 * d0)));
  Pt = (Pt + vcross(Nt, (C1 * d1)));
  Pt = (Pt + vcross(Nt, (C2 * d2)));
  double3 J0 = vcross(e2, e1);
  double3 J1 = vcross(e0, e2);
  double3 J2 = vcross(e1, e0);
  double ps = 0.0L;
  if ((!inside)) {
    ps = ((((-3.0L) * so) * sv) / At);
  }
  ps = (ps - (C0 * vdot(J0, Nt)));
  ps = (ps - (C1 * vdot(J1, Nt)));
  ps = (ps - (C2 * vdot(J2, Nt)));
  psi = ps;
  phi = double3((vdot(Pt, J0) / (2.0L * At)), (vdot(Pt, J1) / (2.0L * At)), (vdot(Pt, J2) / (2.0L * At)));
  return;
}

double bh_sub(double3 eta, double3 v0, double3 v1, double3 n, double d) {
  double3 ue = vdir((v1 - v0));
  double z0 = vdot((eta - v0), ue);
  double z1 = vdot((eta - v1), ue);
  double a = vdot(vcross(n, ue), (eta - v0));
  double3 w = (eta - v0);
  double3 Dv = double3(((((1.0L - (ue.x * ue.x)) * w.x) + ((0.0L - (ue.x * ue.y)) * w.y)) + ((0.0L - (ue.x * ue.z)) * w.z)), ((((0.0L - (ue.y * ue.x)) * w.x) + ((1.0L - (ue.y * ue.y)) * w.y)) + ((0.0L - (ue.y * ue.z)) * w.z)), ((((0.0L - (ue.z * ue.x)) * w.x) + ((0.0L - (ue.z * ue.y)) * w.y)) + ((1.0L - (ue.z * ue.z)) * w.z)));
  double D2 = vdot(Dv, Dv);
  double le0 = vnorm((v0 - eta));
  double le1 = vnorm((v1 - eta));
  double al = (((2.0L * d) * d) + D2);
  if ((al < 1.0e-11L)) {
    return 0.0L;
  }
  return (((0.5L * a) * ((((((2.0L * d) * d) + D2) * dlog(((le1 - z1) / (le0 - z0)))) + (le0 * z0)) - (le1 * z1))) / 3.0L);
}

double bh_psi(double3 eta, double3 v0, double3 v1, double3 v2, bool inside) {
  double3 n = vdir(vcross((v1 - v0), (v2 - v0)));
  double d = vdot((eta - v0), n);
  double s = ((bh_sub(eta, v0, v1, n, d) + bh_sub(eta, v1, v2, n, d)) + bh_sub(eta, v2, v0, n, d));
  if ((!inside)) {
    double om = solid_angle((v2 - eta), (v0 - eta), (v1 - eta));
    s = (s + ((((d * d) * d) * om) / 3.0L));
  }
  return ((1.0L / (8.0L * 3.141592653589793L)) * s);
}

double3 bh_grad_sub(double3 eta, double3 v0, double3 v1, double d, double3 n) {
  double3 ue = vdir((v1 - v0));
  double3 re = vcross(n, ue);
  double z0 = vdot((eta - v0), ue);
  double z1 = vdot((eta - v1), ue);
  double l0 = vnorm((v0 - eta));
  double l1 = vnorm((v1 - eta));
  double el = vnorm((v1 - v0));
  double3 u0 = vdir((eta - v0));
  double3 u1 = vdir((eta - v1));
  double ae = vdot(re, (eta - v0));
  double3 w = (eta - v0);
  double3 Dv = double3(((((1.0L - (ue.x * ue.x)) * w.x) + ((0.0L - (ue.x * ue.y)) * w.y)) + ((0.0L - (ue.x * ue.z)) * w.z)), ((((0.0L - (ue.y * ue.x)) * w.x) + ((1.0L - (ue.y * ue.y)) * w.y)) + ((0.0L - (ue.y * ue.z)) * w.z)), ((((0.0L - (ue.z * ue.x)) * w.x) + ((0.0L - (ue.z * ue.y)) * w.y)) + ((1.0L - (ue.z * ue.z)) * w.z)));
  double D2 = vdot(Dv, Dv);
  double lg = dlog(((l1 - z1) / (l0 - z0)));
  double3 g = ((((-((2.0L * el) * (l0 + l1))) * (((d * d) * d) * d)) * re) / ((((3.0L * ((l0 + l1) + el)) * ((l0 + l1) - el)) * l0) * l1));
  g = (g + ((((((3.0L * D2) * lg) - (l1 * z1)) + (l0 * z0)) * re) / 6.0L));
  g = (g + (((ae / 6.0L) * (((2.0L * d) * d) + D2)) * (((u1 - ue) / (l1 - z1)) - ((u0 - ue) / (l0 - z0)))));
  g = (g + ((ae / 6.0L) * (((z0 * u0) - (z1 * u1)) + ((l0 - l1) * ue))));
  return g;
}

double3 bh_grad_tp(double3 eta, double3 v0, double3 v1, double3 v2) {
  double3 n = vdir(vcross((v1 - v0), (v2 - v0)));
  double d = vdot((eta - v0), n);
  return ((1.0L / (8.0L * 3.141592653589793L)) * ((bh_grad_sub(eta, v0, v1, d, n) + bh_grad_sub(eta, v1, v2, d, n)) + bh_grad_sub(eta, v2, v0, d, n)));
}

void bh_phi(double3 eta, double3 v0, double3 v1, double3 v2, double hpsi, out double3 phi) {
  double3 n = vdir(vcross((v1 - v0), (v2 - v0)));
  double d = vdot((eta - v0), n);
  double3 r0 = vcross(n, (v2 - v1));
  r0 = (r0 / vdot(r0, (v0 - v1)));
  double3 r1 = vcross(n, (v0 - v2));
  r1 = (r1 / vdot(r1, (v1 - v2)));
  double3 r2 = vcross(n, (v1 - v0));
  r2 = (r2 / vdot(r2, (v2 - v0)));
  double g0 = vdot((eta - v1), r0);
  double g1 = vdot((eta - v2), r1);
  double g2 = vdot((eta - v0), r2);
  if ((abs(d) < 1.0e-8L)) {
    if (((((g0 < 0.0L) || (g0 > 1.0L)) || ((g1 < 0.0L) || (g1 > 1.0L))) || ((g2 < 0.0L) || (g2 > 1.0L)))) {
      phi = double3(0.0L, 0.0L, 0.0L);
      return;
    }
  }
  double3 gr = bh_grad_tp(eta, v0, v1, v2);
  phi = double3((((-d) * vdot(gr, r0)) + (((0.5L * d) * g0) * hpsi)), (((-d) * vdot(gr, r1)) + (((0.5L * d) * g1) * hpsi)), (((-d) * vdot(gr, r2)) + (((0.5L * d) * g2) * hpsi)));
  return;
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  uint base = (i * params.stride);
  uint K = (params.nV + params.nT);
  for (uint k = 0u; k < params.stride; ++k) {
    rows[(base + k)] = 0.0L;
  }
  double3 eta = double3(pts[(i * 3u)], pts[((i * 3u) + 1u)], pts[((i * 3u) + 2u)]);
  uint ot = own[i];
  for (uint t = 0u; t < params.nT; ++t) {
    uint a = tris[(t * 3u)];
    uint b = tris[((t * 3u) + 1u)];
    uint c = tris[((t * 3u) + 2u)];
    double3 v0 = double3(cage[(tris[((t * 3u) + 0u)] * 3u)], cage[((tris[((t * 3u) + 0u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 0u)] * 3u) + 2u)]);
    double3 v1 = double3(cage[(tris[((t * 3u) + 1u)] * 3u)], cage[((tris[((t * 3u) + 1u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 1u)] * 3u) + 2u)]);
    double3 v2 = double3(cage[(tris[((t * 3u) + 2u)] * 3u)], cage[((tris[((t * 3u) + 2u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 2u)] * 3u) + 2u)]);
    bool ins = (ot == t);
    double hpsi;
    double3 hphi;
    h_coords(eta, v0, v1, v2, ins, hpsi, hphi);
    double bpsi = bh_psi(eta, v0, v1, v2, ins);
    double3 bphi = double3(0.0L, 0.0L, 0.0L);
    if ((!ins)) {
      bh_phi(eta, v0, v1, v2, hpsi, bphi);
    }
    rows[((base + params.nV) + t)] = hpsi;
    rows[(base + a)] = (rows[(base + a)] + hphi.x);
    rows[(base + b)] = (rows[(base + b)] + hphi.y);
    rows[(base + c)] = (rows[(base + c)] + hphi.z);
    rows[(((base + K) + params.nV) + t)] = bpsi;
    rows[((base + K) + a)] = (rows[((base + K) + a)] + bphi.x);
    rows[((base + K) + b)] = (rows[((base + K) + b)] + bphi.y);
    rows[((base + K) + c)] = (rows[((base + K) + c)] + bphi.z);
  }
  if ((ot != 4294967295u)) {
    uint oa = tris[(ot * 3u)];
    uint ob = tris[((ot * 3u) + 1u)];
    uint oc = tris[((ot * 3u) + 2u)];
    rows[(base + oa)] = (rows[(base + oa)] + (-gamma[(i * 3u)]));
    rows[(base + ob)] = (rows[(base + ob)] + (-gamma[((i * 3u) + 1u)]));
    rows[(base + oc)] = (rows[(base + oc)] + (-gamma[((i * 3u) + 2u)]));
  }
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.BhcCoords
