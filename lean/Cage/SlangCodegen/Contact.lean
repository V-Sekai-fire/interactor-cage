import Cage.SlangCodegen.Normals

/-!
# `Cage.SlangCodegen.Contact` — the in-motion clearance hinge and its gradient

One thread per (frame p, garment vertex i), `z = z_{p,i}` (`cage_lbs`).
The signed distance `d_p(z)` to the body skinned to frame p is the
closest point `q` over every body triangle (Ericson, *Real-Time Collision
Detection* §5.1.5, which also names the feature: the face, one of the
three edges, one of the three vertices), signed by the feature's
pseudonormal (Bærentzen and Aanæs 2005): the face normal; the sum of the
two faces' normals across an edge (`nbr`); the angle-weighted vertex
pseudonormal (`cage_body_vnormals`). The first strictly nearer triangle
wins, so ties go to the lower index on both targets.

    d = s |z − q|,   s = sign((z − q) · pn)  (+1 at |z − q| = 0)
    ∇d = s (z − q) / |z − q|   (pn / |pn| when |z − q| ≤ 1e-12)
    h = max(0, m − d)
    hb = √w_p · h                       (so Σ hb² = Σ_p w_p Σ_i h²)
    g_z = −2 w_p h ∇d

This is the closest-point-with-pseudonormals form of `d_p`; RFD 2277
names fit.elf's brick-grid SDF and the Lean tricubic sampler, which this
base does not carry into a guest outside fit.elf. Brute force over the
body's triangles: a gate-sized body, not an avatar's.

Bindings (set 0):

  0  ConstantBuffer<CageContactParams> { uint F; uint P; uint T; uint BV; float margin; }
  1  StructuredBuffer<float>   z     (3 F P)
  2  StructuredBuffer<float>   bv    (3 F BV)
  3  StructuredBuffer<uint>    tris  (3 T)
  4  StructuredBuffer<float>   fn    (3 F T)
  5  StructuredBuffer<float>   vn    (3 F BV)
  6  StructuredBuffer<uint>    nbr   (3 T; the triangle across edge (k, k+1))
  7  StructuredBuffer<float>   fw    (F)
  8  RWStructuredBuffer<float> hb    (F P)
  9  RWStructuredBuffer<float> gz    (3 F P)
 10  RWStructuredBuffer<float> dist  (F P)
-/

namespace Cage.SlangCodegen.Contact

open LeanSlang
open Drape.SlangCodegen.Dsl

def f3T : SlangType := .vec .float 3
def f3 (a b c : E) : E := call "float3" [a, b, c]
def cx (e : E) : E := .member e "x"
def cy (e : E) : E := .member e "y"
def cz (e : E) : E := .member e "z"
def dotv (a b : E) : E := cx a * cx b + cy a * cy b + cz a * cz b
def retE (e : E) : St := .ret (some e)

/-- Ericson's ClosestPtPointTriangle. Returns the feature: 0 the face,
    1 + k corner k, 4 + k the edge (k, k+1). -/
def closestTri : SlangFunctionDecl :=
  { retType := uT, name := "closest_tri"
  , params := [arg "pt" f3T, arg "a" f3T, arg "b" f3T, arg "c" f3T, arg "q" f3T .qOut]
  , body :=
      [ let_ f3T "ab" (v "b" - v "a")
      , let_ f3T "ac" (v "c" - v "a")
      , let_ f3T "ap" (v "pt" - v "a")
      , let_ fT "d1" (dotv (v "ab") (v "ap"))
      , let_ fT "d2" (dotv (v "ac") (v "ap"))
      , if_ (and_ (le (v "d1") (fl 0.0)) (le (v "d2") (fl 0.0))) [ setv "q" (v "a"), retE (u 1) ]
      , let_ f3T "bp" (v "pt" - v "b")
      , let_ fT "d3" (dotv (v "ab") (v "bp"))
      , let_ fT "d4" (dotv (v "ac") (v "bp"))
      , if_ (and_ (ge (v "d3") (fl 0.0)) (le (v "d4") (v "d3"))) [ setv "q" (v "b"), retE (u 2) ]
      , let_ fT "vc" (v "d1" * v "d4" - v "d3" * v "d2")
      , if_ (and_ (and_ (le (v "vc") (fl 0.0)) (ge (v "d1") (fl 0.0))) (le (v "d3") (fl 0.0)))
          [ let_ fT "t1" (v "d1" / (v "d1" - v "d3"))
          , setv "q" (v "a" + v "t1" * v "ab"), retE (u 4) ]
      , let_ f3T "cp" (v "pt" - v "c")
      , let_ fT "d5" (dotv (v "ab") (v "cp"))
      , let_ fT "d6" (dotv (v "ac") (v "cp"))
      , if_ (and_ (ge (v "d6") (fl 0.0)) (le (v "d5") (v "d6"))) [ setv "q" (v "c"), retE (u 3) ]
      , let_ fT "vb" (v "d5" * v "d2" - v "d1" * v "d6")
      , if_ (and_ (and_ (le (v "vb") (fl 0.0)) (ge (v "d2") (fl 0.0))) (le (v "d6") (fl 0.0)))
          [ let_ fT "t2" (v "d2" / (v "d2" - v "d6"))
          , setv "q" (v "a" + v "t2" * v "ac"), retE (u 6) ]
      , let_ fT "va" (v "d3" * v "d6" - v "d5" * v "d4")
      , if_ (and_ (and_ (le (v "va") (fl 0.0)) (ge (v "d4" - v "d3") (fl 0.0))) (ge (v "d5" - v "d6") (fl 0.0)))
          [ let_ fT "t3" ((v "d4" - v "d3") / ((v "d4" - v "d3") + (v "d5" - v "d6")))
          , setv "q" (v "b" + v "t3" * (v "c" - v "b")), retE (u 5) ]
      , let_ fT "dn" (fl 1.0 / (v "va" + v "vb" + v "vc"))
      , let_ fT "sv" (v "vb" * v "dn")
      , let_ fT "sw" (v "vc" * v "dn")
      , setv "q" (v "a" + v "ab" * v "sv" + v "ac" * v "sw")
      , retE (u 0) ] }

def ld (buf : String) (idx : E) : E :=
  f3 (at_ buf (idx * u 3)) (at_ buf (idx * u 3 + u 1)) (at_ buf (idx * u 3 + u 2))

def shader : SlangShaderModule :=
  { structs := [ { name := "CageContactParams"
                 , fields := [fld "F" uT, fld "P" uT, fld "T" uT, fld "BV" uT, fld "margin" fT] } ]
  , globals := [ paramsCB "CageContactParams", roF "z" 1, roF "bv" 2, roU "tris" 3, roF "fn" 4, roF "vn" 5,
                 roU "nbr" 6, roF "fw" 7, rwF "hb" 8, rwF "gz" 9, rwF "dist" 10 ]
  , functions :=
      [ closestTri
      , entry 64 [dtid]
          [ let_ uT "id" (.member (v "tid") "x")
          , if_ (ge (v "id") (p "F" * p "P")) [ ret ]
          , let_ uT "fp" (v "id" / p "P")
          , let_ uT "o" (v "fp" * p "BV")
          , let_ f3T "pt" (ld "z" (v "id"))
          , let_ fT "best" fltMax
          , let_ uT "bt" (u 0)
          , let_ uT "bf" (u 0)
          , let_ f3T "bq" (v "pt")
          , for_ "t" (u 0) (p "T")
              [ decl f3T "q"
              , let_ uT "ft" (call "closest_tri"
                  [ v "pt", ld "bv" (v "o" + at_ "tris" (v "t" * u 3)), ld "bv" (v "o" + at_ "tris" (v "t" * u 3 + u 1))
                  , ld "bv" (v "o" + at_ "tris" (v "t" * u 3 + u 2)), v "q" ])
              , let_ f3T "dq" (v "pt" - v "q")
              , let_ fT "d2" (dotv (v "dq") (v "dq"))
              , if_ (lt (v "d2") (v "best"))
                  [ setv "best" (v "d2"), setv "bt" (v "t"), setv "bf" (v "ft"), setv "bq" (v "q") ] ]
          , let_ uT "ftb" (v "fp" * p "T" + v "bt")
          , let_ f3T "pn" (ld "fn" (v "ftb"))
          , if_ (and_ (ge (v "bf") (u 1)) (le (v "bf") (u 3)))
              [ setv "pn" (ld "vn" (v "o" + at_ "tris" (v "bt" * u 3 + v "bf" - u 1))) ]
          , if_ (ge (v "bf") (u 4))
              [ setv "pn" (v "pn" + ld "fn" (v "fp" * p "T" + at_ "nbr" (v "bt" * u 3 + v "bf" - u 4))) ]
          , let_ f3T "dv" (v "pt" - v "bq")
          , let_ fT "dl" (call "sqrt" [v "best"])
          , let_ fT "sg" (sel (lt (dotv (v "dv") (v "pn")) (fl 0.0)) (fl (-1.0)) (fl 1.0))
          , let_ f3T "gr" (v "pn" / fmax (call "sqrt" [dotv (v "pn") (v "pn")]) (.litFloatExact 1e-20))
          , if_ (gt (v "dl") (.litFloatExact 1e-12)) [ setv "gr" (v "sg" * v "dv" / v "dl") ]
          , let_ fT "sd" (v "sg" * v "dl")
          , let_ fT "h" (fmax (p "margin" - v "sd") (fl 0.0))
          , let_ fT "wp" (at_ "fw" (v "fp"))
          , setAt "hb" (v "id") (call "sqrt" [v "wp"] * v "h")
          , let_ fT "k" (fl (-2.0) * v "wp" * v "h")
          , setAt "gz" (v "id" * u 3) (v "k" * cx (v "gr"))
          , setAt "gz" (v "id" * u 3 + u 1) (v "k" * cy (v "gr"))
          , setAt "gz" (v "id" * u 3 + u 2) (v "k" * cz (v "gr"))
          , setAt "dist" (v "id") (v "sd") ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageContactParams {
  uint F;
  uint P;
  uint T;
  uint BV;
  float margin;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageContactParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> z;
[[vk::binding(2, 0)]]
StructuredBuffer<float> bv;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(4, 0)]]
StructuredBuffer<float> fn;
[[vk::binding(5, 0)]]
StructuredBuffer<float> vn;
[[vk::binding(6, 0)]]
StructuredBuffer<uint> nbr;
[[vk::binding(7, 0)]]
StructuredBuffer<float> fw;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float> hb;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> gz;
[[vk::binding(10, 0)]]
RWStructuredBuffer<float> dist;

uint closest_tri(float3 pt, float3 a, float3 b, float3 c, out float3 q) {
  float3 ab = (b - a);
  float3 ac = (c - a);
  float3 ap = (pt - a);
  float d1 = (((ab.x * ap.x) + (ab.y * ap.y)) + (ab.z * ap.z));
  float d2 = (((ac.x * ap.x) + (ac.y * ap.y)) + (ac.z * ap.z));
  if (((d1 <= 0.000000) && (d2 <= 0.000000))) {
    q = a;
    return 1u;
  }
  float3 bp = (pt - b);
  float d3 = (((ab.x * bp.x) + (ab.y * bp.y)) + (ab.z * bp.z));
  float d4 = (((ac.x * bp.x) + (ac.y * bp.y)) + (ac.z * bp.z));
  if (((d3 >= 0.000000) && (d4 <= d3))) {
    q = b;
    return 2u;
  }
  float vc = ((d1 * d4) - (d3 * d2));
  if ((((vc <= 0.000000) && (d1 >= 0.000000)) && (d3 <= 0.000000))) {
    float t1 = (d1 / (d1 - d3));
    q = (a + (t1 * ab));
    return 4u;
  }
  float3 cp = (pt - c);
  float d5 = (((ab.x * cp.x) + (ab.y * cp.y)) + (ab.z * cp.z));
  float d6 = (((ac.x * cp.x) + (ac.y * cp.y)) + (ac.z * cp.z));
  if (((d6 >= 0.000000) && (d5 <= d6))) {
    q = c;
    return 3u;
  }
  float vb = ((d5 * d2) - (d1 * d6));
  if ((((vb <= 0.000000) && (d2 >= 0.000000)) && (d6 <= 0.000000))) {
    float t2 = (d2 / (d2 - d6));
    q = (a + (t2 * ac));
    return 6u;
  }
  float va = ((d3 * d6) - (d5 * d4));
  if ((((va <= 0.000000) && ((d4 - d3) >= 0.000000)) && ((d5 - d6) >= 0.000000))) {
    float t3 = ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    q = (b + (t3 * (c - b)));
    return 5u;
  }
  float dn = (1.000000 / ((va + vb) + vc));
  float sv = (vb * dn);
  float sw = (vc * dn);
  q = ((a + (ab * sv)) + (ac * sw));
  return 0u;
}

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.P))) {
    return;
  }
  uint fp = (id / params.P);
  uint o = (fp * params.BV);
  float3 pt = float3(z[(id * 3u)], z[((id * 3u) + 1u)], z[((id * 3u) + 2u)]);
  float best = asfloat(2139095039u);
  uint bt = 0u;
  uint bf = 0u;
  float3 bq = pt;
  for (uint t = 0u; t < params.T; ++t) {
    float3 q;
    uint ft = closest_tri(pt, float3(bv[((o + tris[(t * 3u)]) * 3u)], bv[(((o + tris[(t * 3u)]) * 3u) + 1u)], bv[(((o + tris[(t * 3u)]) * 3u) + 2u)]), float3(bv[((o + tris[((t * 3u) + 1u)]) * 3u)], bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 1u)], bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 2u)]), float3(bv[((o + tris[((t * 3u) + 2u)]) * 3u)], bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 1u)], bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 2u)]), q);
    float3 dq = (pt - q);
    float d2 = (((dq.x * dq.x) + (dq.y * dq.y)) + (dq.z * dq.z));
    if ((d2 < best)) {
      best = d2;
      bt = t;
      bf = ft;
      bq = q;
    }
  }
  uint ftb = ((fp * params.T) + bt);
  float3 pn = float3(fn[(ftb * 3u)], fn[((ftb * 3u) + 1u)], fn[((ftb * 3u) + 2u)]);
  if (((bf >= 1u) && (bf <= 3u))) {
    pn = float3(vn[((o + tris[(((bt * 3u) + bf) - 1u)]) * 3u)], vn[(((o + tris[(((bt * 3u) + bf) - 1u)]) * 3u) + 1u)], vn[(((o + tris[(((bt * 3u) + bf) - 1u)]) * 3u) + 2u)]);
  }
  if ((bf >= 4u)) {
    pn = (pn + float3(fn[(((fp * params.T) + nbr[(((bt * 3u) + bf) - 4u)]) * 3u)], fn[((((fp * params.T) + nbr[(((bt * 3u) + bf) - 4u)]) * 3u) + 1u)], fn[((((fp * params.T) + nbr[(((bt * 3u) + bf) - 4u)]) * 3u) + 2u)]));
  }
  float3 dv = (pt - bq);
  float dl = sqrt(best);
  float sg = (((((dv.x * pn.x) + (dv.y * pn.y)) + (dv.z * pn.z)) < 0.000000) ? -1.000000 : 1.000000);
  float3 gr = (pn / max(sqrt((((pn.x * pn.x) + (pn.y * pn.y)) + (pn.z * pn.z))), 1.0e-20f));
  if ((dl > 1.0e-12f)) {
    gr = ((sg * dv) / dl);
  }
  float sd = (sg * dl);
  float h = max((params.margin - sd), 0.000000);
  float wp = fw[fp];
  hb[id] = (sqrt(wp) * h);
  float k = ((-2.000000 * wp) * h);
  gz[(id * 3u)] = (k * gr.x);
  gz[((id * 3u) + 1u)] = (k * gr.y);
  gz[((id * 3u) + 2u)] = (k * gr.z);
  dist[id] = sd;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.Contact
