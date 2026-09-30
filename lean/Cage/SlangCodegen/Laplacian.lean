import Drape.SlangCodegen.Dsl

/-!
# `Cage.SlangCodegen.Laplacian` — the cage's uniform Laplacian and its transpose

One thread per cage vertex i over 3-vectors, the one-ring in CSR
(`rowptr`, `col`, symmetric), `deg_i = rowptr[i+1] − rowptr[i]`:

    transpose = 0:  (L x)_i  = x_i − (Σ_{j∈N(i)} x_j) / deg_i
    transpose = 1:  (Lᵀx)_i  = x_i − Σ_{j∈N(i)} x_j / deg_j

    combinatorial = 1:  (L x)_i = Σ_{j∈N(i)} (x_j − x_i)   (symmetric: transpose ignored)

    y_i = (accumulate ? y_i : 0) + scale · (L or Lᵀ) x_i

so `‖L u‖²`'s gradient `2 w_L LᵀL u` is two dispatches: `t = L u`, then
`g += (2 w_L) Lᵀ t`. A vertex of degree 0 is its own row (x_i). The
combinatorial form is the G3 host oracle's (tests/cage_oracle).

Bindings (set 0):

  0  ConstantBuffer<CageLapParams> { uint n; uint transpose; uint accumulate; float scale; uint combinatorial; }
  1  StructuredBuffer<uint>    rowptr  (n + 1)
  2  StructuredBuffer<uint>    col     (nnz)
  3  StructuredBuffer<float>   x       (3 n)
  4  RWStructuredBuffer<float> y       (3 n)
-/

namespace Cage.SlangCodegen.Laplacian

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "CageLapParams", fields := [fld "n" uT, fld "transpose" uT, fld "accumulate" uT, fld "scale" fT, fld "combinatorial" uT] } ]
  , globals := [ paramsCB "CageLapParams", roU "rowptr" 1, roU "col" 2, roF "x" 3, rwF "y" 4 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "n")) [ ret ]
          , let_ uT "r0" (at_ "rowptr" (v "i"))
          , let_ uT "r1" (at_ "rowptr" (v "i" + u 1))
          , let_ fT "sx" (fl 0.0)
          , let_ fT "sy" (fl 0.0)
          , let_ fT "sz" (fl 0.0)
          , for_ "k" (v "r0") (v "r1")
              [ let_ uT "j" (at_ "col" (v "k"))
              , let_ fT "wj" (fl 1.0)
              , if_ (and_ (ne (p "transpose") (u 0)) (eq (p "combinatorial") (u 0)))
                  [ setv "wj" (fl 1.0 / toF (at_ "rowptr" (v "j" + u 1) - at_ "rowptr" (v "j"))) ]
              , setv "sx" (v "sx" + v "wj" * at_ "x" (v "j" * u 3))
              , setv "sy" (v "sy" + v "wj" * at_ "x" (v "j" * u 3 + u 1))
              , setv "sz" (v "sz" + v "wj" * at_ "x" (v "j" * u 3 + u 2)) ]
          , if_ (and_ (and_ (eq (p "transpose") (u 0)) (eq (p "combinatorial") (u 0))) (gt (v "r1") (v "r0")))
              [ let_ fT "dg" (toF (v "r1" - v "r0"))
              , setv "sx" (v "sx" / v "dg")
              , setv "sy" (v "sy" / v "dg")
              , setv "sz" (v "sz" / v "dg") ]
          , let_ bT "acc" (ne (p "accumulate") (u 0))
          , let_ fT "rx" (at_ "x" (v "i" * u 3) - v "sx")
          , let_ fT "ry" (at_ "x" (v "i" * u 3 + u 1) - v "sy")
          , let_ fT "rz" (at_ "x" (v "i" * u 3 + u 2) - v "sz")
          , if_ (ne (p "combinatorial") (u 0))
              [ let_ fT "dc" (toF (v "r1" - v "r0"))
              , setv "rx" (v "sx" - v "dc" * at_ "x" (v "i" * u 3))
              , setv "ry" (v "sy" - v "dc" * at_ "x" (v "i" * u 3 + u 1))
              , setv "rz" (v "sz" - v "dc" * at_ "x" (v "i" * u 3 + u 2)) ]
          , setAt "y" (v "i" * u 3) (sel (v "acc") (at_ "y" (v "i" * u 3)) (fl 0.0) + p "scale" * v "rx")
          , setAt "y" (v "i" * u 3 + u 1) (sel (v "acc") (at_ "y" (v "i" * u 3 + u 1)) (fl 0.0) + p "scale" * v "ry")
          , setAt "y" (v "i" * u 3 + u 2) (sel (v "acc") (at_ "y" (v "i" * u 3 + u 2)) (fl 0.0) + p "scale" * v "rz") ] ] }

-- BEGIN PIN
def expected : String :=
"struct CageLapParams {
  uint n;
  uint transpose;
  uint accumulate;
  float scale;
  uint combinatorial;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLapParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> rowptr;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> col;
[[vk::binding(3, 0)]]
StructuredBuffer<float> x;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> y;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  uint r0 = rowptr[i];
  uint r1 = rowptr[(i + 1u)];
  float sx = 0.000000;
  float sy = 0.000000;
  float sz = 0.000000;
  for (uint k = r0; k < r1; ++k) {
    uint j = col[k];
    float wj = 1.000000;
    if (((params.transpose != 0u) && (params.combinatorial == 0u))) {
      wj = (1.000000 / float((rowptr[(j + 1u)] - rowptr[j])));
    }
    sx = (sx + (wj * x[(j * 3u)]));
    sy = (sy + (wj * x[((j * 3u) + 1u)]));
    sz = (sz + (wj * x[((j * 3u) + 2u)]));
  }
  if ((((params.transpose == 0u) && (params.combinatorial == 0u)) && (r1 > r0))) {
    float dg = float((r1 - r0));
    sx = (sx / dg);
    sy = (sy / dg);
    sz = (sz / dg);
  }
  bool acc = (params.accumulate != 0u);
  float rx = (x[(i * 3u)] - sx);
  float ry = (x[((i * 3u) + 1u)] - sy);
  float rz = (x[((i * 3u) + 2u)] - sz);
  if ((params.combinatorial != 0u)) {
    float dc = float((r1 - r0));
    rx = (sx - (dc * x[(i * 3u)]));
    ry = (sy - (dc * x[((i * 3u) + 1u)]));
    rz = (sz - (dc * x[((i * 3u) + 2u)]));
  }
  y[(i * 3u)] = ((acc ? y[(i * 3u)] : 0.000000) + (params.scale * rx));
  y[((i * 3u) + 1u)] = ((acc ? y[((i * 3u) + 1u)] : 0.000000) + (params.scale * ry));
  y[((i * 3u) + 2u)] = ((acc ? y[((i * 3u) + 2u)] : 0.000000) + (params.scale * rz));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Cage.SlangCodegen.Laplacian
