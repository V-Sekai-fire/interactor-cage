import Drape.SlangCodegen.Dsl

/-!
# `Cage.SlangCodegen.Common` — double-precision helpers for the (1,3) bind

Not a kernel. The Slang helper functions the bind kernels
(`Cage.SlangCodegen.BhcCoords`, …) emit ahead of their entry point:

* 3-vector algebra on `double3`, written out component by component in
  the order `point3.h` of the reference evaluates it (`vdot`, `vcross`,
  `vnorm`, `vdir`), so the kernels round like BHC.h does;
* `dlog` and `datan2` in double from `+ − × ÷`, `sqrt` and `frexp` only.
  SPIR-V's GLSL.std.450 `Log` and `Atan2` take 16- and 32-bit floats
  only (`spirv-val`: "expected Result Type to be a 16 or 32-bit scalar"),
  so the two transcendentals the coordinates need are series here and
  both targets compute the same thing:
  - `dlog x`: `x = m·2^e` (`frexp`), `m ∈ [√½, √2)`, `s = (m−1)/(m+1)`,
    `log m = 2 atanh s = 2 Σ_{k≤12} s^{2k+1}/(2k+1)`; `|s| ≤ 0.1716`, so
    the first dropped term is below 2⁻⁵⁶ relative.
  - `datan01 z` (`z ∈ [0,1]`): two half-angle reductions
    `atan z = 2 atan(z / (1 + √(1+z²)))` bring `z` below `tan(π/16)`,
    then 14 terms of the alternating series.
  - `datan2 y x` from `datan01` by octant, as `atan2` for finite inputs
    (`atan2(±0, +0) = 0`).
* the per-triangle terms of Thiery, Michel and Chen's biharmonic
  coordinates for triangular 3D cages (SIGGRAPH 2024), as `BHC.h` of
  V-Sekai-fire/interactor-tool-godot-cage-deformer (MIT) writes them:
  `solid_angle` (`get_signed_solid_angle`), `h_coords` (`h_coordinates`
  and, with `inside`, `h_coordinates_inside_triangle`), `bh_psi`
  (`bh_psi`, `bh_psi_inside_triangle`), `bh_grad_tp`
  (`bh_psi_gradient_in_tangent_plane`) and `bh_phi` (`bh_phi`, which
  takes the harmonic ψ that `h_coords` already computed: `h_psi` in
  BHC.h is the same expression).
-/

namespace Cage.SlangCodegen.Common

open LeanSlang
open Drape.SlangCodegen.Dsl

def dT : SlangType := .scalar .double
def d3T : SlangType := .vec .double 3

/-- An exact double literal. -/
def dl (x : Float) : E := .litDoubleExact x
def toD (e : E) : E := .cast dT e
def cx (e : E) : E := .member e "x"
def cy (e : E) : E := .member e "y"
def cz (e : E) : E := .member e "z"
def d3 (a b c : E) : E := call "double3" [a, b, c]
def retE (e : E) : St := .ret (some e)

/-- A helper (non-entry) function. -/
def fn (ret : SlangType) (name : String) (params : List SlangBinding) (body : List St) : SlangFunctionDecl :=
  { retType := ret, name := name, params := params, body := body }

def pD (n : String) : SlangBinding := arg n dT
def p3 (n : String) : SlangBinding := arg n d3T

/-- π and ln 2 as the nearest doubles (M_PI, M_LN2). -/
def piD : E := dl 3.141592653589793
def ln2D : E := dl 0.6931471805599453
def sqrtHalfD : E := dl 0.7071067811865476

def vdot : SlangFunctionDecl :=
  fn dT "vdot" [p3 "a", p3 "b"]
    [ retE (cx (v "a") * cx (v "b") + cy (v "a") * cy (v "b") + cz (v "a") * cz (v "b")) ]

def vcross : SlangFunctionDecl :=
  fn d3T "vcross" [p3 "a", p3 "b"]
    [ retE (d3 (cy (v "a") * cz (v "b") - cz (v "a") * cy (v "b"))
               (cz (v "a") * cx (v "b") - cx (v "a") * cz (v "b"))
               (cx (v "a") * cy (v "b") - cy (v "a") * cx (v "b"))) ]

def vnorm : SlangFunctionDecl :=
  fn dT "vnorm" [p3 "a"]
    [ retE (call "sqrt" [cx (v "a") * cx (v "a") + cy (v "a") * cy (v "a") + cz (v "a") * cz (v "a")]) ]

/-- `point3::direction()`: each component divided by the norm. -/
def vdir : SlangFunctionDecl :=
  fn d3T "vdir" [p3 "a"]
    [ let_ dT "n" (call "vnorm" [v "a"])
    , retE (d3 (cx (v "a") / v "n") (cy (v "a") / v "n") (cz (v "a") / v "n")) ]

/-- `1/(2k+1)` for the series. -/
private def inv (k : Nat) : E := dl (1.0 / (2.0 * k.toFloat + 1.0))

def dlog : SlangFunctionDecl :=
  let horner : List St := (List.range 12).reverse.map fun k =>
    setv "p" (inv k + v "s2" * v "p")
  fn dT "dlog" [pD "x"]
    ([ let_ iT "e" (.litInt 0)
     , let_ dT "m" (call "frexp" [v "x", v "e"])
     , if_ (lt (v "m") sqrtHalfD)
         [ setv "m" (v "m" * dl 2.0), setv "e" (v "e" - .litInt 1) ]
     , let_ dT "s" ((v "m" - dl 1.0) / (v "m" + dl 1.0))
     , let_ dT "s2" (v "s" * v "s")
     , let_ dT "p" (inv 12) ] ++ horner ++
     [ retE (dl 2.0 * v "s" * v "p" + toD (v "e") * ln2D) ])

/-- `(−1)^k/(2k+1)`. -/
private def alt (k : Nat) : E := dl ((if k % 2 == 0 then 1.0 else -1.0) / (2.0 * k.toFloat + 1.0))

def datan01 : SlangFunctionDecl :=
  let horner : List St := (List.range 13).reverse.map fun k =>
    setv "p" (alt k + v "t2" * v "p")
  fn dT "datan01" [pD "z"]
    ([ let_ dT "t" (v "z" / (dl 1.0 + call "sqrt" [dl 1.0 + v "z" * v "z"]))
     , setv "t" (v "t" / (dl 1.0 + call "sqrt" [dl 1.0 + v "t" * v "t"]))
     , let_ dT "t2" (v "t" * v "t")
     , let_ dT "p" (alt 13) ] ++ horner ++
     [ retE (dl 4.0 * v "t" * v "p") ])

def datan2 : SlangFunctionDecl :=
  fn dT "datan2" [pD "y", pD "x"]
    [ let_ dT "ax" (fabs (v "x"))
    , let_ dT "ay" (fabs (v "y"))
    , let_ dT "lo" (fmin (v "ax") (v "ay"))
    , let_ dT "hi" (fmax (v "ax") (v "ay"))
    , let_ dT "r" (dl 0.0)
    , if_ (gt (v "hi") (dl 0.0)) [ setv "r" (call "datan01" [v "lo" / v "hi"]) ]
    , if_ (gt (v "ay") (v "ax")) [ setv "r" (piD / dl 2.0 - v "r") ]
    , if_ (lt (v "x") (dl 0.0)) [ setv "r" (piD - v "r") ]
    , if_ (lt (v "y") (dl 0.0)) [ setv "r" (-(v "r")) ]
    , retE (v "r") ]

/-- `get_signed_solid_angle`. -/
def solidAngle : SlangFunctionDecl :=
  fn dT "solid_angle" [p3 "a", p3 "b", p3 "c"]
    [ let_ dT "det" (call "vdot" [v "a", call "vcross" [v "b", v "c"]])
    , if_ (lt (fabs (v "det")) (dl 1e-10)) [ retE (dl 0.0) ]
    , let_ dT "al" (call "vnorm" [v "a"])
    , let_ dT "bl" (call "vnorm" [v "b"])
    , let_ dT "cl" (call "vnorm" [v "c"])
    , let_ dT "dv" (v "al" * v "bl" * v "cl" + call "vdot" [v "a", v "b"] * v "cl"
                    + call "vdot" [v "a", v "c"] * v "bl" + call "vdot" [v "b", v "c"] * v "al")
    , let_ dT "at" (call "datan2" [fabs (v "det"), v "dv"])
    , if_ (lt (v "at") (dl 0.0)) [ setv "at" (v "at" + piD) ]
    , let_ dT "om" (dl 2.0 * v "at")
    , if_ (gt (v "det") (dl 0.0)) [ retE (v "om") ]
    , retE (-(v "om")) ]

/-- `h_coordinates` (and, with `inside`, `h_coordinates_inside_triangle`):
    the harmonic ψ of triangle (t0, t1, t2) and its three φ at `eta`. -/
def hCoords : SlangFunctionDecl :=
  let e (k : Nat) := v ("e" ++ toString k)
  let en (k : Nat) := v ("en" ++ toString k)
  let tv (k : Nat) := v ("t" ++ toString k)
  let dd (k : Nat) := v ("d" ++ toString k)
  let dn (k : Nat) := v ("dn" ++ toString k)
  let cc (k : Nat) := v ("C" ++ toString k)
  let jj (k : Nat) := v ("J" ++ toString k)
  let r3 := [0, 1, 2]
  fn (.named "void") "h_coords"
    [p3 "eta", p3 "t0", p3 "t1", p3 "t2", arg "inside" bT, arg "psi" dT .qOut, arg "phi" d3T .qOut]
    ([ let_ d3T "Nt" (call "vcross" [v "t1" - v "t0", v "t2" - v "t0"])
     , let_ dT "NtNorm" (call "vnorm" [v "Nt"])
     , let_ dT "At" (v "NtNorm" / dl 2.0)
     , setv "Nt" (v "Nt" / v "NtNorm") ] ++
     r3.map (fun k => let_ d3T ("e" ++ toString k) (tv k - v "eta")) ++
     r3.map (fun k => let_ dT ("en" ++ toString k) (call "vnorm" [e k])) ++
     r3.map (fun k => let_ d3T ("u" ++ toString k) (e k / en k)) ++
     [ let_ dT "so" (dl 0.5)
     , if_ (not_ (v "inside"))
         [ setv "so" (call "solid_angle" [v "u0", v "u1", v "u2"] / (dl 4.0 * piD)) ]
     , let_ dT "sv" (call "vdot" [call "vcross" [e 0, e 1], e 2] / dl 6.0) ] ++
     r3.map (fun k => let_ dT ("R" ++ toString k) (en ((k + 1) % 3) + en ((k + 2) % 3))) ++
     r3.map (fun k => let_ d3T ("d" ++ toString k) (tv ((k + 1) % 3) - tv ((k + 2) % 3))) ++
     r3.map (fun k => let_ dT ("dn" ++ toString k) (call "vnorm" [dd k])) ++
     r3.map (fun k =>
       let R := v ("R" ++ toString k)
       let_ dT ("C" ++ toString k)
         (call "dlog" [(R + dn k) / (R - dn k)] / (dl 4.0 * piD * dn k))) ++
     [ let_ d3T "Pt" ((-(v "so")) * v "Nt") ] ++
     r3.map (fun k => setv "Pt" (v "Pt" + call "vcross" [v "Nt", cc k * dd k])) ++
     r3.map (fun k => let_ d3T ("J" ++ toString k) (call "vcross" [e ((k + 2) % 3), e ((k + 1) % 3)])) ++
     [ let_ dT "ps" (dl 0.0)
     , if_ (not_ (v "inside")) [ setv "ps" (dl (-3.0) * v "so" * v "sv" / v "At") ] ] ++
     r3.map (fun k => setv "ps" (v "ps" - cc k * call "vdot" [jj k, v "Nt"])) ++
     [ setv "psi" (v "ps")
     , setv "phi" (d3 (call "vdot" [v "Pt", jj 0] / (dl 2.0 * v "At"))
                      (call "vdot" [v "Pt", jj 1] / (dl 2.0 * v "At"))
                      (call "vdot" [v "Pt", jj 2] / (dl 2.0 * v "At")))
     , ret ])

/-- `bh_psi_subroutine_without_solid_angle`. -/
def bhSub : SlangFunctionDecl :=
  let m (i j : E) (i' j' : Nat) : E :=
    if i' == j' then dl 1.0 - i * j else dl 0.0 - i * j
  let row (i : Nat) : E :=
    let ui := [cx (v "ue"), cy (v "ue"), cz (v "ue")]
    let ws := [cx (v "w"), cy (v "w"), cz (v "w")]
    let t (j : Nat) := m (ui.getD i (dl 0.0)) (ui.getD j (dl 0.0)) i j * ws.getD j (dl 0.0)
    t 0 + t 1 + t 2
  fn dT "bh_sub" [p3 "eta", p3 "v0", p3 "v1", p3 "n", pD "d"]
    [ let_ d3T "ue" (call "vdir" [v "v1" - v "v0"])
    , let_ dT "z0" (call "vdot" [v "eta" - v "v0", v "ue"])
    , let_ dT "z1" (call "vdot" [v "eta" - v "v1", v "ue"])
    , let_ dT "a" (call "vdot" [call "vcross" [v "n", v "ue"], v "eta" - v "v0"])
    , let_ d3T "w" (v "eta" - v "v0")
    , let_ d3T "Dv" (d3 (row 0) (row 1) (row 2))
    , let_ dT "D2" (call "vdot" [v "Dv", v "Dv"])
    , let_ dT "le0" (call "vnorm" [v "v0" - v "eta"])
    , let_ dT "le1" (call "vnorm" [v "v1" - v "eta"])
    , let_ dT "al" (dl 2.0 * v "d" * v "d" + v "D2")
    , if_ (lt (v "al") (dl 1e-11)) [ retE (dl 0.0) ]
    , retE (dl 0.5 * v "a" * ((dl 2.0 * v "d" * v "d" + v "D2")
              * call "dlog" [(v "le1" - v "z1") / (v "le0" - v "z0")]
              + v "le0" * v "z0" - v "le1" * v "z1") / dl 3.0) ]

/-- `bh_psi` (and, with `inside`, `bh_psi_inside_triangle`). -/
def bhPsi : SlangFunctionDecl :=
  fn dT "bh_psi" [p3 "eta", p3 "v0", p3 "v1", p3 "v2", arg "inside" bT]
    [ let_ d3T "n" (call "vdir" [call "vcross" [v "v1" - v "v0", v "v2" - v "v0"]])
    , let_ dT "d" (call "vdot" [v "eta" - v "v0", v "n"])
    , let_ dT "s" (call "bh_sub" [v "eta", v "v0", v "v1", v "n", v "d"]
                  + call "bh_sub" [v "eta", v "v1", v "v2", v "n", v "d"]
                  + call "bh_sub" [v "eta", v "v2", v "v0", v "n", v "d"])
    , if_ (not_ (v "inside"))
        [ let_ dT "om" (call "solid_angle" [v "v2" - v "eta", v "v0" - v "eta", v "v1" - v "eta"])
        , setv "s" (v "s" + v "d" * v "d" * v "d" * v "om" / dl 3.0) ]
    , retE (dl 1.0 / (dl 8.0 * piD) * v "s") ]

/-- `bh_psi_gradient_in_tangent_plane_subroutine`. -/
def bhGradSub : SlangFunctionDecl :=
  let m (i j : E) (i' j' : Nat) : E :=
    if i' == j' then dl 1.0 - i * j else dl 0.0 - i * j
  let row (i : Nat) : E :=
    let ui := [cx (v "ue"), cy (v "ue"), cz (v "ue")]
    let ws := [cx (v "w"), cy (v "w"), cz (v "w")]
    let t (j : Nat) := m (ui.getD i (dl 0.0)) (ui.getD j (dl 0.0)) i j * ws.getD j (dl 0.0)
    t 0 + t 1 + t 2
  fn d3T "bh_grad_sub" [p3 "eta", p3 "v0", p3 "v1", pD "d", p3 "n"]
    [ let_ d3T "ue" (call "vdir" [v "v1" - v "v0"])
    , let_ d3T "re" (call "vcross" [v "n", v "ue"])
    , let_ dT "z0" (call "vdot" [v "eta" - v "v0", v "ue"])
    , let_ dT "z1" (call "vdot" [v "eta" - v "v1", v "ue"])
    , let_ dT "l0" (call "vnorm" [v "v0" - v "eta"])
    , let_ dT "l1" (call "vnorm" [v "v1" - v "eta"])
    , let_ dT "el" (call "vnorm" [v "v1" - v "v0"])
    , let_ d3T "u0" (call "vdir" [v "eta" - v "v0"])
    , let_ d3T "u1" (call "vdir" [v "eta" - v "v1"])
    , let_ dT "ae" (call "vdot" [v "re", v "eta" - v "v0"])
    , let_ d3T "w" (v "eta" - v "v0")
    , let_ d3T "Dv" (d3 (row 0) (row 1) (row 2))
    , let_ dT "D2" (call "vdot" [v "Dv", v "Dv"])
    , let_ dT "lg" (call "dlog" [(v "l1" - v "z1") / (v "l0" - v "z0")])
    , let_ d3T "g" ((-(dl 2.0 * v "el" * (v "l0" + v "l1"))) * (v "d" * v "d" * v "d" * v "d") * v "re"
                    / (dl 3.0 * (v "l0" + v "l1" + v "el") * (v "l0" + v "l1" - v "el") * v "l0" * v "l1"))
    , setv "g" (v "g" + (dl 3.0 * v "D2" * v "lg" - v "l1" * v "z1" + v "l0" * v "z0") * v "re" / dl 6.0)
    , setv "g" (v "g" + v "ae" / dl 6.0 * (dl 2.0 * v "d" * v "d" + v "D2")
                  * ((v "u1" - v "ue") / (v "l1" - v "z1") - (v "u0" - v "ue") / (v "l0" - v "z0")))
    , setv "g" (v "g" + v "ae" / dl 6.0 * (v "z0" * v "u0" - v "z1" * v "u1" + (v "l0" - v "l1") * v "ue"))
    , retE (v "g") ]

/-- `bh_psi_gradient_in_tangent_plane`. -/
def bhGrad : SlangFunctionDecl :=
  fn d3T "bh_grad_tp" [p3 "eta", p3 "v0", p3 "v1", p3 "v2"]
    [ let_ d3T "n" (call "vdir" [call "vcross" [v "v1" - v "v0", v "v2" - v "v0"]])
    , let_ dT "d" (call "vdot" [v "eta" - v "v0", v "n"])
    , retE (dl 1.0 / (dl 8.0 * piD) *
        (call "bh_grad_sub" [v "eta", v "v0", v "v1", v "d", v "n"]
         + call "bh_grad_sub" [v "eta", v "v1", v "v2", v "d", v "n"]
         + call "bh_grad_sub" [v "eta", v "v2", v "v0", v "d", v "n"])) ]

/-- `bh_phi`, given the harmonic ψ `hpsi` of the same triangle at `eta`. -/
def bhPhi : SlangFunctionDecl :=
  let outside (g : String) : E := or_ (lt (v g) (dl 0.0)) (gt (v g) (dl 1.0))
  fn (.named "void") "bh_phi" [p3 "eta", p3 "v0", p3 "v1", p3 "v2", pD "hpsi", arg "phi" d3T .qOut]
    [ let_ d3T "n" (call "vdir" [call "vcross" [v "v1" - v "v0", v "v2" - v "v0"]])
    , let_ dT "d" (call "vdot" [v "eta" - v "v0", v "n"])
    , let_ d3T "r0" (call "vcross" [v "n", v "v2" - v "v1"])
    , setv "r0" (v "r0" / call "vdot" [v "r0", v "v0" - v "v1"])
    , let_ d3T "r1" (call "vcross" [v "n", v "v0" - v "v2"])
    , setv "r1" (v "r1" / call "vdot" [v "r1", v "v1" - v "v2"])
    , let_ d3T "r2" (call "vcross" [v "n", v "v1" - v "v0"])
    , setv "r2" (v "r2" / call "vdot" [v "r2", v "v2" - v "v0"])
    , let_ dT "g0" (call "vdot" [v "eta" - v "v1", v "r0"])
    , let_ dT "g1" (call "vdot" [v "eta" - v "v2", v "r1"])
    , let_ dT "g2" (call "vdot" [v "eta" - v "v0", v "r2"])
    , if_ (lt (fabs (v "d")) (dl 1e-8))
        [ if_ (or_ (or_ (outside "g0") (outside "g1")) (outside "g2"))
            [ setv "phi" (d3 (dl 0.0) (dl 0.0) (dl 0.0)), ret ] ]
    , let_ d3T "gr" (call "bh_grad_tp" [v "eta", v "v0", v "v1", v "v2"])
    , setv "phi" (d3 ((-(v "d")) * call "vdot" [v "gr", v "r0"] + dl 0.5 * v "d" * v "g0" * v "hpsi")
                     ((-(v "d")) * call "vdot" [v "gr", v "r1"] + dl 0.5 * v "d" * v "g1" * v "hpsi")
                     ((-(v "d")) * call "vdot" [v "gr", v "r2"] + dl 0.5 * v "d" * v "g2" * v "hpsi"))
    , ret ]

/-- Every helper, in dependency order (Slang needs no forward
    declarations, but this keeps the emitted file readable top-down). -/
def bhcHelpers : List SlangFunctionDecl :=
  [ vdot, vcross, vnorm, vdir, dlog, datan01, datan2, solidAngle, hCoords, bhSub, bhPsi, bhGradSub, bhGrad, bhPhi ]

end Cage.SlangCodegen.Common
