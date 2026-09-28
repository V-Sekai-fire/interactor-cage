# Cage-fit oracle (Gate 9, G3; host-native)

`gen.cpp` restates RFD 2277's cage fit in double and solves it with unmodified
LBFGSpp 0.3.0, so the Lean-emitted fit in `cage.elf` is checked against an
independent solve:

    y_i  = sum_k Phi_ik (c0_k + u_k) + sum_t Psi_it n_t(c0 + u)     bind space
    x_ip = sum_j w_ij (R_pj y_i + T_pj)                             pose p
    E(u) = wp sum_p sum_i max(0, m - d(x_ip))^2 + wL |L u|^2 + wr |u|^2

`Phi` and `Psi` are the (1,3) biharmonic coordinates from BHC.h, the only parity
reference. `d` is the exact signed distance to the closed body mesh, signed by
angle-weighted pseudonormals. `L` is the uniform graph Laplacian of the cage.
Frozen cage vertices are held by `lb = ub = 0`.

The fixture:
- a body icosphere (r = 0.1 m, 1280 triangles);
- the garment, the upper cap of a 0.0985 m sphere (65 points, 1.5 mm inside the
  body), with a band skinned to a second bone;
- two poses: rest, and bone 1 tilted 10 degrees and dropped 3 mm;
- a closed cage (an icosphere, r = 0.16 m, 42 vertices), whose bottom
  (z < -0.08) is frozen.

It reports:

- **identity**: the undeformed cage reproduces the garment;
- **gradcheck**: central differences at a random `u`, as `|fd - g| / max|g|`,
  gated at 1e-4;
- **negative control**: the normal pull-back's sign flipped must fail that
  check, so the check can fail;
- **the solve**: the energy, the projected gradient, and, per pose, the points
  inside the body and within the margin before and after.

The signed distance has kinks where the closest feature switches, so near the
optimum LBFGSpp's line search can stall. That stop is reported, and the last
accepted iterate is kept.

## Run

```sh
tests/cage_oracle/build.sh                 # writes gates/9-cage/oracle/
OUT=/some/dir tests/cage_oracle/build.sh   # writes elsewhere, e.g. to diff -r
```

It clones `V-Sekai-fire/interactor-aria-lbfgspp` @ `10086b6b` (LBFGSpp and
Eigen, the pin `tests/lbfgsb_oracle` uses) and
`V-Sekai-fire/interactor-tool-godot-cage-deformer` @ `f0b27162` (`src/BHC.h`,
`src/point3.h`) into `.deps/` (gitignored). `CXX` overrides the compiler.

Outputs:
- `cage.obj`, `body.obj`;
- `phi.txt`, `psi.txt`: one row per garment point, `%.17g`;
- `problem.txt`: `m wp wL wr`, the bone-1 weights, the frozen flags, and one
  `pose` line per pose (two bones, each as rows of R then T);
- `solution.txt`: the stop and the energies, then `u`, one coordinate per line;
- `gen.log`.

## Scope

LBFGSpp (MIT), Eigen (MPL-2.0) and BHC.h (MIT; cite Thiery, Michel and Chen,
"Biharmonic Coordinates and their Derivatives for Triangular 3D Cages",
SIGGRAPH 2024) are used only here, by a host-native test tool. None of them
reaches a guest ELF: the cage bind and fit in `cage.elf` are written in Lean
and emitted through Slang (AGENTS.md rules 2 and 3).
