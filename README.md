# interactor-cage

The cage stage as a godot-sandbox guest: the (1,3) biharmonic bind and the in-motion cage fit on L-BFGS-B.

## What it is for

The Lean tree states the cage kernels and emits them as Slang, which compiles for both the CPU and the GPU, so every kernel the guest runs comes from Lean. It builds against the repositories it needs as sibling checkouts at their goal-manifest paths, and `transport-meshing-pen` builds the guest ELFs. RFD 2277 owns the design.

## Build and run

```sh
cd lean
lake build
```

`kernels/cage/gen.sh` regenerates the kernels from Lean.

## Licence

MIT. See [LICENSE](LICENSE).
