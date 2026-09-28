// cage_kernels -- the CPU path of kernels/cage (and the two kernels it
// reuses: kernels/anny's anny_csr_gemv3 and kernels/drape's saxpby): each
// function dispatches one Lean-emitted kernel's slangc -target cpp emit over
// caller-owned arrays, one thread after another (the guest/drape/vec_cpu.cpp
// pattern, AGENTS.md rule 2). No arithmetic lives here: only buffer binding
// and the dispatch grid.
//
// Sizes are the kernels' params (see each lean/Cage/SlangCodegen/<Module>.lean
// for the bindings). Pointers may point into the middle of a larger array, so
// a caller can dispatch a chunk of rows (the bind's per-tick slices).
// std types only: it compiles into cage.elf and into the native gates alike.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstddef>
#include <cstdint>

namespace cagek {

constexpr uint32_t kNone = 0xFFFFFFFFu;

// ---- the bind, double precision -------------------------------------------------

void bind_samples(uint32_t nT, uint32_t nB, const double *cage, size_t nV, const uint32_t *tris,
		const double *bary, double *pts, double *gamma, uint32_t *own);
void bhc_coords(uint32_t P, uint32_t nV, uint32_t nT, const double *pts, const double *cage, const uint32_t *tris,
		const uint32_t *own, const double *gamma, double *rows);
// Entries [ij0, ij0 + count) of the K x K system (G and rhs are the whole arrays).
void bind_gram(uint32_t S, uint32_t K, uint32_t ij0, uint32_t count, const double *rows, double *G, double *rhs);
// Returns the kernel's status word: 1 when every pivot was non-zero.
uint32_t dense_lu(uint32_t K, double *G, uint32_t *piv);
void dense_lu_solve(uint32_t K, uint32_t ncols, const double *G, const uint32_t *piv, double *X);
void bind_blend(uint32_t P, uint32_t K, const double *rows, const double *X, double *wd, float *wf);

// ---- the fit, float --------------------------------------------------------------

void normals(uint32_t nT, const float *c, size_t nV, const uint32_t *tris, float *n, float *nlen);
void normal_vjp(uint32_t nT, const float *c, size_t nV, const uint32_t *tris, const float *n, const float *nlen,
		const float *gn, float *gcorner);
// combinatorial: (L x)_i = sum_j (x_j - x_i) (symmetric), else x_i - mean of the one-ring.
void laplacian(uint32_t n, bool transpose, bool accumulate, float scale, const uint32_t *rowptr, const uint32_t *col,
		size_t nnz, const float *x, float *y, bool combinatorial = false);
// Σ x_k² in df32; returns hi + lo in double.
double sumsq(uint32_t n, const float *x);
void bone_blend(uint32_t F, uint32_t P, uint32_t NB, const uint32_t *rowptr, const uint32_t *col, const float *w,
		size_t nnz, const float *bones, float *blend);
void lbs(uint32_t F, uint32_t P, const float *blend, const float *y, float *z);
void lbs_t(uint32_t F, uint32_t P, bool accumulate, const float *blend, const float *gz, float *gy);
void body_normals(uint32_t F, uint32_t T, uint32_t BV, const float *bv, const uint32_t *tris, float *fn, float *ang);
void body_vnormals(uint32_t F, uint32_t T, uint32_t BV, const uint32_t *rowptr, const uint32_t *corner,
		const float *fn, const float *ang, float *vn);
void contact(uint32_t F, uint32_t P, uint32_t T, uint32_t BV, float margin, const float *z, const float *bv,
		const uint32_t *tris, const float *fn, const float *vn, const uint32_t *nbr, const float *fw, float *hb,
		float *gz, float *dist);

// ---- the curvenet cage, double ------------------------------------------------------

// The generalized winding number of each point (1 inside a closed outward cage, 0 outside).
void winding(uint32_t P, uint32_t nT, const double *pts, const double *cage, size_t nV, const uint32_t *tris, double *w);

// ---- reused ----------------------------------------------------------------------

// kernels/anny anny_csr_gemv3: y (+)= M x over 3-vectors, M in CSR (rows x cols).
void csr_gemv3(uint32_t rows, bool accumulate, const uint32_t *rowptr, const uint32_t *col, const float *val,
		size_t nnz, const float *x, size_t cols, float *y);
// kernels/drape saxpby: dst = fma(alpha, x, beta * y).
void saxpby(uint32_t n, float alpha, const float *x, float beta, const float *y, float *dst);

} // namespace cagek
