// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "cage_kernels.h"

#include "slang-cpp-prelude.h"

// Each slangc emit puts main_0 / GlobalParams_0 at file scope under
// extern "C"; emptying the EXTERN_C macros lets every emit live in its own
// namespace in one TU (guest/drape/vec_cpu.cpp, tests/anny_kernels).
#undef SLANG_PRELUDE_EXTERN_C
#undef SLANG_PRELUDE_EXTERN_C_START
#undef SLANG_PRELUDE_EXTERN_C_END
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END

namespace k_samples {
#include "../../../kernels/cage/cpp/cage_bind_samples_emit.cpp"
}
namespace k_bhc {
#include "../../../kernels/cage/cpp/cage_bhc_coords_emit.cpp"
}
namespace k_gram {
#include "../../../kernels/cage/cpp/cage_bind_gram_emit.cpp"
}
namespace k_lu {
#include "../../../kernels/cage/cpp/cage_dense_lu_emit.cpp"
}
namespace k_lus {
#include "../../../kernels/cage/cpp/cage_dense_lu_solve_emit.cpp"
}
namespace k_blend {
#include "../../../kernels/cage/cpp/cage_bind_blend_emit.cpp"
}
namespace k_nrm {
#include "../../../kernels/cage/cpp/cage_normals_emit.cpp"
}
namespace k_nvjp {
#include "../../../kernels/cage/cpp/cage_normal_vjp_emit.cpp"
}
namespace k_lap {
#include "../../../kernels/cage/cpp/cage_laplacian_emit.cpp"
}
namespace k_ssq {
#include "../../../kernels/cage/cpp/cage_sumsq_emit.cpp"
}
namespace k_bb {
#include "../../../kernels/cage/cpp/cage_bone_blend_emit.cpp"
}
namespace k_lbs {
#include "../../../kernels/cage/cpp/cage_lbs_emit.cpp"
}
namespace k_lbst {
#include "../../../kernels/cage/cpp/cage_lbs_t_emit.cpp"
}
namespace k_bn {
#include "../../../kernels/cage/cpp/cage_body_normals_emit.cpp"
}
namespace k_bvn {
#include "../../../kernels/cage/cpp/cage_body_vnormals_emit.cpp"
}
namespace k_ct {
#include "../../../kernels/cage/cpp/cage_contact_emit.cpp"
}
namespace k_wind {
#include "../../../kernels/cage/cpp/cage_winding_emit.cpp"
}
namespace k_csr {
#include "../../../kernels/anny/cpp/anny_csr_gemv3_emit.cpp"
}
namespace k_sx {
#include "../../../kernels/drape/cpp/saxpby_emit.cpp"
}

namespace cagek {
namespace {

using KernelFn = void (*)(ComputeVaryingInput *, void *, void *);

// The whole grid of `threads` threads in groups of `group` (the kernel's
// numthreads); every kernel bounds-checks its own id.
void dispatch(KernelFn fn, void *gp, size_t threads, uint32_t group) {
	if (threads == 0) {
		return;
	}
	ComputeVaryingInput vi{};
	vi.startGroupID = uint3(0u, 0u, 0u);
	vi.endGroupID = uint3(uint32_t((threads + group - 1) / group), 1u, 1u);
	fn(&vi, nullptr, gp);
}

template <class B, class T>
void bind(B &b, const T *p, size_t count) {
	b.data = reinterpret_cast<decltype(b.data)>(const_cast<T *>(p));
	b.count = count;
}

} // namespace

void bind_samples(uint32_t nT, uint32_t nB, const double *cage, size_t nV, const uint32_t *tris, const double *bary,
		double *pts, double *gamma, uint32_t *own) {
	k_samples::CageSamplesParams_0 prm{ nT, nB };
	k_samples::GlobalParams_0 g{};
	g.params_0 = &prm;
	const size_t S = size_t(nT) * nB;
	bind(g.cage_0, cage, 3 * nV);
	bind(g.tris_0, tris, 3 * size_t(nT));
	bind(g.bary_0, bary, 3 * size_t(nB));
	bind(g.pts_0, pts, 3 * S);
	bind(g.gamma_0, gamma, 3 * S);
	bind(g.own_0, own, S);
	dispatch(k_samples::main_0, &g, S, 64);
}

void bhc_coords(uint32_t P, uint32_t nV, uint32_t nT, const double *pts, const double *cage, const uint32_t *tris,
		const uint32_t *own, const double *gamma, double *rows) {
	const uint32_t stride = 2 * (nV + nT);
	k_bhc::CageBhcParams_0 prm{ P, nV, nT, stride };
	k_bhc::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.pts_0, pts, 3 * size_t(P));
	bind(g.cage_0, cage, 3 * size_t(nV));
	bind(g.tris_0, tris, 3 * size_t(nT));
	bind(g.own_0, own, P);
	bind(g.gamma_0, gamma, 3 * size_t(P));
	bind(g.rows_0, rows, size_t(P) * stride);
	dispatch(k_bhc::main_0, &g, P, 64);
}

void bind_gram(uint32_t S, uint32_t K, uint32_t ij0, uint32_t count, const double *rows, double *G, double *rhs) {
	k_gram::CageGramParams_0 prm{ S, K, 2 * K, ij0, count };
	k_gram::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.rows_0, rows, size_t(S) * 2 * K);
	bind(g.G_0, G, size_t(K) * K);
	bind(g.rhs_0, rhs, size_t(K) * K);
	dispatch(k_gram::main_0, &g, count, 64);
}

uint32_t dense_lu(uint32_t K, double *G, uint32_t *piv) {
	uint32_t status = 0;
	k_lu::CageLuParams_0 prm{ K };
	k_lu::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.G_0, G, size_t(K) * K);
	bind(g.piv_0, piv, K);
	bind(g.status_0, &status, 1);
	dispatch(k_lu::main_0, &g, 1, 1);
	return status;
}

void dense_lu_solve(uint32_t K, uint32_t ncols, const double *G, const uint32_t *piv, double *X) {
	k_lus::CageLuSolveParams_0 prm{ K, ncols };
	k_lus::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.G_0, G, size_t(K) * K);
	bind(g.piv_0, piv, K);
	bind(g.X_0, X, size_t(K) * ncols);
	dispatch(k_lus::main_0, &g, ncols, 64);
}

void bind_blend(uint32_t P, uint32_t K, const double *rows, const double *X, double *wd, float *wf) {
	k_blend::CageBlendParams_0 prm{ P, K, 2 * K };
	k_blend::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.rows_0, rows, size_t(P) * 2 * K);
	bind(g.X_0, X, size_t(K) * K);
	bind(g.wd_0, wd, size_t(P) * K);
	bind(g.wf_0, wf, size_t(P) * K);
	dispatch(k_blend::main_0, &g, P, 64);
}

void normals(uint32_t nT, const float *c, size_t nV, const uint32_t *tris, float *n, float *nlen) {
	k_nrm::CageNormalsParams_0 prm{ nT };
	k_nrm::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.c_0, c, 3 * nV);
	bind(g.tris_0, tris, 3 * size_t(nT));
	bind(g.n_0, n, 3 * size_t(nT));
	bind(g.nlen_0, nlen, nT);
	dispatch(k_nrm::main_0, &g, nT, 64);
}

void normal_vjp(uint32_t nT, const float *c, size_t nV, const uint32_t *tris, const float *n, const float *nlen,
		const float *gn, float *gcorner) {
	k_nvjp::CageNormalsParams_0 prm{ nT };
	k_nvjp::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.c_0, c, 3 * nV);
	bind(g.tris_0, tris, 3 * size_t(nT));
	bind(g.n_0, n, 3 * size_t(nT));
	bind(g.nlen_0, nlen, nT);
	bind(g.gn_0, gn, 3 * size_t(nT));
	bind(g.gcorner_0, gcorner, 9 * size_t(nT));
	dispatch(k_nvjp::main_0, &g, nT, 64);
}

void laplacian(uint32_t n, bool transpose, bool accumulate, float scale, const uint32_t *rowptr, const uint32_t *col,
		size_t nnz, const float *x, float *y, bool combinatorial) {
	k_lap::CageLapParams_0 prm{ n, transpose ? 1u : 0u, accumulate ? 1u : 0u, scale, combinatorial ? 1u : 0u };
	k_lap::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.rowptr_0, rowptr, size_t(n) + 1);
	bind(g.col_0, col, nnz);
	bind(g.x_0, x, 3 * size_t(n));
	bind(g.y_0, y, 3 * size_t(n));
	dispatch(k_lap::main_0, &g, n, 64);
}

double sumsq(uint32_t n, const float *x) {
	float out[2] = { 0.0f, 0.0f };
	k_ssq::CageSumSqParams_0 prm{ n, 0u };
	k_ssq::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.x_0, x, n);
	bind(g.out_0, out, 2);
	dispatch(k_ssq::main_0, &g, 1, 1);
	return double(out[0]) + double(out[1]);
}

void bone_blend(uint32_t F, uint32_t P, uint32_t NB, const uint32_t *rowptr, const uint32_t *col, const float *w,
		size_t nnz, const float *bones, float *blend) {
	k_bb::CageBoneBlendParams_0 prm{ F, P, NB };
	k_bb::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.rowptr_0, rowptr, size_t(P) + 1);
	bind(g.col_0, col, nnz);
	bind(g.w_0, w, nnz);
	bind(g.bones_0, bones, size_t(F) * NB * 12);
	bind(g.blend_0, blend, size_t(F) * P * 12);
	dispatch(k_bb::main_0, &g, size_t(F) * P, 64);
}

void lbs(uint32_t F, uint32_t P, const float *blend, const float *y, float *z) {
	k_lbs::CageLbsParams_0 prm{ F, P };
	k_lbs::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.blend_0, blend, size_t(F) * P * 12);
	bind(g.y_0, y, 3 * size_t(P));
	bind(g.z_0, z, 3 * size_t(F) * P);
	dispatch(k_lbs::main_0, &g, size_t(F) * P, 64);
}

void lbs_t(uint32_t F, uint32_t P, bool accumulate, const float *blend, const float *gz, float *gy) {
	k_lbst::CageLbsTParams_0 prm{ F, P, accumulate ? 1u : 0u };
	k_lbst::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.blend_0, blend, size_t(F) * P * 12);
	bind(g.gz_0, gz, 3 * size_t(F) * P);
	bind(g.gy_0, gy, 3 * size_t(P));
	dispatch(k_lbst::main_0, &g, P, 64);
}

void body_normals(uint32_t F, uint32_t T, uint32_t BV, const float *bv, const uint32_t *tris, float *fn, float *ang) {
	k_bn::CageBodyParams_0 prm{ F, T, BV };
	k_bn::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.bv_0, bv, 3 * size_t(F) * BV);
	bind(g.tris_0, tris, 3 * size_t(T));
	bind(g.fn_0, fn, 3 * size_t(F) * T);
	bind(g.ang_0, ang, 3 * size_t(F) * T);
	dispatch(k_bn::main_0, &g, size_t(F) * T, 64);
}

void body_vnormals(uint32_t F, uint32_t T, uint32_t BV, const uint32_t *rowptr, const uint32_t *corner,
		const float *fn, const float *ang, float *vn) {
	k_bvn::CageBodyParams_0 prm{ F, T, BV };
	k_bvn::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.rowptr_0, rowptr, size_t(BV) + 1);
	bind(g.corner_0, corner, 3 * size_t(T));
	bind(g.fn_0, fn, 3 * size_t(F) * T);
	bind(g.ang_0, ang, 3 * size_t(F) * T);
	bind(g.vn_0, vn, 3 * size_t(F) * BV);
	dispatch(k_bvn::main_0, &g, size_t(F) * BV, 64);
}

void contact(uint32_t F, uint32_t P, uint32_t T, uint32_t BV, float margin, const float *z, const float *bv,
		const uint32_t *tris, const float *fn, const float *vn, const uint32_t *nbr, const float *fw, float *hb,
		float *gz, float *dist) {
	k_ct::CageContactParams_0 prm{ F, P, T, BV, margin };
	k_ct::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.z_0, z, 3 * size_t(F) * P);
	bind(g.bv_0, bv, 3 * size_t(F) * BV);
	bind(g.tris_0, tris, 3 * size_t(T));
	bind(g.fn_0, fn, 3 * size_t(F) * T);
	bind(g.vn_0, vn, 3 * size_t(F) * BV);
	bind(g.nbr_0, nbr, 3 * size_t(T));
	bind(g.fw_0, fw, F);
	bind(g.hb_0, hb, size_t(F) * P);
	bind(g.gz_0, gz, 3 * size_t(F) * P);
	bind(g.dist_0, dist, size_t(F) * P);
	dispatch(k_ct::main_0, &g, size_t(F) * P, 64);
}

void winding(uint32_t P, uint32_t nT, const double *pts, const double *cage, size_t nV, const uint32_t *tris, double *w) {
	k_wind::CageWindingParams_0 prm{ P, nT };
	k_wind::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.pts_0, pts, 3 * size_t(P));
	bind(g.cage_0, cage, 3 * nV);
	bind(g.tris_0, tris, 3 * size_t(nT));
	bind(g.w_0, w, P);
	dispatch(k_wind::main_0, &g, P, 64);
}

void csr_gemv3(uint32_t rows, bool accumulate, const uint32_t *rowptr, const uint32_t *col, const float *val,
		size_t nnz, const float *x, size_t cols, float *y) {
	k_csr::AnnyCsrParams_0 prm{ rows, accumulate ? 1u : 0u };
	k_csr::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.rowptr_0, rowptr, size_t(rows) + 1);
	bind(g.col_0, col, nnz);
	bind(g.val_0, val, nnz);
	bind(g.x_0, x, 3 * cols);
	bind(g.y_0, y, 3 * size_t(rows));
	dispatch(k_csr::main_0, &g, rows, 64);
}

void saxpby(uint32_t n, float alpha, const float *x, float beta, const float *y, float *dst) {
	k_sx::SaxpbyParams_0 prm{ n, alpha, beta };
	k_sx::GlobalParams_0 g{};
	g.params_0 = &prm;
	bind(g.x_0, x, n);
	bind(g.y_0, y, n);
	bind(g.dst_0, dst, n);
	dispatch(k_sx::main_0, &g, n, 256);
}

} // namespace cagek
