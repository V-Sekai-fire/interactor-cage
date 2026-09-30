// cage_fit -- the in-motion cage fit (RFD 2277, "The guest"): move the cage
// in bind space so the garment clears the body in every sampled frame.
//
//   c = c0 + u                    the cage (knots) displaced by u, 3 nV
//   y = Phi c + Psi n(c)          the garment in bind space (bhc13's deform)
//   z_p = LBS_p(y)                skinned to frame p with the garment's own
//                                 weights and frame p's bone matrices
//   f(u) = sum_p w_p sum_i max(0, m - d_p(z_{p,i}))^2 + w_L |L u|^2 + w_r |u|^2
//
// d_p is the signed distance to the body skinned to frame p: the closest point
// over the body's triangles, signed by its angle-weighted pseudonormal
// (cage_contact). RFD 2277 names fit.elf's brick-grid SDF and the Lean
// tricubic sampler; that pair lives inside fit.elf (PolyFEM) on this base, so
// this is the stated fallback. The gradient: each frame's g_z carried back
// through LBS_p's linear part (cage_lbs_t), then Phi^T g plus the normal
// pull-back (bhc13::jacobian_points), plus 2 w_L L^T L u and 2 w_r u.
// Frozen cage vertices get lb = ub = 0. Every vector operation is a Lean
// kernel (guest/common/deform/cage_kernels.h); only the three df32 sums are
// combined here, in double. The cage lives in bind space, so the result
// bakes straight into the garment mesh (no unskin).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bhc13/bhc13.h"
#include "fit_driver.h"

namespace cagefit {

// The body skinned to each sampled frame (the host skins it, or cage.elf's
// caller does; RFD 2277's cage_body_frames).
struct Body {
	uint32_t F = 0, BV = 0, T = 0;
	std::vector<float> xyz;        // F x BV x 3
	std::vector<uint32_t> tris;    // 3 T, CCW-outward (mesh_wire)
	std::vector<float> frame_w;    // F (empty: all 1)
};

// The garment's own skinning: frame p's bones and its skin weights (CSR, one
// row per bound vertex).
struct Skin {
	uint32_t NB = 0;
	std::vector<float> bones;      // F x NB x 12, [R | t] row-major (translation column 3)
	std::vector<uint32_t> rowptr;  // P + 1
	std::vector<uint32_t> col;     // bone index
	std::vector<float> w;
};

struct Params {
	float margin = 0.002f;
	float wL = 1.0f;
	float wr = 1e-3f;
	// The Laplacian: false the normalized uniform one (x_i minus the one-ring mean),
	// true the combinatorial sum_j (x_j - x_i) (the G3 oracle's).
	bool lap_combinatorial = false;
	// |u| bound per coordinate (0: unbounded); frozen knots are always 0.
	float box = 0.0f;
	LbfgsbParams lb;
};

struct Report {
	double f = 0.0, contact = 0.0, lap = 0.0, reg = 0.0;
	float dmin = 0.0f;              // min over frames and vertices of d_p
	uint32_t below_margin = 0;      // (p, i) with d < m
	uint32_t inside = 0;            // (p, i) with d < 0
	uint32_t worst_frame = 0, worst_vertex = 0;
};

class Problem : public fitd::Objective {
public:
	bool setup(const deform::Bind &bind, const Body &body, const Skin &skin, const std::vector<uint8_t> &frozen,
			const Params &prm, std::string &err);
	bool evaluate(const float *u, float *g, double &f, std::string &err) override;
	// The terms and the clearance at u (runs the forward kernels only).
	Report report(const float *u);
	// The clearance of an explicit bind-space garment y (3 P): the forward
	// kernels from LBS on, with the terms that need u left at 0.
	Report report_y(const float *y);
	// The garment in bind space at u (3 P).
	void garment(const float *u, std::vector<float> &y);
	// L-BFGS-B bounds: 0 for frozen coordinates, unbounded otherwise.
	void bounds(std::vector<float> &lb, std::vector<float> &ub) const;
	uint32_t n() const { return 3 * (b_ ? b_->nV : 0); }
	const Params &params() const { return prm_; }
	// Per (frame, vertex) signed distances of the last forward pass.
	const std::vector<float> &distances() const { return dist_; }

private:
	void forward(const float *u);
	const deform::Bind *b_ = nullptr;
	Body body_;
	Skin skin_;
	Params prm_;
	std::vector<uint8_t> frozen_;
	// Per-frame, once: the blended garment bones, the body's normals.
	std::vector<float> blend_, fn_, ang_, vn_, fw_;
	std::vector<uint32_t> vrow_, vcorner_, nbr_;
	// Per evaluation.
	std::vector<float> c_, y_, z_, hb_, gz_, dist_, gy_, lu_, gc_;
	deform::DeformScratch ds_;
	double contact_ = 0.0, lap_ = 0.0, reg_ = 0.0;
};

} // namespace cagefit
