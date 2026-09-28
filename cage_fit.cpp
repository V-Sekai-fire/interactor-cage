// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "cage_fit.h"

#include <cfloat>
#include <map>
#include <utility>

#include "cage_kernels.h"

namespace cagefit {

bool Problem::setup(const deform::Bind &bind, const Body &body, const Skin &skin, const std::vector<uint8_t> &frozen,
		const Params &prm, std::string &err) {
	b_ = &bind;
	body_ = body;
	skin_ = skin;
	prm_ = prm;
	const uint32_t P = bind.P, nV = bind.nV, F = body.F;
	if (P == 0 || nV == 0) {
		err = "cage fit: empty bind";
		return false;
	}
	if (F == 0 || body.xyz.size() != size_t(F) * body.BV * 3 || body.tris.size() != 3 * size_t(body.T) || body.T == 0) {
		err = "cage fit: body frames must be F x BV x 3 floats with 3 T triangle indices";
		return false;
	}
	for (uint32_t i : body.tris) {
		if (i >= body.BV) {
			err = "cage fit: body triangle index out of range";
			return false;
		}
	}
	if (skin.bones.size() != size_t(F) * skin.NB * 12 || skin.rowptr.size() != size_t(P) + 1 ||
			skin.col.size() != skin.w.size() || skin.rowptr.back() != skin.col.size()) {
		err = "cage fit: garment skin must be F x NB x 12 bone floats and a CSR of P = " + std::to_string(P) +
				" rows";
		return false;
	}
	for (uint32_t c : skin.col) {
		if (c >= skin.NB) {
			err = "cage fit: garment skin bone index out of range";
			return false;
		}
	}
	if (!frozen.empty() && frozen.size() != nV) {
		err = "cage fit: frozen must hold one flag per cage vertex (" + std::to_string(nV) + ")";
		return false;
	}
	frozen_ = frozen;
	fw_ = body.frame_w;
	if (fw_.empty()) {
		fw_.assign(F, 1.0f);
	}
	if (fw_.size() != F) {
		err = "cage fit: frame weights must be one per frame";
		return false;
	}
	// Body topology: each vertex's corners (for the vertex pseudonormals) and
	// each triangle's neighbour across edge (k, k+1) (for the edge ones). An
	// edge with no twin (an open body) signs with its own face.
	const uint32_t T = body.T, BV = body.BV;
	{
		std::vector<uint32_t> cnt(BV, 0u);
		for (uint32_t i : body.tris) {
			++cnt[i];
		}
		vrow_.assign(size_t(BV) + 1, 0u);
		for (uint32_t v = 0; v < BV; ++v) {
			vrow_[v + 1] = vrow_[v] + cnt[v];
		}
		vcorner_.assign(3 * size_t(T), 0u);
		std::vector<uint32_t> at(vrow_.begin(), vrow_.end() - 1);
		for (uint32_t t = 0; t < T; ++t) {
			for (uint32_t k = 0; k < 3; ++k) {
				vcorner_[at[body.tris[3 * t + k]]++] = 3 * t + k;
			}
		}
		std::map<std::pair<uint32_t, uint32_t>, uint32_t> owner;
		for (uint32_t t = 0; t < T; ++t) {
			for (uint32_t k = 0; k < 3; ++k) {
				owner[{ body.tris[3 * t + k], body.tris[3 * t + (k + 1) % 3] }] = t;
			}
		}
		nbr_.assign(3 * size_t(T), 0u);
		for (uint32_t t = 0; t < T; ++t) {
			for (uint32_t k = 0; k < 3; ++k) {
				auto it = owner.find({ body.tris[3 * t + (k + 1) % 3], body.tris[3 * t + k] });
				nbr_[3 * t + k] = it == owner.end() ? t : it->second;
			}
		}
	}
	// Once per frame set: the garment's blended bones, the body's normals.
	blend_.assign(size_t(F) * P * 12, 0.0f);
	cagek::bone_blend(F, P, skin.NB, skin.rowptr.data(), skin.col.data(), skin.w.data(), skin.col.size(),
			skin.bones.data(), blend_.data());
	fn_.assign(3 * size_t(F) * T, 0.0f);
	ang_.assign(3 * size_t(F) * T, 0.0f);
	vn_.assign(3 * size_t(F) * BV, 0.0f);
	cagek::body_normals(F, T, BV, body.xyz.data(), body.tris.data(), fn_.data(), ang_.data());
	cagek::body_vnormals(F, T, BV, vrow_.data(), vcorner_.data(), fn_.data(), ang_.data(), vn_.data());
	c_.assign(3 * size_t(nV), 0.0f);
	y_.assign(3 * size_t(P), 0.0f);
	z_.assign(3 * size_t(F) * P, 0.0f);
	hb_.assign(size_t(F) * P, 0.0f);
	gz_.assign(3 * size_t(F) * P, 0.0f);
	dist_.assign(size_t(F) * P, 0.0f);
	gy_.assign(3 * size_t(P), 0.0f);
	lu_.assign(3 * size_t(nV), 0.0f);
	gc_.assign(3 * size_t(nV), 0.0f);
	return true;
}

void Problem::bounds(std::vector<float> &lb, std::vector<float> &ub) const {
	const uint32_t n3 = n();
	lb.assign(n3, prm_.box > 0 ? -prm_.box : -FLT_MAX);
	ub.assign(n3, prm_.box > 0 ? prm_.box : FLT_MAX);
	for (size_t v = 0; v < frozen_.size(); ++v) {
		if (frozen_[v]) {
			for (int k = 0; k < 3; ++k) {
				lb[3 * v + k] = 0.0f;
				ub[3 * v + k] = 0.0f;
			}
		}
	}
}

void Problem::forward(const float *u) {
	const deform::Bind &b = *b_;
	const uint32_t P = b.P, nV = b.nV, F = body_.F;
	cagek::saxpby(3 * nV, 1.0f, b.rest.data(), 1.0f, u, c_.data());
	deform::deform_points(b, c_.data(), y_.data(), ds_);
	cagek::lbs(F, P, blend_.data(), y_.data(), z_.data());
	cagek::contact(F, P, body_.T, body_.BV, prm_.margin, z_.data(), body_.xyz.data(), body_.tris.data(), fn_.data(),
			vn_.data(), nbr_.data(), fw_.data(), hb_.data(), gz_.data(), dist_.data());
	contact_ = cagek::sumsq(F * P, hb_.data());
	cagek::laplacian(nV, false, false, 1.0f, b.ring.rowptr.data(), b.ring.col.data(), b.ring.col.size(), u,
			lu_.data(), prm_.lap_combinatorial);
	lap_ = cagek::sumsq(3 * nV, lu_.data());
	reg_ = cagek::sumsq(3 * nV, u);
}

bool Problem::evaluate(const float *u, float *g, double &f, std::string &err) {
	if (!b_) {
		err = "cage fit: not set up";
		return false;
	}
	const deform::Bind &b = *b_;
	const uint32_t P = b.P, nV = b.nV, F = body_.F;
	forward(u);
	f = contact_ + double(prm_.wL) * lap_ + double(prm_.wr) * reg_;
	cagek::lbs_t(F, P, false, blend_.data(), gz_.data(), gy_.data());
	deform::jacobian_points(b, c_.data(), gy_.data(), gc_.data(), ds_);
	cagek::laplacian(nV, true, true, 2.0f * prm_.wL, b.ring.rowptr.data(), b.ring.col.data(), b.ring.col.size(),
			lu_.data(), gc_.data(), prm_.lap_combinatorial);
	cagek::saxpby(3 * nV, 1.0f, gc_.data(), 2.0f * prm_.wr, u, g);
	return true;
}

Report Problem::report(const float *u) {
	Report r;
	if (!b_) {
		return r;
	}
	forward(u);
	r.contact = contact_;
	r.lap = lap_;
	r.reg = reg_;
	r.f = contact_ + double(prm_.wL) * lap_ + double(prm_.wr) * reg_;
	r.dmin = FLT_MAX;
	const uint32_t P = b_->P;
	for (size_t k = 0; k < dist_.size(); ++k) {
		const float d = dist_[k];
		if (d < r.dmin) {
			r.dmin = d;
			r.worst_frame = uint32_t(k / P);
			r.worst_vertex = uint32_t(k % P);
		}
		r.below_margin += d < prm_.margin ? 1u : 0u;
		r.inside += d < 0.0f ? 1u : 0u;
	}
	return r;
}

Report Problem::report_y(const float *y) {
	Report r;
	if (!b_) {
		return r;
	}
	const uint32_t P = b_->P, F = body_.F;
	cagek::lbs(F, P, blend_.data(), y, z_.data());
	cagek::contact(F, P, body_.T, body_.BV, prm_.margin, z_.data(), body_.xyz.data(), body_.tris.data(), fn_.data(),
			vn_.data(), nbr_.data(), fw_.data(), hb_.data(), gz_.data(), dist_.data());
	r.contact = cagek::sumsq(F * P, hb_.data());
	r.f = r.contact;
	r.dmin = FLT_MAX;
	for (size_t k = 0; k < dist_.size(); ++k) {
		const float d = dist_[k];
		if (d < r.dmin) {
			r.dmin = d;
			r.worst_frame = uint32_t(k / P);
			r.worst_vertex = uint32_t(k % P);
		}
		r.below_margin += d < prm_.margin ? 1u : 0u;
		r.inside += d < 0.0f ? 1u : 0u;
	}
	return r;
}

void Problem::garment(const float *u, std::vector<float> &y) {
	const deform::Bind &b = *b_;
	cagek::saxpby(3 * b.nV, 1.0f, b.rest.data(), 1.0f, u, c_.data());
	y.assign(3 * size_t(b.P), 0.0f);
	deform::deform_points(b, c_.data(), y.data(), ds_);
}

} // namespace cagefit
