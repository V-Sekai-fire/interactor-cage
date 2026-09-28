// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "bhc13.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <utility>

#include "../cage_kernels.h"

namespace deform {

namespace {

std::string fmt3(const char *f, double a, double b = 0, double c = 0, double d = 0) {
	char buf[256];
	std::snprintf(buf, sizeof buf, f, a, b, c, d);
	return buf;
}

// A dense (row-major rows x cols) float matrix as CSR, and its transpose.
void dense_csr(const std::vector<float> &W, uint32_t P, uint32_t K, uint32_t off, uint32_t cols, Bind::Csr &m,
		Bind::Csr &mt) {
	m.rowptr.resize(size_t(P) + 1);
	m.col.resize(size_t(P) * cols);
	m.val.resize(size_t(P) * cols);
	for (uint32_t i = 0; i < P; ++i) {
		m.rowptr[i] = i * cols;
		for (uint32_t j = 0; j < cols; ++j) {
			m.col[size_t(i) * cols + j] = j;
			m.val[size_t(i) * cols + j] = W[size_t(i) * K + off + j];
		}
	}
	m.rowptr[P] = P * cols;
	mt.rowptr.resize(size_t(cols) + 1);
	mt.col.resize(size_t(P) * cols);
	mt.val.resize(size_t(P) * cols);
	for (uint32_t j = 0; j < cols; ++j) {
		mt.rowptr[j] = j * P;
		for (uint32_t i = 0; i < P; ++i) {
			mt.col[size_t(j) * P + i] = i;
			mt.val[size_t(j) * P + i] = W[size_t(i) * K + off + j];
		}
	}
	mt.rowptr[cols] = cols * P;
}

} // namespace

bool net_from_mesh(const std::vector<float> &xyz, const std::vector<int32_t> &faces, const std::vector<int32_t> &counts,
		Net &out, std::string &err) {
	if (xyz.size() % 3 != 0 || xyz.empty()) {
		err = "net_from_mesh: xyz must hold 3 floats a knot (got " + std::to_string(xyz.size()) + ")";
		return false;
	}
	const int64_t nV = int64_t(xyz.size() / 3);
	size_t total = 0;
	for (size_t f = 0; f < counts.size(); ++f) {
		if (counts[f] < 3) {
			err = "net_from_mesh: polygon " + std::to_string(f) + " has " + std::to_string(counts[f]) + " corners (< 3)";
			return false;
		}
		total += size_t(counts[f]);
	}
	if (total != faces.size()) {
		err = "net_from_mesh: counts sum to " + std::to_string(total) + " but faces holds " + std::to_string(faces.size());
		return false;
	}
	for (size_t i = 0; i < faces.size(); ++i) {
		if (faces[i] < 0 || faces[i] >= nV) {
			err = "net_from_mesh: faces[" + std::to_string(i) + "] = " + std::to_string(faces[i]) + " out of range [0, " +
					std::to_string(nV) + ")";
			return false;
		}
	}
	for (float x : xyz) {
		if (!std::isfinite(x)) {
			err = "net_from_mesh: a knot coordinate is not finite";
			return false;
		}
	}
	out.xyz = xyz;
	out.faces = faces;
	out.counts = counts;
	return true;
}

bool check_net(const Net &net, std::vector<uint32_t> &tris, std::vector<uint32_t> &tri_poly, double &volume,
		std::string &err) {
	const uint32_t nV = net.knot_count();
	tris.clear();
	tri_poly.clear();
	size_t at = 0;
	for (size_t f = 0; f < net.counts.size(); ++f) {
		const int32_t n = net.counts[f];
		for (int32_t k = 1; k + 1 < n; ++k) {
			tris.push_back(uint32_t(net.faces[at]));
			tris.push_back(uint32_t(net.faces[at + k]));
			tris.push_back(uint32_t(net.faces[at + k + 1]));
			tri_poly.push_back(uint32_t(f));
		}
		at += size_t(n);
	}
	const uint32_t nT = uint32_t(tri_poly.size());
	if (nT < 4) {
		err = "bind: not a closed surface: " + std::to_string(nT) + " triangles (a closed surface needs at least 4)";
		return false;
	}
	auto P = [&](uint32_t v, int k) { return double(net.xyz[3 * size_t(v) + k]); };
	// Degenerate triangles: a repeated corner or zero area.
	for (uint32_t t = 0; t < nT; ++t) {
		const uint32_t a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
		if (a == b || b == c || a == c) {
			err = "bind: degenerate triangle " + std::to_string(t) + " (polygon " + std::to_string(tri_poly[t]) +
					") repeats a knot";
			return false;
		}
		const double e[3] = { P(b, 0) - P(a, 0), P(b, 1) - P(a, 1), P(b, 2) - P(a, 2) };
		const double f[3] = { P(c, 0) - P(a, 0), P(c, 1) - P(a, 1), P(c, 2) - P(a, 2) };
		const double n[3] = { e[1] * f[2] - e[2] * f[1], e[2] * f[0] - e[0] * f[2], e[0] * f[1] - e[1] * f[0] };
		if (n[0] * n[0] + n[1] * n[1] + n[2] * n[2] == 0.0) {
			err = "bind: degenerate triangle " + std::to_string(t) + " (polygon " + std::to_string(tri_poly[t]) +
					") has zero area";
			return false;
		}
	}
	// Closed and consistently oriented: every directed edge once, and its twin once.
	std::map<std::pair<uint32_t, uint32_t>, uint32_t> directed;
	for (uint32_t t = 0; t < nT; ++t) {
		for (int k = 0; k < 3; ++k) {
			const uint32_t a = tris[3 * t + k], b = tris[3 * t + (k + 1) % 3];
			if (++directed[{ a, b }] > 1) {
				err = "bind: inconsistently oriented (or non-manifold): edge " + std::to_string(a) + "->" +
						std::to_string(b) + " is walked the same way by two triangles";
				return false;
			}
		}
	}
	for (const auto &kv : directed) {
		if (directed.find({ kv.first.second, kv.first.first }) == directed.end()) {
			err = "bind: not closed: edge " + std::to_string(kv.first.first) + "-" + std::to_string(kv.first.second) +
					" borders one triangle only (an open boundary)";
			return false;
		}
	}
	// Every knot on a face, and each knot's triangles one fan (vertex-manifold).
	std::vector<uint32_t> deg(nV, 0);
	for (uint32_t t = 0; t < nT; ++t) {
		for (int k = 0; k < 3; ++k) {
			++deg[tris[3 * t + k]];
		}
	}
	for (uint32_t v = 0; v < nV; ++v) {
		if (deg[v] == 0) {
			err = "bind: knot " + std::to_string(v) + " is on no face";
			return false;
		}
	}
	{
		// next[(v, a)] = b for the corner (a -> v -> b) around v; walking from one
		// outgoing edge must visit all deg[v] triangles.
		std::map<std::pair<uint32_t, uint32_t>, uint32_t> next;
		for (uint32_t t = 0; t < nT; ++t) {
			for (int k = 0; k < 3; ++k) {
				const uint32_t v = tris[3 * t + k], b = tris[3 * t + (k + 1) % 3], a = tris[3 * t + (k + 2) % 3];
				next[{ v, b }] = a;
			}
		}
		std::vector<uint32_t> first(nV, UINT32_MAX);
		for (uint32_t t = 0; t < nT; ++t) {
			for (int k = 0; k < 3; ++k) {
				const uint32_t v = tris[3 * t + k];
				if (first[v] == UINT32_MAX) {
					first[v] = tris[3 * t + (k + 1) % 3];
				}
			}
		}
		for (uint32_t v = 0; v < nV; ++v) {
			uint32_t b = first[v], walked = 0;
			do {
				auto it = next.find({ v, b });
				if (it == next.end()) {
					break;
				}
				b = it->second;
				++walked;
			} while (b != first[v] && walked <= deg[v]);
			if (walked != deg[v]) {
				err = "bind: not vertex-manifold: knot " + std::to_string(v) + "'s " + std::to_string(deg[v]) +
						" triangles form more than one fan";
				return false;
			}
		}
	}
	volume = 0.0;
	for (uint32_t t = 0; t < nT; ++t) {
		const uint32_t a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
		const double cx = P(b, 1) * P(c, 2) - P(b, 2) * P(c, 1);
		const double cy = P(b, 2) * P(c, 0) - P(b, 0) * P(c, 2);
		const double cz = P(b, 0) * P(c, 1) - P(b, 1) * P(c, 0);
		volume += (P(a, 0) * cx + P(a, 1) * cy + P(a, 2) * cz) / 6.0;
	}
	if (!(volume > 0.0)) {
		err = "bind: inverted: signed volume " + fmt3("%.6g", volume) + " <= 0 (faces must wind outward)";
		return false;
	}
	return true;
}

std::vector<double> subdivision_barycentrics(int levels) {
	struct B {
		double v[3][3];
	};
	std::vector<B> tris(1);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			tris[0].v[i][j] = (i == j) ? 1.0 : 0.0;
		}
	}
	auto mid = [](const double *a, const double *b, double *o) {
		for (int k = 0; k < 3; ++k) {
			o[k] = (a[k] + b[k]) / 2;
		}
	};
	for (int l = 0; l < levels; ++l) {
		std::vector<B> old = tris;
		tris.clear();
		for (const B &bt : old) {
			double v01[3], v02[3], v21[3];
			mid(bt.v[0], bt.v[1], v01);
			mid(bt.v[0], bt.v[2], v02);
			mid(bt.v[2], bt.v[1], v21);
			auto push = [&](const double *a, const double *b, const double *c) {
				B n;
				for (int k = 0; k < 3; ++k) {
					n.v[0][k] = a[k];
					n.v[1][k] = b[k];
					n.v[2][k] = c[k];
				}
				tris.push_back(n);
			};
			push(bt.v[0], v01, v02);
			push(bt.v[1], v21, v01);
			push(bt.v[2], v02, v21);
			push(v01, v21, v02);
		}
	}
	std::vector<double> out;
	for (const B &bt : tris) {
		for (int k = 0; k < 3; ++k) {
			out.push_back((bt.v[0][k] + bt.v[1][k] + bt.v[2][k]) / 3.0);
		}
	}
	return out;
}

// ---- the staged bind --------------------------------------------------------------

const char *BindJob::stage_name() const {
	switch (stage_) {
		case S_IDLE: return "idle";
		case S_SAMPLES: return "samples";
		case S_SAMPLE_ROWS: return "sample_rows";
		case S_GRAM: return "gram";
		case S_LU: return "lu";
		case S_SOLVE: return "solve";
		case S_MESH: return "mesh";
		case S_CSR: return "csr";
		case S_DONE: return "done";
		case S_FAILED: return "failed";
	}
	return "?";
}

bool BindJob::fail(const std::string &why) {
	err_ = why;
	stage_ = S_FAILED;
	return false;
}

bool BindJob::begin(const Net &net, const std::vector<float> &mesh_xyz, const std::string &method,
		const BindOptions &opt, std::string &err) {
	return begin(net, std::vector<double>(mesh_xyz.begin(), mesh_xyz.end()), method, opt, err);
}

bool BindJob::begin(const Net &net, const std::vector<double> &mesh_xyz, const std::string &method,
		const BindOptions &opt, std::string &err) {
	b_ = Bind();
	err_.clear();
	opt_ = opt;
	if (opt_.chunk == 0) {
		opt_.chunk = 1;
	}
	if (opt_.gram_chunk == 0) {
		opt_.gram_chunk = 1;
	}
	if (method != "bhc13") {
		if (method == "harmonic" || method == "idw" || method == "lbs") {
			err = "bind: method \"" + method + "\" is RFD 2279's and not built here (only \"bhc13\")";
		} else {
			err = "bind: unknown method \"" + method + "\" (only \"bhc13\")";
		}
		stage_ = S_FAILED;
		err_ = err;
		return false;
	}
	if (mesh_xyz.size() % 3 != 0) {
		err = "bind: mesh xyz must hold 3 floats a vertex";
		stage_ = S_FAILED;
		err_ = err;
		return false;
	}
	if (!check_net(net, b_.tris, b_.tri_poly, b_.volume, err)) {
		stage_ = S_FAILED;
		err_ = err;
		return false;
	}
	b_.method = method;
	b_.nV = net.knot_count();
	b_.nT = uint32_t(b_.tri_poly.size());
	b_.P = uint32_t(mesh_xyz.size() / 3);
	b_.rest = net.xyz;
	if (net.xyz_d.size() == net.xyz.size()) {
		cage_ = net.xyz_d;
	} else {
		cage_.assign(net.xyz.begin(), net.xyz.end());
	}
	mesh_ = mesh_xyz;
	bary_ = subdivision_barycentrics(3);
	const uint32_t nB = uint32_t(bary_.size() / 3);
	S_ = b_.nT * nB;
	const uint32_t K = b_.K();
	spts_.assign(3 * size_t(S_), 0.0);
	sgamma_.assign(3 * size_t(S_), 0.0);
	sown_.assign(S_, 0u);
	srows_.assign(size_t(S_) * 2 * K, 0.0);
	G_.assign(size_t(K) * K, 0.0);
	X_.assign(size_t(K) * K, 0.0);
	piv_.assign(K, 0u);
	b_.W.assign(size_t(b_.P) * K, 0.0f);
	b_.Wd.assign(size_t(b_.P) * K, 0.0);
	b_.winding.assign(b_.P, 0.0f);
	none_.assign(opt_.chunk, cagek::kNone);
	zeros_.assign(3 * size_t(opt_.chunk), 0.0);
	rows_.assign(size_t(opt_.chunk) * 2 * K, 0.0);
	cursor_ = 0;
	stage_ = S_SAMPLES;
	return true;
}

bool BindJob::step() {
	const uint32_t K = b_.K();
	switch (stage_) {
		case S_SAMPLES: {
			const uint32_t nB = uint32_t(bary_.size() / 3);
			cagek::bind_samples(b_.nT, nB, cage_.data(), b_.nV, b_.tris.data(), bary_.data(), spts_.data(),
					sgamma_.data(), sown_.data());
			cursor_ = 0;
			stage_ = S_SAMPLE_ROWS;
			return true;
		}
		case S_SAMPLE_ROWS: {
			const uint32_t n = std::min(opt_.chunk, S_ - cursor_);
			cagek::bhc_coords(n, b_.nV, b_.nT, spts_.data() + 3 * size_t(cursor_), cage_.data(), b_.tris.data(),
					sown_.data() + cursor_, sgamma_.data() + 3 * size_t(cursor_),
					srows_.data() + size_t(cursor_) * 2 * K);
			cursor_ += n;
			if (cursor_ >= S_) {
				cursor_ = 0;
				stage_ = S_GRAM;
			}
			return true;
		}
		case S_GRAM: {
			const uint32_t total = K * K;
			const uint32_t n = std::min(opt_.gram_chunk, total - cursor_);
			cagek::bind_gram(S_, K, cursor_, n, srows_.data(), G_.data(), X_.data());
			cursor_ += n;
			if (cursor_ >= total) {
				cursor_ = 0;
				stage_ = S_LU;
			}
			return true;
		}
		case S_LU: {
			if (cagek::dense_lu(K, G_.data(), piv_.data()) != 1u) {
				return fail("bind: the (1,3) constraint system is singular (a zero pivot in its LU)");
			}
			stage_ = S_SOLVE;
			return true;
		}
		case S_SOLVE: {
			cagek::dense_lu_solve(K, K, G_.data(), piv_.data(), X_.data());
			for (double x : X_) {
				if (!std::isfinite(x)) {
					return fail("bind: the constraint solve produced a non-finite value");
				}
			}
			// The sample rows are done with.
			std::vector<double>().swap(srows_);
			cursor_ = 0;
			stage_ = S_MESH;
			return true;
		}
		case S_MESH: {
			const uint32_t n = std::min(opt_.chunk, b_.P - cursor_);
			if (n > 0) {
				cagek::bhc_coords(n, b_.nV, b_.nT, mesh_.data() + 3 * size_t(cursor_), cage_.data(), b_.tris.data(),
						none_.data(), zeros_.data(), rows_.data());
				cagek::bind_blend(n, K, rows_.data(), X_.data(), b_.Wd.data() + size_t(cursor_) * K,
						b_.W.data() + size_t(cursor_) * K);
				// Generalized winding: the harmonic vertex coordinates sum to 1
				// inside a closed cage and to 0 outside.
				for (uint32_t i = 0; i < n; ++i) {
					double w = 0.0;
					for (uint32_t v = 0; v < b_.nV; ++v) {
						w += rows_[size_t(i) * 2 * K + v];
					}
					b_.winding[cursor_ + i] = float(w);
					if (!(std::fabs(w - 1.0) <= 0.5)) {
						++b_.outside;
					}
				}
			}
			cursor_ += n;
			if (cursor_ >= b_.P) {
				stage_ = S_CSR;
			}
			return true;
		}
		case S_CSR: {
			if (opt_.fault == 1 && !b_.W.empty()) {
				b_.W[0] += 0.01f;
				b_.Wd[0] += 0.01;
			}
			for (float w : b_.W) {
				if (!std::isfinite(w)) {
					return fail("bind: a weight is not finite (a bound vertex on the cage surface?)");
				}
			}
			if (opt_.require_contained && b_.outside > 0) {
				uint32_t first = 0;
				while (first < b_.P && std::fabs(b_.winding[first] - 1.0f) <= 0.5f) {
					++first;
				}
				return fail("bind: " + std::to_string(b_.outside) + " of " + std::to_string(b_.P) +
						" bound vertices lie outside the cage (first: vertex " + std::to_string(first) +
						fmt3(", winding %.3f)", b_.winding[first]));
			}
			dense_csr(b_.W, b_.P, K, 0, b_.nV, b_.phi, b_.phiT);
			dense_csr(b_.W, b_.P, K, b_.nV, b_.nT, b_.psi, b_.psiT);
			// Corner gather: vertex v sums the per-corner terms (3t + k) of its corners.
			{
				std::vector<std::vector<uint32_t>> at(b_.nV); // local only (AGENTS.md: keep long-lived data flat)
				for (uint32_t t = 0; t < b_.nT; ++t) {
					for (uint32_t k = 0; k < 3; ++k) {
						at[b_.tris[3 * t + k]].push_back(3 * t + k);
					}
				}
				b_.corners.rowptr.assign(1, 0u);
				b_.corners.col.clear();
				for (uint32_t v = 0; v < b_.nV; ++v) {
					b_.corners.col.insert(b_.corners.col.end(), at[v].begin(), at[v].end());
					b_.corners.rowptr.push_back(uint32_t(b_.corners.col.size()));
				}
				b_.corners.val.assign(b_.corners.col.size(), 1.0f);
				std::vector<std::vector<uint32_t>> nb(b_.nV);
				for (uint32_t t = 0; t < b_.nT; ++t) {
					for (uint32_t k = 0; k < 3; ++k) {
						const uint32_t a = b_.tris[3 * t + k], c = b_.tris[3 * t + (k + 1) % 3];
						nb[a].push_back(c);
						nb[c].push_back(a);
					}
				}
				b_.ring.rowptr.assign(1, 0u);
				b_.ring.col.clear();
				for (uint32_t v = 0; v < b_.nV; ++v) {
					std::sort(nb[v].begin(), nb[v].end());
					nb[v].erase(std::unique(nb[v].begin(), nb[v].end()), nb[v].end());
					b_.ring.col.insert(b_.ring.col.end(), nb[v].begin(), nb[v].end());
					b_.ring.rowptr.push_back(uint32_t(b_.ring.col.size()));
				}
			}
			std::vector<double>().swap(G_);
			std::vector<double>().swap(rows_);
			stage_ = S_DONE;
			return true;
		}
		default:
			return false;
	}
}

bool bind(const Net &net, const std::vector<float> &mesh_xyz, const std::string &method, Bind &out, std::string &err,
		const BindOptions &opt) {
	return bind(net, std::vector<double>(mesh_xyz.begin(), mesh_xyz.end()), method, out, err, opt);
}

bool bind(const Net &net, const std::vector<double> &mesh_xyz, const std::string &method, Bind &out, std::string &err,
		const BindOptions &opt) {
	BindJob job;
	if (!job.begin(net, mesh_xyz, method, opt, err)) {
		return false;
	}
	while (job.step()) {
		if (job.done()) {
			break;
		}
	}
	if (!job.done()) {
		err = job.error();
		return false;
	}
	out = std::move(job.result());
	return true;
}

const std::vector<float> &weights(const Bind &b) {
	return b.W;
}

bool knot_positions(const Bind &b, const std::vector<float> &knots_posed, std::vector<float> &c, std::string &err) {
	if (knots_posed.size() != size_t(b.nV) * 12) {
		err = "deform: knots_posed must be nV x 12 = " + std::to_string(size_t(b.nV) * 12) + " floats (got " +
				std::to_string(knots_posed.size()) + ")";
		return false;
	}
	c.resize(3 * size_t(b.nV));
	for (uint32_t v = 0; v < b.nV; ++v) {
		for (int r = 0; r < 3; ++r) {
			c[3 * v + r] = knots_posed[12 * size_t(v) + 4 * r + 3];
		}
	}
	return true;
}

void deform_points(const Bind &b, const float *c, float *y, DeformScratch &s) {
	s.n.resize(3 * size_t(b.nT));
	s.nlen.resize(b.nT);
	cagek::normals(b.nT, c, b.nV, b.tris.data(), s.n.data(), s.nlen.data());
	cagek::csr_gemv3(b.P, false, b.phi.rowptr.data(), b.phi.col.data(), b.phi.val.data(), b.phi.col.size(), c, b.nV, y);
	cagek::csr_gemv3(b.P, true, b.psi.rowptr.data(), b.psi.col.data(), b.psi.val.data(), b.psi.col.size(),
			s.n.data(), b.nT, y);
}

void jacobian_points(const Bind &b, const float *c, const float *dy, float *dc, DeformScratch &s) {
	s.gn.resize(3 * size_t(b.nT));
	s.gcorner.resize(9 * size_t(b.nT));
	cagek::csr_gemv3(b.nV, false, b.phiT.rowptr.data(), b.phiT.col.data(), b.phiT.val.data(), b.phiT.col.size(), dy,
			b.P, dc);
	cagek::csr_gemv3(b.nT, false, b.psiT.rowptr.data(), b.psiT.col.data(), b.psiT.val.data(), b.psiT.col.size(), dy,
			b.P, s.gn.data());
	cagek::normal_vjp(b.nT, c, b.nV, b.tris.data(), s.n.data(), s.nlen.data(), s.gn.data(), s.gcorner.data());
	cagek::csr_gemv3(b.nV, true, b.corners.rowptr.data(), b.corners.col.data(), b.corners.val.data(),
			b.corners.col.size(), s.gcorner.data(), 3 * size_t(b.nT), dc);
}

bool deform(const Bind &b, const std::vector<float> &knots_posed, std::vector<float> &out, std::string &err) {
	std::vector<float> c;
	if (!knot_positions(b, knots_posed, c, err)) {
		return false;
	}
	DeformScratch s;
	out.resize(3 * size_t(b.P));
	deform_points(b, c.data(), out.data(), s);
	return true;
}

bool jacobian(const Bind &b, const std::vector<float> &knots_posed, const std::vector<float> &dy,
		std::vector<float> &dknots, std::string &err) {
	std::vector<float> c;
	if (!knot_positions(b, knots_posed, c, err)) {
		return false;
	}
	if (dy.size() != 3 * size_t(b.P)) {
		err = "jacobian: dy must be 3 P = " + std::to_string(3 * size_t(b.P)) + " floats";
		return false;
	}
	DeformScratch s;
	std::vector<float> y(3 * size_t(b.P));
	deform_points(b, c.data(), y.data(), s);
	dknots.assign(3 * size_t(b.nV), 0.0f);
	jacobian_points(b, c.data(), dy.data(), dknots.data(), s);
	return true;
}

} // namespace deform
