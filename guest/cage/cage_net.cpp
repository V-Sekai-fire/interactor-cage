// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "cage_net.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <tuple>
#include <utility>

#include "cage_kernels.h"
#include "pmp/algorithms/decimation.h"
#include "pmp/surface_mesh.h"

namespace cagenet {

namespace {

std::string fmt(const char *f, double a = 0, double b = 0, double c = 0, double d = 0) {
	char buf[256];
	std::snprintf(buf, sizeof buf, f, a, b, c, d);
	return buf;
}

} // namespace

std::vector<double> winding(const std::vector<float> &cage_xyz, const std::vector<uint32_t> &tris,
		const std::vector<float> &points) {
	const std::vector<double> c(cage_xyz.begin(), cage_xyz.end()), p(points.begin(), points.end());
	std::vector<double> w(points.size() / 3, 0.0);
	cagek::winding(uint32_t(w.size()), uint32_t(tris.size() / 3), p.data(), c.data(), cage_xyz.size() / 3,
			tris.data(), w.data());
	return w;
}

bool Builder::fail(const std::string &why) {
	err_ = why;
	stage_ = S_FAILED;
	return false;
}

uint32_t Builder::count_outside(const std::vector<float> &c, double &wmin, double &wmax) const {
	uint32_t outside = 0;
	wmin = 1e300;
	wmax = -1e300;
	if (bound_.empty()) {
		wmin = wmax = 1.0;
		return outside;
	}
	for (double w : winding(c, ctri_, bound_)) {
		outside += std::fabs(w - 1.0) > 0.5 ? 1u : 0u;
		wmin = std::min(wmin, w);
		wmax = std::max(wmax, w);
	}
	return outside;
}

bool Builder::begin(const std::vector<float> &vertices, const std::vector<int32_t> &triangles,
		const std::vector<float> &bound, const Options &opt, std::string &err) {
	out_ = Result();
	err_.clear();
	opt_ = opt;
	if (vertices.size() % 3 != 0 || triangles.size() % 3 != 0 || triangles.empty()) {
		err = "cage_net: vertices must be 3 floats a vertex and triangles 3 indices a triangle";
		stage_ = S_FAILED;
		err_ = err;
		return false;
	}
	for (int32_t i : triangles) {
		if (i < 0 || size_t(i) >= vertices.size() / 3) {
			err = "cage_net: triangle index out of range";
			stage_ = S_FAILED;
			err_ = err;
			return false;
		}
	}
	vin_ = vertices;
	tin_ = triangles;
	bound_ = bound;
	stage_ = S_PREPARE;
	return true;
}

bool Builder::step() {
	switch (stage_) {
		case S_PREPARE:
			if (!prepare()) {
				return false;
			}
			stage_ = (out_.outside > 0 && opt_.inflate_max > 0 && opt_.inflate_step > 0) ? S_INFLATE : S_FINISH;
			d_ = 0.0f;
			return true;
		case S_INFLATE: {
			// One offset a step: cage + d * vn, the winding of every bound vertex.
			d_ += opt_.inflate_step;
			const uint32_t nV = uint32_t(cage_.size() / 3);
			moved_ = cage_;
			cagek::saxpby(3 * nV, 1.0f, cage_.data(), d_, vn_.data(), moved_.data());
			out_.outside = count_outside(moved_, out_.wmin, out_.wmax);
			out_.inflate = d_;
			if (out_.outside == 0 || d_ + opt_.inflate_step > opt_.inflate_max + 1e-9f) {
				cage_ = moved_;
				probe_.xyz = cage_;
				std::vector<uint32_t> tris, tri_poly;
				if (!deform::check_net(probe_, tris, tri_poly, out_.volume, err_)) {
					return fail("cage_net: the inflated cage fails bhc13's checks: " + err_);
				}
				stage_ = S_FINISH;
			}
			return true;
		}
		case S_FINISH: {
			std::vector<int32_t> faces(ctri_.begin(), ctri_.end()), counts(ctri_.size() / 3, 3);
			out_.vertices = uint32_t(cage_.size() / 3);
			out_.triangles = uint32_t(ctri_.size() / 3);
			if (!deform::net_from_mesh(cage_, faces, counts, out_.net, err_)) {
				return fail(err_);
			}
			out_.log = fmt("patch surface %.0f v %.0f t", out_.in_vertices, out_.in_triangles) +
					fmt(", %.0f loops capped (%.0f cap triangles)", out_.loops, out_.cap_triangles) +
					fmt(", %.0f v -> decimated %.0f v", out_.capped_vertices, out_.vertices) +
					fmt(" %.0f t; volume %.6g", out_.triangles, out_.volume) +
					fmt("; outside %.0f -> %.0f after inflate %.3f m", out_.outside_before, out_.outside, out_.inflate) +
					fmt("; winding [%.4f, %.4f]", out_.wmin, out_.wmax);
			stage_ = S_DONE;
			return false;
		}
		default:
			return false;
	}
}

bool build_cage(const std::vector<float> &vertices, const std::vector<int32_t> &triangles,
		const std::vector<float> &bound, const Options &opt, Result &out, std::string &err) {
	Builder b;
	if (!b.begin(vertices, triangles, bound, opt, err)) {
		return false;
	}
	while (b.step()) {
	}
	if (!b.done()) {
		err = b.error();
		return false;
	}
	out = b.result();
	return true;
}

// Weld, cap, decimate, check, and the containment before any inflation.
bool Builder::prepare() {
	const std::vector<float> &vertices = vin_;
	const std::vector<int32_t> &triangles = tin_;
	const Options &opt = opt_;
	Result &out = out_;
	std::string err;
	const size_t n0 = vertices.size() / 3;
	// 1. Weld (first vertex within weld_eps wins) and drop degenerate triangles.
	std::vector<uint32_t> remap(n0);
	std::vector<double> xyz;
	{
		std::map<std::tuple<int64_t, int64_t, int64_t>, std::vector<uint32_t>> grid;
		const double h = opt.weld_eps > 0 ? opt.weld_eps : 1e-12;
		for (size_t i = 0; i < n0; ++i) {
			const double p[3] = { vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2] };
			const int64_t g[3] = { int64_t(std::floor(p[0] / h)), int64_t(std::floor(p[1] / h)),
				int64_t(std::floor(p[2] / h)) };
			uint32_t hit = UINT32_MAX;
			for (int dx = -1; dx <= 1 && hit == UINT32_MAX; ++dx) {
				for (int dy = -1; dy <= 1 && hit == UINT32_MAX; ++dy) {
					for (int dz = -1; dz <= 1 && hit == UINT32_MAX; ++dz) {
						auto it = grid.find({ g[0] + dx, g[1] + dy, g[2] + dz });
						if (it == grid.end()) {
							continue;
						}
						for (uint32_t j : it->second) {
							const double ex = xyz[3 * j] - p[0], ey = xyz[3 * j + 1] - p[1], ez = xyz[3 * j + 2] - p[2];
							if (ex * ex + ey * ey + ez * ez <= opt.weld_eps * opt.weld_eps) {
								hit = j;
								break;
							}
						}
					}
				}
			}
			if (hit == UINT32_MAX) {
				hit = uint32_t(xyz.size() / 3);
				xyz.insert(xyz.end(), p, p + 3);
				grid[{ g[0], g[1], g[2] }].push_back(hit);
			}
			remap[i] = hit;
		}
	}
	std::vector<uint32_t> tri;
	for (size_t t = 0; t < triangles.size(); t += 3) {
		const uint32_t a = remap[triangles[t]], b = remap[triangles[t + 1]], c = remap[triangles[t + 2]];
		if (a != b && b != c && a != c) {
			tri.insert(tri.end(), { a, b, c });
		}
	}
	out.in_vertices = uint32_t(xyz.size() / 3);
	out.in_triangles = uint32_t(tri.size() / 3);
	// 2. Cap the boundary loops. A boundary edge a->b is owned by one triangle;
	// its loop continues from b. The cap is wound b->a.
	{
		std::map<std::pair<uint32_t, uint32_t>, int> directed;
		for (size_t t = 0; t < tri.size(); t += 3) {
			for (int k = 0; k < 3; ++k) {
				++directed[{ tri[t + k], tri[t + (k + 1) % 3] }];
			}
		}
		std::map<uint32_t, uint32_t> next; // boundary a -> b
		for (const auto &kv : directed) {
			if (kv.second != 1) {
				err = fmt("cage_net: patch surface is not manifold: edge %.0f->%.0f walked %.0f times",
						kv.first.first, kv.first.second, kv.second);
				return fail(err);
			}
			if (directed.find({ kv.first.second, kv.first.first }) == directed.end()) {
				if (next.count(kv.first.first)) {
					err = fmt("cage_net: boundary vertex %.0f starts two boundary edges (a pinched boundary)",
							kv.first.first);
					return fail(err);
				}
				next[kv.first.first] = kv.first.second;
			}
		}
		while (!next.empty()) {
			std::vector<uint32_t> loop;
			uint32_t a = next.begin()->first;
			while (next.count(a)) {
				loop.push_back(a);
				const uint32_t b = next[a];
				next.erase(a);
				a = b;
			}
			if (loop.size() < 3 || a != loop.front()) {
				err = "cage_net: a boundary does not close into a loop";
				return fail(err);
			}
			double c[3] = { 0, 0, 0 };
			for (uint32_t v : loop) {
				for (int k = 0; k < 3; ++k) {
					c[k] += xyz[3 * v + k];
				}
			}
			const uint32_t ci = uint32_t(xyz.size() / 3);
			for (int k = 0; k < 3; ++k) {
				xyz.push_back(c[k] / double(loop.size()));
			}
			for (size_t i = 0; i < loop.size(); ++i) {
				tri.insert(tri.end(), { loop[(i + 1) % loop.size()], loop[i], ci });
			}
			++out.loops;
			out.cap_triangles += uint32_t(loop.size());
		}
	}
	out.capped_vertices = uint32_t(xyz.size() / 3);
	// 3. Decimate with PMP.
	pmp::SurfaceMesh mesh;
	{
		std::vector<pmp::Vertex> vh;
		for (size_t i = 0; i < xyz.size(); i += 3) {
			vh.push_back(mesh.add_vertex(pmp::Point(xyz[i], xyz[i + 1], xyz[i + 2])));
		}
		for (size_t t = 0; t < tri.size(); t += 3) {
			if (!mesh.add_triangle(vh[tri[t]], vh[tri[t + 1]], vh[tri[t + 2]]).is_valid()) {
				err = fmt("cage_net: the capped surface is not manifold at triangle %.0f", double(t / 3));
				return fail(err);
			}
		}
		if (opt.target_vertices > 0 && mesh.n_vertices() > opt.target_vertices) {
			pmp::decimate(mesh, opt.target_vertices);
			mesh.garbage_collection();
		}
	}
	std::vector<float> cage;
	std::vector<uint32_t> ctri;
	for (auto v : mesh.vertices()) {
		const pmp::Point &p = mesh.position(v);
		cage.insert(cage.end(), { float(p[0]), float(p[1]), float(p[2]) });
	}
	for (auto f : mesh.faces()) {
		for (auto v : mesh.vertices(f)) {
			ctri.push_back(uint32_t(v.idx()));
		}
	}
	// 4. As bhc13 will check it.
	deform::Net probe;
	std::vector<int32_t> faces(ctri.begin(), ctri.end()), counts(ctri.size() / 3, 3);
	if (!deform::net_from_mesh(cage, faces, counts, probe, err)) {
		return fail(err);
	}
	std::vector<uint32_t> tris, tri_poly;
	if (!deform::check_net(probe, tris, tri_poly, out.volume, err)) {
		err = "cage_net: the coarse cage fails bhc13's checks: " + err;
		return fail(err);
	}
	// 5. Containment, then inflation along the vertex normals if asked.
	const uint32_t nV = uint32_t(cage.size() / 3), nT = uint32_t(ctri.size() / 3);
	cage_ = cage;
	ctri_ = ctri;
	probe_ = probe;
	out.outside_before = count_outside(cage_, out.wmin, out.wmax);
	out.outside = out.outside_before;
	if (out.outside > 0 && opt.inflate_max > 0 && opt.inflate_step > 0) {
		// Area-weighted vertex normals: the face normals (cage_normals) summed
		// with weight |N| onto the corners (anny_csr_gemv3).
		std::vector<float> n(3 * size_t(nT)), nlen(nT);
		vn_.assign(3 * size_t(nV), 0.0f);
		cagek::normals(nT, cage.data(), nV, ctri.data(), n.data(), nlen.data());
		std::vector<std::vector<uint32_t>> at(nV);
		for (uint32_t t = 0; t < nT; ++t) {
			for (int k = 0; k < 3; ++k) {
				at[ctri[3 * t + k]].push_back(t);
			}
		}
		std::vector<uint32_t> rowptr(1, 0u), col;
		std::vector<float> val;
		for (uint32_t v = 0; v < nV; ++v) {
			for (uint32_t t : at[v]) {
				col.push_back(t);
				val.push_back(nlen[t]);
			}
			rowptr.push_back(uint32_t(col.size()));
		}
		cagek::csr_gemv3(nV, false, rowptr.data(), col.data(), val.data(), col.size(), n.data(), nT, vn_.data());
		for (uint32_t v = 0; v < nV; ++v) {
			const float l = std::sqrt(vn_[3 * v] * vn_[3 * v] + vn_[3 * v + 1] * vn_[3 * v + 1] + vn_[3 * v + 2] * vn_[3 * v + 2]);
			for (int k = 0; k < 3; ++k) {
				vn_[3 * v + k] = l > 0 ? vn_[3 * v + k] / l : 0.0f;
			}
		}
	}
	return true;
}

} // namespace cagenet
