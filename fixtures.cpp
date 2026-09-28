// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "fixtures.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <utility>

namespace cagefix {

namespace {

// Midpoint-subdivided icosahedron on the unit sphere, CCW-outward. Only
// sqrt, so every build (guest libstdc++, native libc++/libstdc++) makes the
// same floats.
void unit_icosphere(int levels, std::vector<double> &v, std::vector<uint32_t> &f) {
	const double t = (1.0 + std::sqrt(5.0)) / 2.0;
	const double base[12][3] = { { -1, t, 0 }, { 1, t, 0 }, { -1, -t, 0 }, { 1, -t, 0 }, { 0, -1, t }, { 0, 1, t },
		{ 0, -1, -t }, { 0, 1, -t }, { t, 0, -1 }, { t, 0, 1 }, { -t, 0, -1 }, { -t, 0, 1 } };
	v.clear();
	for (auto &p : base) {
		const double n = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
		v.insert(v.end(), { p[0] / n, p[1] / n, p[2] / n });
	}
	f = { 0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3,
		4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1 };
	for (int l = 0; l < levels; ++l) {
		std::map<std::pair<uint32_t, uint32_t>, uint32_t> mid;
		auto m = [&](uint32_t a, uint32_t b) {
			const auto key = std::make_pair(std::min(a, b), std::max(a, b));
			auto it = mid.find(key);
			if (it != mid.end()) {
				return it->second;
			}
			double p[3];
			for (int k = 0; k < 3; ++k) {
				p[k] = (v[3 * a + k] + v[3 * b + k]) / 2.0;
			}
			const double n = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
			const uint32_t id = uint32_t(v.size() / 3);
			v.insert(v.end(), { p[0] / n, p[1] / n, p[2] / n });
			mid[key] = id;
			return id;
		};
		std::vector<uint32_t> g;
		for (size_t i = 0; i < f.size(); i += 3) {
			const uint32_t a = f[i], b = f[i + 1], c = f[i + 2];
			const uint32_t ab = m(a, b), bc = m(b, c), ca = m(c, a);
			g.insert(g.end(), { a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca });
		}
		f = g;
	}
}

float smoothstep(float e0, float e1, float x) {
	float t = (x - e0) / (e1 - e0);
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	return t * t * (3.0f - 2.0f * t);
}

} // namespace

void icosphere(float r, int levels, std::vector<float> &v, std::vector<int32_t> &f) {
	std::vector<double> u;
	std::vector<uint32_t> g;
	unit_icosphere(levels, u, g);
	v.clear();
	for (double x : u) {
		v.push_back(float(double(r) * x));
	}
	f.assign(g.begin(), g.end());
}

void uv_sphere(float r, int nlat, int nlon, std::vector<float> &v, std::vector<uint32_t> &f) {
	// Kept for callers that want rows; the gates use icospheres (sqrt only).
	std::vector<double> u;
	std::vector<uint32_t> g;
	unit_icosphere(nlat, u, g);
	(void)nlon;
	v.clear();
	for (double x : u) {
		v.push_back(float(double(r) * x));
	}
	f = g;
}

void cylinder(float r, float y0, float y1, int nseg, int nrow, std::vector<float> &v, std::vector<int32_t> &f) {
	// guest/curvenet/checks.cpp's capped cylinder, verbatim.
	v.clear();
	f.clear();
	for (int j = 0; j <= nrow; ++j) {
		const float y = y0 + (y1 - y0) * float(j) / float(nrow);
		for (int i = 0; i < nseg; ++i) {
			const double a = 2 * M_PI * i / nseg;
			v.insert(v.end(), { float(r * std::cos(a)), y, float(r * std::sin(a)) });
		}
	}
	auto id = [nseg](int i, int j) { return int32_t(j * nseg + i % nseg); };
	for (int j = 0; j < nrow; ++j) {
		for (int i = 0; i < nseg; ++i) {
			const int32_t a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
			f.insert(f.end(), { a, d, b, b, d, c });
		}
	}
	const int32_t bot = int32_t(v.size() / 3), top = bot + 1;
	v.insert(v.end(), { 0, y0, 0, 0, y1, 0 });
	for (int i = 0; i < nseg; ++i) {
		f.insert(f.end(), { bot, id(i, 0), id(i + 1, 0), top, id(i + 1, nrow), id(i, nrow) });
	}
}

SphereScene sphere_scene() {
	SphereScene s;
	// Body: icosphere r 0.1 (3 subdivisions: 642 vertices, 1280 triangles).
	std::vector<double> u;
	std::vector<uint32_t> g;
	unit_icosphere(3, u, g);
	const uint32_t BV = uint32_t(u.size() / 3);
	const float shift[2][3] = { { 0.002f, 0.0f, 0.0f }, { -0.001f, 0.0f, 0.002f } };
	s.body.F = 2;
	s.body.BV = BV;
	s.body.T = uint32_t(g.size() / 3);
	s.body.tris = g;
	for (int p = 0; p < 2; ++p) {
		for (uint32_t i = 0; i < BV; ++i) {
			for (int k = 0; k < 3; ++k) {
				s.body.xyz.push_back(float(0.1 * u[3 * i + k]) + shift[p][k]);
			}
		}
	}
	s.body.frame_w = { 1.0f, 1.0f };
	// Garment: the r 0.1015 shell's points with |y| <= 0.866 r (the -60..+60
	// degree band) from a 4-subdivision icosphere, dented where x > 0.08,
	// |y| < 0.03.
	std::vector<double> gu;
	std::vector<uint32_t> gg;
	unit_icosphere(3, gu, gg);
	for (size_t i = 0; i < gu.size(); i += 3) {
		if (std::fabs(gu[i + 1]) > 0.866) {
			continue;
		}
		double r = 0.1015;
		if (0.1015 * gu[i] > 0.08 && std::fabs(0.1015 * gu[i + 1]) < 0.03) {
			r = 0.0995;
		}
		for (int k = 0; k < 3; ++k) {
			s.garment.push_back(float(r * gu[i + k]));
		}
	}
	const uint32_t P = uint32_t(s.garment.size() / 3);
	// Garment skin: bone 0 fixed, bone 1 turned about z by +-20 degrees.
	const float c20 = 0.9396926207859084f, s20 = 0.3420201433256687f;
	s.skin.NB = 2;
	const float rz[2][2] = { { c20, s20 }, { c20, -s20 } };
	for (int p = 0; p < 2; ++p) {
		const float id[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
		s.skin.bones.insert(s.skin.bones.end(), id, id + 12);
		const float c = rz[p][0], sn = rz[p][1];
		const float rot[12] = { c, -sn, 0, 0, sn, c, 0, 0, 0, 0, 1, 0 };
		s.skin.bones.insert(s.skin.bones.end(), rot, rot + 12);
	}
	s.skin.rowptr.push_back(0);
	for (uint32_t i = 0; i < P; ++i) {
		const float w1 = smoothstep(-0.03f, 0.03f, s.garment[3 * i + 1]);
		if (w1 < 1.0f) {
			s.skin.col.push_back(0);
			s.skin.w.push_back(1.0f - w1);
		}
		if (w1 > 0.0f) {
			s.skin.col.push_back(1);
			s.skin.w.push_back(w1);
		}
		s.skin.rowptr.push_back(uint32_t(s.skin.col.size()));
	}
	// Cage: icosphere r 0.13, one subdivision; knots below y = -0.1 frozen.
	icosphere(0.13f, 1, s.cage, s.faces);
	s.counts.assign(s.faces.size() / 3, 3);
	for (size_t i = 0; i < s.cage.size(); i += 3) {
		s.frozen.push_back(s.cage[i + 1] < -0.1f ? 1 : 0);
	}
	s.params.margin = 0.002f;
	s.params.wL = 0.001f;
	s.params.wr = 1e-4f;
	s.params.lb.m = 6;
	s.params.lb.epsilon = 1e-7;
	s.params.lb.epsilon_rel = 0.0;
	s.params.lb.past = 1;
	s.params.lb.delta = 1e-10;
	s.params.lb.max_iterations = 200;
	return s;
}

DressPen dress_pen() {
	DressPen d;
	cylinder(0.15f, 0.3f, 1.1f, 64, 16, d.body, d.body_tris);
	const float neck_y = 0.9f, hem_y = 0.5f, r = 0.175f;
	auto at = [r](float y, double a) {
		return std::vector<float>{ float(r * std::cos(a)), y, float(r * std::sin(a)), 0.5f };
	};
	auto arc = [&](float y, double a0, double a1, int n) {
		std::vector<float> out;
		for (int i = 0; i <= n; ++i) {
			const std::vector<float> p = at(y, a0 + (a1 - a0) * double(i) / double(n));
			out.insert(out.end(), p.begin(), p.end());
		}
		return out;
	};
	auto line = [&](float ya, float yb, double a, int n) {
		std::vector<float> out;
		const std::vector<float> A = at(ya, a), B = at(yb, a);
		for (int i = 0; i <= n; ++i) {
			const float t = float(double(i) / double(n));
			for (int k = 0; k < 3; ++k) {
				out.push_back(i == n ? B[k] : A[k] + (B[k] - A[k]) * t);
			}
			out.push_back(0.5f);
		}
		return out;
	};
	d.strokes.push_back({ "neckline_front", true, arc(neck_y, 0.0, M_PI, 24) });
	d.strokes.push_back({ "neckline_back", true, arc(neck_y, 0.0, -M_PI, 24) });
	d.strokes.push_back({ "hem_front", true, arc(hem_y, 0.0, M_PI, 24) });
	d.strokes.push_back({ "hem_back", true, arc(hem_y, 0.0, -M_PI, 24) });
	d.strokes.push_back({ "strap_right", false, line(neck_y, hem_y, 0.0, 12) });
	d.strokes.push_back({ "strap_left", false, line(neck_y, hem_y, M_PI, 12) });
	// The bodice: a tube r 0.158, y 0.55 to 0.85, 48 around, 13 rows.
	for (int j = 0; j <= 12; ++j) {
		const float y = 0.55f + 0.3f * float(j) / 12.0f;
		for (int i = 0; i < 48; ++i) {
			const double a = 2 * M_PI * i / 48;
			d.bound.insert(d.bound.end(), { float(0.158 * std::cos(a)), y, float(0.158 * std::sin(a)) });
		}
	}
	return d;
}

std::string DressPen::describe() const {
	std::string s = "scripted pen (a stand-in for a person) on the capped cylinder r 0.15 m, y 0.3..1.1:";
	for (const PenStroke &st : strokes) {
		char b[160];
		const size_t n = st.xyzp.size() / 4;
		std::snprintf(b, sizeof b, " %s%s (%zu samples, (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f));", st.name.c_str(),
				st.boundary ? " [boundary]" : "", n, st.xyzp[0], st.xyzp[1], st.xyzp[2], st.xyzp[4 * (n - 1)],
				st.xyzp[4 * (n - 1) + 1], st.xyzp[4 * (n - 1) + 2]);
		s += b;
	}
	char b[128];
	std::snprintf(b, sizeof b, " snap_radius %.3f, surface_offset %.3f", snap_radius, surface_offset);
	return s + b;
}

std::vector<float> encode_pen(const DressPen &p) {
	std::vector<float> out;
	out.push_back(float(p.strokes.size()));
	for (const PenStroke &s : p.strokes) {
		out.push_back(s.boundary ? 1.0f : 0.0f);
		out.push_back(float(s.xyzp.size() / 4));
		out.insert(out.end(), s.xyzp.begin(), s.xyzp.end());
	}
	return out;
}

} // namespace cagefix
