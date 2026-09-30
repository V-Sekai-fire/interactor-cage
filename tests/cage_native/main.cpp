// cage_native -- cage.elf's native flat control and the gates that read host
// files (RFD 2277 Phase A, gates/9-cage). The same TUs as the ELF (guest/cage,
// guest/common/deform, guest/drape's L-BFGS-B), compiled for the host.
//
//   cage_native checks                        G2, G3 (guest half), G4: the lines cage_check_all answers
//   cage_native g1 <cage.obj> <mesh.obj> <deformed.obj> <oracle out> [fault]
//                                             G1: the Lean bind's weights against bhc_oracle's
//                                             (<oracle out>.weights), and the deformed positions;
//                                             G2 on the demo; with "fault", the corrupted-weight
//                                             control (every comparison must FAIL)
//   cage_native g3 <oracle dir> <out dir>     G3: tests/cage_oracle's fixture bound by the Lean bind
//                                             and fitted by the guest's L-BFGS-B, against LBFGSpp's
//                                             solution.txt; writes phi.txt / psi.txt (%.9g) and u.txt
//   cage_native cn                            curvenet to cage: the scripted pen through curvenet's
//                                             code (curvenet_core), capped, coarsened, bound
// Exit 1 on a failed gate (a control that passes is a failure).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bhc13/bhc13.h"
#include "cage_api.h"
#include "cage_fit.h"
#include "checks.h"
#include "curvenet_api.h"
#include "fit_driver.h"
#include "fixtures.h"

namespace {

double now_ms() {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Obj {
	std::vector<double> v;
	std::vector<int32_t> f, c;
};

bool read_obj(const char *p, Obj &o) {
	std::ifstream in(p);
	if (!in) {
		return false;
	}
	std::string line;
	while (std::getline(in, line)) {
		std::istringstream s(line);
		std::string tag;
		s >> tag;
		if (tag == "v") {
			double x, y, z;
			s >> x >> y >> z;
			o.v.insert(o.v.end(), { x, y, z });
		} else if (tag == "f") {
			int n = 0;
			std::string tok;
			while (s >> tok) {
				o.f.push_back(std::stoi(tok.substr(0, tok.find('/'))) - 1);
				++n;
			}
			o.c.push_back(n);
		}
	}
	return true;
}

std::vector<float> to_f(const std::vector<double> &d) {
	return std::vector<float>(d.begin(), d.end());
}

std::vector<double> read_numbers(const std::string &path) {
	std::ifstream in(path);
	std::vector<double> out;
	double x;
	while (in >> x) {
		out.push_back(x);
	}
	return out;
}

std::vector<float> knots_of(const std::vector<float> &c) {
	std::vector<float> k(4 * c.size(), 0.0f);
	for (size_t v = 0; v < c.size() / 3; ++v) {
		k[12 * v] = k[12 * v + 5] = k[12 * v + 10] = 1.0f;
		for (int r = 0; r < 3; ++r) {
			k[12 * v + 4 * r + 3] = c[3 * v + r];
		}
	}
	return k;
}

int cmd_checks() {
	int bad = 0;
	for (const std::string &n : cagecheck::names()) {
		const double t0 = now_ms();
		const std::string line = cagecheck::run(n);
		std::printf("%s [%.0f ms]\n", line.c_str(), now_ms() - t0);
		std::fflush(stdout);
		bad += line.rfind("PASS", 0) == 0 ? 0 : 1;
	}
	std::printf("checks: %d/%zu\n", int(cagecheck::names().size()) - bad, cagecheck::names().size());
	return bad ? 1 : 0;
}

// G1 tolerances: float32 weights against the double oracle.
constexpr double kWTol = 1e-6;   // weights (|w| <= ~0.7 here: ~16 float ulps)
constexpr double kPTol = 1e-5;   // positions, m

int cmd_g1(int argc, char **argv) {
	if (argc < 6) {
		std::fprintf(stderr, "usage: g1 <cage.obj> <mesh.obj> <deformed.obj> <oracle out> [fault]\n");
		return 2;
	}
	const bool fault = argc > 6 && std::strcmp(argv[6], "fault") == 0;
	Obj cage, mesh, def;
	if (!read_obj(argv[2], cage) || !read_obj(argv[3], mesh) || !read_obj(argv[4], def)) {
		std::printf("FAIL g1: cannot read the OBJs\n");
		return 1;
	}
	const std::vector<double> ow = read_numbers(std::string(argv[5]) + ".weights");
	const std::vector<double> op = read_numbers(argv[5]);
	deform::BindOptions opt;
	opt.fault = fault ? 1 : 0;
	int failures = 0;
	std::printf("G1%s: cage %s (%zu knots, %zu polygons), mesh %s (%zu vertices); oracle %s (%zu weights, %zu positions)\n",
			fault ? " CONTROL (fault switch on: W[0][0] += 0.01)" : "", argv[2], cage.v.size() / 3, cage.c.size(),
			argv[3], mesh.v.size() / 3, argv[5], ow.size(), op.size() / 3);
	// Two input paths: the OBJ's doubles as the oracle read them (the Lean
	// bind's own error), and the float32 wire form the engine hands cage.elf.
	for (int wire = 0; wire < 2; ++wire) {
		deform::Net net;
		std::string err;
		deform::net_from_mesh(to_f(cage.v), cage.f, cage.c, net, err);
		if (!wire) {
			net.xyz_d = cage.v;
		}
		deform::Bind b;
		const double t0 = now_ms();
		const bool ok = wire ? deform::bind(net, to_f(mesh.v), "bhc13", b, err, opt)
							 : deform::bind(net, mesh.v, "bhc13", b, err, opt);
		const double ms = now_ms() - t0;
		if (!ok) {
			std::printf("FAIL g1 %s: bind refused: %s\n", wire ? "wire" : "exact", err.c_str());
			++failures;
			continue;
		}
		const size_t K = b.K();
		if (ow.size() != size_t(b.P) * K) {
			std::printf("FAIL g1: oracle has %zu weights, the bind %zu\n", ow.size(), size_t(b.P) * K);
			return 1;
		}
		double ef = 0, ed = 0;
		size_t at = 0;
		for (size_t i = 0; i < ow.size(); ++i) {
			const double e = std::fabs(double(b.W[i]) - ow[i]);
			if (e > ef) {
				ef = e;
				at = i;
			}
			ed = std::max(ed, std::fabs(b.Wd[i] - ow[i]));
		}
		// Deformed positions: the oracle deformed by cage_deformed.obj.
		std::vector<float> y;
		deform::deform(b, knots_of(to_f(def.v)), y, err);
		double ep = 0;
		for (size_t i = 0; i < y.size() && i < op.size(); ++i) {
			ep = std::max(ep, std::fabs(double(y[i]) - op[i]));
		}
		// G2 on the demo: the rest cage returns the mesh.
		std::vector<float> y0;
		deform::deform(b, knots_of(to_f(cage.v)), y0, err);
		double e0 = 0;
		for (size_t i = 0; i < y0.size(); ++i) {
			e0 = std::max(e0, std::fabs(double(y0[i]) - mesh.v[i]));
		}
		const bool pw = ef <= kWTol, pp = ep <= kPTol, p0 = e0 <= kPTol;
		std::printf("%s g1 %-5s inputs: bind %.0f ms, K = %zu (%u knots + %u triangles), %u triangles x 64 constraint samples, "
					"outside %u; max|W_float - oracle| = %.3e (at vertex %zu, weight %zu; tolerance %.0e), "
					"max|W_double - oracle| = %.3e\n",
				pw ? "PASS" : "FAIL", wire ? "wire" : "exact", ms, K, b.nV, b.nT, b.nT, b.outside, ef, at / K, at % K,
				kWTol, ed);
		std::printf("%s g1 %-5s deformed by cage_deformed.obj: max|y - oracle| = %.3e m (tolerance %.0e)\n",
				pp ? "PASS" : "FAIL", wire ? "wire" : "exact", ep, kPTol);
		std::printf("%s g2 %-5s demo, zero displacement: max|y - x| = %.3e m (tolerance %.0e)\n", p0 ? "PASS" : "FAIL",
				wire ? "wire" : "exact", e0, kPTol);
		failures += (pw ? 0 : 1) + (pp ? 0 : 1) + (p0 ? 0 : 1);
	}
	if (fault) {
		// The control: every comparison must have failed.
		std::printf("%s g1 control: %d of 6 comparisons failed with the fault switch on (6 must)\n",
				failures == 6 ? "PASS" : "FAIL", failures);
		return failures == 6 ? 0 : 1;
	}
	return failures ? 1 : 0;
}

// G3 against tests/cage_oracle (gates/9-cage/oracle).
int cmd_g3(int argc, char **argv) {
	if (argc < 4) {
		std::fprintf(stderr, "usage: g3 <oracle dir> <out dir>\n");
		return 2;
	}
	const std::string od = argv[2], out = argv[3];
	Obj cage, body;
	if (!read_obj((od + "/cage.obj").c_str(), cage) || !read_obj((od + "/body.obj").c_str(), body)) {
		std::printf("FAIL g3: cannot read %s/{cage,body}.obj\n", od.c_str());
		return 1;
	}
	// problem.txt
	std::ifstream pin(od + "/problem.txt");
	std::string line, tag;
	double m = 0, wp = 0, wL = 0, wr = 0;
	std::vector<double> w1, poses;
	std::vector<int> frozen;
	while (std::getline(pin, line)) {
		std::istringstream s(line);
		s >> tag;
		if (tag == "m") {
			std::string k;
			s >> m >> k >> wp >> k >> wL >> k >> wr;
		} else if (tag == "w1") {
			double x;
			while (s >> x) {
				w1.push_back(x);
			}
		} else if (tag == "frozen") {
			int x;
			while (s >> x) {
				frozen.push_back(x);
			}
		} else if (tag == "pose") {
			double x;
			while (s >> x) {
				poses.push_back(x);
			}
		}
	}
	const std::vector<double> ophi = read_numbers(od + "/phi.txt"), opsi = read_numbers(od + "/psi.txt");
	const size_t nV = cage.v.size() / 3, nT = cage.c.size(), P = w1.size(), F = poses.size() / 24;
	if (ophi.size() != P * nV || opsi.size() != P * nT || frozen.size() != nV || F != 2) {
		std::printf("FAIL g3: oracle sizes: phi %zu, psi %zu, P %zu, nV %zu, nT %zu, poses %zu\n", ophi.size(),
				opsi.size(), P, nV, nT, F);
		return 1;
	}
	// The garment points: the oracle writes Phi and Psi, not the points; they
	// are Phi c0 + Psi n(c0) (its identity check: 1.5e-16 m), in double.
	std::vector<double> x(3 * P, 0.0);
	{
		std::vector<double> n(3 * nT);
		for (size_t t = 0; t < nT; ++t) {
			const double *a = &cage.v[3 * cage.f[3 * t]], *b = &cage.v[3 * cage.f[3 * t + 1]],
						 *c = &cage.v[3 * cage.f[3 * t + 2]];
			const double e[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, f[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
			const double N[3] = { e[1] * f[2] - e[2] * f[1], e[2] * f[0] - e[0] * f[2], e[0] * f[1] - e[1] * f[0] };
			const double l = std::sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
			for (int k = 0; k < 3; ++k) {
				n[3 * t + k] = N[k] / l;
			}
		}
		for (size_t i = 0; i < P; ++i) {
			for (int k = 0; k < 3; ++k) {
				double s = 0;
				for (size_t v = 0; v < nV; ++v) {
					s += ophi[i * nV + v] * cage.v[3 * v + k];
				}
				for (size_t t = 0; t < nT; ++t) {
					s += opsi[i * nT + t] * n[3 * t + k];
				}
				x[3 * i + k] = s;
			}
		}
	}
	{
		FILE *fg = std::fopen((out + "/garment.txt").c_str(), "w");
		for (size_t i = 0; i < P; ++i) {
			std::fprintf(fg, "%.17g %.17g %.17g\n", x[3 * i], x[3 * i + 1], x[3 * i + 2]);
		}
		std::fclose(fg);
	}
	// The Lean bind on those points.
	deform::Net net;
	std::string err;
	deform::net_from_mesh(to_f(cage.v), cage.f, cage.c, net, err);
	net.xyz_d = cage.v;
	deform::Bind b;
	if (!deform::bind(net, x, "bhc13", b, err)) {
		std::printf("FAIL g3: bind: %s\n", err.c_str());
		return 1;
	}
	double ephi = 0, epsi = 0;
	{
		FILE *fp = std::fopen((out + "/phi.txt").c_str(), "w"), *fs = std::fopen((out + "/psi.txt").c_str(), "w");
		for (size_t i = 0; i < P; ++i) {
			for (size_t v = 0; v < nV; ++v) {
				const float w = b.W[i * b.K() + v];
				ephi = std::max(ephi, std::fabs(double(w) - ophi[i * nV + v]));
				std::fprintf(fp, "%s%.9g", v ? " " : "", w);
			}
			std::fprintf(fp, "\n");
			for (size_t t = 0; t < nT; ++t) {
				const float w = b.W[i * b.K() + nV + t];
				epsi = std::max(epsi, std::fabs(double(w) - opsi[i * nT + t]));
				std::fprintf(fs, "%s%.9g", t ? " " : "", w);
			}
			std::fprintf(fs, "\n");
		}
		std::fclose(fp);
		std::fclose(fs);
	}
	std::printf("g3 bind: Lean bind of the oracle's %zu garment points in its %zu-knot cage: max|Phi - oracle| = %.3e, "
				"max|Psi - oracle| = %.3e (wrote %s/phi.txt, psi.txt, %%.9g)\n",
			P, nV, ephi, epsi, out.c_str());
	// The problem in cage_fit's terms: the body is not posed (one mesh for both
	// frames), bones are (R rows, T) per bone per pose, hinge weight wp per
	// frame, the combinatorial Laplacian, |u| <= 0.05 as the oracle bounds it.
	cagefit::Body body_f;
	body_f.F = 2;
	body_f.BV = uint32_t(body.v.size() / 3);
	body_f.T = uint32_t(body.c.size());
	body_f.tris.assign(body.f.begin(), body.f.end());
	for (int p = 0; p < 2; ++p) {
		for (double v : body.v) {
			body_f.xyz.push_back(float(v));
		}
	}
	body_f.frame_w = { float(wp), float(wp) };
	cagefit::Skin skin;
	skin.NB = 2;
	for (size_t p = 0; p < F; ++p) {
		for (int j = 0; j < 2; ++j) {
			const double *q = &poses[24 * p + 12 * j];
			for (int r = 0; r < 3; ++r) {
				skin.bones.insert(skin.bones.end(), { float(q[3 * r]), float(q[3 * r + 1]), float(q[3 * r + 2]), float(q[9 + r]) });
			}
		}
	}
	skin.rowptr.push_back(0);
	for (size_t i = 0; i < P; ++i) {
		skin.col.insert(skin.col.end(), { 0u, 1u });
		skin.w.insert(skin.w.end(), { float(1.0 - w1[i]), float(w1[i]) });
		skin.rowptr.push_back(uint32_t(skin.col.size()));
	}
	cagefit::Params prm;
	prm.margin = float(m);
	prm.wL = float(wL);
	prm.wr = float(wr);
	prm.lap_combinatorial = true;
	prm.box = 0.05f;
	// LBFGSpp's settings in tests/cage_oracle/gen.cpp.
	prm.lb.m = 6;
	prm.lb.delta = 0;
	prm.lb.max_linesearch = 40;
	prm.lb.max_iterations = 2000;
	prm.lb.epsilon = 1e-12;
	prm.lb.epsilon_rel = 1e-10;
	std::vector<uint8_t> fz(frozen.begin(), frozen.end());
	cagefit::Problem prob;
	if (!prob.setup(b, body_f, skin, fz, prm, err)) {
		std::printf("FAIL g3: setup: %s\n", err.c_str());
		return 1;
	}
	const uint32_t n = prob.n();
	std::vector<float> u(n, 0.0f), lb, ub;
	prob.bounds(lb, ub);
	const cagefit::Report r0 = prob.report(u.data());
	fitd::Driver d;
	if (!d.begin(n, u, lb, ub, prm.lb, err)) {
		std::printf("FAIL g3: %s\n", err.c_str());
		return 1;
	}
	const double t0 = now_ms();
	int ticks = 0;
	fitd::Driver::State s = fitd::Driver::RUNNING;
	while (s == fitd::Driver::RUNNING && ticks < 5000) {
		s = d.tick(prob);
		++ticks;
	}
	const double ms = now_ms() - t0;
	u = d.x();
	const cagefit::Report r1 = prob.report(u.data());
	// Per-pose clearance.
	const std::vector<float> &dist = prob.distances();
	uint32_t inside[2] = { 0, 0 }, below[2] = { 0, 0 };
	float deep[2] = { 0, 0 };
	for (int p = 0; p < 2; ++p) {
		for (size_t i = 0; i < P; ++i) {
			const float dd = dist[p * P + i];
			inside[p] += dd < 0 ? 1 : 0;
			below[p] += dd < m ? 1 : 0;
			deep[p] = std::min(deep[p], dd);
		}
	}
	// solution.txt: the header line, then u.
	std::ifstream sin(od + "/solution.txt");
	std::getline(sin, line);
	double E0o = 0, Eo = 0;
	{
		std::istringstream s2(line);
		std::string k;
		while (s2 >> k) {
			if (k == "E0") {
				s2 >> E0o;
			} else if (k == "E") {
				s2 >> Eo;
			}
		}
	}
	std::vector<double> uo;
	double xv;
	while (sin >> xv) {
		uo.push_back(xv);
	}
	double du = 0, umax = 0;
	size_t duat = 0;
	for (size_t k = 0; k < n && k < uo.size(); ++k) {
		const double e = std::fabs(double(u[k]) - uo[k]);
		if (e > du) {
			du = e;
			duat = k;
		}
		umax = std::max(umax, double(std::fabs(u[k])));
	}
	FILE *fu = std::fopen((out + "/u.txt").c_str(), "w");
	for (float v : u) {
		std::fprintf(fu, "%.9g\n", v);
	}
	std::fclose(fu);
	std::printf("g3 fit (guest code, float32/df32, CPU kernels): %s after %d iterations, %d evaluations, %d ticks, %.0f ms "
				"(%s); E %.9g -> %.9g (oracle %.9g -> %.9g, relative difference %.3e)\n",
			s == fitd::Driver::FAILED && d.error().find("line search") != std::string::npos ? "stopped, the last accepted iterate kept" : fitd::Driver::state_name(s), d.iterations(), d.nfev(), ticks, ms,
			s == fitd::Driver::FAILED ? d.error().c_str() : d.reason().c_str(), r0.f, r1.f, E0o, Eo,
			std::fabs(r1.f - Eo) / Eo);
	std::printf("g3 u: max|u - u*| = %.3e m (coordinate %zu, knot %zu), max|u| %.3f mm (u* %zu coordinates)\n", du, duat,
			duat / 3, 1e3 * umax, uo.size());
	for (int p = 0; p < 2; ++p) {
		std::printf("g3 pose %d: inside %u, deepest %.3f mm, within the %.0f mm margin %u of %zu\n", p, inside[p],
				-1e3 * deep[p], 1e3 * m, below[p], P);
	}
	const bool zero_inside = inside[0] == 0 && inside[1] == 0;
	std::printf("%s g3: 0 inside per pose %s; the u and E differences above are reported, not gated\n",
			zero_inside ? "PASS" : "FAIL", zero_inside ? "holds" : "does NOT hold");
	return zero_inside ? 0 : 1;
}

// Curvenet to cage: the scripted pen through curvenet_core, then cage_api.
int cmd_cn() {
	const cagefix::DressPen pen = cagefix::dress_pen();
	std::printf("cn pen: %s\n", pen.describe().c_str());
	cn::reset();
	cn::set_param("snap_radius", pen.snap_radius);
	cn::set_param("surface_offset", pen.surface_offset);
	std::printf("cn set_body: %s\n", cn::set_body(pen.body, pen.body_tris).c_str());
	const double t0 = now_ms();
	std::string last;
	for (const cagefix::PenStroke &s : pen.strokes) {
		cn::set_param("boundary", s.boundary ? 1 : 0);
		const int id = cn::pen_begin(s.xyzp[0], s.xyzp[1], s.xyzp[2], s.xyzp[3]);
		for (size_t i = 4; i < s.xyzp.size(); i += 4) {
			cn::pen_point(id, s.xyzp[i], s.xyzp[i + 1], s.xyzp[i + 2], s.xyzp[i + 3]);
		}
		last = cn::pen_end(id);
		std::printf("cn %s: %s\n", s.name.c_str(), last.c_str());
	}
	cn::set_param("boundary", 0);
	const std::string mb = cn::mesh_build(0.0, 1e-5);
	std::printf("cn mesh_build(0, 1e-5): %s [%.0f ms]\n", mb.c_str(), now_ms() - t0);
	const std::vector<float> mv = cn::mesh_vertices();
	const std::vector<int32_t> mi = cn::mesh_indices();
	const double t1 = now_ms();
	std::string cage = cageapi::net_from_patches(mv, mi, pen.bound, 100);
	int nticks = 0;
	while (cage.rfind("ok", 0) == 0 || cage.rfind("state=net ", 0) == 0) {
		cage = cageapi::tick(0, 1);
		++nticks;
	}
	std::printf("cn cage: %s [%d ticks, %.0f ms]\n", cage.c_str(), nticks, now_ms() - t1);
	if (cage.rfind("state=net_done", 0) != 0) {
		std::printf("FAIL cn: no cage\n");
		return 1;
	}
	const double t2 = now_ms();
	std::string b = cageapi::bind(pen.bound, true);
	int ticks = 0;
	while (b.rfind("FAIL", 0) != 0 && b.find("state=bound") == std::string::npos && ticks < 100000) {
		b = cageapi::tick(0, 1);
		++ticks;
	}
	std::printf("cn bind of the bodice (%zu vertices): %s [%d ticks, %.0f ms]\n", pen.bound.size() / 3, b.c_str(), ticks,
			now_ms() - t2);
	const bool ok = b.find("state=bound") != std::string::npos && cage.find(" outside=0 ") != std::string::npos;
	std::printf("%s cn: the curvenet cage is closed, manifold, consistently oriented, of positive volume, and contains "
				"every bound vertex (bhc13 bound it with require_contained)\n",
			ok ? "PASS" : "FAIL");
	// G2 through the API on this cage: the rest knots return the bodice.
	const std::vector<float> knots = knots_of(cageapi::net_vertices());
	const std::vector<float> y = cageapi::deform(knots);
	double e = 0;
	for (size_t i = 0; i < y.size(); ++i) {
		e = std::max(e, double(std::fabs(y[i] - pen.bound[i])));
	}
	std::printf("%s cn g2: zero displacement of the curvenet cage returns the bodice: max|y - x| = %.3e m\n",
			e <= kPTol && !y.empty() ? "PASS" : "FAIL", e);
	// Controls: the same patches uncapped (no net_from_patches) must be refused by
	// bind as an open net.
	{
		std::vector<int32_t> counts(mi.size() / 3, 3);
		cageapi::net_from_mesh(mv, mi, counts);
		std::string r = cageapi::bind(pen.bound, true);
		while (r.rfind("FAIL", 0) != 0 && r.find("state=bound") == std::string::npos) {
			r = cageapi::tick(0, 1);
		}
		const bool refused = r.rfind("FAIL", 0) == 0 && r.find("not closed") != std::string::npos;
		std::printf("%s cn control: the uncapped patch surface as a net: %s\n", refused ? "PASS" : "FAIL", r.c_str());
		return ok && refused && e <= kPTol ? 0 : 1;
	}
}

#include "gas.inc"

} // namespace

int main(int argc, char **argv) {
	const std::string cmd = argc > 1 ? argv[1] : "checks";
	if (cmd == "checks") {
		return cmd_checks();
	}
	if (cmd == "g1") {
		return cmd_g1(argc, argv);
	}
	if (cmd == "g3") {
		return cmd_g3(argc, argv);
	}
	if (cmd == "gas") {
		return cmd_gas(argc, argv);
	}
	if (cmd == "cn") {
		return cmd_cn();
	}
	std::fprintf(stderr, "usage: cage_native checks | g1 ... | g3 <oracle dir> <out dir> | cn\n");
	return 2;
}
