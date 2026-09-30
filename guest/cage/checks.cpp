// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "checks.h"

#include "skin_bake.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>

#include "common/blake3.h"
#include "bhc13/bhc13.h"
#include "cage_fit.h"
#include "fit_driver.h"
#include "fixtures.h"

namespace cagecheck {

namespace {

std::string fmt(const char *f, ...) {
	char b[2048];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

std::string fsig(const std::vector<float> &v) {
	blake3::Ctx h;
	for (float x : v) {
		uint32_t bits;
		std::memcpy(&bits, &x, 4);
		unsigned char le[4];
		for (int i = 0; i < 4; ++i) {
			le[i] = (unsigned char)(bits >> (8 * i));
		}
		h.update(le, 4);
	}
	return h.hex().substr(0, 12);
}

struct Out {
	bool pass = false;
	std::vector<long long> ints;
	std::vector<float> floats;
	std::string detail;
};

// The sphere scene, bound once per process and shared by the checks; its
// bind is a resumable job (scene_step), one BindJob step a call.
struct Scene {
	cagefix::SphereScene s;
	deform::Net net;
	deform::BindJob job;
	deform::Bind bind;
	std::string err;
	bool started = false, done = false, ok = false;
};

Scene &scene() {
	static std::unique_ptr<Scene> sc;
	if (!sc) {
		sc.reset(new Scene());
	}
	return *sc;
}

// Advances the scene's bind by one step; false once it is done (ok or not).
bool scene_step() {
	Scene &sc = scene();
	if (sc.done) {
		return false;
	}
	if (!sc.started) {
		sc.started = true;
		sc.s = cagefix::sphere_scene();
		if (!deform::net_from_mesh(sc.s.cage, sc.s.faces, sc.s.counts, sc.net, sc.err) ||
				!sc.job.begin(sc.net, sc.s.garment, "bhc13", deform::BindOptions(), sc.err)) {
			sc.done = true;
			return false;
		}
		return true;
	}
	sc.job.step();
	if (sc.job.failed()) {
		sc.err = sc.job.error();
		sc.done = true;
	} else if (sc.job.done()) {
		sc.bind = std::move(sc.job.result());
		sc.ok = true;
		sc.done = true;
	}
	return !sc.done;
}

double max_diff(const std::vector<float> &a, const std::vector<float> &b) {
	double m = 0.0;
	for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
		m = std::max(m, double(std::fabs(a[i] - b[i])));
	}
	return m;
}

std::vector<float> rest_knots(const deform::Bind &b) {
	std::vector<float> k(12 * size_t(b.nV), 0.0f);
	for (uint32_t v = 0; v < b.nV; ++v) {
		k[12 * v + 0] = k[12 * v + 5] = k[12 * v + 10] = 1.0f;
		for (int r = 0; r < 3; ++r) {
			k[12 * v + 4 * r + 3] = b.rest[3 * v + r];
		}
	}
	return k;
}

constexpr double kG2Tol = 1e-5;

// The clearance test G3 and its push control share: nothing inside the body
// and the closest vertex within 0.1 mm of the margin. The quadratic hinge
// balances its last few vertices against the regularizers a hair below m,
// so "0 below m" is reported, not required (RFD 2277 gate D asks it of the
// avatar, where the fit margin will carry that headroom).
constexpr float kClearTol = 1e-4f;
bool clears(const cagefit::Report &r, float m) {
	return r.inside == 0 && r.dmin >= m - kClearTol;
}


// S1: bake_skin on the sphere scene. Knots with x < 0 carry bone 0 and the rest
// bone 1, so every point in the outer quarter of each side is mostly its own
// side's bone and every row
// must sum to 1. The control hands every knot bone 0 and must FAIL the split.
static Out s1_run(bool control) {
	Out o;
	Scene &sc = scene();
	if (!sc.ok) {
		o.detail = "bind: " + sc.err;
		return o;
	}
	const deform::Bind &b = sc.bind;
	// Two slots a knot: knots on the plane (|x| < 1 mm) split evenly, so the check is symmetric.
	std::vector<int32_t> kb(size_t(b.nV) * 2, -1);
	std::vector<float> kw(size_t(b.nV) * 2, 0.0f);
	uint32_t on_plane = 0;
	for (uint32_t v = 0; v < b.nV; ++v) {
		const float x = b.rest[3 * v];
		if (control) {
			kb[2 * v] = 0;
			kw[2 * v] = 1.0f;
		} else if (std::fabs(x) < 1e-3f) {
			kb[2 * v] = 0; kw[2 * v] = 0.5f; kb[2 * v + 1] = 1; kw[2 * v + 1] = 0.5f;
			++on_plane;
		} else {
			kb[2 * v] = x < 0.0f ? 0 : 1;
			kw[2 * v] = 1.0f;
		}
	}
	deform::Skin sk;
	std::string err;
	if (!deform::bake_skin(b, kb, kw, 2, 4, sk, err)) {
		o.detail = err;
		return o;
	}
	float xmin = 1e9f, xmax = -1e9f;
	for (uint32_t p = 0; p < b.P; ++p) {
		xmin = std::min(xmin, sc.s.garment[3 * p]);
		xmax = std::max(xmax, sc.s.garment[3 * p]);
	}
	const float band = 0.25f * (xmax - xmin);
	uint32_t left = 0, left_ok = 0, right = 0, right_ok = 0;
	double worst_sum = 0.0;
	for (uint32_t p = 0; p < b.P; ++p) {
		float w0 = 0.0f, w1 = 0.0f, sum = 0.0f;
		for (uint32_t k = 0; k < sk.influences; ++k) {
			const int32_t bone = sk.bones[size_t(p) * sk.influences + k];
			const float w = sk.weights[size_t(p) * sk.influences + k];
			sum += w;
			if (bone == 0) w0 += w;
			if (bone == 1) w1 += w;
		}
		worst_sum = std::max(worst_sum, double(std::fabs(sum - 1.0f)));
		const float x = sc.s.garment[3 * p];
		if (x < xmin + band) { ++left; left_ok += w0 > 0.5f; }
		if (x > xmax - band) { ++right; right_ok += w1 > 0.5f; }
	}
	o.pass = sk.empty == 0 && worst_sum < 1e-5 && left > 0 && right > 0 && left_ok == left && right_ok == right;
	o.ints = { b.P, sk.empty, on_plane, left, left_ok, right, right_ok };
	o.floats = sk.weights;
	o.detail = fmt("%s: %u points (%u knots on the plane), empty %u, |row sum - 1| max %.2e; left band %u/%u bone 0 > 0.5, right band %u/%u bone 1 > 0.5",
			control ? "control (every knot bone 0)" : "split at x = 0", b.P, on_plane, sk.empty, worst_sum, left_ok, left, right_ok, right);
	return o;
}

static Out s1_skin() { return s1_run(false); }
static Out s1_skin_control() { Out o = s1_run(true); o.pass = !o.pass; o.detail = "must FAIL the split: " + o.detail; return o; }

Out g2_zero() {
	Out o;
	Scene &sc = scene();
	if (!sc.ok) {
		o.detail = "bind: " + sc.err;
		return o;
	}
	std::vector<float> y;
	std::string err;
	if (!deform::deform(sc.bind, rest_knots(sc.bind), y, err)) {
		o.detail = err;
		return o;
	}
	const double e = max_diff(y, sc.s.garment);
	o.pass = e <= kG2Tol;
	o.ints = { sc.bind.nV, sc.bind.nT, sc.bind.P, sc.bind.outside };
	o.floats = y;
	o.detail = fmt("sphere scene: cage %u knots %u triangles, garment %u vertices, all inside (winding); "
				   "zero displacement: max|y - x| = %.3e m (tolerance %.0e)",
			sc.bind.nV, sc.bind.nT, sc.bind.P, e, kG2Tol);
	return o;
}

// The fit, run once per process and shared by g3 and g4_push; resumable
// (fit_step): one driver step (one evaluation or one solver phase) a call.
struct Fit {
	bool started = false, done = false, ok = false;
	std::string err, reason;
	cagefit::Problem prob;
	fitd::Driver drv;
	cagefit::Report before, after;
	std::vector<float> u, y;
	int iterations = 0, nfev = 0, ticks = 0;
};

Fit &fit() {
	static std::unique_ptr<Fit> f;
	if (!f) {
		f.reset(new Fit());
	}
	return *f;
}

// Advances the fit by one step (the scene's bind first); false once done.
bool fit_step() {
	Fit &f = fit();
	if (f.done) {
		return false;
	}
	if (scene_step()) {
		return true;
	}
	Scene &sc = scene();
	if (!f.started) {
		f.started = true;
		if (!sc.ok) {
			f.err = "bind: " + sc.err;
			f.done = true;
			return false;
		}
		if (!f.prob.setup(sc.bind, sc.s.body, sc.s.skin, sc.s.frozen, sc.s.params, f.err)) {
			f.done = true;
			return false;
		}
		const uint32_t n = f.prob.n();
		std::vector<float> u0(n, 0.0f), lb, ub;
		f.prob.bounds(lb, ub);
		f.before = f.prob.report(u0.data());
		if (!f.drv.begin(n, u0, lb, ub, sc.s.params.lb, f.err)) {
			f.done = true;
			return false;
		}
		return true;
	}
	const fitd::Driver::State st = f.drv.step(f.prob);
	// A tick is an accepted iterate or the end (fit_driver.h's tick()).
	f.ticks += (f.drv.accepted() || st != fitd::Driver::RUNNING) ? 1 : 0;
	if (st == fitd::Driver::RUNNING && f.ticks < 1000) {
		return true;
	}
	f.done = true;
	f.u = f.drv.x();
	f.iterations = f.drv.iterations();
	f.nfev = f.drv.nfev();
	f.reason = f.drv.reason();
	if (st != fitd::Driver::CONVERGED) {
		f.err = st == fitd::Driver::FAILED ? f.drv.error() : "no convergence in 1000 ticks";
		return false;
	}
	f.after = f.prob.report(f.u.data());
	f.prob.garment(f.u.data(), f.y);
	f.ok = true;
	return false;
}

Out g3_fit() {
	Out o;
	Fit &f = fit();
	if (!f.ok) {
		o.detail = f.err;
		return o;
	}
	const float m = scene().s.params.margin;
	double umax = 0.0;
	for (float x : f.u) {
		umax = std::max(umax, double(std::fabs(x)));
	}
	o.pass = clears(f.after, m) && f.after.f < 0.05 * f.before.f;
	o.ints = { f.iterations, f.nfev, f.ticks, f.before.below_margin, f.before.inside, f.after.below_margin,
		f.after.inside };
	o.floats = f.u;
	o.detail = fmt("in-motion fit, 2 frames, m = %.3f m, w_L %.3g, w_r %.3g (clears: 0 inside, d_min >= m - 0.1 mm): f %.6e -> %.6e (contact %.3e, |Lu|^2 %.3e, "
				   "|u|^2 %.3e); below margin %u -> %u, inside %u -> %u of %u; d_min %.3f -> %.3f mm; "
				   "%d iterations, %d evaluations, %d ticks (%s); max|u| %.3f mm",
			m, scene().s.params.wL, scene().s.params.wr, f.before.f, f.after.f, f.after.contact, f.after.lap,
			f.after.reg, f.before.below_margin, f.after.below_margin, f.before.inside, f.after.inside,
			unsigned(f.u.empty() ? 0 : scene().bind.P * scene().s.body.F), 1e3 * f.before.dmin, 1e3 * f.after.dmin,
			f.iterations, f.nfev, f.ticks, f.reason.c_str(), 1e3 * umax);
	return o;
}

// G4's corrupted bind, its own resumable job (the fault switch on).
struct Corrupt {
	deform::BindJob job;
	bool started = false, done = false;
	std::string err;
};

Corrupt &corrupt() {
	static std::unique_ptr<Corrupt> c;
	if (!c) {
		c.reset(new Corrupt());
	}
	return *c;
}

bool corrupt_step() {
	if (scene_step()) {
		return true;
	}
	Corrupt &c = corrupt();
	if (c.done) {
		return false;
	}
	if (!c.started) {
		c.started = true;
		deform::BindOptions opt;
		opt.fault = 1;
		if (!c.job.begin(scene().net, scene().s.garment, "bhc13", opt, c.err)) {
			c.done = true;
		}
		return !c.done;
	}
	c.job.step();
	c.done = c.job.done() || c.job.failed();
	if (c.job.failed()) {
		c.err = c.job.error();
	}
	return !c.done;
}

Out g4_corrupt() {
	Out o;
	Scene &sc = scene();
	Corrupt &c = corrupt();
	if (!c.job.done()) {
		o.detail = "bind: " + c.err;
		return o;
	}
	const deform::Bind &b = c.job.result();
	std::string err;
	std::vector<float> y;
	deform::deform(b, rest_knots(b), y, err);
	const double e = max_diff(y, sc.s.garment);
	const bool g2_fails = !(e <= kG2Tol);
	o.pass = g2_fails;
	o.ints = { g2_fails ? 1 : 0 };
	o.floats = { float(e) };
	o.detail = fmt("fault switch on (W[0][0] += 0.01): G2's max|y - x| = %.3e m %s the %.0e tolerance, so G2 %s",
			e, g2_fails ? "exceeds" : "is within", kG2Tol, g2_fails ? "FAILS as it must" : "PASSES: the gate is blind");
	return o;
}

Out g4_push() {
	Out o;
	Fit &f = fit();
	if (!f.ok) {
		o.detail = f.err;
		return o;
	}
	// Push the vertex with the most clearance 5 mm toward the body's centre.
	const float m = scene().s.params.margin;
	cagefit::Report r0 = f.prob.report_y(f.y.data());
	const std::vector<float> &d = f.prob.distances();
	const uint32_t P = scene().bind.P;
	uint32_t pick = 0;
	for (uint32_t i = 0; i < P; ++i) {
		if (d[i] > d[pick]) {
			pick = i;
		}
	}
	std::vector<float> y = f.y;
	const float *p = &y[3 * pick];
	const float l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
	const float dir[3] = { p[0] / l, p[1] / l, p[2] / l };
	const float clear0 = d[pick];
	for (int k = 0; k < 3; ++k) {
		y[3 * pick + k] -= (clear0 + 0.005f) * dir[k];
	}
	cagefit::Report r1 = f.prob.report_y(y.data());
	const bool clear_fails = !clears(r1, m);
	o.pass = clears(r0, m) && clear_fails;
	o.ints = { pick, r0.below_margin, r1.below_margin, r1.inside };
	o.floats = { r1.dmin };
	o.detail = fmt("fitted garment: %u below margin (d_min %.3f mm); vertex %u pushed %.3f mm inward: %u below margin, "
				   "%u inside, d_min %.3f mm, so the clearance test %s",
			r0.below_margin, 1e3 * r0.dmin, pick, 1e3 * (clear0 + 0.005f), r1.below_margin, r1.inside, 1e3 * r1.dmin,
			clear_fails ? "FAILS as it must" : "PASSES: the gate is blind");
	if (!clears(r0, m)) {
		o.detail += " (not a control: the fitted garment itself fails the clearance test)";
	}
	return o;
}

Out g4_open() {
	Out o;
	Scene &sc = scene();
	std::vector<int32_t> faces(sc.s.faces.begin() + 3, sc.s.faces.end());
	std::vector<int32_t> counts(sc.s.counts.begin() + 1, sc.s.counts.end());
	deform::Net net;
	deform::Bind b;
	std::string err;
	if (!deform::net_from_mesh(sc.s.cage, faces, counts, net, err)) {
		o.detail = "net_from_mesh: " + err;
		return o;
	}
	const bool bound = deform::bind(net, sc.s.garment, "bhc13", b, err);
	const bool named = err.find("not closed") != std::string::npos;
	o.pass = !bound && named;
	o.ints = { bound ? 1 : 0, named ? 1 : 0 };
	o.detail = bound ? "bind ACCEPTED an open net: the gate is blind"
					 : "triangle 0 removed: bind refused: " + err;
	return o;
}

using CheckFn = Out (*)();
using StepFn = bool (*)();
struct Entry {
	const char *name;
	StepFn prep; // the resumable work the check needs, one step a call
	CheckFn fn;  // the verdict, once prep is done (short)
};
const Entry kChecks[] = {
	{ "g2_zero", scene_step, g2_zero },
	{ "g3_fit", fit_step, g3_fit },
	{ "g4_corrupt", corrupt_step, g4_corrupt },
	{ "g4_push", fit_step, g4_push },
	{ "g4_open", scene_step, g4_open },
	{ "s1_skin", scene_step, s1_skin },
	{ "s1_skin_control", scene_step, s1_skin_control },
};

std::string line_of(const std::string &name, const Out &o) {
	std::string ints;
	for (size_t i = 0; i < o.ints.size(); ++i) {
		ints += (i ? "," : "") + std::to_string(o.ints[i]);
	}
	return fmt("%s %s ints=%s fsig=%s/%d :: ", o.pass ? "PASS" : "FAIL", name.c_str(), ints.c_str(),
				   fsig(o.floats).c_str(), int(o.floats.size())) +
			o.detail;
}

} // namespace

std::vector<std::string> names() {
	std::vector<std::string> n;
	for (const Entry &e : kChecks) {
		n.push_back(e.name);
	}
	return n;
}

Job::Job(const std::string &name) :
		name_(name) {
	for (size_t i = 0; i < sizeof(kChecks) / sizeof(kChecks[0]); ++i) {
		if (name == kChecks[i].name) {
			entry_ = int(i);
		}
	}
	if (entry_ < 0) {
		line_ = "FAIL " + name + " ints= fsig=000000000000/0 :: unknown check";
	}
}

bool Job::step() {
	if (!line_.empty()) {
		return false;
	}
	++steps_;
	if (kChecks[entry_].prep()) {
		return true;
	}
	line_ = line_of(name_, kChecks[entry_].fn());
	return false;
}

std::string run(const std::string &name) {
	Job j(name);
	while (j.step()) {
	}
	return j.line();
}

std::string run_all() {
	std::string out;
	int passed = 0, total = 0;
	for (const Entry &e : kChecks) {
		const std::string line = run(e.name);
		passed += line.rfind("PASS", 0) == 0 ? 1 : 0;
		++total;
		out += line + "\n";
	}
	return out + "checks: " + std::to_string(passed) + "/" + std::to_string(total);
}

} // namespace cagecheck
