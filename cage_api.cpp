// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "cage_api.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <sstream>

#include "bhc13/bhc13.h"
#include "cage_fit.h"
#include "cage_net.h"
#include "checks.h"
#include "fit_driver.h"
#include "fixtures.h"

namespace cageapi {

namespace {

std::string fmt(const char *f, ...) {
	char b[2048];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

// Gas: in the guest, instructions retired in 2^20 units (execution_timeout's
// unit; libriscv counts rdinstret from the vmcall's start). Natively there is
// no instruction counter and one step is one unit, so a native gas of N is N
// steps.
uint64_t gas_now(uint64_t steps) {
#if defined(__riscv)
	uint64_t v;
	asm volatile("rdinstret %0" : "=r"(v));
	(void)steps;
	return v >> 20;
#else
	return steps;
#endif
}

enum Job { J_NONE, J_NET, J_BINDING, J_BOUND, J_FITTING, J_DONE, J_FAILED };

const char *job_name(Job j) {
	switch (j) {
		case J_NONE: return "idle";
		case J_NET: return "net";
		case J_BINDING: return "binding";
		case J_BOUND: return "bound";
		case J_FITTING: return "fitting";
		case J_DONE: return "done";
		case J_FAILED: return "failed";
	}
	return "?";
}

struct State {
	deform::Net net;
	bool have_net = false;
	cagenet::Builder builder;
	std::string net_line;
	deform::BindJob bindjob;
	deform::Bind bind;
	cagefit::Body body;
	cagefit::Skin skin;
	std::vector<uint8_t> frozen;
	cagefit::Params prm;
	bool have_fit = false;
	cagefit::Problem prob;
	fitd::Driver drv;
	std::vector<float> u;
	std::string fit_line;
	Job job = J_NONE;
	std::string err;
	uint64_t steps = 0; // work units since the job was queued
	// Checks, queued (cage_check / cage_check_all); they run before the job.
	std::deque<std::string> checks;
	std::unique_ptr<cagecheck::Job> check;
	int checks_passed = 0, checks_run = 0;
};

std::unique_ptr<State> &holder() {
	static std::unique_ptr<State> s(new State());
	return s;
}

State &st() {
	return *holder();
}

std::string failed(const std::string &why) {
	State &s = st();
	s.err = why;
	s.job = J_FAILED;
	return "FAIL: " + why;
}

// Bound vertices inside the body, per pose, at the last forward pass.
std::string pose_line(State &s) {
	const std::vector<float> &d = s.prob.distances();
	std::string out;
	const uint32_t P = s.bind.P;
	for (uint32_t p = 0; P && p < s.body.F && d.size() >= size_t(s.body.F) * P; ++p) {
		uint32_t in = 0;
		for (uint32_t i = 0; i < P; ++i) {
			in += d[size_t(p) * P + i] < 0.0f ? 1u : 0u;
		}
		out += (p ? "," : "") + std::to_string(in);
	}
	return out;
}

// One unit of work of whatever is queued; running is false once nothing is.
std::string step_once(bool &running) {
	State &s = st();
	running = true;
	++s.steps;
	if (s.check || !s.checks.empty()) {
		if (!s.check) {
			s.check.reset(new cagecheck::Job(s.checks.front()));
			s.checks.pop_front();
		}
		if (s.check->step()) {
			return "state=checking check=" + s.check->name() + fmt(" check_steps=%d", s.check->steps());
		}
		const std::string line = s.check->line();
		s.checks_passed += line.rfind("PASS", 0) == 0 ? 1 : 0;
		++s.checks_run;
		s.check.reset();
		running = !s.checks.empty() || (s.job == J_NET || s.job == J_BINDING || s.job == J_FITTING);
		return "state=checked " + line;
	}
	switch (s.job) {
		case J_NET: {
			const bool more = s.builder.step();
			if (s.builder.failed()) {
				running = false;
				return failed(s.builder.error());
			}
			if (more) {
				return std::string("state=net stage=") + s.builder.stage_name();
			}
			const cagenet::Result &r = s.builder.result();
			s.net = r.net;
			s.have_net = true;
			s.job = J_NONE;
			running = false;
			s.net_line = fmt("state=net_done knots=%u triangles=%u loops=%u volume=%.6g outside_before=%u outside=%u "
							 "inflate=%.4f wmin=%.4f wmax=%.4f :: ",
								 r.vertices, r.triangles, r.loops, r.volume, r.outside_before, r.outside, r.inflate,
								 r.wmin, r.wmax) +
					r.log;
			return s.net_line;
		}
		case J_BINDING: {
			s.bindjob.step();
			if (s.bindjob.failed()) {
				running = false;
				return failed(s.bindjob.error());
			}
			if (s.bindjob.done()) {
				s.bind = std::move(s.bindjob.result());
				s.job = J_BOUND;
				running = false;
				return fmt("state=bound P=%u knots=%u triangles=%u outside=%u volume=%.6g", s.bind.P, s.bind.nV,
						s.bind.nT, s.bind.outside, s.bind.volume);
			}
			return std::string("state=binding stage=") + s.bindjob.stage_name();
		}
		case J_FITTING: {
			const fitd::Driver::State d = s.drv.step(s.prob);
			if (d == fitd::Driver::FAILED && !s.drv.stalled()) {
				running = false;
				return failed(s.drv.error());
			}
			if (d == fitd::Driver::RUNNING && !s.drv.accepted()) {
				return fmt("state=fitting iteration=%d nfev=%d", s.drv.iterations(), s.drv.nfev());
			}
			s.u = s.drv.x();
			// The clearance at the accepted iterate: a forward pass (the
			// objective recomputes its own scratch at its next evaluation).
			const cagefit::Report r = s.prob.report(s.u.data());
			std::string how = "fitting";
			if (d != fitd::Driver::RUNNING) {
				s.job = J_DONE;
				running = false;
				how = d == fitd::Driver::CONVERGED ? "done reason=" + s.drv.reason()
												   : std::string("done reason=line_search_stalled_last_accepted_kept");
			}
			s.fit_line = fmt("state=%s iteration=%d nfev=%d E=%.9e pg=%.3e inside_per_pose=%s dmin=%.5f", how.c_str(),
					s.drv.iterations(), s.drv.nfev(), s.drv.fx(), s.drv.pgNorm(), pose_line(s).c_str(), r.dmin);
			return s.fit_line;
		}
		default:
			running = false;
			return std::string("state=") + job_name(s.job) + (s.job == J_FAILED ? " error=" + s.err : "");
	}
}

} // namespace

std::string reset() {
	holder().reset(new State());
	return "ok";
}

std::string net_from_mesh(const std::vector<float> &xyz, const std::vector<int32_t> &faces,
		const std::vector<int32_t> &counts) {
	State &s = st();
	std::string err;
	if (!deform::net_from_mesh(xyz, faces, counts, s.net, err)) {
		return "FAIL: " + err;
	}
	s.have_net = true;
	s.job = J_NONE;
	return fmt("ok knots=%u polygons=%zu", s.net.knot_count(), s.net.counts.size());
}

std::string net_from_patches(const std::vector<float> &vertices, const std::vector<int32_t> &triangles,
		const std::vector<float> &bound, int target_vertices) {
	State &s = st();
	cagenet::Options opt;
	if (target_vertices > 0) {
		opt.target_vertices = uint32_t(target_vertices);
	}
	std::string err;
	s.have_net = false;
	if (!s.builder.begin(vertices, triangles, bound, opt, err)) {
		return "FAIL: " + err;
	}
	s.job = J_NET;
	s.steps = 0;
	return fmt("ok queued net P=%zu patch_vertices=%zu patch_triangles=%zu", bound.size() / 3, vertices.size() / 3,
			triangles.size() / 3);
}

std::vector<float> net_vertices() {
	return st().net.xyz;
}

std::vector<int32_t> net_indices() {
	return st().net.faces;
}

std::string bind(const std::vector<float> &mesh_xyz, bool require_contained) {
	State &s = st();
	if (!s.have_net) {
		return "FAIL: no cage net (net_from_mesh, or net_from_patches ticked to state=net_done, first)";
	}
	deform::BindOptions opt;
	opt.require_contained = require_contained;
	std::string err;
	s.have_fit = false;
	if (!s.bindjob.begin(s.net, mesh_xyz, "bhc13", opt, err)) {
		return failed(err);
	}
	s.job = J_BINDING;
	s.steps = 0;
	return fmt("ok queued bind P=%zu K=%u samples=%u", mesh_xyz.size() / 3, s.bindjob.result().K(),
			s.bindjob.samples());
}

std::string body_frames(const std::vector<float> &xyz, const std::vector<int32_t> &tris, int frames,
		const std::vector<float> &frame_w) {
	State &s = st();
	if (frames <= 0 || xyz.size() % (3 * size_t(frames)) != 0 || tris.size() % 3 != 0 || tris.empty()) {
		return "FAIL: body_frames wants frames x BV x 3 floats and 3 indices a triangle";
	}
	s.body.F = uint32_t(frames);
	s.body.BV = uint32_t(xyz.size() / (3 * size_t(frames)));
	s.body.T = uint32_t(tris.size() / 3);
	s.body.xyz = xyz;
	s.body.tris.assign(tris.begin(), tris.end());
	s.body.frame_w = frame_w;
	return fmt("ok frames=%u body_vertices=%u body_triangles=%u", s.body.F, s.body.BV, s.body.T);
}

std::string fit_set(const std::vector<float> &bones, const std::vector<int32_t> &rowptr,
		const std::vector<int32_t> &col, const std::vector<float> &w, const std::vector<int32_t> &frozen,
		const std::string &params) {
	State &s = st();
	if (s.body.F == 0) {
		return "FAIL: body_frames first";
	}
	if (bones.size() % (12 * size_t(s.body.F)) != 0) {
		return "FAIL: bones must be frames x bones x 12 floats";
	}
	s.skin.NB = uint32_t(bones.size() / (12 * size_t(s.body.F)));
	s.skin.bones = bones;
	s.skin.rowptr.assign(rowptr.begin(), rowptr.end());
	s.skin.col.assign(col.begin(), col.end());
	s.skin.w = w;
	s.frozen.assign(frozen.begin(), frozen.end());
	cagefit::Params p;
	std::istringstream ss(params);
	std::string k, v, rest;
	while (ss >> k) {
		if (!(ss >> v)) {
			return "FAIL: param " + k + " has no value";
		}
		if (k == "margin") {
			p.margin = std::strtof(v.c_str(), nullptr);
		} else if (k == "w_L") {
			p.wL = std::strtof(v.c_str(), nullptr);
		} else if (k == "w_r") {
			p.wr = std::strtof(v.c_str(), nullptr);
		} else if (k == "box") {
			p.box = std::strtof(v.c_str(), nullptr);
		} else if (k == "laplacian") {
			if (v != "normalized" && v != "combinatorial") {
				return "FAIL: laplacian is normalized or combinatorial";
			}
			p.lap_combinatorial = v == "combinatorial";
		} else {
			rest += k + " " + v + " ";
		}
	}
	std::string err;
	if (!p.lb.parse(rest, err)) {
		return "FAIL: " + err;
	}
	s.prm = p;
	s.have_fit = true;
	return fmt("ok bones=%u margin=%.4f w_L=%g w_r=%g :: %s", s.skin.NB, p.margin, p.wL, p.wr, p.lb.dump().c_str());
}

std::string queue_fit() {
	State &s = st();
	if (s.job != J_BOUND && s.job != J_DONE) {
		return std::string("FAIL: the bind is not done (state ") + job_name(s.job) + ")";
	}
	if (!s.have_fit) {
		return "FAIL: fit_set first";
	}
	std::string err;
	if (!s.prob.setup(s.bind, s.body, s.skin, s.frozen, s.prm, err)) {
		return failed(err);
	}
	std::vector<float> lb, ub;
	s.prob.bounds(lb, ub);
	s.u.assign(s.prob.n(), 0.0f);
	if (!s.drv.begin(s.prob.n(), s.u, lb, ub, s.prm.lb, err)) {
		return failed(err);
	}
	s.job = J_FITTING;
	s.steps = 0;
	const cagefit::Report r = s.prob.report(s.u.data());
	return fmt("ok queued fit n=%u E0=%.9e below_margin=%u inside_per_pose=%s dmin=%.5f", s.prob.n(), r.f,
			r.below_margin, pose_line(s).c_str(), r.dmin);
}

std::string tick(uint64_t host_us, uint64_t gas) {
	State &s = st();
	(void)host_us; // the host's clock, for its own timing; the guest clock is not a clock
	const uint64_t budget = gas == 0 ? 1 : gas;
	uint64_t n = 0;
	const uint64_t g0 = gas_now(0);
	std::string line;
	bool running = true;
	do {
		line = step_once(running);
		++n;
		// A finished check, fit iterate or net is worth a host look: return.
		if (line.rfind("state=checked", 0) == 0 || line.rfind("state=net_done", 0) == 0 ||
				(line.rfind("state=fitting iteration", 0) == 0 && line.find(" E=") != std::string::npos) ||
				line.rfind("FAIL", 0) == 0) {
			break;
		}
	} while (running && gas_now(n) - g0 < budget);
	return line + fmt(" | steps=%llu total_steps=%llu", (unsigned long long)n, (unsigned long long)s.steps);
}

std::vector<float> result() {
	State &s = st();
	std::vector<float> y;
	if (s.job == J_DONE) {
		s.prob.garment(s.u.data(), y);
	}
	return y;
}

std::vector<float> result_u() {
	State &s = st();
	return s.job == J_DONE ? s.u : std::vector<float>();
}

std::string report() {
	State &s = st();
	if (s.check || !s.checks.empty()) {
		return fmt("progress checks: %d run, %d passed, %zu queued", s.checks_run, s.checks_passed,
				s.checks.size() + (s.check ? 1 : 0));
	}
	switch (s.job) {
		case J_DONE: {
			const cagefit::Report r = s.prob.report(s.u.data());
			return fmt("ok f=%.9e contact=%.3e lap=%.3e reg=%.3e below_margin=%u inside=%u inside_per_pose=%s "
					   "dmin=%.5f worst=(frame %u, vertex %u)",
					r.f, r.contact, r.lap, r.reg, r.below_margin, r.inside, pose_line(s).c_str(), r.dmin,
					r.worst_frame, r.worst_vertex);
		}
		case J_FITTING:
			return "progress " + s.fit_line + fmt(" steps=%llu", (unsigned long long)s.steps);
		case J_BINDING:
			return std::string("progress binding stage=") + s.bindjob.stage_name() +
					fmt(" steps=%llu", (unsigned long long)s.steps);
		case J_NET:
			return std::string("progress net stage=") + s.builder.stage_name();
		default:
			return std::string("progress state=") + job_name(s.job) + (s.job == J_FAILED ? " error=" + s.err : "") +
					(s.net_line.empty() ? "" : " last_net: " + s.net_line);
	}
}

std::vector<float> weights() {
	State &s = st();
	return s.job >= J_BOUND && s.job != J_FAILED ? deform::weights(s.bind) : std::vector<float>();
}

std::vector<float> deform(const std::vector<float> &knots_posed) {
	State &s = st();
	std::vector<float> y;
	std::string err;
	if (s.job >= J_BOUND && s.job != J_FAILED) {
		deform::deform(s.bind, knots_posed, y, err);
	}
	return y;
}

std::string build() {
	return "FAIL: cage_build is Phase C's (the NDMF pass, RFD 2277); not built in Phase A";
}

std::string preview() {
	return "FAIL: cage_preview is Phase C's (an IRenderFilter, RFD 2277); not built in Phase A";
}

std::string dump() {
	return "FAIL: cage_dump is Phase C's (its hash must equal the build's, RFD 2277); not built in Phase A";
}

std::vector<float> pen_script() {
	return cagefix::encode_pen(cagefix::dress_pen());
}

std::vector<float> pen_body_vertices() {
	return cagefix::dress_pen().body;
}

std::vector<int32_t> pen_body_indices() {
	return cagefix::dress_pen().body_tris;
}

std::vector<float> pen_bound() {
	return cagefix::dress_pen().bound;
}

std::string pen_describe() {
	return cagefix::dress_pen().describe();
}

std::vector<std::string> check_names() {
	return cagecheck::names();
}

std::string check(const std::string &name) {
	st().checks.push_back(name);
	return "ok queued check " + name;
}

std::string check_all() {
	for (const std::string &n : cagecheck::names()) {
		st().checks.push_back(n);
	}
	return fmt("ok queued %zu checks", cagecheck::names().size());
}

} // namespace cageapi
