// fit_driver -- a generic L-BFGS-B fit in reverse communication, one solver
// iteration per host tick (AGENTS.md rule 4).
//
// dress-on's in-guest L-BFGS-B (guest/drape/lbfgsb.h: LBFGSpp 0.3.0's
// LBFGSBSolver over the Lean-emitted kernels, float32 vectors with df32
// sums, double scalars) on its CPU backend (VecCpu), driven by an Objective
// the caller implements. Nothing here knows what is being fitted: cage.elf's
// in-motion cage fit (guest/cage/cage_fit.h) and the ANNY head fit (hf_*)
// both sit on it.
//
//   Driver d;
//   d.begin(n, x0, lb, ub, params, err);        // lb == ub freezes a coordinate
//   while (d.tick(obj) == Driver::RUNNING) {}    // one call per host tick
//   d.x();                                       // the result (or the last iterate)
//
// tick() runs the solver until it accepts an iterate (one L-BFGS-B
// iteration: the direction phase, then the line search's evaluations), or
// converges, or fails; each evaluation the solver asks for is the
// Objective's evaluate() at that point. std types only.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lbfgsb.h"
#include "vec_cpu.h"

namespace fitd {

class Objective {
public:
	virtual ~Objective() = default;
	// f and its gradient g (n floats) at x (n floats).
	virtual bool evaluate(const float *x, float *g, double &f, std::string &err) = 0;
};

struct Iterate {
	int iteration = 0; // 0: the start point
	int nfev = 0;
	double f = 0.0;
	double pg = 0.0;   // ||P(x - g) - x||_inf
};

class Driver {
public:
	enum State { IDLE, RUNNING, CONVERGED, FAILED };
	static const char *state_name(State s);

	bool begin(uint32_t n, const std::vector<float> &x0, const std::vector<float> &lb, const std::vector<float> &ub,
			const LbfgsbParams &p, std::string &err);
	// One solver iteration (until the next accepted iterate), or the end.
	State tick(Objective &obj);
	// The smallest resumable unit: one objective evaluation, or one solver
	// phase between them. A host with an instruction budget (gas) calls this
	// until the budget is spent; tick() is step() until an iterate is accepted.
	// accepted() says whether the last step accepted one.
	State step(Objective &obj);
	bool accepted() const { return accepted_; }
	State state() const { return state_; }

	// The current point: the result once CONVERGED; after FAILED, the last
	// accepted iterate (x0 before any), as LBFGSpp's callers keep it.
	const std::vector<float> &x();
	// FAILED because the line search stalled (a kink of the objective, e.g. a
	// signed distance switching features): the last accepted iterate stands.
	bool stalled() const { return state_ == FAILED && err_.find("line search") != std::string::npos; }
	const std::vector<Iterate> &trace() const { return trace_; }
	int iterations() const { return lb_.iterations(); }
	int nfev() const { return lb_.nfev(); }
	double fx() const { return lb_.fx(); }
	double pgNorm() const { return lb_.pgNorm(); }
	const std::string &reason() const { return lb_.reason(); }
	const std::string &error() const { return err_; }

private:
	State fail(const std::string &why);
	VecCpu vec_;
	Lbfgsb lb_{ vec_ };
	Lbfgsb::Status st_ = Lbfgsb::FAIL;
	State state_ = IDLE;
	uint32_t n_ = 0;
	std::vector<float> x_, g_, xacc_;
	bool accepted_ = false;
	std::vector<Iterate> trace_;
	std::string err_;
};

} // namespace fitd
