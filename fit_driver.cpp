// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "fit_driver.h"

namespace fitd {

const char *Driver::state_name(State s) {
	switch (s) {
		case IDLE: return "idle";
		case RUNNING: return "running";
		case CONVERGED: return "converged";
		case FAILED: return "failed";
	}
	return "?";
}

Driver::State Driver::fail(const std::string &why) {
	err_ = why;
	state_ = FAILED;
	return state_;
}

bool Driver::begin(uint32_t n, const std::vector<float> &x0, const std::vector<float> &lb, const std::vector<float> &ub,
		const LbfgsbParams &p, std::string &err) {
	trace_.clear();
	err_.clear();
	n_ = n;
	if (x0.size() != n || lb.size() != n || ub.size() != n) {
		err = "fit: x0, lb and ub must each hold n = " + std::to_string(n) + " floats";
		state_ = FAILED;
		err_ = err;
		return false;
	}
	x_ = x0;
	xacc_ = x0;
	g_.assign(n, 0.0f);
	st_ = lb_.start(n, x0.data(), lb.data(), ub.data(), p);
	if (st_ == Lbfgsb::FAIL) {
		err = "fit: " + lb_.error();
		state_ = FAILED;
		err_ = err;
		return false;
	}
	state_ = RUNNING;
	return true;
}

Driver::State Driver::step(Objective &obj) {
	accepted_ = false;
	if (state_ != RUNNING) {
		return state_;
	}
	switch (st_) {
		case Lbfgsb::BUSY:
		case Lbfgsb::ACCEPT:
			st_ = lb_.next();
			break;
		case Lbfgsb::NEED_EVAL:
		case Lbfgsb::TRY: {
			if (!lb_.readX(x_)) {
				return fail("fit: readX: " + vec_.error());
			}
			double f = 0.0;
			std::string err;
			if (!obj.evaluate(x_.data(), g_.data(), f, err)) {
				return fail("fit: evaluate: " + err);
			}
			if (!lb_.setGradient(g_.data())) {
				return fail("fit: setGradient: " + vec_.error());
			}
			if (st_ == Lbfgsb::NEED_EVAL) {
				trace_.push_back({ 0, 1, f, 0.0 });
			}
			st_ = lb_.next(f);
			break;
		}
		case Lbfgsb::CONVERGED:
		case Lbfgsb::FAIL:
			break;
	}
	switch (st_) {
		case Lbfgsb::ACCEPT:
			lb_.readX(xacc_);
			trace_.push_back({ lb_.iterations(), lb_.nfev(), lb_.fx(), lb_.pgNorm() });
			accepted_ = true;
			return state_;
		case Lbfgsb::CONVERGED:
			trace_.push_back({ lb_.iterations(), lb_.nfev(), lb_.fx(), lb_.pgNorm() });
			lb_.readX(x_);
			state_ = CONVERGED;
			return state_;
		case Lbfgsb::FAIL:
			return fail("fit: " + lb_.error());
		default:
			return state_;
	}
}

Driver::State Driver::tick(Objective &obj) {
	State s = state_;
	do {
		s = step(obj);
	} while (s == RUNNING && !accepted_);
	return s;
}

const std::vector<float> &Driver::x() {
	if (state_ == RUNNING || state_ == CONVERGED) {
		lb_.readX(x_);
	} else if (state_ == FAILED) {
		x_ = xacc_;
	}
	return x_;
}

} // namespace fitd
