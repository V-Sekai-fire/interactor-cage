// checks -- cage.elf's gates that need no host data, run the same way in the
// guest (cage_check / cage_check_all) and natively (tests/cage_native, the
// flat control). One line each:
//   "PASS <name> ints=a,b,c fsig=<12 hex>/<count> :: <detail>"
// fsig is BLAKE3 (first 12 hex digits) over the bit patterns of the check's
// float outputs, so the gate compares guest and native bit for bit.
//
//   g2_zero      G2: the sphere scene's garment bound and deformed by the rest
//                cage comes back within 1e-5 m
//   g3_fit       G3's guest half: the in-motion fit of the sphere scene, from
//                u = 0 to convergence, clears the body by the margin in both
//                frames (the LBFGSpp host oracle, tests/cage_oracle, is the
//                other half)
//   g4_corrupt   G4: with the bind's fault switch on (one weight +0.01), G2's
//                comparison must FAIL
//   g4_push      G4: the fitted garment with one vertex pushed 5 mm into the
//                body must FAIL the clearance test
//   g4_open      G4: the scene's cage with one triangle removed must be
//                refused by bind, naming an open boundary
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <string>
#include <vector>

namespace cagecheck {

std::vector<std::string> names();

// One check as a resumable job: step() does one unit of its work (one bind
// step, one fit evaluation, ...) and returns true while there is more; the
// verdict line is ready once it returns false. The scene's bind and fit are
// shared by the checks and kept across jobs.
class Job {
public:
	explicit Job(const std::string &name);
	bool step();
	const std::string &line() const { return line_; }
	const std::string &name() const { return name_; }
	int steps() const { return steps_; }

private:
	std::string name_, line_;
	int entry_ = -1;
	int steps_ = 0;
};

// The whole check at once (Job run to the end).
std::string run(const std::string &name);
// Every check, one line each, then "checks: <passed>/<total>".
std::string run_all();

} // namespace cagecheck
