// cage_net -- curvenet patches to a closed cage (RFD 2277, "Curvenet to cage").
//
// curvenet.elf's mesh_build merges its patches (welded, oriented away from the
// body, CCW-outward per mesh_wire); that surface is open where the pen drew
// boundary strokes (a neckline, a hem). This:
//   1. welds within weld_eps and drops degenerate triangles;
//   2. caps every boundary loop with a fan about the loop's centroid, wound
//      against the loop so the capped surface stays consistently oriented;
//   3. coarsens to about target_vertices with PMP's quadric decimation
//      (vendor/pmp-subset, the curvenet stage's PMP);
//   4. checks the result as bhc13's bind will (deform::check_net: closed, edge-
//      and vertex-manifold, consistently oriented, positive volume);
//   5. tests containment of every bound vertex with the generalized winding
//      number (cage_winding, a Lean kernel) and, when some lie outside and
//      inflate_max > 0, offsets the cage along its area-weighted vertex
//      normals in inflate_step increments until every one is inside (the
//      offset is reported: a coarse cage's flat faces cut inside the patch
//      surface they approximate);
//   6. hands the triangles to deform::net_from_mesh.
// std types only (PMP inside), so cage.elf and the native gates share it.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bhc13/bhc13.h"

namespace cagenet {

struct Options {
	uint32_t target_vertices = 100;
	double weld_eps = 1e-5;
	float inflate_step = 0.002f;
	float inflate_max = 0.05f;
};

struct Result {
	deform::Net net;
	uint32_t in_vertices = 0, in_triangles = 0;   // after the weld
	uint32_t loops = 0, cap_triangles = 0;        // boundary loops capped
	uint32_t capped_vertices = 0;                 // before decimation
	uint32_t vertices = 0, triangles = 0;         // the cage
	double volume = 0.0;
	uint32_t outside_before = 0, outside = 0;     // bound vertices outside, before / after inflation
	float inflate = 0.0f;                         // the offset applied (m)
	double wmin = 0.0, wmax = 0.0;                // winding range over the bound vertices
	std::string log;
};

// The same, resumable (AGENTS.md rule 4; cage.elf's gas): step() does one stage
// (weld + cap + decimate + check; then one inflation offset and its
// containment test a step; then the net) and returns true while there is more.
class Builder {
public:
	bool begin(const std::vector<float> &vertices, const std::vector<int32_t> &triangles,
			const std::vector<float> &bound, const Options &opt, std::string &err);
	bool step();
	bool done() const { return stage_ == S_DONE; }
	bool failed() const { return stage_ == S_FAILED; }
	const std::string &error() const { return err_; }
	Result &result() { return out_; }
	const char *stage_name() const {
		return stage_ == S_PREPARE ? "prepare" : stage_ == S_INFLATE ? "inflate" : stage_ == S_FINISH ? "finish" : stage_ == S_DONE ? "done" : stage_ == S_FAILED ? "failed" : "idle";
	}

private:
	enum Stage { S_IDLE, S_PREPARE, S_INFLATE, S_FINISH, S_DONE, S_FAILED };
	bool prepare();
	bool fail(const std::string &why);
	uint32_t count_outside(const std::vector<float> &c, double &wmin, double &wmax) const;
	Stage stage_ = S_IDLE;
	Options opt_;
	Result out_;
	std::string err_;
	std::vector<float> vin_, bound_, cage_, vn_, moved_;
	std::vector<int32_t> tin_;
	std::vector<uint32_t> ctri_;
	deform::Net probe_;
	float d_ = 0.0f;
};

// The whole build at once (Builder run to the end).
bool build_cage(const std::vector<float> &vertices, const std::vector<int32_t> &triangles,
		const std::vector<float> &bound, const Options &opt, Result &out, std::string &err);

// The winding number of each bound vertex in a triangle cage (cage_winding).
std::vector<double> winding(const std::vector<float> &cage_xyz, const std::vector<uint32_t> &tris,
		const std::vector<float> &points);

} // namespace cagenet
