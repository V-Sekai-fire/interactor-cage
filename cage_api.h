// cage_api -- cage.elf's whole surface in std types (the curvenet_api split:
// guest/cage/main.cpp is the only TU that sees the sandbox's api.hpp; this and
// everything under it compiles natively too, tests/cage_native).
//
// The engine is reached only through these entries (RFD 2277: the Godot-shaped
// sandbox API, so the same calls work under Godot and under the Unity
// sandbox). Arrays follow guest/common/mesh_wire.h (metres, CCW-outward).
// Strings answer "ok ..." or "FAIL: ...". One stateful stage:
//
//   cage net      net_from_mesh(xyz, faces, counts)            polygons of any size
//                 net_from_patches(verts, tris, bound, target)  curvenet patches -> capped,
//                                                               coarsened, contained cage
//   bind          bind(mesh_xyz, require_contained)             queues bhc13's bind (ticks)
//   frames        body_frames(xyz, tris, frames, frame_w)       the body skinned per frame
//   fit           fit_set(bones, rowptr, col, w, frozen, params)
//                 queue_fit()
//   tick          tick(host_us, gas)                            gassed slices of the queued job
//                                                               (AGENTS.md rule 4; resumable)
//   results       result() (y), result_u() -- empty until the fit is done --,
//                 report() (the progress until then), weights(), deform(knots)
//   checks        check(name), check_all() queue gate checks; ticks run them
//   stubs         build(), preview(), dump()                    Phase C's (NDMF) entries
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cageapi {

std::string reset();
std::string net_from_mesh(const std::vector<float> &xyz, const std::vector<int32_t> &faces,
		const std::vector<int32_t> &counts);
std::string net_from_patches(const std::vector<float> &vertices, const std::vector<int32_t> &triangles,
		const std::vector<float> &bound, int target_vertices);
std::vector<float> net_vertices();
std::vector<int32_t> net_indices();
std::string bind(const std::vector<float> &mesh_xyz, bool require_contained);
std::string body_frames(const std::vector<float> &xyz, const std::vector<int32_t> &tris, int frames,
		const std::vector<float> &frame_w);
// params: "margin 0.002 w_L 0.001 w_r 0.0001" plus LbfgsbParams keys ("m 6 epsilon 1e-7 ...").
std::string fit_set(const std::vector<float> &bones, const std::vector<int32_t> &rowptr,
		const std::vector<int32_t> &col, const std::vector<float> &w, const std::vector<int32_t> &frozen,
		const std::string &params);
std::string queue_fit();
// Advance the queued job (queued checks first) by steps until gas is spent,
// one step at least, and stop early at a finished check, an accepted fit
// iterate or a finished net. Gas: in the guest, instructions in 2^20 units
// (rdinstret, execution_timeout's unit); natively, steps. A step: one bind
// chunk (64 points or 512 Gram entries, or the LU), one fit evaluation or
// solver phase, one cage-builder stage, one check step. Job state stays in the
// guest between calls, so a spent budget only suspends the job. host_us is the
// host clock (the guest clock is not a clock); it is not used for the work.
std::string tick(uint64_t host_us, uint64_t gas);
std::vector<float> result();
std::vector<float> result_u();
std::string report();
std::vector<float> weights();
std::vector<float> deform(const std::vector<float> &knots_posed);
std::string build();
std::string preview();
std::string dump();
// The scripted pen (cagefix::dress_pen) for a host that drives curvenet.elf:
// [n, then per stroke: boundary, samples, 4 floats a sample]; its body and
// bound garment; and a one-line description (printed in the results).
std::vector<float> pen_script();
std::vector<float> pen_body_vertices();
std::vector<int32_t> pen_body_indices();
std::vector<float> pen_bound();
std::string pen_describe();
std::vector<std::string> check_names();
std::string check(const std::string &name);
std::string check_all();

} // namespace cageapi
