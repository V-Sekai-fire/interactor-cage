// cage.elf -- the curvenet-cage refit stage (RFD 2277 Phase A): bhc13's bind
// (guest/common/deform/bhc13), the in-motion cage fit on dress-on's
// L-BFGS-B (guest/cage/cage_fit.h, fit_driver.h) and curvenet patches to a
// closed cage (cage_net.h).
//
// The api.hpp split (guest/curvenet/main.cpp's): this TU is the only one that
// sees the sandbox's api.hpp; cage_api.cpp and below are std types and link
// into tests/cage_native too. The engine is reached only through these
// entries and the Godot-shaped sandbox API, so the same ELF runs under Godot
// and under the Unity sandbox (RFD 2277 Phase B). Every ADD_API_FUNCTION has
// a no-argument wrapper in project/main.gd (AGENTS.md rule 8). No GPU: the CPU
// path of every Lean kernel (their SPIR-V is emitted and validated, not yet
// dispatched). Rule 4 and the operator's gas rule: cage_bind, cage_queue_fit,
// cage_net_from_patches and cage_check only queue; cage_tick(host_us, gas)
// advances the queued job by a gas slice, resumably (job state stays here).

#include <api.hpp>

#include <string>
#include <vector>

#include "cage_api.h"

static Variant text(const std::string &s) {
	return Variant(String(s));
}

template <typename T>
static Variant packed(const std::vector<T> &v) {
	return Variant(PackedArray<T>(v));
}

static Variant cage_reset() {
	return text(cageapi::reset());
}

static Variant cage_net_from_mesh(PackedArray<float> xyz, PackedArray<int32_t> faces, PackedArray<int32_t> counts) {
	return text(cageapi::net_from_mesh(xyz.fetch(), faces.fetch(), counts.fetch()));
}

static Variant cage_net_from_patches(PackedArray<float> vertices, PackedArray<int32_t> triangles,
		PackedArray<float> bound, int target) {
	return text(cageapi::net_from_patches(vertices.fetch(), triangles.fetch(), bound.fetch(), target));
}

static Variant cage_net_vertices() {
	return packed(cageapi::net_vertices());
}

static Variant cage_net_indices() {
	return packed(cageapi::net_indices());
}

static Variant cage_bind(PackedArray<float> mesh, int require_contained) {
	return text(cageapi::bind(mesh.fetch(), require_contained != 0));
}

static Variant cage_body_frames(PackedArray<float> xyz, PackedArray<int32_t> triangles, int frames,
		PackedArray<float> frame_w) {
	return text(cageapi::body_frames(xyz.fetch(), triangles.fetch(), frames, frame_w.fetch()));
}

static Variant cage_fit_set(PackedArray<float> bones, PackedArray<int32_t> rowptr, PackedArray<int32_t> col,
		PackedArray<float> w, PackedArray<int32_t> frozen, String params) {
	return text(cageapi::fit_set(bones.fetch(), rowptr.fetch(), col.fetch(), w.fetch(), frozen.fetch(), params.utf8()));
}

static Variant cage_queue_fit() {
	return text(cageapi::queue_fit());
}

static Variant cage_tick(double host_us, int gas) {
	return text(cageapi::tick(uint64_t(host_us), gas > 0 ? uint64_t(gas) : 1u));
}

static Variant cage_result() {
	return packed(cageapi::result());
}

static Variant cage_result_u() {
	return packed(cageapi::result_u());
}

static Variant cage_report() {
	return text(cageapi::report());
}

static Variant cage_weights() {
	return packed(cageapi::weights());
}

static Variant cage_bake_skin(PackedArray<int32_t> knot_bones, PackedArray<float> knot_weights, int per_knot,
		int max_influences) {
	return text(cageapi::bake_skin(knot_bones.fetch(), knot_weights.fetch(), per_knot, max_influences));
}

static Variant cage_skin_bones() {
	return packed(cageapi::skin_bones());
}

static Variant cage_skin_weights() {
	return packed(cageapi::skin_weights());
}

static Variant cage_deform(PackedArray<float> knots_posed) {
	return packed(cageapi::deform(knots_posed.fetch()));
}

static Variant cage_build() {
	return text(cageapi::build());
}

static Variant cage_preview() {
	return text(cageapi::preview());
}

static Variant cage_dump() {
	return text(cageapi::dump());
}

static Variant cage_pen_script() {
	return packed(cageapi::pen_script());
}

static Variant cage_pen_body_vertices() {
	return packed(cageapi::pen_body_vertices());
}

static Variant cage_pen_body_indices() {
	return packed(cageapi::pen_body_indices());
}

static Variant cage_pen_bound() {
	return packed(cageapi::pen_bound());
}

static Variant cage_pen_describe() {
	return text(cageapi::pen_describe());
}

static Variant cage_check(String name) {
	return text(cageapi::check(name.utf8()));
}

static Variant cage_check_all() {
	return text(cageapi::check_all());
}

static Variant cage_check_names() {
	std::string out;
	for (const std::string &n : cageapi::check_names()) {
		out += (out.empty() ? "" : " ") + n;
	}
	return text(out);
}

int main() {
	ADD_API_FUNCTION(cage_reset, "String", "", "Drop the net, bind, frames and fit");
	ADD_API_FUNCTION(cage_net_from_mesh, "String", "PackedFloat32Array xyz, PackedInt32Array faces, PackedInt32Array counts",
			"A cage net from polygons of any size (bhc13's net_from_mesh)");
	ADD_API_FUNCTION(cage_net_from_patches, "String",
			"PackedFloat32Array vertices, PackedInt32Array triangles, PackedFloat32Array bound, int target",
			"Queue: curvenet patches -> capped, coarsened cage containing bound");
	ADD_API_FUNCTION(cage_net_vertices, "PackedFloat32Array", "", "The cage's knots");
	ADD_API_FUNCTION(cage_net_indices, "PackedInt32Array", "", "The cage's polygon corners");
	ADD_API_FUNCTION(cage_bind, "String", "PackedFloat32Array mesh, int require_contained",
			"Queue bhc13's bind of the mesh (cage_tick advances it)");
	ADD_API_FUNCTION(cage_body_frames, "String",
			"PackedFloat32Array xyz, PackedInt32Array triangles, int frames, PackedFloat32Array frame_w",
			"The body skinned to each sampled frame");
	ADD_API_FUNCTION(cage_fit_set, "String",
			"PackedFloat32Array bones, PackedInt32Array rowptr, PackedInt32Array col, PackedFloat32Array w, PackedInt32Array frozen, String params",
			"Garment bones per frame, its skin weights (CSR), frozen knots, loss and solver params");
	ADD_API_FUNCTION(cage_queue_fit, "String", "", "Start the in-motion fit (cage_tick advances it)");
	ADD_API_FUNCTION(cage_tick, "String", "float host_us, int gas",
			"Advance the queued job by gas (2^20-instruction units); resumable");
	ADD_API_FUNCTION(cage_result, "PackedFloat32Array", "", "The refitted garment in bind space");
	ADD_API_FUNCTION(cage_result_u, "PackedFloat32Array", "", "The cage displacement u");
	ADD_API_FUNCTION(cage_report, "String", "", "The loss terms and clearance at u");
	ADD_API_FUNCTION(cage_weights, "PackedFloat32Array", "", "(Phi | Psi), P x (nV + nT)");
	ADD_API_FUNCTION(cage_bake_skin, "String",
			"PackedInt32Array knot_bones, PackedFloat32Array knot_weights, int per_knot, int max_influences",
			"RFD 2279 bake_skin: knot bone weights through Phi to the bound points");
	ADD_API_FUNCTION(cage_skin_bones, "PackedInt32Array", "", "The baked bones, P x max_influences (-1 unused)");
	ADD_API_FUNCTION(cage_skin_weights, "PackedFloat32Array", "", "The baked weights, P x max_influences");
	ADD_API_FUNCTION(cage_deform, "PackedFloat32Array", "PackedFloat32Array knots_posed",
			"bhc13's deform: knots as nV x 12 [R | t]; translations only");
	ADD_API_FUNCTION(cage_build, "String", "", "Phase C (NDMF build); a stub here");
	ADD_API_FUNCTION(cage_preview, "String", "", "Phase C (preview); a stub here");
	ADD_API_FUNCTION(cage_dump, "String", "", "Phase C (dump); a stub here");
	ADD_API_FUNCTION(cage_pen_script, "PackedFloat32Array", "", "The scripted pen's strokes for curvenet.elf");
	ADD_API_FUNCTION(cage_pen_body_vertices, "PackedFloat32Array", "", "The scripted pen's body");
	ADD_API_FUNCTION(cage_pen_body_indices, "PackedInt32Array", "", "The scripted pen's body triangles");
	ADD_API_FUNCTION(cage_pen_bound, "PackedFloat32Array", "", "The garment the curvenet cage must contain");
	ADD_API_FUNCTION(cage_pen_describe, "String", "", "The scripted pen, in words (printed in the results)");
	ADD_API_FUNCTION(cage_check, "String", "String name", "Queue one Gate 9 check (cage_tick runs it)");
	ADD_API_FUNCTION(cage_check_all, "String", "", "Queue every Gate 9 check");
	ADD_API_FUNCTION(cage_check_names, "String", "", "The Gate 9 check names");
	halt();
}
