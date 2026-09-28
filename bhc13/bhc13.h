// bhc13 -- the (1,3) biharmonic cage deform: RFD 2279's `bhc13` method, the
// internal C++ API its wrappers sit on (RFD 2277 owns it).
//
// Thiery, Michel and Chen, "Biharmonic Coordinates and their Derivatives for
// Triangular 3D Cages", SIGGRAPH 2024; restated from BHC.h of
// V-Sekai-fire/interactor-tool-godot-cage-deformer (MIT). Every number is a
// Lean-emitted kernel (lean/Cage, kernels/cage; guest/common/deform/
// cage_kernels.h dispatches their cpp emits): the bind in double, the deform
// and its Jacobian in float. This file holds topology checks, buffer layout
// and staging only.
//
// The surface (RFD 2279's names):
//   net_from_mesh(xyz, faces, counts)  a net from polygons of any size; they are
//                                      triangulated (fanned from each polygon's
//                                      first vertex) at bind time.
//   bind(net, mesh, "bhc13")           refuses, naming the reason, unless the
//                                      triangulated cycles form a closed,
//                                      edge- and vertex-manifold, consistently and
//                                      outward oriented surface (signed volume > 0)
//                                      with every knot on a face, and (unless
//                                      waived) every bound vertex inside it.
//   weights(bind)                      P x (nV + nT) floats, row-major: the vertex
//                                      weights Phi, then the normal weights Psi.
//   deform(bind, knots_posed)          knots_posed is nV x 12 ([R | t] row-major,
//                                      translation in column 3); bhc13 reads only
//                                      the translations. y = Phi c + Psi n(c).
//   jacobian(bind, knots_posed, dy)    the transpose product: dL/dc from dL/dy,
//                                      Phi^T dy plus the normal pull-back
//                                      (I - n n^T) (Psi^T dy)_t / |N_t| onto each
//                                      triangle's vertices. nV x 3 (rotations get
//                                      no cotangent: bhc13 ignores them).
// Only "bhc13" is implemented; "harmonic", "idw", "lbs" and bake_skin are
// RFD 2279's and are refused by name here.
//
// The guest has no filesystem; everything is std types, so the same code is
// cage.elf's and the native gates' (tests/cage_native).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace deform {

struct Net {
	std::vector<float> xyz;       // 3 nV: the knots (cage vertices)
	std::vector<int32_t> faces;   // polygon corners, flattened
	std::vector<int32_t> counts;  // corners per polygon (>= 3)
	// Optional: the knots in double (a gate binding exactly what a double
	// oracle read). Used by bind when it has 3 nV entries; xyz stays the wire form.
	std::vector<double> xyz_d;
	uint32_t knot_count() const { return uint32_t(xyz.size() / 3); }
};

bool net_from_mesh(const std::vector<float> &xyz, const std::vector<int32_t> &faces, const std::vector<int32_t> &counts,
		Net &out, std::string &err);

struct BindOptions {
	// Refuse when a bound vertex lies outside the cage (generalized winding
	// |w - 1| > 0.5, w = the sum of its harmonic vertex coordinates).
	bool require_contained = true;
	// Points of the mesh (or samples) per bind step (BindJob::step).
	uint32_t chunk = 64;
	// Gram entries per bind step.
	uint32_t gram_chunk = 512;
	// The gates' fault switch (G4): 1 adds 0.01 to the first bound vertex's
	// first weight after the solve, which G1 and G2 must then catch.
	int fault = 0;
};

struct Bind {
	std::string method;                 // "bhc13"
	uint32_t nV = 0, nT = 0, P = 0;     // knots, triangles, bound vertices
	uint32_t K() const { return nV + nT; }
	std::vector<float> rest;            // 3 nV: the bind-pose knots
	std::vector<uint32_t> tris;         // 3 nT: the triangulated cycles
	std::vector<uint32_t> tri_poly;     // nT: the net polygon each triangle came from
	std::vector<float> W;               // P x K: (Phi | Psi), float
	std::vector<double> Wd;             // P x K: the same in double (gates)
	std::vector<float> winding;         // P: generalized winding of each bound vertex
	double volume = 0.0;                // signed volume of the cage (> 0)
	uint32_t outside = 0;               // bound vertices with |winding - 1| > 0.5
	// CSR views for the deform and its transpose (anny_csr_gemv3's layout).
	struct Csr {
		std::vector<uint32_t> rowptr, col;
		std::vector<float> val;
	};
	Csr phi, psi;        // P x nV, P x nT
	Csr phiT, psiT;      // nV x P, nT x P
	Csr corners;         // nV x 3 nT: sums a triangle's per-corner terms onto its vertices
	Csr ring;            // nV one-rings (val unused): the uniform Laplacian
};

// Checks the triangulated net (the refusals bind() names). tris/tri_poly/
// volume are filled on success.
bool check_net(const Net &net, std::vector<uint32_t> &tris, std::vector<uint32_t> &tri_poly, double &volume,
		std::string &err);

// The bind, staged: begin(), then step() until done() (each step one chunk of
// one stage, sized by BindOptions); failed() with error() on a refusal.
class BindJob {
public:
	bool begin(const Net &net, const std::vector<float> &mesh_xyz, const std::string &method,
			const BindOptions &opt, std::string &err);
	// The same with the bound vertices in double (gates).
	bool begin(const Net &net, const std::vector<double> &mesh_xyz, const std::string &method,
			const BindOptions &opt, std::string &err);
	bool step();                   // false once done or failed
	bool done() const { return stage_ == S_DONE; }
	bool failed() const { return stage_ == S_FAILED; }
	const std::string &error() const { return err_; }
	const char *stage_name() const;
	Bind &result() { return b_; }
	// The constraint system, for gates: S samples, K unknowns.
	uint32_t samples() const { return S_; }

private:
	enum Stage { S_IDLE, S_SAMPLES, S_SAMPLE_ROWS, S_GRAM, S_LU, S_SOLVE, S_MESH, S_CSR, S_DONE, S_FAILED };
	bool fail(const std::string &why);
	Stage stage_ = S_IDLE;
	std::string err_;
	BindOptions opt_;
	Bind b_;
	std::vector<double> cage_, mesh_, bary_, spts_, sgamma_, srows_, G_, X_, rows_;
	std::vector<uint32_t> sown_, piv_, none_;
	std::vector<double> zeros_;
	uint32_t S_ = 0, cursor_ = 0;
};

// The whole bind at once (BindJob run to the end).
bool bind(const Net &net, const std::vector<float> &mesh_xyz, const std::string &method, Bind &out, std::string &err,
		const BindOptions &opt = BindOptions());
bool bind(const Net &net, const std::vector<double> &mesh_xyz, const std::string &method, Bind &out, std::string &err,
		const BindOptions &opt = BindOptions());

const std::vector<float> &weights(const Bind &b);

// Knot positions (3 nV) from nV x 12 transforms (translation column).
bool knot_positions(const Bind &b, const std::vector<float> &knots_posed, std::vector<float> &c, std::string &err);

bool deform(const Bind &b, const std::vector<float> &knots_posed, std::vector<float> &out, std::string &err);
bool jacobian(const Bind &b, const std::vector<float> &knots_posed, const std::vector<float> &dy,
		std::vector<float> &dknots, std::string &err);

// The same on knot positions c (3 nV), and the scratch the fit reuses.
struct DeformScratch {
	std::vector<float> n, nlen, gn, gcorner;
};
void deform_points(const Bind &b, const float *c, float *y, DeformScratch &s);
// dc = Phi^T dy + the normal pull-back of Psi^T dy; overwrites dc (3 nV).
// Needs s.n / s.nlen from a deform_points at the same c.
void jacobian_points(const Bind &b, const float *c, const float *dy, float *dc, DeformScratch &s);

// The midpoint subdivision table of computeConstrainedBiharmonicMatrices_13:
// 4^levels sub-triangle centroids (3 barycentric doubles each), in BHC.h's order.
std::vector<double> subdivision_barycentrics(int levels);

} // namespace deform
