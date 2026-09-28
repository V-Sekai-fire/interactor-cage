// fixtures -- the scenes cage.elf's gates run on, generated (the guest has no
// filesystem), so the guest and the native control build the same bytes.
//
// sphere_scene (G2, G3, G4): a sphere body skinned to two frames, a garment
// shell that sinks into it in both, and an icosphere cage around the shell.
//   body     UV sphere r 0.1 m (16 x 32), frame 0 shifted (+2, 0, 0) mm,
//            frame 1 shifted (-1, 0, +2) mm (the body as the host skins it);
//   garment  a band of the sphere r 0.1015 m between latitudes -60 and +60
//            degrees (13 rows x 32), dented to r 0.0995 m where x > 0.08 and
//            |y| < 0.03 (it sinks in at rest too); two bones: bone 0 fixed,
//            bone 1 turned about z by +20 degrees in frame 0 and -20 in frame 1,
//            weights w1 = smoothstep(-0.03, 0.03, y), w0 = 1 - w1 (the LBS
//            blend pulls the band's middle inward: it sinks in both poses);
//   cage     icosphere r 0.13 m, one subdivision (42 knots, 80 triangles);
//            knots with y < -0.1 frozen.
//
// dress_pen (the curvenet cage): a scripted pen, the stand-in for a person,
// on the skirt check's capped cylinder body (guest/curvenet/checks.cpp,
// r 0.15 m, y 0.3 to 1.1): the neckline at y 0.9 and the hem at y 0.5, each
// drawn as two half rings of r 0.175 from +x through +z / -z to -x as
// boundary strokes, then the left and right straps from neckline to hem at
// -x and +x as ordinary strokes (the skirt fixture's order and sampling:
// 24 samples a half ring, 12 a strap). curvenet params: snap_radius 0.03,
// surface_offset 0.02 (the curves ride 2 cm off the body). The bound
// garment is a bodice tube r 0.158 m, y 0.55 to 0.85 (48 x 13).
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cage_fit.h"

namespace cagefix {

struct SphereScene {
	std::vector<float> garment;        // 3 P, bind space
	std::vector<float> cage;           // 3 nV
	std::vector<int32_t> faces, counts; // the cage's triangles
	cagefit::Body body;
	cagefit::Skin skin;
	std::vector<uint8_t> frozen;
	cagefit::Params params;
};
SphereScene sphere_scene();

void uv_sphere(float r, int nlat, int nlon, std::vector<float> &v, std::vector<uint32_t> &f);
void icosphere(float r, int levels, std::vector<float> &v, std::vector<int32_t> &f);
void cylinder(float r, float y0, float y1, int nseg, int nrow, std::vector<float> &v, std::vector<int32_t> &f);

struct PenStroke {
	std::string name;
	bool boundary = false;
	std::vector<float> xyzp; // 4 floats a sample
};
struct DressPen {
	std::vector<PenStroke> strokes;
	std::vector<float> body;          // the capped cylinder
	std::vector<int32_t> body_tris;
	std::vector<float> bound;         // the bodice tube's vertices
	float snap_radius = 0.03f, surface_offset = 0.02f;
	std::string describe() const;
};
DressPen dress_pen();
// [n, then per stroke: boundary, samples, 4 floats a sample] for the host.
std::vector<float> encode_pen(const DressPen &p);

} // namespace cagefix
