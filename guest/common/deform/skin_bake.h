// SPDX-License-Identifier: Apache-2.0 OR MIT
// RFD 2279's bake_skin for the bhc13 bind: each bound point takes the Phi-weighted
// blend of its knots' bone weights, clamped at zero, cut to the strongest
// max_influences and renormalized.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bhc13/bhc13.h"

namespace deform {

struct Skin {
	uint32_t influences = 0;          // per point
	std::vector<int32_t> bones;       // P x influences, -1 where unused
	std::vector<float> weights;       // P x influences, each row sums to 1
	uint32_t empty = 0;               // points whose clamped blend was zero
};

// knot_bones / knot_weights: nV x per_knot, -1 for an unused slot.
bool bake_skin(const Bind &b, const std::vector<int32_t> &knot_bones, const std::vector<float> &knot_weights,
		uint32_t per_knot, uint32_t max_influences, Skin &out, std::string &err);

} // namespace deform
