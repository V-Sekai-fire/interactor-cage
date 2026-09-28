// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "skin_bake.h"

#include <algorithm>
#include <map>

#include "cage_kernels.h"

namespace deform {

bool bake_skin(const Bind &b, const std::vector<int32_t> &knot_bones, const std::vector<float> &knot_weights,
		uint32_t per_knot, uint32_t max_influences, Skin &out, std::string &err) {
	if (per_knot == 0 || max_influences == 0 || knot_bones.size() != size_t(b.nV) * per_knot ||
			knot_weights.size() != knot_bones.size()) {
		err = "bake_skin: knot_bones and knot_weights must be nV x per_knot = " + std::to_string(size_t(b.nV) * per_knot);
		return false;
	}
	std::map<int32_t, uint32_t> column;
	for (int32_t bone : knot_bones) {
		if (bone >= 0 && !column.count(bone)) {
			const uint32_t c = uint32_t(column.size());
			column[bone] = c;
		}
	}
	const uint32_t B = uint32_t(column.size());
	std::vector<int32_t> bone_of(B);
	for (auto &kv : column) {
		bone_of[kv.second] = kv.first;
	}
	const uint32_t groups = (B + 2) / 3;
	std::vector<float> x(size_t(b.nV) * 3), y(size_t(b.P) * 3), acc(size_t(b.P) * B, 0.0f);
	for (uint32_t g = 0; g < groups; ++g) {
		std::fill(x.begin(), x.end(), 0.0f);
		for (uint32_t v = 0; v < b.nV; ++v) {
			for (uint32_t s = 0; s < per_knot; ++s) {
				const int32_t bone = knot_bones[size_t(v) * per_knot + s];
				if (bone < 0) {
					continue;
				}
				const uint32_t c = column[bone];
				if (c / 3 == g) {
					x[3 * size_t(v) + c % 3] += knot_weights[size_t(v) * per_knot + s];
				}
			}
		}
		cagek::csr_gemv3(b.P, false, b.phi.rowptr.data(), b.phi.col.data(), b.phi.val.data(), b.phi.col.size(),
				x.data(), b.nV, y.data());
		for (uint32_t p = 0; p < b.P; ++p) {
			for (uint32_t k = 0; k < 3 && 3 * g + k < B; ++k) {
				acc[size_t(p) * B + 3 * g + k] = y[3 * size_t(p) + k];
			}
		}
	}
	out.influences = max_influences;
	out.bones.assign(size_t(b.P) * max_influences, -1);
	out.weights.assign(size_t(b.P) * max_influences, 0.0f);
	out.empty = 0;
	std::vector<std::pair<float, uint32_t>> row(B);
	for (uint32_t p = 0; p < b.P; ++p) {
		for (uint32_t c = 0; c < B; ++c) {
			row[c] = { std::max(0.0f, acc[size_t(p) * B + c]), c };
		}
		std::sort(row.begin(), row.end(), [](auto &a, auto &c) { return a.first > c.first || (a.first == c.first && a.second < c.second); });
		const uint32_t n = std::min(max_influences, B);
		float sum = 0.0f;
		for (uint32_t k = 0; k < n; ++k) {
			sum += row[k].first;
		}
		if (sum <= 0.0f) {
			++out.empty;
			continue;
		}
		for (uint32_t k = 0; k < n && row[k].first > 0.0f; ++k) {
			out.bones[size_t(p) * max_influences + k] = bone_of[row[k].second];
			out.weights[size_t(p) * max_influences + k] = row[k].first / sum;
		}
	}
	return true;
}

} // namespace deform
