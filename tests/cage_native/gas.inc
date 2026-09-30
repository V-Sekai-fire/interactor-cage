// The gas gate: the same job, sliced finely or coarsely, must give the same
// bits. tests/cage_oracle's G3 problem through the API (cage_bind, then
// cage_queue_fit), ticked with gas 1 (one step a call: many resumes) and with
// gas 1e9 (one call a phase); the weights and u are compared bit for bit.
struct GasRun {
	std::vector<float> W, u;
	uint64_t calls = 0;
	std::string last;
};

bool gas_run(const std::string &od, const std::string &gd, uint64_t gas, GasRun &out) {
	Obj cage, body;
	read_obj((od + "/cage.obj").c_str(), cage);
	read_obj((od + "/body.obj").c_str(), body);
	const std::vector<double> x = read_numbers(gd + "/garment.txt");
	std::ifstream pin(od + "/problem.txt");
	std::string line, tag, params;
	std::vector<double> w1, poses;
	std::vector<int32_t> frozen;
	double m = 0, wp = 0, wL = 0, wr = 0;
	while (std::getline(pin, line)) {
		std::istringstream s(line);
		s >> tag;
		double v;
		if (tag == "m") {
			std::string k;
			s >> m >> k >> wp >> k >> wL >> k >> wr;
		} else if (tag == "w1") {
			while (s >> v) w1.push_back(v);
		} else if (tag == "frozen") {
			int z;
			while (s >> z) frozen.push_back(z);
		} else if (tag == "pose") {
			while (s >> v) poses.push_back(v);
		}
	}
	if (cage.v.empty() || x.empty() || w1.empty()) {
		return false;
	}
	cageapi::reset();
	std::vector<int32_t> counts(cage.c.size(), 3);
	cageapi::net_from_mesh(to_f(cage.v), cage.f, counts);
	std::string r = cageapi::bind(to_f(x), true);
	while (r.find("state=bound") == std::string::npos && r.rfind("FAIL", 0) != 0) {
		r = cageapi::tick(0, gas);
		++out.calls;
	}
	out.W = cageapi::weights();
	std::vector<float> bxyz = to_f(body.v);
	bxyz.insert(bxyz.end(), bxyz.begin(), bxyz.end());
	cageapi::body_frames(bxyz, body.f, 2, { float(wp), float(wp) });
	std::vector<float> bones, w;
	std::vector<int32_t> rowptr(1, 0), col;
	for (size_t p = 0; p < poses.size() / 24; ++p) {
		for (int j = 0; j < 2; ++j) {
			const double *q = &poses[24 * p + 12 * j];
			for (int rr = 0; rr < 3; ++rr) {
				bones.insert(bones.end(), { float(q[3 * rr]), float(q[3 * rr + 1]), float(q[3 * rr + 2]), float(q[9 + rr]) });
			}
		}
	}
	for (double a : w1) {
		col.insert(col.end(), { 0, 1 });
		w.insert(w.end(), { float(1.0 - a), float(a) });
		rowptr.push_back(int32_t(col.size()));
	}
	char pb[512];
	std::snprintf(pb, sizeof pb,
			"margin %.9g w_L %.9g w_r %.9g box 0.05 laplacian combinatorial m 6 delta 0 max_linesearch 40 "
			"max_iterations 2000 epsilon 1e-12 epsilon_rel 1e-10",
			m, wL, wr);
	cageapi::fit_set(bones, rowptr, col, w, frozen, pb);
	r = cageapi::queue_fit();
	while (r.rfind("state=done", 0) != 0 && r.rfind("FAIL", 0) != 0) {
		r = cageapi::tick(0, gas);
		++out.calls;
	}
	out.last = r;
	out.u = cageapi::result_u();
	return r.rfind("state=done", 0) == 0;
}

int cmd_gas(int argc, char **argv) {
	if (argc < 4) {
		std::fprintf(stderr, "usage: gas <oracle dir> <g3 out dir>\n");
		return 2;
	}
	GasRun a, b;
	const bool oa = gas_run(argv[2], argv[3], 1, a), ob = gas_run(argv[2], argv[3], 1000000000ull, b);
	const bool same_w = a.W.size() == b.W.size() && !a.W.empty() &&
			std::memcmp(a.W.data(), b.W.data(), 4 * a.W.size()) == 0;
	const bool same_u = a.u.size() == b.u.size() && !a.u.empty() &&
			std::memcmp(a.u.data(), b.u.data(), 4 * a.u.size()) == 0;
	std::printf("gas 1: %llu calls, %s\ngas 1e9: %llu calls, %s\n", (unsigned long long)a.calls, a.last.c_str(),
			(unsigned long long)b.calls, b.last.c_str());
	const bool ok = oa && ob && same_w && same_u && a.calls > b.calls;
	std::printf("%s gas: the G3 bind and fit ticked with gas 1 (%llu resumes) and gas 1e9 (%llu): weights %s, u %s\n",
			ok ? "PASS" : "FAIL", (unsigned long long)a.calls, (unsigned long long)b.calls,
			same_w ? "bit-identical" : "DIFFER", same_u ? "bit-identical" : "DIFFER");
	return ok ? 0 : 1;
}
