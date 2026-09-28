// Cage-fit oracle (Gate 9, G3): host-native, double throughout. Restates the
// cage fit of RFD 2277 and solves it with unmodified LBFGSpp 0.3.0, so the
// Lean-emitted fit in cage.elf can be checked against an independent solve.
//
//   y_i  = sum_k Phi_ik (c0_k + u_k) + sum_t Psi_it n_t(c0 + u)      bind space
//   x_ip = sum_j w_ij (R_pj y_i + T_pj)                              pose p (LBS)
//   E(u) = wp sum_p sum_i max(0, m - d(x_ip))^2 + wL |L u|^2 + wr |u|^2
//
// Phi, Psi: the (1,3) biharmonic coordinates of BHC.h (the only parity
// reference). d: exact signed distance to the closed body mesh, the sign from
// angle-weighted pseudonormals. Frozen cage vertices are held by lb = ub = 0.
//
// The gradient is checked against central differences, and a planted wrong
// gradient (the normal pull-back's sign flipped) must fail the same check, so
// the check can fail.
//
// Usage: gen <outdir>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <Eigen/Core>
#include "BHC.h"
#define private public
#include <LBFGSB.h>
#undef private

using Vec = Eigen::VectorXd;
struct V3 {
    double x = 0, y = 0, z = 0;
    V3() = default;
    V3(double a, double b, double c) : x(a), y(b), z(c) {}
    V3 operator+(const V3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    V3 operator-(const V3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    V3 operator*(double s) const { return {x * s, y * s, z * s}; }
    V3& operator+=(const V3& o) { x += o.x; y += o.y; z += o.z; return *this; }
};
static double dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(const V3& a, const V3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static double norm(const V3& a) { return std::sqrt(dot(a, a)); }
static V3 unit(const V3& a) { double n = norm(a); return n > 0 ? a * (1.0 / n) : a; }
using Tri = std::array<int, 3>;

// ---------------------------------------------------------------- meshes
// Icosphere: level 0 is the icosahedron; each level splits every triangle in four.
static void icosphere(int level, double r, std::vector<V3>& v, std::vector<Tri>& f) {
    const double t = (1 + std::sqrt(5.0)) / 2;
    v = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
         {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    f = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
         {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
         {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    for (auto& p : v) p = unit(p);
    for (int l = 0; l < level; ++l) {
        std::map<std::pair<int, int>, int> mid;
        auto midpoint = [&](int a, int b) {
            auto key = std::minmax(a, b);
            auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            v.push_back(unit((v[a] + v[b]) * 0.5));
            return mid[key] = int(v.size()) - 1;
        };
        std::vector<Tri> g;
        for (auto& tr : f) {
            int a = midpoint(tr[0], tr[1]), b = midpoint(tr[1], tr[2]), c = midpoint(tr[2], tr[0]);
            g.push_back({tr[0], a, c}); g.push_back({tr[1], b, a});
            g.push_back({tr[2], c, b}); g.push_back({a, b, c});
        }
        f = g;
    }
    for (auto& p : v) p = p * r;
}

static double signed_volume(const std::vector<V3>& v, const std::vector<Tri>& f) {
    double s = 0;
    for (auto& t : f) s += dot(v[t[0]], cross(v[t[1]], v[t[2]])) / 6.0;
    return s;
}

// ---------------------------------------------------------------- signed distance
// Exact closest point on a closed triangle mesh; the sign comes from the
// angle-weighted pseudonormal of the closest feature (face, edge or vertex).
struct Body {
    std::vector<V3> v; std::vector<Tri> f;
    std::vector<V3> fn, vn; std::map<std::pair<int, int>, V3> en;
    void build() {
        fn.assign(f.size(), {}); vn.assign(v.size(), {});
        for (size_t t = 0; t < f.size(); ++t) {
            auto& tr = f[t];
            fn[t] = unit(cross(v[tr[1]] - v[tr[0]], v[tr[2]] - v[tr[0]]));
            for (int k = 0; k < 3; ++k) {
                V3 a = unit(v[tr[(k + 1) % 3]] - v[tr[k]]), b = unit(v[tr[(k + 2) % 3]] - v[tr[k]]);
                vn[tr[k]] += fn[t] * std::acos(std::clamp(dot(a, b), -1.0, 1.0));
                en[std::minmax(tr[k], tr[(k + 1) % 3])] += fn[t];
            }
        }
    }
    // Returns d; writes the unit gradient of d (direction from the closest point).
    double sdf(const V3& p, V3& grad) const {
        double best = 1e300; V3 cp, pn;
        for (size_t t = 0; t < f.size(); ++t) {
            const V3 &a = v[f[t][0]], &b = v[f[t][1]], &c = v[f[t][2]];
            // Ericson, Real-Time Collision Detection 5.1.5, with the feature kept.
            V3 ab = b - a, ac = c - a, ap = p - a, q; V3 n;
            double d1 = dot(ab, ap), d2 = dot(ac, ap);
            V3 bp = p - b; double d3 = dot(ab, bp), d4 = dot(ac, bp);
            V3 cpp = p - c; double d5 = dot(ab, cpp), d6 = dot(ac, cpp);
            double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
            if (d1 <= 0 && d2 <= 0) { q = a; n = vn[f[t][0]]; }
            else if (d3 >= 0 && d4 <= d3) { q = b; n = vn[f[t][1]]; }
            else if (d6 >= 0 && d5 <= d6) { q = c; n = vn[f[t][2]]; }
            else if (vc <= 0 && d1 >= 0 && d3 <= 0) { q = a + ab * (d1 / (d1 - d3)); n = en.at(std::minmax(f[t][0], f[t][1])); }
            else if (vb <= 0 && d2 >= 0 && d6 <= 0) { q = a + ac * (d2 / (d2 - d6)); n = en.at(std::minmax(f[t][0], f[t][2])); }
            else if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
                q = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))); n = en.at(std::minmax(f[t][1], f[t][2]));
            } else {
                double den = 1.0 / (va + vb + vc);
                q = a + ab * (vb * den) + ac * (vc * den); n = fn[t];
            }
            double dd = dot(p - q, p - q);
            if (dd < best) { best = dd; cp = q; pn = n; }
        }
        double d = std::sqrt(best);
        double s = dot(p - cp, pn) >= 0 ? 1.0 : -1.0;
        grad = d > 0 ? (p - cp) * (s / d) : unit(pn);
        return s * d;
    }
};

// ---------------------------------------------------------------- the problem
struct Pose { V3 R[2][3]; V3 T[2]; };  // two bones; R as rows
static V3 mul(const V3 R[3], const V3& y) { return {dot(R[0], y), dot(R[1], y), dot(R[2], y)}; }
static V3 mulT(const V3 R[3], const V3& g) { return R[0] * g.x + R[1] * g.y + R[2] * g.z; }

struct Problem {
    std::vector<V3> c0; std::vector<Tri> ct;          // cage
    std::vector<std::vector<double>> phi, psi;        // n_points x n_cage_v, n_points x n_cage_t
    std::vector<double> w1;                           // bone-1 weight per point (bone 0 gets 1 - w1)
    std::vector<Pose> poses;
    std::vector<std::vector<int>> nbr;                // cage vertex adjacency
    std::vector<int> frozen;                          // 1 = held at 0
    Body body;
    double m = 0.002, wp = 1e3, wL = 1.0, wr = 0.1;
    bool plant_wrong = false;                         // flips the normal pull-back's sign

    size_t nc() const { return c0.size(); }
    size_t np() const { return phi.size(); }

    void cage_at(const Vec& u, std::vector<V3>& c) const {
        c.resize(nc());
        for (size_t k = 0; k < nc(); ++k) c[k] = c0[k] + V3(u[3 * k], u[3 * k + 1], u[3 * k + 2]);
    }
    void bind_points(const std::vector<V3>& c, std::vector<V3>& y, std::vector<V3>& N) const {
        N.resize(ct.size());
        for (size_t t = 0; t < ct.size(); ++t) N[t] = cross(c[ct[t][1]] - c[ct[t][0]], c[ct[t][2]] - c[ct[t][0]]);
        y.assign(np(), {});
        for (size_t i = 0; i < np(); ++i) {
            for (size_t k = 0; k < nc(); ++k) y[i] += c[k] * phi[i][k];
            for (size_t t = 0; t < ct.size(); ++t) y[i] += unit(N[t]) * psi[i][t];
        }
    }
    V3 skin(size_t i, const Pose& P, const V3& y) const {
        return (mul(P.R[0], y) + P.T[0]) * (1 - w1[i]) + (mul(P.R[1], y) + P.T[1]) * w1[i];
    }
    V3 skinT(size_t i, const Pose& P, const V3& g) const {
        return mulT(P.R[0], g) * (1 - w1[i]) + mulT(P.R[1], g) * w1[i];
    }
    // Per-pose penetration: count below the body surface and the deepest (m).
    void depth(const Vec& u, size_t p, int& inside, double& deepest, int& in_margin) const {
        std::vector<V3> c, y, N; cage_at(u, c); bind_points(c, y, N);
        inside = 0; in_margin = 0; deepest = 0;
        for (size_t i = 0; i < np(); ++i) {
            V3 gd; double d = body.sdf(skin(i, poses[p], y[i]), gd);
            if (d < 0) { ++inside; deepest = std::max(deepest, -d); }
            if (d < m) ++in_margin;
        }
    }
    double operator()(const Vec& u, Vec& g) const {
        std::vector<V3> c, y, N; cage_at(u, c); bind_points(c, y, N);
        g.setZero(u.size());
        double E = 0;
        std::vector<V3> gy(np());
        for (size_t p = 0; p < poses.size(); ++p)
            for (size_t i = 0; i < np(); ++i) {
                V3 gd; double d = body.sdf(skin(i, poses[p], y[i]), gd);
                double h = m - d;
                if (h <= 0) continue;
                E += wp * h * h;
                gy[i] += skinT(i, poses[p], gd * (-2 * wp * h));
            }
        // Phi^T and the normal pull-back to the cage vertices.
        std::vector<V3> gc(nc()), gn(ct.size());
        for (size_t i = 0; i < np(); ++i) {
            for (size_t k = 0; k < nc(); ++k) gc[k] += gy[i] * phi[i][k];
            for (size_t t = 0; t < ct.size(); ++t) gn[t] += gy[i] * psi[i][t];
        }
        for (size_t t = 0; t < ct.size(); ++t) {
            double len = norm(N[t]); V3 n = N[t] * (1 / len);
            V3 gN = (gn[t] - n * dot(n, gn[t])) * (1 / len);
            if (plant_wrong) gN = gN * -1.0;
            const V3 &a = c[ct[t][0]], &b = c[ct[t][1]], &cc = c[ct[t][2]];
            gc[ct[t][0]] += cross(b - cc, gN);
            gc[ct[t][1]] += cross(cc - a, gN);
            gc[ct[t][2]] += cross(a - b, gN);
        }
        for (size_t k = 0; k < nc(); ++k) { g[3 * k] += gc[k].x; g[3 * k + 1] += gc[k].y; g[3 * k + 2] += gc[k].z; }
        // Uniform graph Laplacian (Lu)_k = sum_l (u_l - u_k); L is symmetric.
        Vec Lu = Vec::Zero(u.size()), LLu = Vec::Zero(u.size());
        auto applyL = [&](const Vec& a, Vec& out) {
            for (size_t k = 0; k < nc(); ++k)
                for (int l : nbr[k])
                    for (int d = 0; d < 3; ++d) out[3 * k + d] += a[3 * l + d] - a[3 * k + d];
        };
        applyL(u, Lu); applyL(Lu, LLu);
        E += wL * Lu.squaredNorm() + wr * u.squaredNorm();
        g += 2 * wL * LLu + 2 * wr * u;
        return E;
    }
};

// ---------------------------------------------------------------- the fixture
static Pose rigid(double ax_deg, V3 t1) {
    Pose P;
    P.R[0][0] = {1, 0, 0}; P.R[0][1] = {0, 1, 0}; P.R[0][2] = {0, 0, 1}; P.T[0] = {0, 0, 0};
    double a = ax_deg * M_PI / 180, s = std::sin(a), co = std::cos(a);
    P.R[1][0] = {1, 0, 0}; P.R[1][1] = {0, co, -s}; P.R[1][2] = {0, s, co}; P.T[1] = t1;
    return P;
}

static Problem make_problem() {
    Problem P;
    icosphere(3, 0.10, P.body.v, P.body.f);  // the body: 642 vertices, 1280 triangles, r = 0.1 m
    P.body.build();
    // The garment: the upper cap of a sphere just inside the body, so it starts
    // penetrating; the band nearest the equator follows bone 1.
    std::vector<V3> gv; std::vector<Tri> gf;
    icosphere(2, 0.0985, gv, gf);
    std::vector<V3> pts;
    for (auto& p : gv) if (p.z > 0.02) pts.push_back(p);
    // The cage: a closed icosphere around it, outward winding.
    icosphere(1, 0.16, P.c0, P.ct);
    std::vector<point3d> cv; std::vector<std::vector<unsigned>> ctu;
    for (auto& p : P.c0) cv.push_back(point3d(p.x, p.y, p.z));
    for (auto& t : P.ct) ctu.push_back({unsigned(t[0]), unsigned(t[1]), unsigned(t[2])});
    Eigen::MatrixXd C11, C12, C21, C22;
    BiharmonicCoordinates3D::computeConstrainedBiharmonicMatrices_13(ctu, cv, C11, C12, C21, C22, 1.0);
    for (auto& q : pts) {
        std::vector<double> hp, hs, bp, bs, ph, ps;
        BiharmonicCoordinates3D::computeCoordinates(point3d(q.x, q.y, q.z), ctu, cv, hp, hs, bp, bs);
        BiharmonicCoordinates3D::compute_13_blending_from_unconstrained_biharmonics(hp, hs, bp, bs, C11, C12, C21, C22, ph, ps);
        P.phi.push_back(ph); P.psi.push_back(ps);
        double t = std::clamp((0.06 - q.z) / 0.04, 0.0, 1.0);  // smoothstep band 0.02 < z < 0.06
        P.w1.push_back(t * t * (3 - 2 * t));
    }
    // Two poses: rest, and bone 1 tilted 10 degrees and dropped 3 mm, which drives
    // the band deeper into the body.
    P.poses = {rigid(0, {0, 0, 0}), rigid(10, {0, 0, -0.003})};
    P.nbr.assign(P.c0.size(), {});
    for (auto& t : P.ct)
        for (int k = 0; k < 3; ++k) {
            int a = t[k], b = t[(k + 1) % 3];
            if (std::find(P.nbr[a].begin(), P.nbr[a].end(), b) == P.nbr[a].end()) { P.nbr[a].push_back(b); P.nbr[b].push_back(a); }
        }
    for (auto& p : P.c0) P.frozen.push_back(p.z < -0.08 ? 1 : 0);  // the bottom of the cage is held
    return P;
}

// ---------------------------------------------------------------- checks and output
static double gradcheck(const Problem& P, const Vec& u, int& worst_k) {
    Vec g(u.size()), gtmp(u.size());
    P(u, g);
    // Error relative to the gradient's largest entry, so small entries count
    // without their round-off dominating: |fd_k - g_k| / max_j |g_j|.
    double worst = 0; worst_k = -1;
    const double h = 1e-6, gmax = std::max(g.cwiseAbs().maxCoeff(), 1e-300);
    for (int k = 0; k < u.size(); ++k) {
        Vec a = u, b = u; a[k] += h; b[k] -= h;
        double fd = (P(a, gtmp) - P(b, gtmp)) / (2 * h);
        double rel = std::fabs(fd - g[k]) / gmax;
        if (rel > worst) { worst = rel; worst_k = k; }
    }
    return worst;
}

static void write_matrix(const std::string& path, const std::vector<std::vector<double>>& M) {
    FILE* f = std::fopen(path.c_str(), "w");
    for (auto& row : M) { for (size_t k = 0; k < row.size(); ++k) std::fprintf(f, k ? " %.17g" : "%.17g", row[k]); std::fprintf(f, "\n"); }
    std::fclose(f);
}
static void write_obj(const std::string& path, const std::vector<V3>& v, const std::vector<Tri>& f) {
    FILE* o = std::fopen(path.c_str(), "w");
    for (auto& p : v) std::fprintf(o, "v %.17g %.17g %.17g\n", p.x, p.y, p.z);
    for (auto& t : f) std::fprintf(o, "f %d %d %d\n", t[0] + 1, t[1] + 1, t[2] + 1);
    std::fclose(o);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: gen <outdir>\n"); return 2; }
    std::string out = argv[1];
    Problem P = make_problem();
    const int n = int(3 * P.nc());
    std::printf("cage %zu v %zu t, signed volume %.6g m^3; body %zu v %zu t, signed volume %.6g m^3\n",
                P.nc(), P.ct.size(), signed_volume(P.c0, P.ct), P.body.v.size(), P.body.f.size(),
                signed_volume(P.body.v, P.body.f));
    int nfrozen = 0; for (int f : P.frozen) nfrozen += f;
    std::printf("garment %zu points, 2 bones, %zu poses; %d cage vertices frozen; m %.9g wp %.9g wL %.9g wr %.9g\n",
                P.np(), P.poses.size(), nfrozen, P.m, P.wp, P.wL, P.wr);

    // Identity: the undeformed cage reproduces the garment points.
    {
        std::vector<V3> y, N; P.bind_points(P.c0, y, N);
        std::vector<V3> gv; std::vector<Tri> gf; icosphere(2, 0.0985, gv, gf);
        std::vector<V3> pts; for (auto& p : gv) if (p.z > 0.02) pts.push_back(p);
        double e = 0; for (size_t i = 0; i < y.size(); ++i) e = std::max(e, norm(y[i] - pts[i]));
        std::printf("identity: max |y(c0) - x| = %.3e m\n", e);
    }

    // The gradient check, then the planted wrong gradient, which must fail it.
    Vec u0 = Vec::Zero(n);
    for (int k = 0; k < n; ++k) u0[k] = P.frozen[k / 3] ? 0 : 1e-3 * std::sin(0.7 * k + 0.3);
    int wk; double gc = gradcheck(P, u0, wk);
    Problem Pw = P; Pw.plant_wrong = true;
    int wkw; double gcw = gradcheck(Pw, u0, wkw);
    const double gate = 1e-4;
    std::printf("gradcheck at a random u (|u| 1 mm): worst |fd - g| / max|g| = %.3e at coordinate %d (gate %.0e) %s\n", gc, wk, gate, gc < gate ? "PASS" : "FAIL");
    std::printf("negative control, normal pull-back sign flipped: worst %.3e %s\n", gcw,
                gcw >= gate ? "FAILS AS IT MUST" : "PASSED -- the check cannot fail, it proves nothing");

    // The solve.
    Vec lb(n), ub(n);
    for (int k = 0; k < n; ++k) { bool fz = P.frozen[k / 3]; lb[k] = fz ? 0 : -0.05; ub[k] = fz ? 0 : 0.05; }
    LBFGSpp::LBFGSBParam<double> prm;
    prm.m = 6; prm.delta = 0; prm.max_linesearch = 40; prm.max_iterations = 2000; prm.epsilon = 1e-12; prm.epsilon_rel = 1e-10;
    LBFGSpp::LBFGSBSolver<double> solver(prm);
    Vec u = Vec::Zero(n), g(n);
    double E0 = P(u, g), fx = 0;
    int evals = 0;
    std::function<double(const Vec&, Vec&)> fun = [&](const Vec& x, Vec& gg) { ++evals; return P(x, gg); };
    // The signed distance has kinks where the closest feature switches, so near
    // the optimum the line search can stall; that stop is reported, and the last
    // accepted iterate kept (minimize updates u in place).
    int it = -1; std::string stop = "converged";
    try { it = solver.minimize(fun, u, fx, lb, ub); }
    catch (const std::exception& e) { stop = e.what(); Vec gg(n); fx = P(u, gg); }
    Vec gend(n); P(u, gend);
    std::printf("solve: %d iterations, %d evaluations (%s), E %.9g -> %.9g, |proj grad| %.3e\n", it, evals, stop.c_str(), E0, fx, LBFGSpp::LBFGSBSolver<double>::proj_grad_norm(u, gend, lb, ub));
    for (size_t p = 0; p < P.poses.size(); ++p) {
        int in0, in1, mg0, mg1; double d0, d1;
        P.depth(Vec::Zero(n), p, in0, d0, mg0); P.depth(u, p, in1, d1, mg1);
        std::printf("pose %zu: inside %d -> %d, deepest %.3f -> %.3f mm, within margin %d -> %d of %zu\n",
                    p, in0, in1, d0 * 1e3, d1 * 1e3, mg0, mg1, P.np());
    }
    double umax = 0; for (int k = 0; k < n; ++k) umax = std::max(umax, std::fabs(u[k]));
    std::printf("largest cage displacement %.3f mm\n", umax * 1e3);

    // Fixtures: the cage and the body as OBJ, the weights, the problem and the answer.
    write_obj(out + "/cage.obj", P.c0, P.ct);
    write_obj(out + "/body.obj", P.body.v, P.body.f);
    write_matrix(out + "/phi.txt", P.phi);
    write_matrix(out + "/psi.txt", P.psi);
    FILE* f = std::fopen((out + "/problem.txt").c_str(), "w");
    std::fprintf(f, "m %.17g wp %.17g wL %.17g wr %.17g\n", P.m, P.wp, P.wL, P.wr);
    std::fprintf(f, "w1"); for (double w : P.w1) std::fprintf(f, " %.17g", w); std::fprintf(f, "\n");
    std::fprintf(f, "frozen"); for (int z : P.frozen) std::fprintf(f, " %d", z); std::fprintf(f, "\n");
    for (auto& Q : P.poses) {
        std::fprintf(f, "pose");
        for (int b = 0; b < 2; ++b) {
            for (int r = 0; r < 3; ++r) std::fprintf(f, " %.17g %.17g %.17g", Q.R[b][r].x, Q.R[b][r].y, Q.R[b][r].z);
            std::fprintf(f, " %.17g %.17g %.17g", Q.T[b].x, Q.T[b].y, Q.T[b].z);
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    f = std::fopen((out + "/solution.txt").c_str(), "w");
    std::fprintf(f, "iterations %d evaluations %d stop %s E0 %.17g E %.17g\n", it, evals, stop.c_str(), E0, fx);
    for (int k = 0; k < n; ++k) std::fprintf(f, "%.17g\n", u[k]);
    std::fclose(f);
    return (gc < gate && gcw >= gate) ? 0 : 1;
}
