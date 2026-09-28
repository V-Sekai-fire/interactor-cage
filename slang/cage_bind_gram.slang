struct CageGramParams {
  uint S;
  uint K;
  uint stride;
  uint ij0;
  uint count;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageGramParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> rows;
[[vk::binding(2, 0)]]
RWStructuredBuffer<double> G;
[[vk::binding(3, 0)]]
RWStructuredBuffer<double> rhs;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= params.count)) {
    return;
  }
  uint ij = (params.ij0 + id);
  if ((ij >= (params.K * params.K))) {
    return;
  }
  uint i = (ij / params.K);
  uint j = (ij % params.K);
  double ga = 0.0L;
  double gb = 0.0L;
  double r = 0.0L;
  for (uint s = 0u; s < params.S; ++s) {
    uint o = (s * params.stride);
    double ai = rows[((o + params.K) + i)];
    double aj = rows[((o + params.K) + j)];
    double bi = rows[(o + i)];
    double bj = rows[(o + j)];
    ga = (ga + (ai * aj));
    gb = (gb + (bi * bj));
    r = (r - (ai * bj));
  }
  G[ij] = (ga + gb);
  rhs[ij] = r;
}
