struct CageLuSolveParams {
  uint K;
  uint ncols;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLuSolveParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> G;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> piv;
[[vk::binding(3, 0)]]
RWStructuredBuffer<double> X;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint j = tid.x;
  if ((j >= params.ncols)) {
    return;
  }
  for (uint k = 0u; k < params.K; ++k) {
    uint pr = piv[k];
    if ((pr != k)) {
      double tmp = X[((k * params.ncols) + j)];
      X[((k * params.ncols) + j)] = X[((pr * params.ncols) + j)];
      X[((pr * params.ncols) + j)] = tmp;
    }
  }
  for (uint r = 0u; r < params.K; ++r) {
    double s = X[((r * params.ncols) + j)];
    for (uint c = 0u; c < r; ++c) {
      s = (s - (G[((r * params.K) + c)] * X[((c * params.ncols) + j)]));
    }
    X[((r * params.ncols) + j)] = s;
  }
  for (uint q = 0u; q < params.K; ++q) {
    uint r = ((params.K - 1u) - q);
    double s = X[((r * params.ncols) + j)];
    for (uint c = (r + 1u); c < params.K; ++c) {
      s = (s - (G[((r * params.K) + c)] * X[((c * params.ncols) + j)]));
    }
    X[((r * params.ncols) + j)] = (s / G[((r * params.K) + r)]);
  }
}
