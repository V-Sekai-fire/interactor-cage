struct CageLuParams {
  uint K;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLuParams> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<double> G;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> piv;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> status;

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  uint ok = 1u;
  for (uint k = 0u; k < params.K; ++k) {
    uint pr = k;
    double mx = abs(G[((k * params.K) + k)]);
    for (uint r = (k + 1u); r < params.K; ++r) {
      double a = abs(G[((r * params.K) + k)]);
      if ((a > mx)) {
        mx = a;
        pr = r;
      }
    }
    piv[k] = pr;
    if ((mx == 0.0L)) {
      ok = 0u;
    } else {
      if ((pr != k)) {
        for (uint c = 0u; c < params.K; ++c) {
          double tmp = G[((k * params.K) + c)];
          G[((k * params.K) + c)] = G[((pr * params.K) + c)];
          G[((pr * params.K) + c)] = tmp;
        }
      }
      double dk = G[((k * params.K) + k)];
      for (uint r = (k + 1u); r < params.K; ++r) {
        double l = (G[((r * params.K) + k)] / dk);
        G[((r * params.K) + k)] = l;
        for (uint c = (k + 1u); c < params.K; ++c) {
          G[((r * params.K) + c)] = (G[((r * params.K) + c)] - (l * G[((k * params.K) + c)]));
        }
      }
    }
  }
  status[0u] = ok;
}
