struct CageBlendParams {
  uint P;
  uint K;
  uint stride;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageBlendParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> rows;
[[vk::binding(2, 0)]]
StructuredBuffer<double> X;
[[vk::binding(3, 0)]]
RWStructuredBuffer<double> wd;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> wf;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  uint o = (i * params.stride);
  for (uint k = 0u; k < params.K; ++k) {
    double w = rows[(o + k)];
    for (uint m = 0u; m < params.K; ++m) {
      w = (w + (X[((m * params.K) + k)] * rows[((o + params.K) + m)]));
    }
    wd[((i * params.K) + k)] = w;
    wf[((i * params.K) + k)] = float(w);
  }
}
