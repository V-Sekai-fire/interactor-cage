struct CageLbsTParams {
  uint F;
  uint P;
  uint accumulate;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLbsTParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> blend;
[[vk::binding(2, 0)]]
StructuredBuffer<float> gz;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> gy;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  bool acc = (params.accumulate != 0u);
  float s0 = (acc ? gy[((i * 3u) + 0u)] : 0.000000);
  float s1 = (acc ? gy[((i * 3u) + 1u)] : 0.000000);
  float s2 = (acc ? gy[((i * 3u) + 2u)] : 0.000000);
  for (uint fp = 0u; fp < params.F; ++fp) {
    uint id = ((fp * params.P) + i);
    s0 = (s0 + (((blend[((id * 12u) + 0u)] * gz[((id * 3u) + 0u)]) + (blend[((id * 12u) + 4u)] * gz[((id * 3u) + 1u)])) + (blend[((id * 12u) + 8u)] * gz[((id * 3u) + 2u)])));
    s1 = (s1 + (((blend[((id * 12u) + 1u)] * gz[((id * 3u) + 0u)]) + (blend[((id * 12u) + 5u)] * gz[((id * 3u) + 1u)])) + (blend[((id * 12u) + 9u)] * gz[((id * 3u) + 2u)])));
    s2 = (s2 + (((blend[((id * 12u) + 2u)] * gz[((id * 3u) + 0u)]) + (blend[((id * 12u) + 6u)] * gz[((id * 3u) + 1u)])) + (blend[((id * 12u) + 10u)] * gz[((id * 3u) + 2u)])));
  }
  gy[((i * 3u) + 0u)] = s0;
  gy[((i * 3u) + 1u)] = s1;
  gy[((i * 3u) + 2u)] = s2;
}
