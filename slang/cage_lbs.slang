struct CageLbsParams {
  uint F;
  uint P;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLbsParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> blend;
[[vk::binding(2, 0)]]
StructuredBuffer<float> y;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> z;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.P))) {
    return;
  }
  uint i = (id % params.P);
  float y0 = y[(i * 3u)];
  float y1 = y[((i * 3u) + 1u)];
  float y2 = y[((i * 3u) + 2u)];
  z[((id * 3u) + 0u)] = ((((blend[((id * 12u) + 0u)] * y0) + (blend[((id * 12u) + 1u)] * y1)) + (blend[((id * 12u) + 2u)] * y2)) + blend[((id * 12u) + 3u)]);
  z[((id * 3u) + 1u)] = ((((blend[((id * 12u) + 4u)] * y0) + (blend[((id * 12u) + 5u)] * y1)) + (blend[((id * 12u) + 6u)] * y2)) + blend[((id * 12u) + 7u)]);
  z[((id * 3u) + 2u)] = ((((blend[((id * 12u) + 8u)] * y0) + (blend[((id * 12u) + 9u)] * y1)) + (blend[((id * 12u) + 10u)] * y2)) + blend[((id * 12u) + 11u)]);
}
