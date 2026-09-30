struct CageBoneBlendParams {
  uint F;
  uint P;
  uint NB;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageBoneBlendParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> rowptr;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> col;
[[vk::binding(3, 0)]]
StructuredBuffer<float> w;
[[vk::binding(4, 0)]]
StructuredBuffer<float> bones;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> blend;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.P))) {
    return;
  }
  uint fp = (id / params.P);
  uint i = (id % params.P);
  float a0 = 0.000000;
  float a1 = 0.000000;
  float a2 = 0.000000;
  float a3 = 0.000000;
  float a4 = 0.000000;
  float a5 = 0.000000;
  float a6 = 0.000000;
  float a7 = 0.000000;
  float a8 = 0.000000;
  float a9 = 0.000000;
  float a10 = 0.000000;
  float a11 = 0.000000;
  for (uint k = rowptr[i]; k < rowptr[(i + 1u)]; ++k) {
    float wk = w[k];
    uint b = (((fp * params.NB) + col[k]) * 12u);
    a0 = (a0 + (wk * bones[(b + 0u)]));
    a1 = (a1 + (wk * bones[(b + 1u)]));
    a2 = (a2 + (wk * bones[(b + 2u)]));
    a3 = (a3 + (wk * bones[(b + 3u)]));
    a4 = (a4 + (wk * bones[(b + 4u)]));
    a5 = (a5 + (wk * bones[(b + 5u)]));
    a6 = (a6 + (wk * bones[(b + 6u)]));
    a7 = (a7 + (wk * bones[(b + 7u)]));
    a8 = (a8 + (wk * bones[(b + 8u)]));
    a9 = (a9 + (wk * bones[(b + 9u)]));
    a10 = (a10 + (wk * bones[(b + 10u)]));
    a11 = (a11 + (wk * bones[(b + 11u)]));
  }
  blend[((id * 12u) + 0u)] = a0;
  blend[((id * 12u) + 1u)] = a1;
  blend[((id * 12u) + 2u)] = a2;
  blend[((id * 12u) + 3u)] = a3;
  blend[((id * 12u) + 4u)] = a4;
  blend[((id * 12u) + 5u)] = a5;
  blend[((id * 12u) + 6u)] = a6;
  blend[((id * 12u) + 7u)] = a7;
  blend[((id * 12u) + 8u)] = a8;
  blend[((id * 12u) + 9u)] = a9;
  blend[((id * 12u) + 10u)] = a10;
  blend[((id * 12u) + 11u)] = a11;
}
