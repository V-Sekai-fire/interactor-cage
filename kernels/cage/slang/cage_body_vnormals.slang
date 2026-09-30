struct CageBodyParams {
  uint F;
  uint T;
  uint BV;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageBodyParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> rowptr;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> corner;
[[vk::binding(3, 0)]]
StructuredBuffer<float> fn;
[[vk::binding(4, 0)]]
StructuredBuffer<float> ang;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> vn;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.BV))) {
    return;
  }
  uint fp = (id / params.BV);
  uint vi = (id % params.BV);
  float x = 0.000000;
  float y = 0.000000;
  float z = 0.000000;
  for (uint k = rowptr[vi]; k < rowptr[(vi + 1u)]; ++k) {
    uint cn = corner[k];
    uint ft = ((fp * params.T) + (cn / 3u));
    float an = ang[(((fp * params.T) * 3u) + cn)];
    x = (x + (an * fn[(ft * 3u)]));
    y = (y + (an * fn[((ft * 3u) + 1u)]));
    z = (z + (an * fn[((ft * 3u) + 2u)]));
  }
  vn[(id * 3u)] = x;
  vn[((id * 3u) + 1u)] = y;
  vn[((id * 3u) + 2u)] = z;
}
