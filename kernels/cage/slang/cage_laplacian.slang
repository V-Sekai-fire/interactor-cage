struct CageLapParams {
  uint n;
  uint transpose;
  uint accumulate;
  float scale;
  uint combinatorial;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageLapParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> rowptr;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> col;
[[vk::binding(3, 0)]]
StructuredBuffer<float> x;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> y;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  uint r0 = rowptr[i];
  uint r1 = rowptr[(i + 1u)];
  float sx = 0.000000;
  float sy = 0.000000;
  float sz = 0.000000;
  for (uint k = r0; k < r1; ++k) {
    uint j = col[k];
    float wj = 1.000000;
    if (((params.transpose != 0u) && (params.combinatorial == 0u))) {
      wj = (1.000000 / float((rowptr[(j + 1u)] - rowptr[j])));
    }
    sx = (sx + (wj * x[(j * 3u)]));
    sy = (sy + (wj * x[((j * 3u) + 1u)]));
    sz = (sz + (wj * x[((j * 3u) + 2u)]));
  }
  if ((((params.transpose == 0u) && (params.combinatorial == 0u)) && (r1 > r0))) {
    float dg = float((r1 - r0));
    sx = (sx / dg);
    sy = (sy / dg);
    sz = (sz / dg);
  }
  bool acc = (params.accumulate != 0u);
  float rx = (x[(i * 3u)] - sx);
  float ry = (x[((i * 3u) + 1u)] - sy);
  float rz = (x[((i * 3u) + 2u)] - sz);
  if ((params.combinatorial != 0u)) {
    float dc = float((r1 - r0));
    rx = (sx - (dc * x[(i * 3u)]));
    ry = (sy - (dc * x[((i * 3u) + 1u)]));
    rz = (sz - (dc * x[((i * 3u) + 2u)]));
  }
  y[(i * 3u)] = ((acc ? y[(i * 3u)] : 0.000000) + (params.scale * rx));
  y[((i * 3u) + 1u)] = ((acc ? y[((i * 3u) + 1u)] : 0.000000) + (params.scale * ry));
  y[((i * 3u) + 2u)] = ((acc ? y[((i * 3u) + 2u)] : 0.000000) + (params.scale * rz));
}
