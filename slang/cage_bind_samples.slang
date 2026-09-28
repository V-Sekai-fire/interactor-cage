struct CageSamplesParams {
  uint nT;
  uint nB;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageSamplesParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<double> cage;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(3, 0)]]
StructuredBuffer<double> bary;
[[vk::binding(4, 0)]]
RWStructuredBuffer<double> pts;
[[vk::binding(5, 0)]]
RWStructuredBuffer<double> gamma;
[[vk::binding(6, 0)]]
RWStructuredBuffer<uint> own;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint s = tid.x;
  if ((s >= (params.nT * params.nB))) {
    return;
  }
  uint t = (s / params.nB);
  uint k = (s % params.nB);
  double g0 = bary[(k * 3u)];
  double g1 = bary[((k * 3u) + 1u)];
  double g2 = bary[((k * 3u) + 2u)];
  double3 eta = (((g0 * double3(cage[(tris[((t * 3u) + 0u)] * 3u)], cage[((tris[((t * 3u) + 0u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 0u)] * 3u) + 2u)])) + (g1 * double3(cage[(tris[((t * 3u) + 1u)] * 3u)], cage[((tris[((t * 3u) + 1u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 1u)] * 3u) + 2u)]))) + (g2 * double3(cage[(tris[((t * 3u) + 2u)] * 3u)], cage[((tris[((t * 3u) + 2u)] * 3u) + 1u)], cage[((tris[((t * 3u) + 2u)] * 3u) + 2u)])));
  pts[(s * 3u)] = eta.x;
  pts[((s * 3u) + 1u)] = eta.y;
  pts[((s * 3u) + 2u)] = eta.z;
  gamma[(s * 3u)] = g0;
  gamma[((s * 3u) + 1u)] = g1;
  gamma[((s * 3u) + 2u)] = g2;
  own[s] = t;
}
