struct CageNormalsParams {
  uint nT;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageNormalsParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> c;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> n;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> nlen;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  if ((t >= params.nT)) {
    return;
  }
  float a0 = c[((tris[(t * 3u)] * 3u) + 0u)];
  float a1 = c[((tris[(t * 3u)] * 3u) + 1u)];
  float a2 = c[((tris[(t * 3u)] * 3u) + 2u)];
  float b0 = c[((tris[((t * 3u) + 1u)] * 3u) + 0u)];
  float b1 = c[((tris[((t * 3u) + 1u)] * 3u) + 1u)];
  float b2 = c[((tris[((t * 3u) + 1u)] * 3u) + 2u)];
  float q0 = c[((tris[((t * 3u) + 2u)] * 3u) + 0u)];
  float q1 = c[((tris[((t * 3u) + 2u)] * 3u) + 1u)];
  float q2 = c[((tris[((t * 3u) + 2u)] * 3u) + 2u)];
  float e0 = (b0 - a0);
  float e1 = (b1 - a1);
  float e2 = (b2 - a2);
  float f0 = (q0 - a0);
  float f1 = (q1 - a1);
  float f2 = (q2 - a2);
  float N0 = ((e1 * f2) - (e2 * f1));
  float N1 = ((e2 * f0) - (e0 * f2));
  float N2 = ((e0 * f1) - (e1 * f0));
  float l = max(sqrt((((N0 * N0) + (N1 * N1)) + (N2 * N2))), 1.0e-20f);
  n[((t * 3u) + 0u)] = (N0 / l);
  n[((t * 3u) + 1u)] = (N1 / l);
  n[((t * 3u) + 2u)] = (N2 / l);
  nlen[t] = l;
}
