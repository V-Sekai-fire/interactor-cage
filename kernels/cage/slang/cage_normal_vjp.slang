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
StructuredBuffer<float> n;
[[vk::binding(4, 0)]]
StructuredBuffer<float> nlen;
[[vk::binding(5, 0)]]
StructuredBuffer<float> gn;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> gcorner;

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
  float m0 = n[((t * 3u) + 0u)];
  float m1 = n[((t * 3u) + 1u)];
  float m2 = n[((t * 3u) + 2u)];
  float g0 = gn[((t * 3u) + 0u)];
  float g1 = gn[((t * 3u) + 1u)];
  float g2 = gn[((t * 3u) + 2u)];
  float mg = (((m0 * g0) + (m1 * g1)) + (m2 * g2));
  float il = nlen[t];
  float d0 = ((g0 - (m0 * mg)) / il);
  float d1 = ((g1 - (m1 * mg)) / il);
  float d2 = ((g2 - (m2 * mg)) / il);
  float gb0 = ((f1 * d2) - (f2 * d1));
  float gb1 = ((f2 * d0) - (f0 * d2));
  float gb2 = ((f0 * d1) - (f1 * d0));
  float gc0 = ((d1 * e2) - (d2 * e1));
  float gc1 = ((d2 * e0) - (d0 * e2));
  float gc2 = ((d0 * e1) - (d1 * e0));
  gcorner[((t * 9u) + 0u)] = (-(gb0 + gc0));
  gcorner[((t * 9u) + 1u)] = (-(gb1 + gc1));
  gcorner[((t * 9u) + 2u)] = (-(gb2 + gc2));
  gcorner[((t * 9u) + 3u)] = gb0;
  gcorner[((t * 9u) + 4u)] = gb1;
  gcorner[((t * 9u) + 5u)] = gb2;
  gcorner[((t * 9u) + 6u)] = gc0;
  gcorner[((t * 9u) + 7u)] = gc1;
  gcorner[((t * 9u) + 8u)] = gc2;
}
