struct CageBodyParams {
  uint F;
  uint T;
  uint BV;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageBodyParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> bv;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> fn;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> ang;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.T))) {
    return;
  }
  uint fp = (id / params.T);
  uint t = (id % params.T);
  uint o = (fp * params.BV);
  float a0 = bv[(((o + tris[(t * 3u)]) * 3u) + 0u)];
  float a1 = bv[(((o + tris[(t * 3u)]) * 3u) + 1u)];
  float a2 = bv[(((o + tris[(t * 3u)]) * 3u) + 2u)];
  float b0 = bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 0u)];
  float b1 = bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 1u)];
  float b2 = bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 2u)];
  float q0 = bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 0u)];
  float q1 = bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 1u)];
  float q2 = bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 2u)];
  float ab0 = (b0 - a0);
  float ab1 = (b1 - a1);
  float ab2 = (b2 - a2);
  float aq0 = (q0 - a0);
  float aq1 = (q1 - a1);
  float aq2 = (q2 - a2);
  float ba0 = (a0 - b0);
  float ba1 = (a1 - b1);
  float ba2 = (a2 - b2);
  float bq0 = (q0 - b0);
  float bq1 = (q1 - b1);
  float bq2 = (q2 - b2);
  float qa0 = (a0 - q0);
  float qa1 = (a1 - q1);
  float qa2 = (a2 - q2);
  float qb0 = (b0 - q0);
  float qb1 = (b1 - q1);
  float qb2 = (b2 - q2);
  float N0 = ((ab1 * aq2) - (ab2 * aq1));
  float N1 = ((ab2 * aq0) - (ab0 * aq2));
  float N2 = ((ab0 * aq1) - (ab1 * aq0));
  float l = max(sqrt((((N0 * N0) + (N1 * N1)) + (N2 * N2))), 1.0e-20f);
  fn[((id * 3u) + 0u)] = (N0 / l);
  fn[((id * 3u) + 1u)] = (N1 / l);
  fn[((id * 3u) + 2u)] = (N2 / l);
  float cax0 = ((ab1 * aq2) - (ab2 * aq1));
  float cax1 = ((ab2 * aq0) - (ab0 * aq2));
  float cax2 = ((ab0 * aq1) - (ab1 * aq0));
  float t0 = atan2(sqrt((((cax0 * cax0) + (cax1 * cax1)) + (cax2 * cax2))), (((ab0 * aq0) + (ab1 * aq1)) + (ab2 * aq2)));
  float cbx0 = ((bq1 * ba2) - (bq2 * ba1));
  float cbx1 = ((bq2 * ba0) - (bq0 * ba2));
  float cbx2 = ((bq0 * ba1) - (bq1 * ba0));
  float t1 = atan2(sqrt((((cbx0 * cbx0) + (cbx1 * cbx1)) + (cbx2 * cbx2))), (((bq0 * ba0) + (bq1 * ba1)) + (bq2 * ba2)));
  float cqx0 = ((qa1 * qb2) - (qa2 * qb1));
  float cqx1 = ((qa2 * qb0) - (qa0 * qb2));
  float cqx2 = ((qa0 * qb1) - (qa1 * qb0));
  float t2 = atan2(sqrt((((cqx0 * cqx0) + (cqx1 * cqx1)) + (cqx2 * cqx2))), (((qa0 * qb0) + (qa1 * qb1)) + (qa2 * qb2)));
  ang[(id * 3u)] = t0;
  ang[((id * 3u) + 1u)] = t1;
  ang[((id * 3u) + 2u)] = t2;
}
