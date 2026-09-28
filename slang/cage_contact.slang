struct CageContactParams {
  uint F;
  uint P;
  uint T;
  uint BV;
  float margin;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CageContactParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> z;
[[vk::binding(2, 0)]]
StructuredBuffer<float> bv;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> tris;
[[vk::binding(4, 0)]]
StructuredBuffer<float> fn;
[[vk::binding(5, 0)]]
StructuredBuffer<float> vn;
[[vk::binding(6, 0)]]
StructuredBuffer<uint> nbr;
[[vk::binding(7, 0)]]
StructuredBuffer<float> fw;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float> hb;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> gz;
[[vk::binding(10, 0)]]
RWStructuredBuffer<float> dist;

uint closest_tri(float3 pt, float3 a, float3 b, float3 c, out float3 q) {
  float3 ab = (b - a);
  float3 ac = (c - a);
  float3 ap = (pt - a);
  float d1 = (((ab.x * ap.x) + (ab.y * ap.y)) + (ab.z * ap.z));
  float d2 = (((ac.x * ap.x) + (ac.y * ap.y)) + (ac.z * ap.z));
  if (((d1 <= 0.000000) && (d2 <= 0.000000))) {
    q = a;
    return 1u;
  }
  float3 bp = (pt - b);
  float d3 = (((ab.x * bp.x) + (ab.y * bp.y)) + (ab.z * bp.z));
  float d4 = (((ac.x * bp.x) + (ac.y * bp.y)) + (ac.z * bp.z));
  if (((d3 >= 0.000000) && (d4 <= d3))) {
    q = b;
    return 2u;
  }
  float vc = ((d1 * d4) - (d3 * d2));
  if ((((vc <= 0.000000) && (d1 >= 0.000000)) && (d3 <= 0.000000))) {
    float t1 = (d1 / (d1 - d3));
    q = (a + (t1 * ab));
    return 4u;
  }
  float3 cp = (pt - c);
  float d5 = (((ab.x * cp.x) + (ab.y * cp.y)) + (ab.z * cp.z));
  float d6 = (((ac.x * cp.x) + (ac.y * cp.y)) + (ac.z * cp.z));
  if (((d6 >= 0.000000) && (d5 <= d6))) {
    q = c;
    return 3u;
  }
  float vb = ((d5 * d2) - (d1 * d6));
  if ((((vb <= 0.000000) && (d2 >= 0.000000)) && (d6 <= 0.000000))) {
    float t2 = (d2 / (d2 - d6));
    q = (a + (t2 * ac));
    return 6u;
  }
  float va = ((d3 * d6) - (d5 * d4));
  if ((((va <= 0.000000) && ((d4 - d3) >= 0.000000)) && ((d5 - d6) >= 0.000000))) {
    float t3 = ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    q = (b + (t3 * (c - b)));
    return 5u;
  }
  float dn = (1.000000 / ((va + vb) + vc));
  float sv = (vb * dn);
  float sw = (vc * dn);
  q = ((a + (ab * sv)) + (ac * sw));
  return 0u;
}

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id = tid.x;
  if ((id >= (params.F * params.P))) {
    return;
  }
  uint fp = (id / params.P);
  uint o = (fp * params.BV);
  float3 pt = float3(z[(id * 3u)], z[((id * 3u) + 1u)], z[((id * 3u) + 2u)]);
  float best = asfloat(2139095039u);
  uint bt = 0u;
  uint bf = 0u;
  float3 bq = pt;
  for (uint t = 0u; t < params.T; ++t) {
    float3 q;
    uint ft = closest_tri(pt, float3(bv[((o + tris[(t * 3u)]) * 3u)], bv[(((o + tris[(t * 3u)]) * 3u) + 1u)], bv[(((o + tris[(t * 3u)]) * 3u) + 2u)]), float3(bv[((o + tris[((t * 3u) + 1u)]) * 3u)], bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 1u)], bv[(((o + tris[((t * 3u) + 1u)]) * 3u) + 2u)]), float3(bv[((o + tris[((t * 3u) + 2u)]) * 3u)], bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 1u)], bv[(((o + tris[((t * 3u) + 2u)]) * 3u) + 2u)]), q);
    float3 dq = (pt - q);
    float d2 = (((dq.x * dq.x) + (dq.y * dq.y)) + (dq.z * dq.z));
    if ((d2 < best)) {
      best = d2;
      bt = t;
      bf = ft;
      bq = q;
    }
  }
  uint ftb = ((fp * params.T) + bt);
  float3 pn = float3(fn[(ftb * 3u)], fn[((ftb * 3u) + 1u)], fn[((ftb * 3u) + 2u)]);
  if (((bf >= 1u) && (bf <= 3u))) {
    pn = float3(vn[((o + tris[(((bt * 3u) + bf) - 1u)]) * 3u)], vn[(((o + tris[(((bt * 3u) + bf) - 1u)]) * 3u) + 1u)], vn[(((o + tris[(((bt * 3u) + bf) - 1u)]) * 3u) + 2u)]);
  }
  if ((bf >= 4u)) {
    pn = (pn + float3(fn[(((fp * params.T) + nbr[(((bt * 3u) + bf) - 4u)]) * 3u)], fn[((((fp * params.T) + nbr[(((bt * 3u) + bf) - 4u)]) * 3u) + 1u)], fn[((((fp * params.T) + nbr[(((bt * 3u) + bf) - 4u)]) * 3u) + 2u)]));
  }
  float3 dv = (pt - bq);
  float dl = sqrt(best);
  float sg = (((((dv.x * pn.x) + (dv.y * pn.y)) + (dv.z * pn.z)) < 0.000000) ? -1.000000 : 1.000000);
  float3 gr = (pn / max(sqrt((((pn.x * pn.x) + (pn.y * pn.y)) + (pn.z * pn.z))), 1.0e-20f));
  if ((dl > 1.0e-12f)) {
    gr = ((sg * dv) / dl);
  }
  float sd = (sg * dl);
  float h = max((params.margin - sd), 0.000000);
  float wp = fw[fp];
  hb[id] = (sqrt(wp) * h);
  float k = ((-2.000000 * wp) * h);
  gz[(id * 3u)] = (k * gr.x);
  gz[((id * 3u) + 1u)] = (k * gr.y);
  gz[((id * 3u) + 2u)] = (k * gr.z);
  dist[id] = sd;
}
