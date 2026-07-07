// rt_shaders - Aurora (by nimitz 2017 @stormoid)
// Adapted to generic uniforms.
// License: CC BY-NC-SA 3.0 Unported

#ifndef RT_AURORA_GLSL
#define RT_AURORA_GLSL

#include "../core/math.glsl"
#include "atmosphere.glsl"

mat2 mm2(in float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
mat2 m2 = mat2(0.95534, 0.29552, -0.29552, 0.95534);
float tri(in float x) { return clamp(abs(fract(x) - 0.5), 0.01, 0.49); }
vec2 tri2(in vec2 p)  { return vec2(tri(p.x) + tri(p.y), tri(p.y + tri(p.x))); }

float triNoise2d(in vec2 p, in float spd) {
    float z = 1.8, z2 = 2.5, rz = 0.0;
    p *= mm2(p.x * 0.06);
    vec2 bp = p;
    for (uint i = 0u; i < 5u; ++i) {
        vec2 dg = tri2(bp * 1.85) * 0.75;
        dg *= mm2(uTime * spd);
        p -= dg / z2;
        bp *= 1.3;
        z2 *= 0.45;
        z *= 0.42;
        p *= 1.21 + (rz - 1.0) * 0.02;
        p *= -m2;
        rz += tri(p.x + tri(p.y)) * z;
    }
    return clamp(pow(rz * 29.0, -1.3), 0.0, 0.55);
}

float hash21(in vec2 n) { return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453); }

vec4 aurora(in vec3 ro, in vec3 rd) {
    vec4 col = vec4(0.0), avgCol = vec4(0.0);
    for (float i = 0.0; i < 40.0; i++) {
        float of = 0.006 * hash21(gl_FragCoord.xy) * smoothstep(0.0, 15.0, i);
        float pt = ((0.8 + pow(i, 1.4) * 0.002) - ro.y) / (rd.y * 2.0 + 0.4);
        pt -= of;
        vec3 bpos = ro + pt * rd;
        vec2 p = bpos.zx;
        float rzt = triNoise2d(p, 0.1883);
        vec4 col2 = vec4(0.0, 0.0, 0.0, rzt);
        col2.rgb = (sin(1.0 - vec3(2.15, -0.5, 1.2) + i * 0.043) * 0.5 + 0.5) * rzt;
        avgCol = mix(avgCol, col2, 0.5);
        col += avgCol * exp2(-i * 0.065 - 2.5) * smoothstep(0.0, 5.0, i);
    }
    col *= saturate(rd.y * 15.0 + 0.4);
    return col * 1.8;
}

vec3 NightAurora(in vec3 worldDir) {
    if (worldDir.y < 0.0 && uEyeAltitude < 2e4) return vec3(0.0);
    vec3 planeOrigin = vec3(0.0, planetRadius + uEyeAltitude, 0.0);
    vec2 intersection = RaySphereIntersection(planeOrigin, worldDir, planetRadius + 2e4);
    float raylength = intersection.y;
    if (raylength <= 0.0 || raylength > 5e5) return vec3(0.0);
    vec3 rd = worldDir * raylength;
    float fade = fastExp(-raylength * 1e-5);
    vec4 aur = smoothstep(0.0, 2.5, aurora(vec3(0.0, 0.0, -6.7), rd * 1e-5));
    return aur.rgb * fade * auroraAmount;
}

#endif // RT_AURORA_GLSL
