// rt_shaders - Global Illumination via Reflective Shadow Maps — from Derivative Shaders
// Adapted to generic uniforms.
// License: MIT

#ifndef RT_GI_GLSL
#define RT_GI_GLSL

#include "../core/math.glsl"
#include "../core/encode.glsl"
#include "../core/noise.glsl"

vec3 WorldToShadowProjPos(in vec3 worldPos) {
    vec3 shadowPos = transMAD(uShadowMatrixView, worldPos);
    return projMAD(uShadowMatrixProj, shadowPos);
}

vec2 DistortShadowProjPos(in vec2 shadowClipPos) {
    float distortFactor = quarticLength(shadowClipPos * 1.165) * SHADOW_MAP_BIAS + 1.0 - SHADOW_MAP_BIAS;
    shadowClipPos.xy *= rcp(distortFactor);
    return shadowClipPos * 0.5 + 0.5;
}

vec3 CalculateRSM(in vec3 viewPos, in vec3 worldNormal, in float dither) {
    vec3 total = vec3(0.0);
    vec3 worldPos = transMAD(uViewMatrixInv, viewPos);
    vec3 shadowPos = WorldToShadowProjPos(worldPos);

    vec3 shadowNormal = mat3(uShadowMatrixView) * worldNormal;
    shadowNormal.z = -shadowNormal.z;

    const float realShadowMapRes = shadowMapResolution;
    const float scale = GI_RADIUS * rcp(realShadowMapRes);
    const float sqRadius = GI_RADIUS * GI_RADIUS;
    const float radiusAdd = sqrt(sqRadius / GI_SAMPLES);
    const float rSteps = 1.0 / GI_SAMPLES;

    const float goldenAngle = TAU / (PHI1 + 1.0);
    const mat2 goldenRotate = mat2(cos(goldenAngle), -sin(goldenAngle), sin(goldenAngle), cos(goldenAngle));

    vec2 rot = sincos(dither * 64.0) * scale;
    dither *= rSteps;

    for (uint i = 0u; i < GI_SAMPLES; ++i, rot *= goldenRotate) {
        float fi = float(i) * rSteps + dither;
        vec2 coord = shadowPos.xy + rot * fi;
        ivec2 sampleTexel = ivec2(DistortShadowProjPos(coord) * realShadowMapRes);

        float sampleDepth = texelFetch(uShadowNormal, sampleTexel, 0).x * 10.0 - 5.0;
        vec3 sampleVector = vec3(coord, sampleDepth) - shadowPos;
        float sampleDist = dotSelf(sampleVector);
        if (sampleDist > sqRadius) continue;

        vec3 sampleDir = normalize(sampleVector);
        float diffuse = saturate(dot(shadowNormal, sampleDir));
        if (diffuse < 1e-5) continue;

        vec3 sampleColor = texelFetch(uShadowNormal, sampleTexel, 0).rgb;
        vec3 sampleNormal = DecodeNormal(sampleColor.xy);
        sampleNormal.xy = -sampleNormal.xy;

        float bounce = saturate(dot(sampleNormal, sampleDir));
        if (bounce < 1e-5) continue;

        float falloff = rcp(sampleDist + radiusAdd);
        vec3 albedo = pow(texelFetch(uShadowAlbedo, sampleTexel, 0).rgb, vec3(2.2));

        total += albedo * fi * falloff * bounce * diffuse;
    }

    return total * sqRadius * rSteps * 5e-2 * GI_BRIGHTNESS;
}

#endif // RT_GI_GLSL
