// rt_shaders - Screen-space ambient occlusion — from Derivative Shaders
// Adapted to generic uniforms.
// License: MIT

#ifndef RT_SSAO_GLSL
#define RT_SSAO_GLSL

#include "../core/math.glsl"
#include "../core/noise.glsl"

vec3 ScreenToViewSpace(in vec2 coord) {
    vec3 NDCPos = vec3(coord * 2.0 - 1.0, texelFetch(uTexDepth, ivec2(coord * uScreenSize), 0).x * 2.0 - 1.0);
    vec3 viewPos = projMAD(uProjMatrixInv, NDCPos);
    viewPos /= uProjMatrixInv[2].w * NDCPos.z + uProjMatrixInv[3].w;
    return viewPos;
}

float SpiralAO(in vec2 coord, in vec3 viewPos, in vec3 normal, float dither) {
    float rSteps = 1.0 / float(SSAO_SAMPLES);
    float maxSqLen = sqr(viewPos.z) * 0.25;
    vec2 radius = vec2(0.0);
    vec2 rayStep = vec2(0.6 / uAspectRatio, 0.6) / max((uFar - uNear) * -viewPos.z / uFar + uNear, 5.0) * uProjMatrix[1][1];

    const float goldenAngle = TAU / (PHI1 + 1.0);
    const mat2 goldenRotate = mat2(cos(goldenAngle), -sin(goldenAngle), sin(goldenAngle), cos(goldenAngle));

    vec2 rot = sincos(dither * TAU) * rSteps;
    float total = 0.0;

    for (uint i = 0u; i < SSAO_SAMPLES; ++i, rot *= goldenRotate) {
        radius += rayStep;

        vec3 diff = ScreenToViewSpace(coord + rot * radius) - viewPos;
        float diffSqLen = dotSelf(diff);
        if (diffSqLen > 1e-5 && diffSqLen < maxSqLen) {
            float NdotL = saturate(dot(normal, diff * inversesqrt(diffSqLen)));
            total += NdotL * saturate(1.0 - diffSqLen / maxSqLen);
        }

        diff = ScreenToViewSpace(coord - rot * radius) - viewPos;
        diffSqLen = dotSelf(diff);
        if (diffSqLen > 1e-5 && diffSqLen < maxSqLen) {
            float NdotL = saturate(dot(normal, diff * inversesqrt(diffSqLen)));
            total += NdotL * saturate(1.0 - diffSqLen / maxSqLen);
        }
    }

    total = max0(1.0 - total * rSteps * SSAO_STRENGTH);
    return total * sqrt(total);
}

#endif // RT_SSAO_GLSL
