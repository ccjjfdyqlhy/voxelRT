// rt_shaders - PCSS shadow mapping — from Derivative Shaders (SunLighting)
// Adapted to generic uniforms.
// License: MIT

#ifndef RT_SHADOW_GLSL
#define RT_SHADOW_GLSL

#include "../core/math.glsl"
#include "../core/encode.glsl"
#include "../atmosphere/atmosphere.glsl"
#include "../atmosphere/fog.glsl"
#include "brdf.glsl"

#define SHADOW_MAP_BIAS	0.9
// #define COLORED_SHADOWS
// #define SCREEN_SPACE_SHADOWS

float DistortionFactor(vec2 projectedPos) {
    return length(projectedPos) * 0.25 + 0.91;
}

vec3 DistortShadowSpace(vec3 pos, float factor) {
    pos.xy *= rcp(factor);
    return pos;
}

float GetShadowBias() {
    return 1e-4;
}

vec3 WorldPosToShadowProjPosBias(in vec3 worldPos, out float distortFactor) {
    vec3 shadowClipPos = transMAD(uShadowMatrixView, worldPos);
    shadowClipPos = projMAD(uShadowMatrixProj, shadowClipPos);
    distortFactor = DistortionFactor(shadowClipPos.xy);
    return DistortShadowSpace(shadowClipPos, distortFactor) * 0.5 + 0.5;
}

vec2 BlockerSearch(in vec3 shadowProjPos, in float dither) {
    float searchDepth = 0.0, sumWeight = 0.0, sssDepth = 0.0;
    float searchRadius = 2.0 * uShadowMatrixProj[0].x;
    vec2 rot = cossin(dither * TAU) * searchRadius;
    const vec2 angleStep = cossin(TAU * 0.125);
    const mat2 rotStep = mat2(angleStep, -angleStep.y, angleStep.x);
    for (uint i = 0u; i < 8u; ++i, rot *= rotStep) {
        float fi = float(i) + dither;
        vec2 sampleCoord = shadowProjPos.xy + rot * sqrt(fi * 0.125);
        float depthSample = texelFetch(uShadowMap, ivec2(sampleCoord * shadowMapResolution), 0).x;
        float weight = step(depthSample, shadowProjPos.z);
        sssDepth += max0(shadowProjPos.z - depthSample);
        searchDepth += depthSample * weight;
        sumWeight += weight;
    }
    searchDepth *= 1.0 / sumWeight;
    searchDepth = min(2.0 * (shadowProjPos.z - searchDepth) / searchDepth, 1.0);
    return vec2(searchDepth * uShadowMatrixProj[0].x, sssDepth * uShadowMatrixProjInv[2].z);
}

vec3 PercentageCloserFilter(in vec3 shadowProjPos, in float dither, in float penumbraScale) {
    shadowProjPos.z -= GetShadowBias() - dither * 5e-5;
    const float rSteps = 1.0 / float(PCF_SAMPLES);
    vec3 result = vec3(0.0);
    vec2 rot = cossin(dither * TAU) * penumbraScale;
    const vec2 angleStep = cossin(TAU * 0.125);
    const mat2 rotStep = mat2(angleStep, -angleStep.y, angleStep.x);
    for (uint i = 0u; i < PCF_SAMPLES; ++i, rot *= rotStep) {
        float fi = float(i) + dither;
        vec2 sampleCoord = shadowProjPos.xy + rot * sqrt(fi * rSteps);
        float sampleDepth1 = textureLod(uShadowMap, vec3(sampleCoord, shadowProjPos.z), 0).x;
        result += sampleDepth1;
    }
    return result * rSteps;
}

float ScreenSpaceShadow(in vec3 viewPos, in vec3 rayPos, in float dither, in float sssAmount) {
    vec3 lightVector = mat3(uViewMatrix) * uLightDir;
    vec3 position = (uProjMatrix * vec4(lightVector * abs(viewPos.z) * 0.1 + viewPos, 1.0)).xyz;
    vec3 screenDir = normalize(position - rayPos);
    float absorption = pow(sssAmount, sqrt(length(viewPos)) * 0.5);
    screenDir.xy *= uScreenSize;
    rayPos.xy *= uScreenSize;
    vec3 rayStep = screenDir * max(0.01, 0.05 - sssAmount * 0.05) * uProjMatrix[1][1] * rcp(12.0);
    rayPos += rayStep * (1.0 - sssAmount + dither);
    const float zTolerance = 0.025;
    float shadow = 1.0;
    for (uint i = 0u; i < 12u; ++i, rayPos += rayStep) {
        if (clamp(rayPos.xy, vec2(0.0), uScreenSize) != rayPos.xy) break;
        if (rayPos.z >= 1.0) break;
        float depth = texelFetch(uTexDepth, ivec2(rayPos.xy), 0).x;
        if (depth < rayPos.z) {
            float linearSample = GetDepthLinear(depth);
            float currentDepth = GetDepthLinear(rayPos.z);
            if (abs(linearSample - currentDepth) / currentDepth < zTolerance)
                shadow *= absorption;
        }
        if (shadow < 1e-2) break;
    }
    return shadow;
}

float CalculateFakeBouncedLight(in vec3 normal) {
    normal.y = -normal.y;
    vec3 bounceVector = normalize(uLightDir + vec3(0.0, 1.0, 0.0));
    float bounce = saturate(dot(normal, bounceVector) * 0.4 + 0.6);
    return bounce * (2.0 - bounce) * 3e-2;
}

vec3 CalculateSubsurfaceScattering(in vec3 albedo, in float sssAmount, in float sssDepth, in float LdotV) {
    vec3 coeff = albedo * inversesqrt(GetLuminance(albedo) + 1e-6);
    coeff = oneMinus(0.75 * saturate(coeff)) * (28.0 / sssAmount);
    vec3 subsurfaceScattering =  fastExp(0.375 * coeff * sssDepth) * HenyeyGreensteinPhase(-LdotV, 0.6);
    subsurfaceScattering += fastExp(0.125 * coeff * sssDepth) * (0.33 * HenyeyGreensteinPhase(-LdotV, 0.35) + 0.17 * rPI);
    return subsurfaceScattering * sssAmount * PI;
}

#endif // RT_SHADOW_GLSL
