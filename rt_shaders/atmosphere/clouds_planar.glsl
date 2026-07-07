// rt_shaders - Planar clouds (cirrus / cirrocumulus) — from Derivative Shaders
// Adapted to generic uniforms.
// License: MIT

#ifndef RT_CLOUDS_PLANAR_GLSL
#define RT_CLOUDS_PLANAR_GLSL

#include "../core/math.glsl"
#include "../core/noise.glsl"
#include "atmosphere.glsl"

#define CLOUD_PLANE_ALTITUDE 7000
#define CLOUD_PLANE0_DENSITY 1.0
#define CLOUD_PLANE0_COVERY 0.5
#define CLOUD_PLANE1_DENSITY 1.0
#define CLOUD_PLANE1_COVERY 0.5

// --- Cirrus (high-altitude wispy) ---
#if CIRRUS_CLOUDS >= 1
float GetCloudsNoise(vec2 position) { return texture(uNoiseTex, position * 1e-2).a; }

vec4 PlanarSample0(in float dist, in vec2 worldPos, in float LdotV) {
    worldPos /= 1.0 + distance(worldPos, uCameraPos.xz) * 5e-6;
    vec2 position = worldPos * 4e-5 - wind.xz;
    position += texture(uNoiseTex, position * 0.04).y * 0.1;
    float localCoverage = texture(uNoiseTex, position * 2e-3 + 0.15).x;

    const float goldenAngle = TAU / (PHI1 + 1.0);
    const mat2 goldenRotate = mat2(cos(goldenAngle), -sin(goldenAngle), sin(goldenAngle), cos(goldenAngle));

    float amplitude = 0.5;
    float noise = GetCloudsNoise(position);
    for (uint i = 1u; i < 6u; ++i, amplitude *= 0.43) {
        position = goldenRotate * 3.2 * (position - wind.xz);
        noise += GetCloudsNoise(position * (1.0 + vec2(-0.35, 0.05) * sqrt(float(i)))) * amplitude;
    }
    noise -= saturate(localCoverage * 4.0 - 1.6);
    noise = saturate(noise * 1.36 + CLOUD_PLANE0_COVERY - 1.7) * noise;
    if (noise < 1e-5) return vec4(0.0);

    float powder = oneMinus(fastExp(-noise * 2.4)) * 0.7;
    powder /= 1.0 - powder;
    float phase = MiePhaseClouds(LdotV, vec3(-0.2, 0.5, 0.9), vec3(0.3, 0.6, 0.1));
    bool moonlit = uSunDir.y < -0.049;
    vec3 lightColor = phase * (moonlit ? vec3(NIGHT_BRIGHTNESS) : uSunIlluminance) * 17.0;
    lightColor += uSkyIlluminance * 0.1;
    lightColor *= oneMinus(0.8 * uWetness);
    noise = 1.0 - fastExp(-noise * 1.6 * CLOUD_PLANE0_DENSITY);
    return vec4(lightColor * powder * noise, noise);
}
#endif

// --- Cirrocumulus (patchy low-altitude) ---
#ifdef CIRROCUMULUS_CLOUDS
float CloudPlanarDensity(in vec2 worldPos) {
    worldPos /= 1.0 + distance(worldPos, uCameraPos.xz) * 2e-5;
    vec2 position = worldPos * 1e-4 - wind.xz;
    float baseCoverage = curve(texture(uNoiseTex, position * 0.08).z * 0.7 + 0.1);
    baseCoverage *= max0(1.07 - texture(uNoiseTex, position * 0.003).y * 1.4);

    vec2 curl = texture(uNoiseTex, position * 0.05).xy * 0.04;
    curl += texture(uNoiseTex, position * 0.1).xy * 0.02;
    position += curl;
    float noise = 0.5 * texture(uNoiseTex, position * vec2(0.4, 0.16)).z;
    noise += texture(uNoiseTex, position * 0.9).z - 0.24;
    noise = saturate(noise);
    noise *= clamp((baseCoverage + CLOUD_PLANE1_COVERY - 0.6) * 0.9, 0.0, 0.14);
    if (noise < 1e-6) return 0.0;
    position.x += noise * 0.2;
    noise += 0.02 * texture(uNoiseTex, position * 3.0).z;
    noise += 0.01 * texture(uNoiseTex, position * 5.0 + curl).z - 0.05;
    return cube(saturate(noise * 4.0));
}

vec4 PlanarSample1(in float dist, in vec2 worldPos, in float LdotV, in float lightNoise, in vec4 phases, in vec3 worldDir) {
    float density = CloudPlanarDensity(worldPos);
    if (density < 1e-5) return vec4(0.0);

    float rayLength = 60.0;
    vec2 rayPos = worldPos;
    vec3 rayStep = vec3(uLightDir.xz, 1.0) * rayLength;
    float opticalDepth = 0.0;
    for (uint i = 0u; i < 3u; ++i, rayPos += rayStep.xy) {
        rayStep *= 2.0;
        float d = CloudPlanarDensity(rayPos + rayStep.xy * lightNoise);
        if (d < 1e-4) continue;
        opticalDepth += d * rayStep.z;
    }
    opticalDepth *= CLOUD_PLANE1_DENSITY;
    float powder = oneMinus(fastExp(-density * 6e2)) * 0.75;
    powder /= 1.0 - powder;

    float sunlightEnergy =  fastExp(-opticalDepth * 1.0) * phases.x;
    sunlightEnergy +=        fastExp(-opticalDepth * 0.4)  * phases.y;
    sunlightEnergy +=        fastExp(-opticalDepth * 0.15) * phases.z;
    sunlightEnergy +=        fastExp(-opticalDepth * 0.05) * phases.w;

    opticalDepth = 0.0;
    rayLength = 1e2;
    rayStep = vec3(worldDir.xz, 1.0) * rayLength;
    for (uint i = 0u; i < 2u; ++i, worldPos += rayStep.xy) {
        rayStep *= 2.0;
        float d = CloudPlanarDensity(worldPos + rayStep.xy * lightNoise);
        if (d < 1e-4) continue;
        opticalDepth += d * rayStep.z;
    }
    opticalDepth *= CLOUD_PLANE1_DENSITY;
    float skylightEnergy = fastExp(-opticalDepth * 0.15) + 0.2 * fastExp(-opticalDepth * 0.03);
    vec3 scatteringSky = skylightEnergy * 0.3 * uSkyIlluminance;

    density = oneMinus(fastExp(-density * 2e-2 * CLOUD_PLANE1_DENSITY * dist));
    bool moonlit = uSunDir.y < -0.045;
    vec3 scattering = sunlightEnergy * 1.2e2 * (moonlit ? vec3(NIGHT_BRIGHTNESS) : uSunIlluminance);
    scattering += scatteringSky;
    scattering *= oneMinus(0.7 * uWetness);
    return vec4(scattering * powder * density, density);
}
#endif

#endif // RT_CLOUDS_PLANAR_GLSL
