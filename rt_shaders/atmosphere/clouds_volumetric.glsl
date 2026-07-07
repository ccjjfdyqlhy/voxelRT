// rt_shaders - Volumetric clouds (cumulus) — from Derivative Shaders
// Adapted to generic uniforms.
// License: MIT

#ifndef RT_CLOUDS_VOLUMETRIC_GLSL
#define RT_CLOUDS_VOLUMETRIC_GLSL

#include "../core/math.glsl"
#include "../core/noise.glsl"
#include "atmosphere.glsl"

// --- Config ---
#define CLOUD_CUMULUS_CLEAR_ALTITUDE     1000
#define CLOUD_CUMULUS_CLEAR_THICKNESS    1400
#define CLOUD_CUMULUS_CLEAR_COVERY       1.0
#define CLOUD_CUMULUS_CLEAR_DENSITY      1.0
#define CLOUD_CUMULUS_CLEAR_SUNLIGHTING  1.0
#define CLOUD_CUMULUS_CLEAR_SKYLIGHTING  1.0

#define CLOUD_CUMULUS_RAIN_ALTITUDE      800
#define CLOUD_CUMULUS_RAIN_THICKNESS     3000
#define CLOUD_CUMULUS_RAIN_COVERY        1.2
#define CLOUD_CUMULUS_RAIN_DENSITY       1.0
#define CLOUD_CUMULUS_RAIN_SUNLIGHTING   0.3
#define CLOUD_CUMULUS_RAIN_SKYLIGHTING   0.3

#define CLOUD_CUMULUS_SAMPLES            32
#define CLOUD_CUMULUS_SUNLIGHT_SAMPLES   4
#define CLOUD_CUMULUS_SKYLIGHT_SAMPLES   2
#define CLOUD_LOCAL_COVERAGE

// --- Cloud properties ---
struct CloudProperties {
    float altitude;
    float thickness;
    float coverage;
    float density;
    float sunlighting, skylighting;
    float maxAltitude;
    float noiseScale;
    float cloudPeakWeight;
};

CloudProperties GetGlobalCloudProperties() {
    CloudProperties cp;
    cp.altitude    = mix(CLOUD_CUMULUS_CLEAR_ALTITUDE,     CLOUD_CUMULUS_RAIN_ALTITUDE,     uWetness);
    cp.density     = mix(CLOUD_CUMULUS_CLEAR_DENSITY,      CLOUD_CUMULUS_RAIN_DENSITY,      uWetness);
    cp.sunlighting = mix(CLOUD_CUMULUS_CLEAR_SUNLIGHTING,  CLOUD_CUMULUS_RAIN_SUNLIGHTING,  uWetness);
    cp.skylighting = mix(CLOUD_CUMULUS_CLEAR_SKYLIGHTING,  CLOUD_CUMULUS_RAIN_SKYLIGHTING,  uWetness);
    cp.thickness   = mix(CLOUD_CUMULUS_CLEAR_THICKNESS,    CLOUD_CUMULUS_RAIN_THICKNESS,    uWetness);
    cp.coverage    = mix(CLOUD_CUMULUS_CLEAR_COVERY,       CLOUD_CUMULUS_RAIN_COVERY,       uWetness);
    cp.maxAltitude = cp.altitude + cp.thickness;
    cp.noiseScale  = 4e-4 + 6e-5 * uWetness;
    cp.cloudPeakWeight = 0.1 + 0.7 * uWetness;
    return cp;
}

vec3 wind = vec3(2e-3, 2e-4, 1e-3) * uTime * CLOUDS_SPEED;
float cloudForwardG = 0.6 - uWetness * 0.2;
float cloudBackwardG = -0.4 + uWetness * 0.2;
const float cloudBackwardWeight = 0.25, octWeight = 0.5, octScale = 3.0;

float CloudVolumeDensity(in CloudProperties cloudProperties, in vec3 worldPos, in uint steps, in float noiseDetail) {
    vec3 position = worldPos * cloudProperties.noiseScale - wind;
    float density = noiseDetail * 0.03, weight = 0.5;
    for (uint i = 0u; i < steps; ++i, weight *= octWeight) {
        density += weight * Get3DNoiseSmooth(position);
        position = position * octScale - wind;
    }
    density += octWeight / octScale / steps;
    if (density < 1e-6) return 0.0;

    float normalizedHeight  = saturate((worldPos.y - cloudProperties.altitude) * rcp(cloudProperties.thickness));
    float heightAttenuation = saturate(normalizedHeight * 6.6) * saturate(oneMinus(normalizedHeight) * (2.0 + uWetness));
    density = cloudProperties.coverage == 1.0 ? density : saturate((density - 1.0 + cloudProperties.coverage) * rcp(cloudProperties.coverage));
    density *= heightAttenuation * 1.9;
    density -= heightAttenuation * 0.9 + normalizedHeight * 0.5 + 0.1;
    return saturate(density * 3.0 * cloudProperties.density);
}

float GetNoiseDetail(in vec3 worldDir) {
    worldDir *= 48.0;
    float pnoise = Get3DNoise(worldDir - wind);          worldDir += pnoise * 1e-3 - wind;
    pnoise += Get3DNoise(worldDir * 2.0);                worldDir += pnoise * 1e-3 - wind;
    pnoise += Get3DNoise(worldDir * 4.0) * 0.5;          worldDir += pnoise * 1e-3 - wind;
    pnoise += Get3DNoise(worldDir * 8.0) * 0.25;         worldDir += pnoise * 1e-3 - wind;
    pnoise += Get3DNoise(worldDir * 16.0) * 0.125;
    return pnoise - 0.15;
}

float CloudVolumeSunLightOD(in CloudProperties cloudProperties, in vec3 rayPos, in float lightNoise) {
    float rayLength = cloudProperties.thickness * (0.2 / float(CLOUD_CUMULUS_SUNLIGHT_SAMPLES));
    vec4 rayStep = vec4(uLightDir, 1.0) * rayLength;
    float opticalDepth = 0.0;
    for (uint i = 0u; i < CLOUD_CUMULUS_SUNLIGHT_SAMPLES; ++i, rayPos += rayStep.xyz) {
        rayStep *= 2.0;
        float density = CloudVolumeDensity(cloudProperties, rayPos + rayStep.xyz * lightNoise, 5u, 1.0);
        if (density < 1e-4) continue;
        opticalDepth += density;
    }
    return opticalDepth * rayLength * 0.12;
}

float CloudVolumeSkyLightOD(in CloudProperties cloudProperties, in vec3 rayPos, in float lightNoise) {
    float rayLength = cloudProperties.thickness * (0.2 / float(CLOUD_CUMULUS_SKYLIGHT_SAMPLES));
    vec4 rayStep = vec4(vec3(0.0, 1.0, 0.0), 1.0) * rayLength;
    float opticalDepth = 0.0;
    for (uint i = 0u; i < CLOUD_CUMULUS_SKYLIGHT_SAMPLES; ++i, rayPos += rayStep.xyz) {
        rayStep *= 2.0;
        float density = CloudVolumeDensity(cloudProperties, rayPos + rayStep.xyz * lightNoise, 3u, 1.0);
        if (density < 1e-4) continue;
        opticalDepth += density;
    }
    return opticalDepth * rayLength * 0.04;
}

#endif // RT_CLOUDS_VOLUMETRIC_GLSL
