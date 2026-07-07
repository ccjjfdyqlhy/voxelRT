// rt_shaders - Water wave simulation — from Derivative Shaders
// License: MIT

#ifndef RT_WATER_WAVE_GLSL
#define RT_WATER_WAVE_GLSL

#include "../core/math.glsl"
#include "../core/noise.glsl"

float GetWaterHeight(vec3 worldPos) {
    vec2 pos = worldPos.xz * 0.01;
    float wave = 0.0;
    wave += sin( pos.x * 0.5 + pos.y * 0.3 + uTime * 0.5) * 0.5;
    wave += sin( pos.x * 0.8 - pos.y * 0.4 + uTime * 0.7) * 0.3;
    wave += sin( pos.x * 1.2 + pos.y * 0.7 + uTime * 1.1) * 0.2;
    return wave * WATER_WAVE_HEIGHT;
}

vec3 GetWaterNormal(vec3 worldPos, float waveHeight) {
    float eps = 0.1;
    float hx = GetWaterHeight(worldPos + vec3(eps, 0.0, 0.0));
    float hz = GetWaterHeight(worldPos + vec3(0.0, 0.0, eps));
    return normalize(vec3(hx - waveHeight, eps, hz - waveHeight));
}

#endif // RT_WATER_WAVE_GLSL
