// rt_shaders - Rain surface effects — from Derivative Shaders
// License: MIT

#ifndef RT_RAIN_GLSL
#define RT_RAIN_GLSL

#include "../core/math.glsl"
#include "../core/noise.glsl"

float GetWetnessFactor() {
    return uWetness;
}

#endif // RT_RAIN_GLSL
