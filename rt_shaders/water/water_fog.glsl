// rt_shaders - Water fog / absorption — from Derivative Shaders
// License: MIT

#ifndef RT_WATER_FOG_GLSL
#define RT_WATER_FOG_GLSL

#include "../core/math.glsl"

vec3 waterAbsorption = vec3(0.4, 0.14, 0.08);

void ApplyWaterFog(inout vec3 color, float dist) {
    vec3 transmittance = fastExp(-waterAbsorption * dist * WATER_FOG_DENSITY);
    color *= transmittance;
}

#endif // RT_WATER_FOG_GLSL
