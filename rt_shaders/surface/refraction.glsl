// rt_shaders - Screen-space refraction — from Derivative Shaders
// License: MIT

#ifndef RT_REFRACTION_GLSL
#define RT_REFRACTION_GLSL

#include "../core/math.glsl"

vec3 RefractRay(vec3 viewDir, vec3 normal, float ior) {
    float cosI = -dot(viewDir, normal);
    float sinI2 = 1.0 - cosI * cosI;
    float sinR2 = sinI2 / (ior * ior);
    if (sinR2 > 1.0) return reflect(viewDir, normal); // total internal reflection
    float cosR = sqrt(1.0 - sinR2);
    return normalize(viewDir * (1.0 / ior) + normal * (cosI / ior - cosR));
}

vec2 RefractCoord(vec2 screenPos, vec3 refractDir, float dist) {
    vec3 refractPos = refractDir * dist;
    vec4 clipPos = uProjMatrix * vec4(refractPos, 1.0);
    return (clipPos.xy / clipPos.w) * 0.5 + 0.5;
}

#endif // RT_REFRACTION_GLSL
