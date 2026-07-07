// rt_shaders - Screen-space reflections — from Derivative Shaders
// License: MIT

#ifndef RT_REFLECTION_GLSL
#define RT_REFLECTION_GLSL

#include "../core/math.glsl"

vec3 ReflectRay(vec3 viewDir, vec3 normal) {
    return reflect(viewDir, normal);
}

vec2 ReflectCoord(vec2 screenPos, vec3 reflectDir, float rayLength) {
    vec3 reflectPos = reflectDir * rayLength;
    vec4 clipPos = uProjMatrix * vec4(reflectPos, 1.0);
    return (clipPos.xy / clipPos.w) * 0.5 + 0.5;
}

#endif // RT_REFLECTION_GLSL
