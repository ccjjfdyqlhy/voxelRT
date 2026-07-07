// rt_shaders - AgX tone mapping — from Derivative Shaders
// Simplified AgX look
// License: MIT

#ifndef RT_TONEMAP_AGX_GLSL
#define RT_TONEMAP_AGX_GLSL

#include "../core/math.glsl"
#include "../core/encode.glsl"

vec3 AgXMinimal(vec3 color) {
    // AgX minimal transform
    color = max(color, 0.0);
    const mat3 agxMat = mat3(
        0.842479, 0.078328, 0.008922,
        0.042516, 0.878968, 0.065161,
        0.005125, 0.042754, 0.925917
    );
    color = color * agxMat;
    color = (color * (color + 0.0245786) - 0.000090537) / (color * (0.983729 * color + 0.432951) + 0.238081);
    color = pow(color, vec3(2.2));
    return color;
}

#endif // RT_TONEMAP_AGX_GLSL
