// rt_shaders - Depth of Field — from Derivative Shaders
// License: MIT

#ifndef RT_DOF_GLSL
#define RT_DOF_GLSL

#include "../core/math.glsl"

// Simplified CoC-based DoF

float GetCoC(float depth, float focusDist, float aperture) {
    float linearDepth = GetDepthLinear(depth);
    float coc = abs(linearDepth - focusDist) / focusDist * aperture;
    return clamp(coc, 0.0, 1.0);
}

vec3 DoFApply(sampler2D tex, vec2 coord, float coc) {
    if (coc < 1e-4) return texture(tex, coord).rgb;
    vec3 color = vec3(0.0);
    float total = 0.0;
    int samples = int(coc * 16.0) + 1;
    for (int i = -samples; i <= samples; ++i) {
        for (int j = -samples; j <= samples; ++j) {
            vec2 offset = vec2(float(i), float(j)) * uScreenPixelSize * coc;
            float w = exp(-float(i * i + j * j) / (2.0 * coc * coc * 64.0));
            color += texture(tex, coord + offset).rgb * w;
            total += w;
        }
    }
    return color / total;
}

#endif // RT_DOF_GLSL
