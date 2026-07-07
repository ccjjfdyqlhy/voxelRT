// rt_shaders - Gaussian / box blur — from Derivative Shaders
// License: MIT

#ifndef RT_BLUR_GLSL
#define RT_BLUR_GLSL

#include "../core/math.glsl"

vec3 BlurHorizontal(sampler2D tex, vec2 coord, float radius) {
    vec3 color = vec3(0.0);
    float total = 0.0;
    for (int i = -BLUR_SAMPLES; i <= BLUR_SAMPLES; ++i) {
        float w = exp(-float(i * i) / (2.0 * radius * radius));
        color += texture(tex, coord + vec2(float(i) * uScreenPixelSize.x, 0.0)).rgb * w;
        total += w;
    }
    return color / total;
}

vec3 BlurVertical(sampler2D tex, vec2 coord, float radius) {
    vec3 color = vec3(0.0);
    float total = 0.0;
    for (int i = -BLUR_SAMPLES; i <= BLUR_SAMPLES; ++i) {
        float w = exp(-float(i * i) / (2.0 * radius * radius));
        color += texture(tex, coord + vec2(0.0, float(i) * uScreenPixelSize.y)).rgb * w;
        total += w;
    }
    return color / total;
}

vec3 BlurGaussian(sampler2D tex, vec2 coord, float radius) {
    vec3 horz = BlurHorizontal(tex, coord, radius);
    return BlurVertical(tex, coord, radius);
}

#endif // RT_BLUR_GLSL
