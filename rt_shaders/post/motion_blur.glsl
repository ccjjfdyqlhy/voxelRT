// rt_shaders - Motion blur — from Derivative Shaders
// License: MIT

#ifndef RT_MOTION_BLUR_GLSL
#define RT_MOTION_BLUR_GLSL

#include "../core/math.glsl"
#include "../core/encode.glsl"

vec3 MotionBlurApply(sampler2D tex, vec2 coord, vec3 velocity) {
    vec3 color = vec3(0.0);
    float total = 0.0;
    for (int i = 0; i < MOTION_BLUR_SAMPLES; ++i) {
        float t = float(i) / float(MOTION_BLUR_SAMPLES) - 0.5;
        vec2 sampleCoord = coord + velocity.xy * t * MOTION_BLUR_STRENGTH;
        float w = exp(-abs(t) * 4.0);
        color += texture(tex, sampleCoord).rgb * w;
        total += w;
    }
    return color / total;
}

#endif // RT_MOTION_BLUR_GLSL
