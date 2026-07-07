// rt_shaders - Fog utilities — from Derivative Shaders
// Adapted to generic uniforms. MC-specific dimension fog removed.
// License: MIT

#ifndef RT_FOG_GLSL
#define RT_FOG_GLSL

#include "../core/math.glsl"
#include "atmosphere.glsl"

float GetDepthLinear(float depth) {
    return (2.0 * uNear) / (uFar + uNear - depth * (uFar - uNear));
}

void CommonFog(inout vec3 color, in float dist) {
    // Generic exponential fog — replace with your engine's fog logic
    float fogFactor = 1.0 - fastExp(-dist * 0.001 * FOG_TYPE);
    vec3 fogColor = uSkyIlluminance * 0.5;
    color = mix(color, fogColor, fogFactor * 0.3);
}

void TransparentAbsorption(inout vec3 color, in vec4 stainedGlassAlbedo) {
    vec3 stainedGlassColor = normalize(stainedGlassAlbedo.rgb + 1e-6) * pow(dotSelf(stainedGlassAlbedo.rgb), 0.25);
    color *= pow4(mix(vec3(1.0), saturate(stainedGlassColor), pow(stainedGlassAlbedo.a, 0.2)));
}

#endif // RT_FOG_GLSL
