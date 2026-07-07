// rt_shaders - Encoding / color space utilities (from Derivative Shaders Common.inc)
// License: MIT

#ifndef RT_ENCODE_GLSL
#define RT_ENCODE_GLSL

#include "math.glsl"

vec2 EncodeNormal(in vec3 n) {
    n.xy /= abs(n.x) + abs(n.y) + abs(n.z);
    if (n.z <= 0.0) {
        n.xy = (vec2(1.0) - abs(n.yx)) * (step(0.0, n.xy) * 2.0 - 1.0);
    }
    return n.xy * 0.5 + 0.5;
}

vec3 DecodeNormal(in vec2 en) {
    en = en * 2.0 - 1.0;
    vec3 normal = vec3(en, oneMinus(abs(en.x) + abs(en.y)));
    if (normal.z <= 0.0) {
        normal.xy = (vec2(1.0) - abs(en.yx)) * (step(0.0, en) * 2.0 - 1.0);
    }
    return normalize(normal);
}

float PackUnorm2x8(vec2 xy) {
    return dot(floor(255.0 * xy + 0.5), vec2(1.0 / 65535.0, 256.0 / 65535.0));
}
vec2 UnpackUnorm2x8(float pack) {
    vec2 xy; xy.x = modf(pack * 65535.0 / 256.0, xy.y);
    return xy * vec2(256.0 / 255.0, 1.0 / 255.0);
}

vec3 LinearToSRGB(in vec3 color) {
    return mix(color * 12.92, 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, lessThan(vec3(0.0031308), color));
}

vec3 SRGBtoLinear(in vec3 color) {
    return mix(color / 12.92, pow((color + 0.055) / 1.055, vec3(2.4)), lessThan(vec3(0.04045), color));
}

float GetLuminance(in vec3 color) {
    return dot(color, vec3(0.2722, 0.6741, 0.0537));
}

vec3 ColorSaturation(in vec3 color, in const float sat) {
    return mix(vec3(GetLuminance(color)), color, sat);
}

vec3 Blackbody(in float t) {
    vec4 vx = vec4(-0.2661239e9, -0.2343580e6, 0.8776956e3, 0.179910);
    vec4 vy = vec4(-1.1063814,   -1.34811020,  2.18555832, -0.20219683);
    float it = 1.0 / t;
    float it2 = it * it;
    float x = dot(vx, vec4(it * it2, it2, it, 1.0));
    float x2 = x * x;
    float y = dot(vy, vec4(x * x2, x2, x, 1.0));

    mat3 xyzToSrgb = mat3(
         3.2404542,-1.5371385,-0.4985314,
        -0.9692660, 1.8760108, 0.0415560,
         0.0556434,-0.2040259, 1.0572252
    );

    vec3 srgb = vec3(x / y, 1.0, oneMinus(x + y) / y) * xyzToSrgb;
    return max0(srgb);
}

vec4 textureSmoothFilter(in sampler2D tex, in vec2 coord) {
    vec2 res = vec2(textureSize(tex, 0));
    coord = coord * res + 0.5;
    vec2 i, f = modf(coord, i);
    f *= f * f * (f * (f * 6.0 - 15.0) + 10.0);
    coord = i + f;
    coord = (coord - 0.5) / res;
    return texture(tex, coord);
}

vec4 cubic(in float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    vec4 w;
    w.x = -x3 + 3.0 * x2 - 3.0 * x + 1.0;
    w.y = 3.0 * x3 - 6.0 * x2 + 4.0;
    w.z = -3.0 * x3 + 3.0 * x2 + 3.0 * x + 1.0;
    w.w = x3;
    return w * rcp(6.0);
}

vec4 textureBicubic(in sampler2D tex, in vec2 coord) {
    vec2 res = textureSize(tex, 0);
    coord = coord * res - 0.5;
    vec2 fTexel = fract(coord);
    coord -= fTexel;
    vec4 xCubic = cubic(fTexel.x);
    vec4 yCubic = cubic(fTexel.y);
    vec4 c = coord.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xCubic.xz + xCubic.yw, yCubic.xz + yCubic.yw);
    vec4 offset = c + vec4(xCubic.y, xCubic.w, yCubic.y, yCubic.w) / s;
    offset *= 1.0 / res.xxyy;
    vec4 sample0 = texture(tex, offset.xz);
    vec4 sample1 = texture(tex, offset.yz);
    vec4 sample2 = texture(tex, offset.xw);
    vec4 sample3 = texture(tex, offset.yw);
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

#endif // RT_ENCODE_GLSL
