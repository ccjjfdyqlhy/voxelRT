// rt_shaders - Pure math utilities (from Derivative Shaders Common.inc)
// License: MIT

#ifndef RT_MATH_GLSL
#define RT_MATH_GLSL

const float PI       = radians(180.0);
const float rPI      = 1.0 / PI;
const float TAU      = radians(360.0);
const float rTAU     = 1.0 / TAU;
const float rLOG2    = 1.0 / log(2.0);

#define rcp(x)           (1.0 / (x))
#define oneMinus(x)      (1.0 - (x))
#define fastExp(x)       exp2((x) * rLOG2)
#define max0(x)          max(x, 0.0)
#define saturate(x)      clamp(x, 0.0, 1.0)
#define clamp16F(x)      clamp(x, 0.0, 65535.0)

#define transMAD(m, v)   (mat3(m) * (v) + (m)[3].xyz)
#define diagonal2(m)     vec2((m)[0].x, (m)[1].y)
#define diagonal3(m)     vec3((m)[0].x, (m)[1].y, m[2].z)
#define diagonal4(m)     vec4(diagonal3(m), (m)[2].w)
#define projMAD(m, v)    (diagonal3(m) * (v) + (m)[3].xyz)

float maxOf(vec2 v)    { return max(v.x, v.y); }
float maxOf(vec3 v)    { return max(v.x, max(v.y, v.z)); }
float maxOf(vec4 v)    { return max(v.x, max(v.y, max(v.z, v.w))); }
float minOf(vec2 v)    { return min(v.x, v.y); }
float minOf(vec3 v)    { return min(v.x, min(v.y, v.z)); }
float minOf(vec4 v)    { return min(v.x, min(v.y, min(v.z, v.w))); }

float sqr(float x)     { return x * x; }
vec2  sqr(vec2 x)      { return x * x; }
vec3  sqr(vec3 x)      { return x * x; }
vec4  sqr(vec4 x)      { return x * x; }

float cube(float x)    { return x * x * x; }
vec2  cube(vec2 x)     { return x * x * x; }
vec3  cube(vec3 x)     { return x * x * x; }

float pow4(float x)    { return cube(x) * x; }
vec3  pow4(vec3 x)     { return cube(x) * x; }

float pow5(float x)    { return pow4(x) * x; }
vec3  pow5(vec3 x)     { return pow4(x) * x; }

float pow16(float x)   { return sqr(pow4(x)); }

float sqrt2(float c)   { return sqrt(sqrt(c)); }
vec3  sqrt2(vec3 c)    { return sqrt(sqrt(c)); }

float curve(float x)   { return sqr(x) * (3.0 - 2.0 * x); }
vec2  curve(vec2 x)    { return sqr(x) * (3.0 - 2.0 * x); }
vec3  curve(vec3 x)    { return sqr(x) * (3.0 - 2.0 * x); }

float dotSelf(vec2 x)  { return dot(x, x); }
float dotSelf(vec3 x)  { return dot(x, x); }

vec2  sincos(float x)  { return vec2(sin(x), cos(x)); }
vec2  cossin(float x)  { return vec2(cos(x), sin(x)); }

float remap(float e0, float e1, float x) { return saturate((x - e0) / (e1 - e0)); }

float cubeLength(in vec2 v) {
    vec2 t = abs(cube(v));
    return pow(t.x + t.y, 1.0 / 3.0);
}

float quarticLength(in vec2 v) {
    return sqrt2(pow4(v.x) + pow4(v.y));
}

#endif // RT_MATH_GLSL
