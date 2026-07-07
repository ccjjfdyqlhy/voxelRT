// rt_shaders - ACES Filmic Tone Mapping (Academy Fit + Full) — from Derivative Shaders
// Adapted include paths only.
// License: ACES (Academy Color Encoding System) — see original header

#ifndef RT_TONEMAP_ACES_GLSL
#define RT_TONEMAP_ACES_GLSL

#include "../core/math.glsl"
#include "../core/encode.glsl"

// ============ ACES Fit (simplified) ============

vec3 RRTAndODTFit(in vec3 rgb) {
    vec3 a = rgb * (rgb + 0.0245786) - 0.000090537;
    vec3 b = rgb * (0.983729 * rgb + 0.4329510) + 0.238081;
    return a / b;
}

vec3 AcademyFit(in vec3 rgb) {
    rgb *= 1.4;
    // ACES to AP1
    const mat3 ap0ToAp1 = mat3(
        1.0498110175, -0.4959030231,  0.0000000000,
        0.0000000000,  1.3733130458,  0.0000000000,
       -0.0000974845,  0.0982400361,  0.9912520182
    );
    vec3 aces = rgb * ap0ToAp1;
    // Apply RRT+ODT
    aces = RRTAndODTFit(aces);
    // Desaturation
    float luminance = GetLuminance(aces);
    aces = mix(vec3(luminance), aces, 0.93);
    return LinearToSRGB(aces);
}

// ============ ACES Full (reference) ============

float rgbToSaturation(vec3 rgb) {
    return (max(maxOf(rgb), 1e-10) - max(minOf(rgb), 1e-10)) / max(maxOf(rgb), 1e-2);
}

float rgbToHue(vec3 rgb) {
    if (rgb.r == rgb.g && rgb.g == rgb.b) return 0.0;
    float hue = (360.0 / TAU) * atan(2.0 * rgb.r - rgb.g - rgb.b, sqrt(3.0) * (rgb.g - rgb.b));
    if (hue < 0.0) hue += 360.0;
    return hue;
}

float rgbToYc(vec3 rgb) {
    const float yc_radius_weight = 1.75;
    float chroma = sqrt(rgb.b * (rgb.b - rgb.g) + rgb.g * (rgb.g - rgb.r) + rgb.r * (rgb.r - rgb.b));
    return (rgb.r + rgb.g + rgb.b + yc_radius_weight * chroma) / 3.0;
}

const mat3 ap0ToXyz = mat3(
    0.9525523959,  0.0000000000,  0.0000936786,
    0.3439664498,  0.7281660966, -0.0721325464,
    0.0000000000,  0.0000000000,  1.0088251844
);
const mat3 xyzToAp0 = mat3(
    1.0498110175,  0.0000000000, -0.0000974845,
   -0.4959030231,  1.3733130458,  0.0982400361,
    0.0000000000,  0.0000000000,  0.9912520182
);

const mat3 ap1ToXyz = mat3(
    0.6624541811,  0.1340042065,  0.1561876870,
    0.2722287168,  0.6740817658,  0.0536895174,
   -0.0055746495,  0.0040607335,  1.0103391003
);
const mat3 xyzToAp1 = mat3(
    1.6410233797, -0.3248032942, -0.2364246952,
   -0.6636628587,  1.6153315917,  0.0167563477,
    0.0117218943, -0.0082844420,  0.9883948585
);

const mat3 ap0ToAp1 = ap0ToXyz * xyzToAp1;
const mat3 D60ToD65_CAT = mat3(
    0.98722400, -0.00611327, 0.01595330,
   -0.00759836,  1.00186000, 0.00533002,
    0.00307257, -0.00509595, 1.08168000
);

const mat3 xyzToRgb = mat3(
    3.2404542, -0.9692660,  0.0556434,
   -1.5371385,  1.8760108, -0.2040259,
   -0.4985314,  0.0415560,  1.0572252
);

const mat3 sRGBtoACES = mat3(
    0.43963298, 0.08977644, 0.01754117,
    0.38298870, 0.81343943, 0.11154655,
    0.17737832, 0.09678413, 0.87091228
);

const float rrtGlowGain = 0.05;
const float rrtGlowMid = 0.08;
const float rrtRedScale = 0.82;
const float rrtRedPivot = 0.03;
const float rrtRedHue = 0.0;
const float rrtRedWidth = 135.0;
const float rrtSatFactor = 0.96;
const float odtSatFactor = 0.93;

float GlowFwd(float yc_in, float glow_gain_in, const float glow_mid) {
    if (yc_in <= 2.0 / 3.0 * glow_mid) return glow_gain_in;
    if (yc_in >= 2.0 * glow_mid) return 0.0;
    return glow_gain_in * (glow_mid / yc_in - 0.5);
}

float SigmoidShaper(float x) {
    float t = max0(1.0 - abs(0.5 * x));
    float y = 1.0 + sign(x) * oneMinus(t * t);
    return 0.5 * y;
}

float CenterHue(float hue, float centerH) {
    float hueCentered = hue - centerH;
    if (hueCentered < -180.0) hueCentered += 360.0;
    else if (hueCentered > 180.0) hueCentered -= 360.0;
    return hueCentered;
}

float CubicBasisShaperFit(float x, const float width) {
    float radius = 0.5 * width;
    return abs(x) < radius ? sqr(curve(1.0 - abs(x) / radius)) : 0.0;
}

vec3 RRTSweeteners(vec3 aces) {
    float saturation = rgbToSaturation(aces);
    float ycIn = rgbToYc(aces);
    float s = SigmoidShaper(saturation * 5.0 - 2.0);
    float addedGlow = 1.0 + GlowFwd(ycIn, rrtGlowGain * s, rrtGlowMid);
    aces *= addedGlow;

    float hue = rgbToHue(aces);
    float centeredHue = CenterHue(hue, rrtRedHue);
    float hueWeight = CubicBasisShaperFit(centeredHue, rrtRedWidth);
    aces.r += hueWeight * saturation * (rrtRedPivot - aces.r) * oneMinus(rrtRedScale);

    aces = clamp16F(aces);
    vec3 rgbPre = clamp16F(aces * ap0ToAp1);
    float luminance = GetLuminance(rgbPre);
    rgbPre = mix(vec3(luminance), rgbPre, rrtSatFactor);
    return rgbPre;
}

vec3 AcademyFull(vec3 rgb) {
    rgb *= 1.4;
    rgb *= sRGBtoACES;
    rgb = RRTSweeteners(rgb);
    rgb = RRTAndODTFit(rgb);
    float luminance = GetLuminance(rgb);
    rgb = mix(vec3(luminance), rgb, odtSatFactor);
    return LinearToSRGB(rgb);
}

#endif // RT_TONEMAP_ACES_GLSL
