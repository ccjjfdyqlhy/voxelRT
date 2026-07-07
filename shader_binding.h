// shader_binding.h — C++ port of rt_shaders GLSL library
// Direct translation: core/math → encode → brdf → atmosphere → shadow → tonemap
// License: MIT

#ifndef SHADER_BINDING_H
#define SHADER_BINDING_H

#include <cmath>
#include <algorithm>
#include "vec3.h"
#include "ray.h"

// ====================================================================
// 1. CORE MATH — port of rt_shaders/core/math.glsl
// ====================================================================
namespace shader_math {

constexpr double PI       = 3.14159265358979323846;
constexpr double rPI      = 1.0 / PI;
constexpr double TAU      = 2.0 * PI;
constexpr double rTAU     = 1.0 / TAU;
constexpr double rLOG2    = 1.0 / std::log(2.0);

inline double rcp(double x)          { return 1.0 / x; }
inline double oneMinus(double x)     { return 1.0 - x; }
inline double fastExp(double x)      { return std::exp2(x * rLOG2); }
inline double max0(double x)         { return std::max(x, 0.0); }
inline double saturate(double x)     { return std::clamp(x, 0.0, 1.0); }
inline double clamp16F(double x)     { return std::clamp(x, 0.0, 65535.0); }

inline double sqr(double x)          { return x * x; }
inline double cube(double x)         { return x * x * x; }
inline double pow4(double x)         { return cube(x) * x; }
inline double pow5(double x)         { return pow4(x) * x; }
inline double pow16(double x)        { return sqr(pow4(x)); }
inline double curve(double x)        { return sqr(x) * (3.0 - 2.0 * x); }
inline double sqrt2(double x)        { return std::sqrt(std::sqrt(x)); }
inline double dotSelf(const Vec3& v) { return v.dot(v); }

inline double maxOf(const Vec3& v)   { return std::max(v.x, std::max(v.y, v.z)); }
inline double minOf(const Vec3& v)   { return std::min(v.x, std::min(v.y, v.z)); }

inline Vec3 sqr(const Vec3& v)       { return Vec3(v.x*v.x, v.y*v.y, v.z*v.z); }
inline Vec3 cube(const Vec3& v)      { return Vec3(v.x*v.x*v.x, v.y*v.y*v.y, v.z*v.z*v.z); }
inline Vec3 pow4(const Vec3& v)      { return cube(v).mul(v); }
inline Vec3 pow5(const Vec3& v)      { return pow4(v).mul(v); }
inline Vec3 max0(const Vec3& v)      { return Vec3(std::max(v.x,0.0), std::max(v.y,0.0), std::max(v.z,0.0)); }
inline Vec3 saturate(const Vec3& v)  { return Vec3(std::clamp(v.x,0.0,1.0), std::clamp(v.y,0.0,1.0), std::clamp(v.z,0.0,1.0)); }
inline Vec3 clamp16F(const Vec3& v)  { return Vec3(std::clamp(v.x,0.0,65535.0), std::clamp(v.y,0.0,65535.0), std::clamp(v.z,0.0,65535.0)); }
inline Vec3 oneMinus(const Vec3& v)  { return Vec3(1.0-v.x, 1.0-v.y, 1.0-v.z); }
inline Vec3 fastExp(const Vec3& v)   { return Vec3(fastExp(v.x), fastExp(v.y), fastExp(v.z)); }

} // namespace shader_math

// ====================================================================
// 2. ENCODE / COLOR — port of rt_shaders/core/encode.glsl
// ====================================================================
namespace shader_encode {
using namespace shader_math;

// LinearToSRGB
inline Vec3 LinearToSRGB(const Vec3& c) {
    Vec3 result;
    result.x = c.x < 0.0031308 ? c.x * 12.92 : 1.055 * std::pow(c.x, 1.0/2.4) - 0.055;
    result.y = c.y < 0.0031308 ? c.y * 12.92 : 1.055 * std::pow(c.y, 1.0/2.4) - 0.055;
    result.z = c.z < 0.0031308 ? c.z * 12.92 : 1.055 * std::pow(c.z, 1.0/2.4) - 0.055;
    return result;
}

inline Vec3 SRGBtoLinear(const Vec3& c) {
    Vec3 result;
    result.x = c.x < 0.04045 ? c.x / 12.92 : std::pow((c.x + 0.055) / 1.055, 2.4);
    result.y = c.y < 0.04045 ? c.y / 12.92 : std::pow((c.y + 0.055) / 1.055, 2.4);
    result.z = c.z < 0.04045 ? c.z / 12.92 : std::pow((c.z + 0.055) / 1.055, 2.4);
    return result;
}

// Luminance (AP1 weights)
inline double GetLuminance(const Vec3& c) {
    return c.dot(Vec3(0.2722, 0.6741, 0.0537));
}

inline Vec3 ColorSaturation(const Vec3& c, double sat) {
    double l = GetLuminance(c);
    return Vec3(l, l, l) * (1.0 - sat) + c * sat;
}

// Blackbody color (Kelvin → RGB)
inline Vec3 Blackbody(double t) {
    double it = 1.0 / t;
    double it2 = it * it;
    double x = -0.2661239e9 * it * it2
               -0.2343580e6 * it2
               +0.8776956e3 * it
               +0.179910;
    double x2 = x * x;
    double y = -1.1063814 * x * x2
               -1.34811020 * x2
               +2.18555832 * x
               -0.20219683;

    // XYZ → sRGB matrix
    Vec3 srgb;
    srgb.x =  3.2404542 * (x / y) + -1.5371385 * 1.0 + -0.4985314 * ((1.0 - x - y) / y);
    srgb.y = -0.9692660 * (x / y) +  1.8760108 * 1.0 +  0.0415560 * ((1.0 - x - y) / y);
    srgb.z =  0.0556434 * (x / y) + -0.2040259 * 1.0 +  1.0572252 * ((1.0 - x - y) / y);
    return max0(srgb);
}

} // namespace shader_encode

// ====================================================================
// 3. BRDF — port of rt_shaders/lighting/brdf.glsl
// ====================================================================
namespace shader_brdf {
using namespace shader_math;

inline double FresnelSchlick(double cosTheta, double f0) {
    double f = pow5(1.0 - cosTheta);
    return saturate(f + oneMinus(f) * f0);
}

// Smith GGX visibility (inverse form, simplified)
inline double V1SmithGGXInv(double cosTheta, double alpha2) {
    return cosTheta + std::sqrt((cosTheta - alpha2 * cosTheta) * cosTheta + alpha2);
}

// Smith GGX visibility (joint form)
inline double V2SmithGGX(double NdotV, double NdotL, double alpha2) {
    if (NdotL <= 0.0) return 0.0;
    double ggxl = NdotL * std::sqrt(alpha2 + (NdotV - NdotV * alpha2) * NdotV);
    double ggxv = NdotV * std::sqrt(alpha2 + (NdotL - NdotL * alpha2) * NdotL);
    return 0.5 / (ggxl + ggxv);
}

inline double DistributionGGX(double NdotH, double alpha2) {
    return alpha2 * rPI / sqr(1.0 + (NdotH * alpha2 - NdotH) * NdotH);
}

// Hammon diffuse BRDF
inline Vec3 DiffuseHammon(double LdotV, double NdotV, double NdotL,
                          double NdotH, double roughness, const Vec3& albedo) {
    if (NdotL < 1e-6) return Vec3(0,0,0);
    double facing = max0(LdotV) * 0.5 + 0.5;
    double singleSmooth = 1.05 * oneMinus(pow5(1.0 - std::max(NdotL, 1e-2)))
                                * oneMinus(pow5(1.0 - std::max(NdotV, 1e-2)));
    double singleRough = facing * (0.45 - 0.2 * facing) * (rcp(NdotH) + 2.0);
    double single = (singleSmooth * (1.0 - roughness) + singleRough * roughness) * rPI;
    double multi = 0.1159 * roughness;
    return (Vec3(multi, multi, multi).mul(albedo) + Vec3(single, single, single)) * NdotL;
}

// Specular BRDF (returns value, multiply by F0 for final)
inline double SpecularBRDF(double LdotH, double NdotV, double NdotL,
                           double NdotH, double alpha2, double f0) {
    if (NdotL < 1e-5) return 0.0;
    double F = FresnelSchlick(LdotH, f0);
    double D = DistributionGGX(NdotH, alpha2);
    double V = V2SmithGGX(std::max(NdotV, 1e-2), std::max(NdotL, 1e-2), alpha2);
    return std::min(NdotL * D * V * F, 4.0);
}

} // namespace shader_brdf

// ====================================================================
// 4. ATMOSPHERE — port of rt_shaders/atmosphere/atmosphere.glsl (simplified)
// Single-scattering analytic atmosphere, no LUT dependency.
// ====================================================================
namespace shader_atmosphere {
using namespace shader_math;
using namespace shader_encode;

// Phase functions
inline double RayleighPhase(double cosTheta) {
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
}

inline double HenyeyGreensteinPhase(double cosTheta, double g) {
    double gg = g * g;
    double phase = 1.0 + gg - 2.0 * g * cosTheta;
    return (1.0 - gg) / (4.0 * PI * phase * std::sqrt(phase));
}

inline double CornetteShanksPhase(double cosTheta, double g) {
    double gg = g * g;
    double a = (1.0 - gg) / (2.0 + gg) * 3.0 * rPI;
    double b = (1.0 + sqr(cosTheta)) * std::pow(1.0 + gg - 2.0 * g * cosTheta, -1.5);
    return a * b * 0.125;
}

// Precomputed atmosphere parameters (earth-like)
struct AtmosphereParams {
    Vec3 rayleighScattering{0.005802, 0.013558, 0.033100};  // per meter
    Vec3 mieScattering{0.003996, 0.003996, 0.003996};
    Vec3 mieAbsorption{0.001, 0.001, 0.001};
    Vec3 groundAlbedo{0.1, 0.1, 0.1};
    double sunAngularRadius = 0.012;
    double miePhaseG = 0.77;
    double planetRadius = 6371000.0;
    double atmosphereHeight = 100000.0;
    double bottomAltitude = 1000.0;
};

// Get sky color for a view direction given sun direction
// Based on Nishita single scattering + Mie
inline Vec3 GetSkyRadiance(const Vec3& viewDir, const Vec3& sunDir,
                           double eyeAltitude, const AtmosphereParams& params = AtmosphereParams()) {
    using namespace shader_math;

    double cosTheta = viewDir.dot(sunDir);
    double zenithAngle = std::acos(std::clamp(viewDir.y, -1.0, 1.0));
    double sunZenith = std::acos(std::clamp(sunDir.y, -1.0, 1.0));

    // Simplified transmittance: exp(-opticalDepth)
    double opticalDepthRayleigh = std::exp(-eyeAltitude / 8000.0) / std::max(std::cos(zenithAngle), 0.01);
    double opticalDepthMie = std::exp(-eyeAltitude / 1200.0) / std::max(std::cos(zenithAngle), 0.01);

    Vec3 transmittance = fastExp(-(params.rayleighScattering * opticalDepthRayleigh
                                  + params.mieScattering * opticalDepthMie));

    // Sun transmittance
    double sunOpticalRayleigh = std::exp(-eyeAltitude / 8000.0) / std::max(std::cos(sunZenith), 0.01);
    double sunOpticalMie = std::exp(-eyeAltitude / 1200.0) / std::max(std::cos(sunZenith), 0.01);
    Vec3 sunTrans = fastExp(-(params.rayleighScattering * sunOpticalRayleigh
                             + params.mieScattering * sunOpticalMie));

    // Single scattering
    Vec3 sunColor = Vec3(1.0, 0.95, 0.8);  // warm sun
    Vec3 rayleigh = sunColor.mul(params.rayleighScattering) * RayleighPhase(cosTheta);
    Vec3 mie = sunColor.mul(params.mieScattering) * HenyeyGreensteinPhase(cosTheta, params.miePhaseG);

    // Apply transmittance and scale
    Vec3 inScatter = (rayleigh + mie).mul(sunTrans).mul(Vec3(1,1,1) - transmittance);

    // Horizon glow
    double horizonFactor = std::exp(-std::abs(zenithAngle - PI/2) * 40.0);
    Vec3 horizonGlow = Vec3(0.9, 0.6, 0.2) * horizonFactor * 0.3;

    // Ground reflection (very rough)
    double groundVis = viewDir.y > 0 ? 0.0 : 1.0;
    Vec3 groundReflect = params.groundAlbedo * groundVis * 0.02;

    Vec3 result = inScatter + horizonGlow + groundReflect;
    return max0(result);
}

// Simple sky gradient fallback (fast)
inline Vec3 SimpleSky(const Vec3& viewDir, const Vec3& sunDir) {
    double t = 0.5 * (viewDir.y + 1.0);
    Vec3 top(0.2, 0.4, 0.9);
    Vec3 horizon(0.8, 0.8, 0.9);
    Vec3 sky = horizon * (1.0 - t) + top * t;

    // Sun disk
    double sunDot = std::max(0.0, viewDir.dot(sunDir));
    if (sunDot > 0.9998) {
        double sunDisk = (sunDot - 0.9998) / (1.0 - 0.9998);
        sky = sky + Vec3(1, 0.9, 0.6) * sunDisk * 2.0;
    }

    // Sun glow
    double sunGlow = std::pow(std::max(0.0, sunDot), 32.0) * 0.5;
    sky = sky + Vec3(1, 0.7, 0.3) * sunGlow;

    return max0(sky);
}

} // namespace shader_atmosphere

// ====================================================================
// 5. SOFT SHADOW — PCSS-style for voxel ray traversal
// ====================================================================
namespace shader_shadow {
using namespace shader_math;

// Multiple-sample shadow with penumbra estimation
// Returns shadow factor: 1.0 = fully lit, 0.0 = fully occluded
inline double SoftShadowQuery(
    const Vec3& pos, const Vec3& normal, const Vec3& toSun,
    double maxDist, int numSamples, double lightRadius,
    const class VoxelGrid* grid)
{
    // Penumbra estimation via blocker search
    double blockerDist = 0.0;
    int blockerCount = 0;
    for (int i = 0; i < std::min(numSamples, 8); i++) {
        // Jitter sample direction within cone
        double theta = (i + 0.5) * TAU / 8.0;
        double r = lightRadius * std::sqrt(double(i + 1) / 8.0);
        Vec3 jitter(std::cos(theta) * r, 0.0, std::sin(theta) * r);
        Vec3 sampleDir = (toSun + jitter).normalized();

        Ray shadowRay(pos + normal * 1e-3, sampleDir);
        Vec3 hitPos, hitNorm;
        VoxelType hitType;
        if (grid->raycast(shadowRay, maxDist, hitPos, hitNorm, hitType)) {
            double d = (hitPos - pos).length();
            blockerDist += d;
            blockerCount++;
        }
    }

    // No blockers → fully lit
    if (blockerCount == 0) return 1.0;
    blockerDist /= blockerCount;

    // Penumbra width proportional to blocker distance & light size
    double penumbra = (blockerDist / maxDist) * lightRadius * 10.0;
    int pcfSamples = std::min(numSamples, 16);

    // PCF with penumbra-sized filter
    int hits = 0;
    for (int i = 0; i < pcfSamples; i++) {
        double theta = (i + 0.5) * TAU / pcfSamples;
        double r = penumbra * std::sqrt(double(i + 1) / pcfSamples);
        Vec3 jitter(std::cos(theta) * r, 0.0, std::sin(theta) * r);
        Vec3 sampleDir = (toSun + jitter).normalized();

        Ray shadowRay(pos + normal * 1e-3, sampleDir);
        Vec3 hitPos, hitNorm;
        VoxelType hitType;
        if (!grid->raycast(shadowRay, maxDist, hitPos, hitNorm, hitType)) {
            hits++;
        }
    }

    return double(hits) / pcfSamples;
}

} // namespace shader_shadow

// ====================================================================
// 6. TONE MAPPING — port of rt_shaders/post/tonemap_aces.glsl
// ====================================================================
namespace shader_tonemap {
using namespace shader_math;
using namespace shader_encode;

// ACES RRT + ODT fit (Narkowicz 2015)
inline Vec3 RRTAndODTFit(const Vec3& rgb) {
    Vec3 a = rgb.mul(rgb + Vec3(0.0245786, 0.0245786, 0.0245786))
               - Vec3(0.000090537, 0.000090537, 0.000090537);
    Vec3 b = rgb.mul(Vec3(0.983729, 0.983729, 0.983729).mul(rgb)
                  + Vec3(0.4329510, 0.4329510, 0.4329510))
               + Vec3(0.238081, 0.238081, 0.238081);
    return a.mul(b.rcp());  // component-wise division
}

// ACES Filmic Tone Mapping (Academy Fit)
// Takes linear RGB (not sRGB!), outputs sRGB
inline Vec3 AcademyFit(const Vec3& rgb) {
    Vec3 color = rgb * 1.4;

    // AP0 → AP1 matrix
    double m00 = 1.0498110175, m01 = -0.4959030231, m02 = 0.0000000000;
    double m10 = 0.0000000000, m11 = 1.3733130458, m12 = 0.0000000000;
    double m20 = -0.0000974845, m21 = 0.0982400361, m22 = 0.9912520182;

    Vec3 aces;
    aces.x = color.x * m00 + color.y * m01 + color.z * m02;
    aces.y = color.x * m10 + color.y * m11 + color.z * m12;
    aces.z = color.x * m20 + color.y * m21 + color.z * m22;

    aces = RRTAndODTFit(aces);

    double luminance = GetLuminance(aces);
    aces = Vec3(luminance, luminance, luminance) * (1.0 - 0.93) + aces * 0.93;

    return LinearToSRGB(aces);
}

// Reinhard tonemap (fallback)
inline Vec3 Reinhard(const Vec3& rgb) {
    return rgb.mul((Vec3(1.0, 1.0, 1.0) + rgb).rcp()); // x / (1 + x)
}

} // namespace shader_tonemap

// ====================================================================
// 7. FAST RNG — port of rt_shaders/core/noise.glsl
// ====================================================================
namespace shader_noise {

inline uint64_t triple32(uint64_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbu;
    x ^= x >> 11;
    x *= 0xac4c1b51u;
    x ^= x >> 15;
    x *= 0x31848babu;
    x ^= x >> 14;
    return x;
}

inline double hashToFloat(uint64_t state) {
    return double(state) / double(0xFFFFFFFFFFFFFFFFu);
}

} // namespace shader_noise

// ====================================================================
// 8. GI (simple hemisphere sampling, port of GI concept)
// ====================================================================
namespace shader_gi {
using namespace shader_math;

inline Vec3 randomHemisphere(const Vec3& normal, double u, double v) {
    double theta = 2.0 * PI * u;
    double r = std::sqrt(v);
    double x = r * std::cos(theta);
    double y = r * std::sin(theta);
    double z = std::sqrt(1.0 - v);

    Vec3 w = normal;
    Vec3 a = (std::fabs(w.x) > 0.9) ? Vec3(0,1,0) : Vec3(1,0,0);
    Vec3 u_dir = a.cross(w).normalized();
    Vec3 v_dir = w.cross(u_dir);

    return (u_dir * x + v_dir * y + w * z).normalized();
}

} // namespace shader_gi

#endif // SHADER_BINDING_H
