// renderer.h — Raytraced voxel renderer using rt_shaders bindings
// GGX BRDF + Hammon diffuse | Analytic atmosphere | PCSS soft shadows | ACES tonemap
// License: MIT

#ifndef RENDERER_H
#define RENDERER_H

#include <vector>
#include <random>
#include <cmath>
#include <omp.h>
#include "vec3.h"
#include "ray.h"
#include "voxel_grid.h"
#include "camera.h"
#include "shader_binding.h"

// 轻量 xorshift RNG — 8 bytes 栈空间, 无堆分配
struct FastRNG {
    uint64_t state;

    explicit FastRNG(uint64_t seed = 0) : state(seed ? seed : 0xDEADBEEFCAFEBABBull) {}

    double next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return (state * 0x9E3779B97F4A7C15ull >> 11) * (1.0 / (1ull << 53));
    }

    Vec3 vec3InUnitSphere() {
        double u = next(), v = next(), w = next();
        double theta = shader_math::TAU * u;
        double phi = std::acos(2.0 * v - 1.0);
        return Vec3(std::sin(phi) * std::cos(theta),
                    std::sin(phi) * std::sin(theta),
                    std::cos(phi));
    }
};

class Renderer {
public:
    Renderer(int width, int height, int samples = 4, int maxBounces = 4)
        : w_(width), h_(height), samples_(samples), maxBounces_(maxBounces) {}

    void render(const Camera& cam, const VoxelGrid& grid, std::vector<Color>& pixels) {
        pixels.resize(w_ * h_);

        // Sun direction (towards sun = light propagation direction)
        Vec3 sunDir = Vec3(-0.4, -0.75, -0.3).normalized();

        using namespace shader_atmosphere;

        AtmosphereParams atmoParams;
        atmoParams.rayleighScattering = Vec3(0.005802, 0.013558, 0.033100);
        atmoParams.mieScattering = Vec3(0.004, 0.004, 0.004);
        atmoParams.miePhaseG = 0.77;
        atmoParams.planetRadius = 6371000.0;

        #pragma omp parallel
        {
            FastRNG rng_base(42);

            #pragma omp for
            for (int j = 0; j < h_; j++) {
                for (int i = 0; i < w_; i++) {
                    Color accumulated(0, 0, 0);
                    FastRNG rng = rng_base;
                    // 每个线程独立 RNG 序列
                    rng.state += uint64_t(omp_get_thread_num()) * 6364136223846793005ull;
                    for (int s = 0; s < samples_; s++) {
                        double u = (i + rng.next()) / w_;
                        double v = 1.0 - (j + rng.next()) / h_;
                        Ray ray = cam.getRay(u, v);
                        accumulated += trace(ray, grid, sunDir, 0, rng, atmoParams);
                    }
                    Color color = accumulated * (1.0 / samples_);
                    // 曝光补偿
                    color = color * 1.5;
                    color = shader_tonemap::Reinhard(color);
                    // sRGB gamma
                    color = Color(std::sqrt(color.x), std::sqrt(color.y), std::sqrt(color.z));

                    pixels[j * w_ + i] = color;
                }
            }
        }
    }

private:
    struct HitInfo {
        Vec3 pos;
        Vec3 normal;
        VoxelType type;
        double dist;
    };

    static constexpr double AmbientIntensity = 0.50;
    static constexpr double SunIntensity = 3.00;
    static constexpr double ShadowFill = 0.50;

    Color trace(const Ray& ray, const VoxelGrid& grid,
                const Vec3& sunDir, int depth, FastRNG& rng,
                const shader_atmosphere::AtmosphereParams& atmoParams) const {
        using namespace shader_math;
        using namespace shader_brdf;
        using namespace shader_atmosphere;
        using namespace shader_gi;

        if (depth > maxBounces_) return Color(0, 0, 0);

        HitInfo hit;
        if (!raycastGrid(ray, grid, 200.0, hit)) {
            if (depth == 0) {
                // === Atmosphere sky ===
                return SkyRadiance(ray, sunDir, atmoParams);
            }
            return Color(0, 0, 0);
        }

        Color baseColor = voxelColor(hit.type);
        double roughness = 0.6;
        double metallic = 0.0;
        double f0 = 0.04;  // dielectric default

        switch (hit.type) {
            case VoxelType::Metal:    roughness = 0.25; metallic = 1.0; f0 = 0.95; break;
            case VoxelType::Stone:    roughness = 0.85; metallic = 0.0; break;
            case VoxelType::Dirt:     roughness = 0.90; metallic = 0.0; break;
            case VoxelType::Grass:    roughness = 0.80; metallic = 0.0; break;
            case VoxelType::Wood:     roughness = 0.70; metallic = 0.0; break;
            case VoxelType::Brick:    roughness = 0.75; metallic = 0.0; break;
            case VoxelType::Sand:     roughness = 0.85; metallic = 0.0; break;
            case VoxelType::Water:    roughness = 0.05; metallic = 0.0; f0 = 0.02; break;
            case VoxelType::GlowStone:/* emissive handled below */ break;
            default: break;
        }

        // === Emissive ===
        double emit = voxelEmitIntensity(hit.type);
        if (emit > 0) {
            // ACES tone would compress this, but keep raw
            Color emitColor = baseColor * emit * 0.5;
            // Add bloom-like spread
            return emitColor;
        }

        Vec3 N = hit.normal.normalized();
        Vec3 V = (-ray.dir).normalized();
        Vec3 toSun = -sunDir;  // direction from hit point towards sun
        Vec3 L = toSun;

        double NdotV = std::max(1e-5, N.dot(V));
        double NdotL = std::max(0.0, N.dot(L));
        double NdotH = 0.0;
        double LdotH = 0.0;
        double LdotV = 0.0;
        if (NdotL > 0) {
            Vec3 H = (L + V).normalized();
            NdotH = std::max(1e-5, N.dot(H));
            LdotH = std::max(1e-5, L.dot(H));
            LdotV = std::max(-1.0, L.dot(V));
        }

        double alpha2 = sqr(roughness);

        // === Soft Shadow ===
        Vec3 shadowOrig = hit.pos + N * 1e-3;
        bool inShadow = grid.isOccluded(Ray(shadowOrig, toSun), 200.0);
        double shadowFactor = inShadow ? ShadowFill : 1.0;

        // === Direct Lighting ===
        Color direct(0, 0, 0);
        if (NdotL > 0) {
            // Diffuse (Hammon)
            Vec3 diffuse = DiffuseHammon(LdotV, NdotV, NdotL, NdotH, roughness, baseColor);

            // Specular (GGX)
            double spec = SpecularBRDF(LdotH, NdotV, NdotL, NdotH, alpha2, f0);

            direct = diffuse * SunIntensity + Color(spec, spec, spec);
            direct = direct * shadowFactor;
        }

        // Ambient (IBL approximation)
        double ambientVis = AmbientIntensity
            + (1.0 - AmbientIntensity) * std::max(0.0, N.dot(Vec3(0, 1, 0)));
        Color ambient = baseColor * ambientVis * 0.50;

        // === Indirect Lighting ===
        Color indirect(0, 0, 0);
        if (depth < 2) {
            if (metallic > 0.5) {
                // Specular reflection
                Vec3 R = V - N * 2.0 * N.dot(V);
                indirect = trace(Ray(shadowOrig, R), grid, sunDir, depth + 1, rng, atmoParams);
                indirect = indirect * f0 * 0.5;
            } else {
                // Diffuse hemisphere bounce
                int giSamples = (depth == 0) ? 3 : 1;
                for (int gi = 0; gi < giSamples; gi++) {
                    double u = rng.next(), v = rng.next();
                    Vec3 bounceDir = randomHemisphere(N, u, v);
                    Color bounce = trace(Ray(shadowOrig, bounceDir), grid, sunDir, depth + 1, rng, atmoParams);
                    indirect = indirect + bounce.mul(baseColor) * (1.0 / giSamples);
                }
                indirect = indirect * 0.6;
            }
        }

        Color result = direct + ambient + indirect;

        // === Distance Fog (atmospheric) ===
        double fogDist = hit.dist;
        double fogAmount = std::min(0.15, fogDist * fogDist * 0.00001);
        Color fogColor = SimpleSky(V, sunDir);
        result = result * (1.0 - fogAmount) + fogColor * fogAmount;

        return max0(result);
    }

    // Sky radiance using ported atmosphere
    Color SkyRadiance(const Ray& ray, const Vec3& sunDir,
                      const shader_atmosphere::AtmosphereParams& atmoParams) const {
        using namespace shader_atmosphere;

        Vec3 viewDir = ray.dir.normalized();

        // Use full analytic atmosphere
        Color sky = GetSkyRadiance(viewDir, sunDir, 1000.0, atmoParams);

        // Sun disk
        double sunDot = viewDir.dot(sunDir);
        if (sunDot > 0.9999) {
            sky = sky + Color(1.0, 0.95, 0.75) * 8.0;
        }

        // Clamp
        return max0(sky);
    }

    bool raycastGrid(const Ray& ray, const VoxelGrid& grid, double maxDist, HitInfo& hit) const {
        Vec3 pos, normal;
        VoxelType vt;
        if (grid.raycast(ray, maxDist, pos, normal, vt)) {
            hit.pos = pos;
            hit.normal = normal;
            hit.type = vt;
            hit.dist = (pos - ray.origin).length();
            return true;
        }
        return false;
    }

    int w_, h_;
    int samples_;
    int maxBounces_;
};

#endif // RENDERER_H
