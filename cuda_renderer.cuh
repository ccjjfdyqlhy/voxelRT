#ifndef CUDA_RENDERER_CUH
#define CUDA_RENDERER_CUH

#include <cuda_runtime.h>
#include "cuda_vec3.cuh"
#include "cuda_voxel_grid.cuh"

// ====================================================================
// DEVICE MATH (use fmax/fmin to avoid CUDA built-in conflicts)
// ====================================================================
namespace cu_math {
    __device__ constexpr double PI  = 3.14159265358979323846;
    __device__ constexpr double rPI = 0.31830988618379067154;
    __device__ constexpr double TAU = 6.28318530717958647692;

    __device__ inline double rcp(double x) { return 1.0 / x; }
    __device__ inline double sqr(double x) { return x * x; }
    __device__ inline double pow5(double x) { double x2 = x*x; return x2*x2*x; }
    __device__ inline double clamp(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
    __device__ inline double saturate(double x) { return clamp(x, 0.0, 1.0); }
    __device__ inline double max0(double x) { return x > 0 ? x : 0; }

    __device__ inline CudaVec3 max0(const CudaVec3& v) {
        return {fmax(v.x, 0.0), fmax(v.y, 0.0), fmax(v.z, 0.0)};
    }
    __device__ inline CudaVec3 saturate(const CudaVec3& v) {
        return {clamp(v.x,0.0,1.0), clamp(v.y,0.0,1.0), clamp(v.z,0.0,1.0)};
    }
    __device__ inline CudaVec3 oneMinus(const CudaVec3& v) { return {1.0-v.x, 1.0-v.y, 1.0-v.z}; }
    __device__ inline CudaVec3 fastExp(const CudaVec3& v) {
        return {exp(v.x), exp(v.y), exp(v.z)};
    }
}

// ====================================================================
// DEVICE BRDF
// ====================================================================
namespace cu_brdf {
    using namespace cu_math;

    __device__ inline double FresnelSchlick(double cosTheta, double f0) {
        return saturate(pow5(1.0 - cosTheta) + (1.0 - pow5(1.0 - cosTheta)) * f0);
    }

    __device__ inline double DistributionGGX(double NdotH, double alpha2) {
        return alpha2 * rPI / sqr(1.0 + (NdotH * alpha2 - NdotH) * NdotH);
    }

    __device__ inline double V2SmithGGX(double NdotV, double NdotL, double alpha2) {
        if (NdotL <= 0.0) return 0.0;
        double ggxl = NdotL * sqrt(alpha2 + (NdotV - NdotV * alpha2) * NdotV);
        double ggxv = NdotV * sqrt(alpha2 + (NdotL - NdotL * alpha2) * NdotL);
        return 0.5 / (ggxl + ggxv);
    }

    __device__ inline CudaVec3 DiffuseHammon(double LdotV, double NdotV, double NdotL,
                                             double NdotH, double roughness, const CudaVec3& albedo) {
        if (NdotL < 1e-6) return {0,0,0};
        double facing = max0(LdotV) * 0.5 + 0.5;
        double singleSmooth = 1.05 * (1.0 - pow5(1.0 - fmax(NdotL, 1e-2)))
                                   * (1.0 - pow5(1.0 - fmax(NdotV, 1e-2)));
        double singleRough = facing * (0.45 - 0.2 * facing) * (rcp(NdotH) + 2.0);
        double single = (singleSmooth * (1.0 - roughness) + singleRough * roughness) * rPI;
        double multi = 0.1159 * roughness;
        return (CudaVec3(multi, multi, multi).mul(albedo) + CudaVec3(single, single, single)) * NdotL;
    }

    __device__ inline double SpecularBRDF(double LdotH, double NdotV, double NdotL,
                                          double NdotH, double alpha2, double f0) {
        if (NdotL < 1e-5) return 0.0;
        double F = FresnelSchlick(LdotH, f0);
        double D = DistributionGGX(NdotH, alpha2);
        double V = V2SmithGGX(fmax(NdotV, 1e-2), fmax(NdotL, 1e-2), alpha2);
        return fmin(NdotL * D * V * F, 4.0);
    }
}

// ====================================================================
// DEVICE ATMOSPHERE (simplified single scattering)
// ====================================================================
namespace cu_atmo {
    using namespace cu_math;

    struct AtmoParams {
        CudaVec3 rayleighScattering;
        CudaVec3 mieScattering;
        double miePhaseG;
        double planetRadius;
    };

    __host__ __device__ inline AtmoParams DefaultAtmo() {
        AtmoParams p;
        p.rayleighScattering = CudaVec3(0.005802, 0.013558, 0.033100);
        p.mieScattering = CudaVec3(0.004, 0.004, 0.004);
        p.miePhaseG = 0.77;
        p.planetRadius = 6371000.0;
        return p;
    }

    __device__ inline double RayleighPhase(double cosTheta) {
        return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
    }

    __device__ inline double HenyeyGreensteinPhase(double cosTheta, double g) {
        double gg = g * g;
        double phase = 1.0 + gg - 2.0 * g * cosTheta;
        return (1.0 - gg) / (4.0 * PI * phase * sqrt(phase));
    }

    __device__ inline CudaVec3 GetSkyRadiance(double vx, double vy, double vz,
                                              double sx, double sy, double sz,
                                              const AtmoParams& params) {
        double cosTheta = vx*sx + vy*sy + vz*sz;
        double zenithAngle = acos(clamp(vy, -1.0, 1.0));
        double sunZenith = acos(clamp(sy, -1.0, 1.0));

        double odR = exp(-1000.0 / 8000.0) / fmax(cos(zenithAngle), 0.01);
        double odM = exp(-1000.0 / 1200.0) / fmax(cos(zenithAngle), 0.01);

        CudaVec3 trans = fastExp(-(params.rayleighScattering * odR
                                   + params.mieScattering * odM));

        double sunOdR = exp(-1000.0 / 8000.0) / fmax(cos(sunZenith), 0.01);
        double sunOdM = exp(-1000.0 / 1200.0) / fmax(cos(sunZenith), 0.01);
        CudaVec3 sunTrans = fastExp(-(params.rayleighScattering * sunOdR
                                      + params.mieScattering * sunOdM));

        CudaVec3 sunColor(1.0, 0.95, 0.8);
        CudaVec3 rayleigh = sunColor.mul(params.rayleighScattering) * RayleighPhase(cosTheta);
        CudaVec3 mie = sunColor.mul(params.mieScattering) * HenyeyGreensteinPhase(cosTheta, params.miePhaseG);

        CudaVec3 inScatter = (rayleigh + mie).mul(sunTrans).mul(CudaVec3(1,1,1) - trans);

        double horizonFactor = exp(-fabs(zenithAngle - PI/2) * 40.0);
        CudaVec3 horizonGlow(0.9, 0.6, 0.2);
        horizonGlow = horizonGlow * horizonFactor * 0.3;

        CudaVec3 result = inScatter + horizonGlow;
        return max0(result);
    }

    __device__ inline CudaVec3 SimpleSky(double vx, double vy, double vz,
                                         double sx, double sy, double sz) {
        double t = 0.5 * (vy + 1.0);
        CudaVec3 top(0.2, 0.4, 0.9);
        CudaVec3 horizon(0.8, 0.8, 0.9);
        CudaVec3 sky = horizon * (1.0 - t) + top * t;

        double sunDot = fmax(0.0, vx*sx + vy*sy + vz*sz);
        double sunGlow = pow(fmax(0.0, sunDot), 32.0) * 0.5;
        sky = sky + CudaVec3(1.0, 0.7, 0.3) * sunGlow;

        if (sunDot > 0.9998) {
            double sunDisk = (sunDot - 0.9998) / (1.0 - 0.9998);
            sky = sky + CudaVec3(1, 0.9, 0.6) * sunDisk * 2.0;
        }
        return max0(sky);
    }
}

// ====================================================================
// DEVICE GI (hemisphere sampling)
// ====================================================================
namespace cu_gi {
    using namespace cu_math;

    __device__ inline CudaVec3 randomHemisphere(const CudaVec3& normal, double u, double v) {
        double theta = TAU * u;
        double r = sqrt(v);
        double x = r * cos(theta);
        double y = r * sin(theta);
        double z = sqrt(1.0 - v);

        CudaVec3 w = normal;
        CudaVec3 a = (fabs(w.x) > 0.9) ? CudaVec3(0,1,0) : CudaVec3(1,0,0);
        CudaVec3 u_dir = a.cross(w).normalized();
        CudaVec3 v_dir = w.cross(u_dir);

        return (u_dir * x + v_dir * y + w * z).normalized();
    }
}

// ====================================================================
// XorShift RNG (per-thread)
// ====================================================================
struct CudaRNG {
    uint64_t state;

    __device__ explicit CudaRNG(uint64_t seed = 0) : state(seed ? seed : 0xDEADBEEFCAFEBABBull) {}

    __device__ double next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return (state * 0x9E3779B97F4A7C15ull >> 11) * (1.0 / (1ull << 53));
    }
};

// Forward decl of loadAtmo (defined after cuTrace, but needed in it)
__device__ inline cu_atmo::AtmoParams loadAtmo();

// ====================================================================
// DEVICE TRACE (recursive, depth ≤ 3 → safe stack)
// ====================================================================
__device__ CudaColor cuTrace(double ox, double oy, double oz,
                              double dx, double dy, double dz,
                              const CudaVoxelGrid& grid,
                              double sx, double sy, double sz,  // sun dir (towards sun)
                              int depth, int maxDepth,
                              CudaRNG& rng)
{
    using namespace cu_math;
    using namespace cu_brdf;
    using namespace cu_atmo;

    const AtmoParams atmoParams = loadAtmo();

    if (depth > maxDepth) return CudaColor(0, 0, 0);

    // DDA raycast
    double hx, hy, hz, nx, ny, nz;
    CudaVoxelType hitType;
    bool hit = grid.raycast(ox, oy, oz, dx, dy, dz, 200.0,
                            hx, hy, hz, nx, ny, nz, hitType);

    if (!hit) {
        if (depth == 0) {
            return GetSkyRadiance(dx, dy, dz, sx, sy, sz, atmoParams);
        }
        return CudaColor(0, 0, 0);
    }

    CudaColor baseColor = cuVoxelColor(hitType);
    double roughness = 0.6;
    double metallic = 0.0;
    double f0 = 0.04;

    switch (hitType) {
        case CudaVoxelType::Metal:    roughness = 0.25; metallic = 1.0; f0 = 0.95; break;
        case CudaVoxelType::Stone:    roughness = 0.85; metallic = 0.0; break;
        case CudaVoxelType::Dirt:     roughness = 0.90; metallic = 0.0; break;
        case CudaVoxelType::Grass:    roughness = 0.80; metallic = 0.0; break;
        case CudaVoxelType::Wood:     roughness = 0.70; metallic = 0.0; break;
        case CudaVoxelType::Brick:    roughness = 0.75; metallic = 0.0; break;
        case CudaVoxelType::Sand:     roughness = 0.85; metallic = 0.0; break;
        case CudaVoxelType::Water:    roughness = 0.05; metallic = 0.0; f0 = 0.02; break;
        default: break;
    }

    // Emissive
    double emit = cuVoxelEmit(hitType);
    if (emit > 0) {
        return baseColor * emit * 0.5;
    }

    // === Glass / Dielectric ===
    if (hitType == CudaVoxelType::Crystal) {
        if (depth >= maxDepth) return CudaColor(0, 0, 0);
        double ior = 1.5;
        CudaVec3 N(nx, ny, nz);
        N = N.normalized();
        CudaVec3 V(-dx, -dy, -dz);
        V = V.normalized();

        bool entering = N.dot(V) < 0;
        CudaVec3 facingN = entering ? N : -N;
        double cosTheta = fmax(0.0, -facingN.dot(V));
        double eta = entering ? (1.0 / ior) : ior;
        double R0 = (1.0 - ior) / (1.0 + ior);
        R0 = R0 * R0;
        double fresnel = R0 + (1.0 - R0) * pow5(1.0 - cosTheta);

        CudaVec3 R = V - facingN * 2.0 * facingN.dot(V);
        double eps = 1.0;

        double k = 1.0 - eta * eta * (1.0 - cosTheta * cosTheta);
        if (k < 0) {
            double rx = hx + R.x * eps, ry = hy + R.y * eps, rz = hz + R.z * eps;
            return cuTrace(rx, ry, rz, R.x, R.y, R.z,
                           grid, sx, sy, sz, depth + 1, maxDepth, rng);
        }

        CudaVec3 T = V * eta + facingN * (eta * cosTheta - sqrt(k));
        T = T.normalized();

        if (rng.next() < fresnel) {
            double rx = hx + R.x * eps, ry = hy + R.y * eps, rz = hz + R.z * eps;
            return cuTrace(rx, ry, rz, R.x, R.y, R.z,
                           grid, sx, sy, sz, depth + 1, maxDepth, rng) / fresnel;
        } else {
            double tx = hx + T.x * eps, ty = hy + T.y * eps, tz = hz + T.z * eps;
            return cuTrace(tx, ty, tz, T.x, T.y, T.z,
                           grid, sx, sy, sz, depth + 1, maxDepth, rng) / (1.0 - fresnel);
        }
    }

    CudaVec3 N(nx, ny, nz);
    N = N.normalized();
    CudaVec3 V(-dx, -dy, -dz);
    V = V.normalized();
    CudaVec3 toSun(-sx, -sy, -sz);
    CudaVec3 L = toSun;

    double NdotV = fmax(1e-5, N.dot(V));
    double NdotL = fmax(0.0, N.dot(L));
    double NdotH = 0.0, LdotH = 0.0;

    if (NdotL > 0) {
        CudaVec3 H = (L + V).normalized();
        NdotH = fmax(1e-5, N.dot(H));
        LdotH = fmax(1e-5, L.dot(H));
    }

    double alpha2 = sqr(roughness);

    // Shadow
    double shadOx = hx + N.x * 1e-3;
    double shadOy = hy + N.y * 1e-3;
    double shadOz = hz + N.z * 1e-3;
    bool inShadow = grid.isOccluded(shadOx, shadOy, shadOz, toSun.x, toSun.y, toSun.z, 200.0);

    double shadowFactor = inShadow ? 0.25 : 1.0;

    // Direct
    CudaColor direct(0, 0, 0);
    if (NdotL > 0) {
        CudaVec3 diffuse = DiffuseHammon(L.dot(V), NdotV, NdotL, NdotH, roughness, baseColor);
        double spec = SpecularBRDF(LdotH, NdotV, NdotL, NdotH, alpha2, f0);
        direct = (diffuse * 3.0 + CudaColor(spec, spec, spec)) * shadowFactor;
    }

    // Ambient
    double ambientVis = 0.50 + 0.50 * fmax(0.0, N.dot(CudaVec3(0,1,0)));
    CudaColor ambient = baseColor * ambientVis * 0.50;

    // Indirect
    CudaColor indirect(0, 0, 0);
    if (depth < 2) {
        if (metallic > 0.5) {
            CudaVec3 R = V - N * 2.0 * N.dot(V);
            indirect = cuTrace(shadOx, shadOy, shadOz,
                               R.x, R.y, R.z,
                               grid, sx, sy, sz, depth+1, maxDepth,
                               rng);
            indirect = indirect * f0 * 0.5;
        } else {
            int giSamples = (depth == 0) ? 3 : 1;
            for (int gi = 0; gi < giSamples; gi++) {
                double u = rng.next(), v = rng.next();
                CudaVec3 bounceDir = cu_gi::randomHemisphere(N, u, v);
                CudaColor bounce = cuTrace(shadOx, shadOy, shadOz,
                                           bounceDir.x, bounceDir.y, bounceDir.z,
                                           grid, sx, sy, sz, depth+1, maxDepth,
                                           rng);
                indirect = indirect + bounce.mul(baseColor) * (1.0 / giSamples);
            }
            indirect = indirect * 0.6;
        }
    }

    CudaColor result = direct + ambient + indirect;

    // Fog
    double dist = sqrt((hx-ox)*(hx-ox) + (hy-oy)*(hy-oy) + (hz-oz)*(hz-oz));
    double fogAmount = fmin(0.15, dist * dist * 0.00001);
    CudaColor fogColor = SimpleSky(V.x, V.y, V.z, sx, sy, sz);
    result = result * (1.0 - fogAmount) + fogColor * fogAmount;

    return max0(result);
}

// ====================================================================
// CUDA GLOBAL KERNEL
// ====================================================================
struct CudaCamera {
    double posX, posY, posZ;
    double llX, llY, llZ;   // lower-left
    double hX, hY, hZ;      // horizontal
    double vX, vY, vZ;      // vertical
};

__constant__ CudaCamera c_cam;
__constant__ CudaVoxelGrid c_grid;
__constant__ double c_sunDir[3];
__constant__ double c_atmoData[7]; // [rayR,rayG,rayB, mieR,mieG,mieB, miePhaseG]

__device__ inline cu_atmo::AtmoParams loadAtmo() {
    cu_atmo::AtmoParams p;
    p.rayleighScattering = CudaVec3(c_atmoData[0], c_atmoData[1], c_atmoData[2]);
    p.mieScattering = CudaVec3(c_atmoData[3], c_atmoData[4], c_atmoData[5]);
    p.miePhaseG = c_atmoData[6];
    p.planetRadius = 6371000.0;
    return p;
}

__global__ void renderKernel(CudaColor* output, int w, int h, 
                             int spp, int maxBounces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= w || idy >= h) return;

    uint64_t seed = uint64_t(idy * w + idx) * 6364136223846793005ull + 42;
    CudaRNG rng(seed);

    CudaColor accum(0, 0, 0);
    for (int s = 0; s < spp; s++) {
        double u = (idx + rng.next()) / w;
        double v = 1.0 - (idy + rng.next()) / h;

        double rx = c_cam.llX + c_cam.hX * u + c_cam.vX * v - c_cam.posX;
        double ry = c_cam.llY + c_cam.hY * u + c_cam.vY * v - c_cam.posY;
        double rz = c_cam.llZ + c_cam.hZ * u + c_cam.vZ * v - c_cam.posZ;
        double len = sqrt(rx*rx + ry*ry + rz*rz);
        rx /= len; ry /= len; rz /= len;

        CudaColor c = cuTrace(c_cam.posX, c_cam.posY, c_cam.posZ,
                              rx, ry, rz,
                              c_grid,
                              c_sunDir[0], c_sunDir[1], c_sunDir[2],
                              0, maxBounces, rng);
        accum = accum + c;
    }

    accum = accum * (1.0 / spp);
    accum = accum * 1.5;
    accum = accum.mul((CudaVec3(1,1,1) + accum).rcp()); // Reinhard
    output[idy * w + idx] = CudaColor(sqrt(accum.x), sqrt(accum.y), sqrt(accum.z));
}

// ====================================================================
// HOST RESOURCES & LAUNCHER (pre-allocated, no per-frame alloc/free)
// ====================================================================

struct CudaResources {
    CudaVoxelGrid d_grid;
    CudaColor*    d_output;
    int           maxW, maxH;

    CudaResources() : d_output(nullptr), maxW(0), maxH(0) {
        d_grid.data = nullptr;
        d_grid.sx = d_grid.sy = d_grid.sz = 0;
    }
};

// One-time init: upload grid, allocate output buffer, set persistent constants
inline CudaResources cudaInitRenderer(const uint8_t* h_gridData,
                                       int gx, int gy, int gz,
                                       int maxW, int maxH)
{
    CudaResources res;

    // Grid → GPU (persistent)
    cuInitGrid(res.d_grid, gx, gy, gz, h_gridData);

    // Output buffer (reused every frame)
    cudaMalloc(&res.d_output, maxW * maxH * sizeof(CudaColor));
    res.maxW = maxW;
    res.maxH = maxH;

    // === Persistent constant memory (set once) ===
    cudaMemcpyToSymbol(c_grid, &res.d_grid, sizeof(CudaVoxelGrid));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA ERR (grid copy): %s\n", cudaGetErrorString(err));

    double h_sun[3] = {-0.4, -0.75, -0.3};
    cudaMemcpyToSymbol(c_sunDir, h_sun, sizeof(double) * 3);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA ERR (sun copy): %s\n", cudaGetErrorString(err));

    double h_atmo[7] = {0.005802, 0.013558, 0.033100, 0.004, 0.004, 0.004, 0.77};
    cudaMemcpyToSymbol(c_atmoData, h_atmo, sizeof(double) * 7);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA ERR (atmo copy): %s\n", cudaGetErrorString(err));

    return res;
}

// Per-frame render: update camera constant, launch kernel, copy back
// No cudaMalloc/cudaFree — uses pre-allocated buffers
inline void cudaRenderFrame(CudaResources& res,
                            double cx, double cy, double cz,
                            double yaw, double pitch, double fov, double aspect,
                            int w, int h, int spp, int maxBounces,
                            CudaColor* h_output)
{
    // Build camera matrix on host, copy to __constant__
    double cyaw = cos(yaw * M_PI / 180.0);
    double syaw = sin(yaw * M_PI / 180.0);
    double cpitch = cos(pitch * M_PI / 180.0);
    double spitch = sin(pitch * M_PI / 180.0);
    CudaVec3 fwd(syaw * cpitch, spitch, cyaw * cpitch);
    fwd = fwd.normalized();
    CudaVec3 worldUp(0, 1, 0);
    CudaVec3 right = worldUp.cross(fwd).normalized();
    CudaVec3 up = fwd.cross(right).normalized();

    double theta = fov * M_PI / 180.0;
    double hh = tan(theta / 2.0);
    double vp_h = 2.0 * hh;
    double vp_w = aspect * vp_h;

    CudaVec3 horizontal = right * vp_w;
    CudaVec3 vertical = up * vp_h;
    CudaVec3 lowerLeft = CudaVec3(cx, cy, cz) + fwd - horizontal * 0.5 - vertical * 0.5;

    CudaCamera h_cam;
    h_cam.posX = cx; h_cam.posY = cy; h_cam.posZ = cz;
    h_cam.llX = lowerLeft.x; h_cam.llY = lowerLeft.y; h_cam.llZ = lowerLeft.z;
    h_cam.hX = horizontal.x; h_cam.hY = horizontal.y; h_cam.hZ = horizontal.z;
    h_cam.vX = vertical.x; h_cam.vY = vertical.y; h_cam.vZ = vertical.z;

    cudaMemcpyToSymbol(c_cam, &h_cam, sizeof(CudaCamera));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA ERR (c_cam copy): %s\n", cudaGetErrorString(err));

    // Launch kernel
    dim3 block(16, 8);
    dim3 kgrid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    renderKernel<<<kgrid, block>>>(res.d_output, w, h, spp, maxBounces);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA ERR (kernel launch): %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA ERR (sync): %s\n", cudaGetErrorString(err));

    // Copy result to host
    cudaMemcpy(h_output, res.d_output, w * h * sizeof(CudaColor), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA ERR (memcpy D2H): %s\n", cudaGetErrorString(err));
}

// Cleanup
inline void cudaDestroyRenderer(CudaResources& res) {
    if (res.d_grid.data) cudaFree(res.d_grid.data);
    if (res.d_output)    cudaFree(res.d_output);
    res.d_grid.data = nullptr;
    res.d_output = nullptr;
}

// Legacy single-shot render (kept for reference, use init+frame for perf)
inline void cudaRender(const uint8_t* h_gridData, int gx, int gy, int gz,
                       double cx, double cy, double cz,
                       double yaw, double pitch, double fov, double aspect,
                       double sunX, double sunY, double sunZ,
                       int w, int h, int spp, int maxBounces,
                       CudaColor* h_output)
{
    CudaResources res = cudaInitRenderer(h_gridData, gx, gy, gz, w, h);
    cudaRenderFrame(res, cx, cy, cz, yaw, pitch, fov, aspect, w, h, spp, maxBounces, h_output);
    cudaDestroyRenderer(res);
}

#endif // CUDA_RENDERER_CUH
