#ifndef CUDA_VOXEL_GRID_CUH
#define CUDA_VOXEL_GRID_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include "cuda_vec3.cuh"

enum class CudaVoxelType : uint8_t {
    Air = 0, Stone, Dirt, Grass, Wood, Leaf, Sand, Water, Brick, Metal, GlowStone
};

__host__ __device__ inline bool cuIsActive(CudaVoxelType t) { return t != CudaVoxelType::Air; }
__host__ __device__ inline bool cuIsEmissive(CudaVoxelType t) { return t == CudaVoxelType::GlowStone; }

__host__ __device__ inline CudaColor cuVoxelColor(CudaVoxelType t) {
    switch (t) {
        case CudaVoxelType::Stone:    return {0.50, 0.50, 0.50};
        case CudaVoxelType::Dirt:     return {0.60, 0.40, 0.20};
        case CudaVoxelType::Grass:    return {0.25, 0.75, 0.15};
        case CudaVoxelType::Wood:     return {0.55, 0.30, 0.10};
        case CudaVoxelType::Leaf:     return {0.15, 0.55, 0.10};
        case CudaVoxelType::Sand:     return {0.85, 0.75, 0.40};
        case CudaVoxelType::Water:    return {0.25, 0.45, 0.80};
        case CudaVoxelType::Brick:    return {0.70, 0.25, 0.20};
        case CudaVoxelType::Metal:    return {0.65, 0.65, 0.75};
        case CudaVoxelType::GlowStone:return {1.00, 0.80, 0.30};
        default:                      return {1.00, 0.00, 1.00};
    }
}

__host__ __device__ inline double cuVoxelEmit(CudaVoxelType t) {
    return t == CudaVoxelType::GlowStone ? 6.0 : 0.0;
}

// Device-side voxel grid — flat uint8_t array
struct CudaVoxelGrid {
    int sx, sy, sz;
    uint8_t* data;

    __device__ size_t idx(int x, int y, int z) const {
        return size_t(x) * sy * sz + size_t(y) * sz + size_t(z);
    }

    __device__ bool inBounds(int x, int y, int z) const {
        return x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz;
    }

    __device__ CudaVoxelType get(int x, int y, int z) const {
        if (!inBounds(x, y, z)) return CudaVoxelType::Air;
        return static_cast<CudaVoxelType>(data[idx(x, y, z)]);
    }

    __device__ bool isSolid(int x, int y, int z) const {
        return get(x, y, z) != CudaVoxelType::Air;
    }

    // DDA raycast — returns hit pos, normal, type
    __device__ bool raycast(double ox, double oy, double oz,
                            double dx, double dy, double dz,
                            double maxDist,
                            double& hitX, double& hitY, double& hitZ,
                            double& normX, double& normY, double& normZ,
                            CudaVoxelType& hitType,
                            int maxSteps = 200) const
    {
        int ix = int(floor(ox));
        int iy = int(floor(oy));
        int iz = int(floor(oz));

        double tMaxX, tMaxY, tMaxZ;
        double tDeltaX, tDeltaY, tDeltaZ;
        int stepX, stepY, stepZ;
        int outX, outY, outZ;

        if (dx > 0) {
            stepX = 1; outX = sx;
            tDeltaX = 1.0 / dx;
            tMaxX = (ix + 1.0 - ox) / dx;
        } else if (dx < 0) {
            stepX = -1; outX = -1;
            tDeltaX = -1.0 / dx;
            tMaxX = (ox - ix) / (-dx);
        } else {
            stepX = 0; outX = sx;
            tDeltaX = 1e30; tMaxX = 1e30;
        }

        if (dy > 0) {
            stepY = 1; outY = sy;
            tDeltaY = 1.0 / dy;
            tMaxY = (iy + 1.0 - oy) / dy;
        } else if (dy < 0) {
            stepY = -1; outY = -1;
            tDeltaY = -1.0 / dy;
            tMaxY = (oy - iy) / (-dy);
        } else {
            stepY = 0; outY = sy;
            tDeltaY = 1e30; tMaxY = 1e30;
        }

        if (dz > 0) {
            stepZ = 1; outZ = sz;
            tDeltaZ = 1.0 / dz;
            tMaxZ = (iz + 1.0 - oz) / dz;
        } else if (dz < 0) {
            stepZ = -1; outZ = -1;
            tDeltaZ = -1.0 / dz;
            tMaxZ = (oz - iz) / (-dz);
        } else {
            stepZ = 0; outZ = sz;
            tDeltaZ = 1e30; tMaxZ = 1e30;
        }

        int nx = 0, ny = 0, nz = 0;
        double t = 0.0;

        for (int steps = 0; steps < maxSteps; steps++) {
            if (ix >= 0 && ix < sx && iy >= 0 && iy < sy && iz >= 0 && iz < sz) {
                uint8_t vt = data[idx(ix, iy, iz)];
                if (vt != 0) {
                    hitX = double(ix) + 0.5;
                    hitY = double(iy) + 0.5;
                    hitZ = double(iz) + 0.5;
                    normX = double(nx);
                    normY = double(ny);
                    normZ = double(nz);
                    hitType = static_cast<CudaVoxelType>(vt);
                    return true;
                }
            }

            if (tMaxX < tMaxY && tMaxX < tMaxZ) {
                t = tMaxX; ix += stepX;
                nx = -stepX; ny = 0; nz = 0;
                tMaxX += tDeltaX;
                if (stepX > 0 ? ix >= outX : ix <= outX) break;
            } else if (tMaxY < tMaxZ) {
                t = tMaxY; iy += stepY;
                nx = 0; ny = -stepY; nz = 0;
                tMaxY += tDeltaY;
                if (stepY > 0 ? iy >= outY : iy <= outY) break;
            } else {
                t = tMaxZ; iz += stepZ;
                nx = 0; ny = 0; nz = -stepZ;
                tMaxZ += tDeltaZ;
                if (stepZ > 0 ? iz >= outZ : iz <= outZ) break;
            }
            if (t > maxDist) break;
        }
        return false;
    }

    // Lightweight occlusion query
    __device__ bool isOccluded(double ox, double oy, double oz,
                               double dx, double dy, double dz,
                               double maxDist, int maxSteps = 200) const
    {
        int ix = int(floor(ox)), iy = int(floor(oy)), iz = int(floor(oz));
        double tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY, tDeltaZ;
        int stepX, stepY, stepZ, outX, outY, outZ;

        if (dx > 0) { stepX=1; outX=sx; tDeltaX=1.0/dx; tMaxX=(ix+1.0-ox)/dx; }
        else if (dx < 0) { stepX=-1; outX=-1; tDeltaX=-1.0/dx; tMaxX=(ox-ix)/(-dx); }
        else { stepX=0; outX=sx; tDeltaX=1e30; tMaxX=1e30; }

        if (dy > 0) { stepY=1; outY=sy; tDeltaY=1.0/dy; tMaxY=(iy+1.0-oy)/dy; }
        else if (dy < 0) { stepY=-1; outY=-1; tDeltaY=-1.0/dy; tMaxY=(oy-iy)/(-dy); }
        else { stepY=0; outY=sy; tDeltaY=1e30; tMaxY=1e30; }

        if (dz > 0) { stepZ=1; outZ=sz; tDeltaZ=1.0/dz; tMaxZ=(iz+1.0-oz)/dz; }
        else if (dz < 0) { stepZ=-1; outZ=-1; tDeltaZ=-1.0/dz; tMaxZ=(oz-iz)/(-dz); }
        else { stepZ=0; outZ=sz; tDeltaZ=1e30; tMaxZ=1e30; }

        for (int s = 0; s < maxSteps; s++) {
            if (ix >= 0 && ix < sx && iy >= 0 && iy < sy && iz >= 0 && iz < sz) {
                if (data[idx(ix, iy, iz)] != 0) return true;
            }
            if (tMaxX < tMaxY && tMaxX < tMaxZ) {
                double t = tMaxX; ix += stepX;
                tMaxX += tDeltaX;
                if (stepX > 0 ? ix >= outX : ix <= outX) break;
                if (t > maxDist) break;
            } else if (tMaxY < tMaxZ) {
                double t = tMaxY; iy += stepY;
                tMaxY += tDeltaY;
                if (stepY > 0 ? iy >= outY : iy <= outY) break;
                if (t > maxDist) break;
            } else {
                double t = tMaxZ; iz += stepZ;
                tMaxZ += tDeltaZ;
                if (stepZ > 0 ? iz >= outZ : iz <= outZ) break;
                if (t > maxDist) break;
            }
        }
        return false;
    }
};

// Host-side helpers
inline void cuInitGrid(CudaVoxelGrid& dgrid, int sx, int sy, int sz, const uint8_t* h_data) {
    dgrid.sx = sx; dgrid.sy = sy; dgrid.sz = sz;
    size_t bytes = size_t(sx) * sy * sz;
    cudaMalloc(&dgrid.data, bytes);
    cudaMemcpy(dgrid.data, h_data, bytes, cudaMemcpyHostToDevice);
}

inline void cuFreeGrid(CudaVoxelGrid& dgrid) {
    if (dgrid.data) cudaFree(dgrid.data);
    dgrid.data = nullptr;
}

#endif // CUDA_VOXEL_GRID_CUH
