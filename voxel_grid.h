#ifndef VOXEL_GRID_H
#define VOXEL_GRID_H

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <cassert>
#include "vec3.h"
#include "ray.h"

// === 体素类型 ===
enum class VoxelType : uint8_t {
    Air = 0, Stone, Dirt, Grass, Wood, Leaf, Sand, Water, Brick, Metal, GlowStone
};

inline bool isActive(VoxelType t) { return t != VoxelType::Air; }
inline bool isEmissive(VoxelType t) { return t == VoxelType::GlowStone; }

// 体素颜色表
inline Color voxelColor(VoxelType t) {
    switch (t) {
        case VoxelType::Stone:    return {0.50, 0.50, 0.50};
        case VoxelType::Dirt:     return {0.60, 0.40, 0.20};
        case VoxelType::Grass:    return {0.25, 0.75, 0.15};
        case VoxelType::Wood:     return {0.55, 0.30, 0.10};
        case VoxelType::Leaf:     return {0.15, 0.55, 0.10};
        case VoxelType::Sand:     return {0.85, 0.75, 0.40};
        case VoxelType::Water:    return {0.25, 0.45, 0.80};
        case VoxelType::Brick:    return {0.70, 0.25, 0.20};
        case VoxelType::Metal:    return {0.65, 0.65, 0.75};
        case VoxelType::GlowStone:return {1.00, 0.80, 0.30};
        default:                  return {1.00, 0.00, 1.00};
    }
}

inline double voxelEmitIntensity(VoxelType t) {
    return t == VoxelType::GlowStone ? 6.0 : 0.0;
}

// 密集体素网格 — 3D 数组 (42x24x42 ≈ 42K, 缓存友好)
class VoxelGrid {
public:
    VoxelGrid(int sizeX, int sizeY, int sizeZ)
        : sx(sizeX), sy(sizeY), sz(sizeZ),
          data_(size_t(sizeX) * sizeY * sizeZ, VoxelType::Air) {}

    int dimX() const { return sx; }
    int dimY() const { return sy; }
    int dimZ() const { return sz; }

    bool inBounds(int x, int y, int z) const {
        return x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz;
    }

    size_t idx(int x, int y, int z) const {
        return size_t(x) * sy * sz + size_t(y) * sz + size_t(z);
    }

    void set(int x, int y, int z, VoxelType t) {
        if (!inBounds(x, y, z)) return;
        data_[idx(x, y, z)] = t;
    }

    VoxelType get(int x, int y, int z) const {
        if (!inBounds(x, y, z)) return VoxelType::Air;
        return data_[idx(x, y, z)];
    }

    bool isSolid(int x, int y, int z) const {
        return get(x, y, z) != VoxelType::Air;
    }

    // 立方体填充
    void fillBox(int x0, int y0, int z0, int x1, int y1, int z1, VoxelType t) {
        for (int x = x0; x <= x1; x++)
            for (int y = y0; y <= y1; y++)
                for (int z = z0; z <= z1; z++)
                    set(x, y, z, t);
    }

    // 空心立方体
    void hollowBox(int x0, int y0, int z0, int x1, int y1, int z1, VoxelType t) {
        for (int x = x0; x <= x1; x++)
            for (int y = y0; y <= y1; y++)
                for (int z = z0; z <= z1; z++)
                    if (x == x0 || x == x1 || y == y0 || y == y1 || z == z0 || z == z1)
                        set(x, y, z, t);
    }

    // DDA 体素 raycast — 返回 VoxelType + hitPos + normal
    bool raycast(const Ray& ray, double maxDist,
                 Vec3& hitPos, Vec3& normal, VoxelType& hitType,
                 int maxSteps = 200) const {
        double ox = ray.origin.x;
        double oy = ray.origin.y;
        double oz = ray.origin.z;
        double dx = ray.dir.x;
        double dy = ray.dir.y;
        double dz = ray.dir.z;

        int ix = int(std::floor(ox));
        int iy = int(std::floor(oy));
        int iz = int(std::floor(oz));

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

        int steps = 0;
        int nx = 0, ny = 0, nz = 0;
        double t = 0.0;

        while (steps < maxSteps) {
            steps++;

            if (ix >= 0 && ix < sx && iy >= 0 && iy < sy && iz >= 0 && iz < sz) {
                VoxelType vt = data_[idx(ix, iy, iz)];
                if (isActive(vt)) {
                    hitPos = Vec3(double(ix) + 0.5, double(iy) + 0.5, double(iz) + 0.5);
                    normal = Vec3(double(nx), double(ny), double(nz));
                    hitType = vt;
                    return true;
                }
            }

            // 推进
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

    // 轻量遮挡查询 — 只返回 bool，不计算 hitPos/normal/dist
    bool isOccluded(const Ray& ray, double maxDist, int maxSteps = 200) const {
        double ox = ray.origin.x, oy = ray.origin.y, oz = ray.origin.z;
        double dx = ray.dir.x, dy = ray.dir.y, dz = ray.dir.z;

        int ix = int(std::floor(ox)), iy = int(std::floor(oy)), iz = int(std::floor(oz));
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
                if (isActive(data_[idx(ix, iy, iz)])) return true;
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

private:
    int sx, sy, sz;
    std::vector<VoxelType> data_;
};

#endif // VOXEL_GRID_H
