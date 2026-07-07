#ifndef COLLISION_H
#define COLLISION_H

#include <cmath>
#include "vec3.h"
#include "voxel_grid.h"

// === 玩家物理参数 ===
struct PlayerParams {
    double height = 1.8;     // 玩家身高
    double radius = 0.3;     // 玩家半径 (X/Z 半宽)
    double eyeHeight = 1.6;  // 相机距脚底偏移
    double gravity = -25.0;  // 重力加速度 (m/s²)
    double jumpSpeed = 9.0;  // 跳跃初速度 (m/s)
    double stepHeight = 0.5; // 自动跨步高度
};

// === 玩家碰撞体 ===
class PlayerCollider {
public:
    PlayerCollider() : params_() {}

    explicit PlayerCollider(const PlayerParams& params) : params_(params) {}

    const PlayerParams& params() const { return params_; }

    // 获取脚底位置对应 AABB
    void getAABB(const Vec3& feetPos, Vec3& min, Vec3& max) const {
        min = Vec3(feetPos.x - params_.radius, feetPos.y, feetPos.z - params_.radius);
        max = Vec3(feetPos.x + params_.radius, feetPos.y + params_.height, feetPos.z + params_.radius);
    }

    // AABB vs 单个体素精确重叠检测
    // 严格 < 比较上界：AABB 紧贴体素表面(如站在上面)不算重叠
    static bool overlapsVoxel(const Vec3& amin, const Vec3& amax,
                              int vx, int vy, int vz) {
        return amin.x < vx + 1.0 && amax.x > vx &&
               amin.y < vy + 1.0 && amax.y > vy &&
               amin.z < vz + 1.0 && amax.z > vz;
    }

    // 检查脚底位置是否与任何实体体素碰撞
    bool collides(const Vec3& feetPos, const VoxelGrid& grid) const {
        Vec3 min, max;
        getAABB(feetPos, min, max);

        int x0 = (int)std::floor(min.x);
        int y0 = (int)std::floor(min.y);
        int z0 = (int)std::floor(min.z);
        int x1 = (int)std::floor(max.x);
        int y1 = (int)std::floor(max.y);
        int z1 = (int)std::floor(max.z);

        for (int x = x0; x <= x1; x++) {
            for (int y = y0; y <= y1; y++) {
                for (int z = z0; z <= z1; z++) {
                    if (grid.isSolid(x, y, z)) {
                        if (overlapsVoxel(min, max, x, y, z))
                            return true;
                    }
                }
            }
        }
        return false;
    }

    // 检查是否站在地面上 (脚底正下方有体素)
    bool onGround(const Vec3& feetPos, const VoxelGrid& grid) const {
        // 脚底位置在 y = feetPos.y 处
        // 检查脚底下方 0.01 单位处是否有固体
        Vec3 checkPos = feetPos - Vec3(0, 0.01, 0);
        return collides(checkPos, grid);
    }

    // 尝试水平移动 (X→Z)，返回新位置 + 是否完全移动
    Vec3 moveHorizontal(Vec3 pos, double dx, double dz,
                        const VoxelGrid& grid,
                        bool& blocked) const {
        blocked = false;
        if (dx != 0) {
            Vec3 tx = pos + Vec3(dx, 0, 0);
            if (!collides(tx, grid)) pos = tx;
            else blocked = true;
        }
        if (dz != 0) {
            Vec3 tz = pos + Vec3(0, 0, dz);
            if (!collides(tz, grid)) pos = tz;
            else blocked = true;
        }
        return pos;
    }

    // 按轴独立移动 + 碰撞解析 + 自动跨步
    Vec3 resolveMove(const Vec3& feetPos, const Vec3& delta,
                     const VoxelGrid& grid) const {
        // Phase 1: 尝试 X→Z 水平移动 (不含 step-up)
        bool blocked = false;
        Vec3 result = moveHorizontal(feetPos, delta.x, delta.z, grid, blocked);

        // Phase 2: 水平被挡? 尝试跨步 (step-up)
        if (blocked && (delta.x != 0 || delta.z != 0)) {
            Vec3 stepUp = feetPos + Vec3(0, params_.stepHeight, 0);
            if (!collides(stepUp, grid)) {
                bool stepBlocked = false;
                Vec3 stepResult = moveHorizontal(stepUp, delta.x, delta.z, grid, stepBlocked);
                if (!stepBlocked) {
                    result = stepResult; // 跨步成功，位置抬高
                }
            }
        }

        // Phase 3: Y 轴 (重力 / 跳跃)
        // 使用 sub-step swept 检测 + epsilon 兜底，防穿墙
        if (delta.y != 0) {
            const double eps = 1e-6;
            // 子步数：至少 4 步，确保每步 ≤ 0.25 单位，不跳过 1 单位厚体素
            int steps = std::max(4, (int)std::ceil(std::abs(delta.y) / 0.25));
            double stepY = delta.y / steps;
            Vec3 tmp = result;
            for (int i = 0; i < steps; i++) {
                Vec3 next = tmp + Vec3(0, stepY, 0);
                if (!collides(next, grid)) {
                    tmp = next;
                } else {
                    // 碰撞 → 吸附到体素表面
                    if (stepY < 0) {
                        // 下落触底：吸附到脚下最近的体素顶面
                        tmp.y = std::floor(next.y + eps) + 1.0;
                    }
                    // stepY > 0 (撞头): tmp 保留上次有效位置即可
                    break;
                }
            }
            result = tmp;
        }

        return result;
    }

private:
    PlayerParams params_;
};

#endif // COLLISION_H
