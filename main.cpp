#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <X11/keysym.h>

#include "vec3.h"
#include "ray.h"
#include "voxel_grid.h"
#include "camera.h"
#include "renderer.h"
#include "window.h"

// ============ 场景构建 ============

// 本地 RNG (renderer.h 的 FastRNG 也在同一目录，引入即可)
// 实际上 renderer.h 已定义 FastRNG, 这里直接引用

void buildScene(VoxelGrid& grid) {
    // 地面 (草)
    int hw = grid.dimX() / 2 - 2;
    int hz = grid.dimZ() / 2 - 2;
    for (int x = -hw; x < hw; x++)
        for (int z = -hz; z < hz; z++)
            grid.set(x + hw, 0, z + hz, VoxelType::Grass);

    // 地面小起伏
    FastRNG rng(42);
    for (int i = 0; i < 30; i++) {
        int x = int(rng.next() * (hw*2-4) - hw + 2);
        int z = int(rng.next() * (hz*2-4) - hz + 2);
        grid.set(x + hw, 1, z + hz, VoxelType::Dirt);
    }

    // 中心塔 (砖石)
    int cx = grid.dimX() / 2, cz = grid.dimZ() / 2;
    for (int y = 0; y < 7; y++)
        for (int x = -2; x <= 2; x++)
            for (int z = -2; z <= 2; z++)
                grid.set(cx + x, y + 1, cz + z, VoxelType::Stone);

    // 塔顶发光
    for (int x = -1; x <= 1; x++)
        for (int z = -1; z <= 1; z++)
            grid.set(cx + x, 8, cz + z, VoxelType::GlowStone);

    // 金属柱
    for (int i = 0; i < 4; i++) {
        double ang = i * M_PI / 2.0;
        int px = cx + int(7 * std::cos(ang));
        int pz = cz + int(7 * std::sin(ang));
        for (int y = 0; y < 5; y++)
            grid.set(px, y + 1, pz, VoxelType::Metal);
        grid.set(px, 6, pz, VoxelType::GlowStone);
    }

    // 随机小建筑
    for (int i = 0; i < 5; i++) {
        int bx = int(rng.next() * (hw-4) - hw/2 + 2);
        int bz = int(rng.next() * (hz-4) - hz/2 + 2);
        int bh = 2 + int(rng.next() * 2);
        int bw = 1 + int(rng.next() * 2);
        for (int y = 0; y < bh; y++)
            for (int x = -bw; x <= bw; x++)
                for (int z = -bw; z <= bw; z++)
                    if (y == 0 || x == -bw || x == bw || z == -bw || z == bw)
                        grid.set(cx + bx + x, y + 1, cz + bz + z,
                                 rng.next() > 0.5 ? VoxelType::Brick : VoxelType::Dirt);
    }

    // 玻璃方块 (塔前)
    for (int y = 0; y < 4; y++)
        for (int x = -2; x <= 1; x++)
            for (int z = -3; z <= -1; z++)
                grid.set(cx + x + 6, y + 3, cz + z, VoxelType::Crystal);

    // 远处围墙
    int wx = grid.dimX() / 2 - 2;
    int wz = grid.dimZ() / 2 - 2;
    for (int x = -wx; x < wx; x++)
        for (int y = 0; y < 4; y++)
            grid.set(x + hw, y + 1, 0, VoxelType::Stone);
    for (int z = -wz; z < wz; z++)
        for (int y = 0; y < 4; y++)
            grid.set(0, y + 1, z + hz, VoxelType::Stone);
    for (int x = -wx; x < wx; x++)
        for (int y = 0; y < 4; y++)
            grid.set(x + hw, y + 1, grid.dimZ()-1, VoxelType::Stone);
    for (int z = -wz; z < wz; z++)
        for (int y = 0; y < 4; y++)
            grid.set(grid.dimX()-1, y + 1, z + hz, VoxelType::Stone);
}

// ============ 双线性插值上采样 ============

static inline void putPixel(unsigned char* fb, int x, int y, int w, int h,
                             unsigned char r, unsigned char g, unsigned char b) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    int idx = (y * w + x) * 4;
    fb[idx + 0] = b;  // X11: BGR
    fb[idx + 1] = g;
    fb[idx + 2] = r;
    fb[idx + 3] = 0;
}

void bilinearUpscale(const std::vector<Color>& src, int sw, int sh,
                     unsigned char* dst, int dw, int dh) {
    for (int dy = 0; dy < dh; dy++) {
        for (int dx = 0; dx < dw; dx++) {
            double sx = (dx + 0.5) * sw / dw - 0.5;
            double sy = (dy + 0.5) * sh / dh - 0.5;

            int ix = (int)std::floor(sx);
            int iy = (int)std::floor(sy);
            double fx = sx - ix;
            double fy = sy - iy;

            auto clamp = [](int v, int min, int max) {
                return v < min ? min : (v > max ? max : v);
            };

            int x0 = clamp(ix,   0, sw-1);
            int x1 = clamp(ix+1, 0, sw-1);
            int y0 = clamp(iy,   0, sh-1);
            int y1 = clamp(iy+1, 0, sh-1);

            const Color& c00 = src[y0 * sw + x0];
            const Color& c10 = src[y0 * sw + x1];
            const Color& c01 = src[y1 * sw + x0];
            const Color& c11 = src[y1 * sw + x1];

            double r = (1-fy)*((1-fx)*c00.x + fx*c10.x) + fy*((1-fx)*c01.x + fx*c11.x);
            double g = (1-fy)*((1-fx)*c00.y + fx*c10.y) + fy*((1-fx)*c01.y + fx*c11.y);
            double b = (1-fy)*((1-fx)*c00.z + fx*c10.z) + fy*((1-fx)*c01.z + fx*c11.z);

            putPixel(dst, dx, dy, dw, dh,
                     (unsigned char)std::min(255.0, r * 255.0),
                     (unsigned char)std::min(255.0, g * 255.0),
                     (unsigned char)std::min(255.0, b * 255.0));
        }
    }
}

// ============ 保存 PPM (调试用) ============

void savePPM(const std::vector<Color>& pixels, int w, int h, const char* name) {
    std::ofstream f(name);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int j = h-1; j >= 0; j--) {
        for (int i = 0; i < w; i++) {
            const Color& c = pixels[j * w + i];
            unsigned char r = (unsigned char)std::min(255.0, c.x * 255.0);
            unsigned char g = (unsigned char)std::min(255.0, c.y * 255.0);
            unsigned char b = (unsigned char)std::min(255.0, c.z * 255.0);
            f << r << g << b;
        }
    }
}

// ============ 主循环 ============

int main() {
    const int GRID_X = 48, GRID_Y = 32, GRID_Z = 48;

    // 场景
    VoxelGrid grid(GRID_X, GRID_Y, GRID_Z);
    buildScene(grid);

    // 测试：直接渲染一帧保存 PPM
    const int TEST_W = 320, TEST_H = 240;
    Camera testCam(Vec3(GRID_X/2.0 - 4, 6, GRID_Z/2.0 - 10),
                   35, -12, 60, (double)TEST_W / TEST_H);

    // 调试：手动追踪中心光线
    Ray centerRay = testCam.getRay(0.5, 0.5);
    printf("Camera pos: (%.1f, %.1f, %.1f)\n", testCam.position().x, testCam.position().y, testCam.position().z);
    printf("Ray dir: (%.4f, %.4f, %.4f)\n", centerRay.dir.x, centerRay.dir.y, centerRay.dir.z);
    printf("Ray origin voxel: (%d, %d, %d)\n",
           int(floor(centerRay.origin.x)), int(floor(centerRay.origin.y)), int(floor(centerRay.origin.z)));

    Vec3 hitPos, hitNorm;
    VoxelType hitType;
    bool hit = grid.raycast(centerRay, 200.0, hitPos, hitNorm, hitType);
    if (hit) {
        printf("RAY HIT: voxelType=%d at (%.2f, %.2f, %.2f), normal=(%d, %d, %d), dist=%.2f\n",
               (int)hitType, hitPos.x, hitPos.y, hitPos.z,
               (int)hitNorm.x, (int)hitNorm.y, (int)hitNorm.z,
               (hitPos - centerRay.origin).length());
    } else {
        printf("RAY MISS: no intersection within 200 units\n");
    }

    // 再检查从中心向地面打一条射线
    Ray groundRay(Vec3(20, 10, 20), Vec3(0, -1, 0));
    hit = grid.raycast(groundRay, 200.0, hitPos, hitNorm, hitType);
    printf("Ground test from (20,10,20) looking down: %s", hit ? "HIT" : "MISS");
    if (hit) printf(" at y=%.2f type=%d", hitPos.y, (int)hitType);
    printf("\n");

    // 检查塔是否在正确位置
    for (int y = 0; y < 8; y++) {
        VoxelType vt = grid.get(24, y+1, 24);
        printf("Tower center (24,%d,24): type=%d\n", y+1, (int)vt);
    }

    Renderer testRenderer(TEST_W, TEST_H, 4, 2);
    std::vector<Color> testBuf;
    testRenderer.render(testCam, grid, testBuf);
    // Check center pixel value
    const Color& pxCenter = testBuf[TEST_H/2 * TEST_W + TEST_W/2];
    printf("Center pixel after fix: %.4f %.4f %.4f (expect ~0.07 for dark Stone)\n",
           pxCenter.x, pxCenter.y, pxCenter.z);
    savePPM(testBuf, TEST_W, TEST_H, "test_bindings.ppm");
    printf("Saved test_bindings.ppm (%dx%d, 4spp, 2bounces)\n", TEST_W, TEST_H);

    // 尝试窗口模式（如果没有显示器则跳过）
    const int WIN_W = 1600, WIN_H = 900;
    const int RT_W = 800, RT_H = 450;   // CPU 实时渲染分辨率

    WindowX11 win(WIN_W, WIN_H, "Voxel Ray Tracer");
    if (!win.isRunning()) return 0;

    // 相机 (初始在塔前方上空)
    Camera rtCam(Vec3(GRID_X/2.0 - 4, 6, GRID_Z/2.0 - 10),
                 35, -12, 60, (double)RT_W / RT_H);

    // 渲染器
    Renderer fullRenderer(WIN_W, WIN_H, 4, 2);
    Renderer fastRenderer(RT_W, RT_H, 1, 0);  // 1spp, no bounce

    // 像素缓冲
    std::vector<Color> fullBuf;
    std::vector<Color> fastBuf;

    // 时间追踪
    using Clock = std::chrono::steady_clock;
    auto lastMoveTime = Clock::now();
    auto lastFPSTime = Clock::now();
    int frameCount = 0;
    double fps = 0;

    bool fullQuality = false;
    bool firstFrame = true;
    bool frameSaved = false;  // auto-save first full frame

    const double MOVE_SPEED = 5.0;
    const double MOUSE_SENS = 0.15;

    auto lastTime = Clock::now();

    // 暖机
    (void)0;

    while (win.isRunning()) {
        win.processEvents();

        auto now = Clock::now();
        double dt = std::chrono::duration<double>(now - lastTime).count();
        lastTime = now;

        // === 相机控制 ===
        bool cameraMoved = false;
        Vec3 moveDelta(0, 0, 0);
        Vec3 fwd = rtCam.forward();
        Vec3 rgt = rtCam.right();

        if (win.isKeyDown(XK_w))
            moveDelta = moveDelta + Vec3(fwd.x, 0, fwd.z).normalized() * MOVE_SPEED * dt;
        if (win.isKeyDown(XK_s))
            moveDelta = moveDelta - Vec3(fwd.x, 0, fwd.z).normalized() * MOVE_SPEED * dt;
        if (win.isKeyDown(XK_a)) moveDelta = moveDelta - rgt * MOVE_SPEED * dt;
        if (win.isKeyDown(XK_d)) moveDelta = moveDelta + rgt * MOVE_SPEED * dt;
        if (win.isKeyDown(XK_space))   moveDelta = moveDelta + Vec3(0, MOVE_SPEED * dt, 0);
        if (win.isShiftDown())         moveDelta = moveDelta - Vec3(0, MOVE_SPEED * dt, 0);

        // P 键保存 PPM
        static bool prevP = false;
        bool nowP = win.isKeyDown(XK_p);
        if (nowP && !prevP) {
            std::vector<Color> saveBuf;
            Camera saveCam(rtCam.position(), rtCam.yaw(), rtCam.pitch(),
                           60, (double)WIN_W / WIN_H);
            Renderer saveRenderer(WIN_W, WIN_H, 8, 2);
            saveRenderer.render(saveCam, grid, saveBuf);
            savePPM(saveBuf, WIN_W, WIN_H, "screenshot.ppm");
            printf("Saved screenshot.ppm (%dx%d, 8spp)\n", WIN_W, WIN_H);
        }
        prevP = nowP;

        if (moveDelta.length() > 1e-6) {
            rtCam.move(moveDelta);
            cameraMoved = true;
            lastMoveTime = Clock::now();
        }

        int mdx = win.mouseDX();
        int mdy = win.mouseDY();
        if (mdx != 0 || mdy != 0) {
            rtCam.rotate(mdx * MOUSE_SENS, -mdy * MOUSE_SENS);
            cameraMoved = true;
            lastMoveTime = Clock::now();
        }

        // === 渲染决策 ===
        double idleSec = std::chrono::duration<double>(Clock::now() - lastMoveTime).count();
        bool doFull = (firstFrame) || (idleSec > 0.3 && !cameraMoved);

        if (cameraMoved || doFull || firstFrame) {
            firstFrame = false;
            fullQuality = doFull;

            if (doFull) {
                Camera fullCam(rtCam.position(), rtCam.yaw(), rtCam.pitch(),
                               60, (double)WIN_W / WIN_H);
                fullRenderer.render(fullCam, grid, fullBuf);
                bilinearUpscale(fullBuf, WIN_W, WIN_H,
                                (unsigned char*)win.framebuffer(), WIN_W, WIN_H);
                if (!frameSaved) {
                    savePPM(fullBuf, WIN_W, WIN_H, "startup.ppm");
                    frameSaved = true;
                }
            } else {
                fastRenderer.render(rtCam, grid, fastBuf);
                bilinearUpscale(fastBuf, RT_W, RT_H,
                                (unsigned char*)win.framebuffer(), WIN_W, WIN_H);
            }

            win.present();
            frameCount++;
        }

        // === FPS 更新 ===
        double elapsed = std::chrono::duration<double>(Clock::now() - lastFPSTime).count();
        if (elapsed > 1.0) {
            fps = frameCount / elapsed;
            char title[128];
            std::snprintf(title, sizeof(title),
                          "Voxel RT | %s | %.1f FPS | %s",
                          fullQuality ? "FULL 4spp" : "FAST 1spp",
                          fps,
                          win.isMouseGrabbed() ? "ESC to quit" : "Click to grab");
            win.setTitle(title);
            frameCount = 0;
            lastFPSTime = Clock::now();
        }

        // 空闲时降低 CPU 占用
        if (!cameraMoved && !doFull && elapsed < 1.0) {
            struct timespec ts = {0, 8000000};  // 8ms
            nanosleep(&ts, nullptr);
        }
    }

    return 0;
}
