// main.cu — CUDA voxel ray tracer
// Reuses scene building from main.cpp, renders on GPU

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <X11/keysym.h>

#include "vec3.h"
#include "ray.h"
#include "voxel_grid.h"
#include "camera.h"
#include "window.h"
#include "cuda_renderer.cuh"

// ============ 场景构建 (复现 main.cpp) ============
struct FastRNG {
    uint64_t state;
    explicit FastRNG(uint64_t seed = 0) : state(seed ? seed : 0xDEADBEEFCAFEBABBull) {}
    double next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 18;
        return (state * 0x9E3779B97F4A7C15ull >> 11) * (1.0 / (1ull << 53));
    }
};

void buildScene(VoxelGrid& grid) {
    int hw = grid.dimX() / 2 - 2;
    int hz = grid.dimZ() / 2 - 2;
    for (int x = -hw; x < hw; x++)
        for (int z = -hz; z < hz; z++)
            grid.set(x + hw, 0, z + hz, VoxelType::Grass);

    FastRNG rng(42);
    for (int i = 0; i < 30; i++) {
        int x = int(rng.next() * (hw*2-4) - hw + 2);
        int z = int(rng.next() * (hz*2-4) - hz + 2);
        grid.set(x + hw, 1, z + hz, VoxelType::Dirt);
    }

    int cx = grid.dimX() / 2, cz = grid.dimZ() / 2;
    for (int y = 0; y < 7; y++)
        for (int x = -2; x <= 2; x++)
            for (int z = -2; z <= 2; z++)
                grid.set(cx + x, y + 1, cz + z, VoxelType::Stone);

    for (int x = -1; x <= 1; x++)
        for (int z = -1; z <= 1; z++)
            grid.set(cx + x, 8, cz + z, VoxelType::GlowStone);

    for (int i = 0; i < 4; i++) {
        double ang = i * M_PI / 2.0;
        int px = cx + int(7 * std::cos(ang));
        int pz = cz + int(7 * std::sin(ang));
        for (int y = 0; y < 5; y++)
            grid.set(px, y + 1, pz, VoxelType::Metal);
        grid.set(px, 6, pz, VoxelType::GlowStone);
    }

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

// ============ PPM 保存 ============
void savePPM(const std::vector<CudaColor>& pixels, int w, int h, const char* name) {
    std::ofstream f(name);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int j = h-1; j >= 0; j--) {
        for (int i = 0; i < w; i++) {
            const CudaColor& c = pixels[j * w + i];
            unsigned char r = (unsigned char)std::min(255.0, c.x * 255.0);
            unsigned char g = (unsigned char)std::min(255.0, c.y * 255.0);
            unsigned char b = (unsigned char)std::min(255.0, c.z * 255.0);
            f << r << g << b;
        }
    }
}

// ============ 主函数 ============
int main() {
    // 选 GPU
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("CUDA device: %s\n", prop.name);

    const int GRID_X = 48, GRID_Y = 32, GRID_Z = 48;

    // 构建场景
    VoxelGrid grid(GRID_X, GRID_Y, GRID_Z);
    buildScene(grid);

    // 导出体素数据到 uint8_t 数组
    std::vector<uint8_t> h_gridData(GRID_X * GRID_Y * GRID_Z);
    for (int x = 0; x < GRID_X; x++)
        for (int y = 0; y < GRID_Y; y++)
            for (int z = 0; z < GRID_Z; z++)
                h_gridData[x * GRID_Y * GRID_Z + y * GRID_Z + z] =
                    static_cast<uint8_t>(grid.get(x, y, z));

    // Sun direction (towards sun)
    double sunX = -0.4, sunY = -0.75, sunZ = -0.3;

    // ===== 测试渲染 =====
    const int TEST_W = 320, TEST_H = 240;
    Camera testCam(Vec3(GRID_X/2.0 - 4, 6, GRID_Z/2.0 - 10),
                   35, -12, 60, (double)TEST_W / TEST_H);

    printf("GPU rendering test %dx%d 4spp 2bounces...\n", TEST_W, TEST_H);
    std::vector<CudaColor> testBuf(TEST_W * TEST_H);

    auto t0 = std::chrono::steady_clock::now();
    cudaRender(h_gridData.data(), GRID_X, GRID_Y, GRID_Z,
               testCam.position().x, testCam.position().y, testCam.position().z,
               testCam.yaw(), testCam.pitch(), 60, (double)TEST_W / TEST_H,
               sunX, sunY, sunZ,
               TEST_W, TEST_H, 4, 2, testBuf.data());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0;
    printf("GPU test render: %.1f ms\n", ms);

    const CudaColor& centerPx = testBuf[TEST_H/2 * TEST_W + TEST_W/2];
    printf("Center pixel: (%.4f, %.4f, %.4f)\n", centerPx.x, centerPx.y, centerPx.z);
    savePPM(testBuf, TEST_W, TEST_H, "gpu_test.ppm");
    printf("Saved gpu_test.ppm\n");

    // ===== 窗口模式 (高频渲染) =====
    const int WIN_W = 1600, WIN_H = 900;
    const int RT_W = 1600, RT_H = 900;  // 原生渲染

    // 检查 X11 显示
    const char* displayEnv = std::getenv("DISPLAY");
    if (!displayEnv || !displayEnv[0]) {
        printf("No DISPLAY, running headless test only.\n");
        return 0;
    }

    WindowX11 win(WIN_W, WIN_H, "Voxel RT (CUDA)");
    if (!win.isRunning()) return 0;

    Camera rtCam(Vec3(GRID_X/2.0 - 4, 6, GRID_Z/2.0 - 10),
                 35, -12, 60, (double)RT_W / RT_H);

    // ===== 预分配 GPU 资源 (一次分配, 每帧复用) =====
    CudaResources gpuRes = cudaInitRenderer(
        h_gridData.data(), GRID_X, GRID_Y, GRID_Z, RT_W, RT_H);
    printf("GPU resources initialized (%dx%d buffer)\n", RT_W, RT_H);

    std::vector<CudaColor> gpuBuf(RT_W * RT_H);

    using Clock = std::chrono::steady_clock;
    auto lastMoveTime = Clock::now();
    auto lastFPSTime = Clock::now();
    int frameCount = 0;
    double fps = 0;

    const double MOVE_SPEED = 5.0;
    const double MOUSE_SENS = 0.15;
    double renderMS = 0;

    auto lastTime = Clock::now();

    while (win.isRunning()) {
        win.processEvents();

        auto now = Clock::now();
        double dt = std::chrono::duration<double>(now - lastTime).count();
        lastTime = now;

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

        if (moveDelta.length() > 1e-6) {
            rtCam.move(moveDelta);
            lastMoveTime = Clock::now();
        }

        int mdx = win.mouseDX();
        int mdy = win.mouseDY();
        if (mdx != 0 || mdy != 0) {
            rtCam.rotate(mdx * MOUSE_SENS, -mdy * MOUSE_SENS);
            lastMoveTime = Clock::now();
        }

        // P 键保存截图
        static bool prevP = false;
        bool nowP = win.isKeyDown(XK_p);
        if (nowP && !prevP) {
            std::vector<CudaColor> ssBuf(WIN_W * WIN_H);
            cudaRender(h_gridData.data(), GRID_X, GRID_Y, GRID_Z,
                       rtCam.position().x, rtCam.position().y, rtCam.position().z,
                       rtCam.yaw(), rtCam.pitch(), 60, (double)WIN_W / WIN_H,
                       sunX, sunY, sunZ,
                       WIN_W, WIN_H, 16, 3, ssBuf.data());
            savePPM(ssBuf, WIN_W, WIN_H, "gpu_screenshot.ppm");
            printf("Saved gpu_screenshot.ppm (%dx%d 16spp 3b)\n", WIN_W, WIN_H);
        }
        prevP = nowP;

        // === 自适应渲染质量 ===
        double idleSec = std::chrono::duration<double>(Clock::now() - lastMoveTime).count();

        int spp, bounces;
        if (idleSec < 0.1) {
            spp = 1; bounces = 1;
        } else if (idleSec < 0.5) {
            spp = 2; bounces = 1;
        } else if (idleSec < 2.0) {
            spp = 4; bounces = 2;
        } else {
            spp = 8; bounces = 3;
        }

        auto renderStart = Clock::now();
        cudaRenderFrame(gpuRes,
                        rtCam.position().x, rtCam.position().y, rtCam.position().z,
                        rtCam.yaw(), rtCam.pitch(), 60, (double)RT_W / RT_H,
                        RT_W, RT_H, spp, bounces, gpuBuf.data());
        renderMS = std::chrono::duration<double>(Clock::now() - renderStart).count() * 1000.0;

        // Direct blit to framebuffer (native resolution, no upscale)
        for (int j = 0; j < RT_H; j++) {
            for (int i = 0; i < RT_W; i++) {
                const CudaColor& c = gpuBuf[j * RT_W + i];
                unsigned char r = (unsigned char)std::min(255.0, c.x * 255.0);
                unsigned char g = (unsigned char)std::min(255.0, c.y * 255.0);
                unsigned char b = (unsigned char)std::min(255.0, c.z * 255.0);
                unsigned char* fb = (unsigned char*)win.framebuffer();
                int idx = (j * WIN_W + i) * 4;
                fb[idx + 0] = b;
                fb[idx + 1] = g;
                fb[idx + 2] = r;
                fb[idx + 3] = 0;
            }
        }
        win.present();

        // Overlay
        int fh = win.fontHeight();
        char line1[128], line2[128];
        std::snprintf(line1, sizeof(line1),
                      "FPS: %.1f | %dx%d | %dspp %db | GPU",
                      fps, RT_W, RT_H, spp, bounces);
        std::snprintf(line2, sizeof(line2),
                      "IDLE: %.1fs | %.1fms/frame | P:Save PPM | ESC:Quit",
                      idleSec, renderMS);
        win.drawText(10, fh + 4, line1, 0x00CCFF66);
        win.drawText(10, fh * 2 + 8, line2, 0x00CCCCCC);
        win.flush();

        frameCount++;

        // FPS
        double elapsed = std::chrono::duration<double>(Clock::now() - lastFPSTime).count();
        if (elapsed > 1.0) {
            fps = frameCount / elapsed;
            char title[128];
            std::snprintf(title, sizeof(title),
                          "Voxel RT (CUDA) | %dx%d %dspp %db | %.1f FPS | %s",
                          RT_W, RT_H, spp, bounces, fps,
                          win.isMouseGrabbed() ? "ESC to quit" : "Click to grab");
            win.setTitle(title);
            frameCount = 0;
            lastFPSTime = Clock::now();
        }
    }

    // ===== 清理 GPU 资源 =====
    cudaDestroyRenderer(gpuRes);
    return 0;
}
