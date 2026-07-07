// main.cu — CUDA voxel ray tracer
// Reuses scene building from main.cpp, renders on GPU

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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
#include "collision.h"
#include "cuda_renderer.cuh"
#include "imgui/imgui.h"
#include "imgui_impl_x11.h"
#include "imgui_impl_fb.h"

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
    for (int i = 0; i < 60; i++) {
        int x = int(rng.next() * (hw*2-8) - hw + 4);
        int z = int(rng.next() * (hz*2-8) - hz + 4);
        grid.set(x + hw, 1, z + hz, VoxelType::Dirt);
    }

    int cx = grid.dimX() / 2, cz = grid.dimZ() / 2;
    for (int y = 0; y < 14; y++)
        for (int x = -4; x <= 4; x++)
            for (int z = -4; z <= 4; z++)
                grid.set(cx + x, y + 1, cz + z, VoxelType::Stone);

    for (int x = -2; x <= 2; x++)
        for (int z = -2; z <= 2; z++)
            grid.set(cx + x, 16, cz + z, VoxelType::GlowStone);

    for (int i = 0; i < 4; i++) {
        double ang = i * M_PI / 2.0;
        int px = cx + int(14 * std::cos(ang));
        int pz = cz + int(14 * std::sin(ang));
        for (int y = 0; y < 10; y++)
            grid.set(px, y + 1, pz, VoxelType::Metal);
        grid.set(px, 12, pz, VoxelType::GlowStone);
    }

    for (int i = 0; i < 10; i++) {
        int bx = int(rng.next() * (hw-8) - hw/2 + 4);
        int bz = int(rng.next() * (hz-8) - hz/2 + 4);
        int bh = 4 + int(rng.next() * 4);
        int bw = 2 + int(rng.next() * 4);
        for (int y = 0; y < bh; y++)
            for (int x = -bw; x <= bw; x++)
                for (int z = -bw; z <= bw; z++)
                    if (y == 0 || x == -bw || x == bw || z == -bw || z == bw)
                        grid.set(cx + bx + x, y + 1, cz + bz + z,
                                 rng.next() > 0.5 ? VoxelType::Brick : VoxelType::Dirt);
    }

    // 玻璃方块 (塔前)
    for (int y = 0; y < 8; y++)
        for (int x = -4; x <= 2; x++)
            for (int z = -6; z <= -2; z++)
                grid.set(cx + x + 12, y + 6, cz + z, VoxelType::Crystal);

    int wx = grid.dimX() / 2 - 4;
    int wz = grid.dimZ() / 2 - 4;
    int wh = 8;
    for (int x = -wx; x < wx; x++)
        for (int y = 0; y < wh; y++)
            grid.set(x + hw, y + 1, 0, VoxelType::Stone);
    for (int z = -wz; z < wz; z++)
        for (int y = 0; y < wh; y++)
            grid.set(0, y + 1, z + hz, VoxelType::Stone);
    for (int x = -wx; x < wx; x++)
        for (int y = 0; y < wh; y++)
            grid.set(x + hw, y + 1, grid.dimZ()-1, VoxelType::Stone);
    for (int z = -wz; z < wz; z++)
        for (int y = 0; y < wh; y++)
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

// ============ 键名显示 ============

static const char* keyName(KeySym ks) {
    if (ks >= XK_a && ks <= XK_z) {
        static char buf[2] = {};
        buf[0] = 'A' + (ks - XK_a);
        return buf;
    }
    if (ks >= XK_0 && ks <= XK_9) {
        static char buf[2] = {};
        buf[0] = '0' + (ks - XK_0);
        return buf;
    }
    switch (ks) {
        case XK_space: return "Space";
        case XK_Shift_L: return "L-Shift";
        case XK_Shift_R: return "R-Shift";
        case XK_Control_L: return "L-Ctrl";
        case XK_Control_R: return "R-Ctrl";
        case XK_Escape: return "Esc";
        case XK_Tab: return "Tab";
        case XK_Return: return "Enter";
        case XK_BackSpace: return "Back";
        case XK_Up: return "上";
        case XK_Down: return "下";
        case XK_Left: return "左";
        case XK_Right: return "右";
        default: return "?";
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

    const int GRID_X = 96, GRID_Y = 64, GRID_Z = 96;

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
    Camera testCam(Vec3(GRID_X/2.0 - 8, 12, GRID_Z/2.0 - 20),
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

    // === 图形选项 ===
    float fov = 60.0f;
    float mouseSens = 0.15f;

    // === 键位配置 ===
    enum Action {
        ActForward, ActBackward, ActLeft, ActRight,
        ActUp, ActDown, ActScreenshot, ActCount
    };
    static const char* actionNames[] = { "前进", "后退", "左移", "右移", "跳跃", "下降", "截图" };
    KeySym actionKeys[ActCount] = { XK_w, XK_s, XK_a, XK_d, XK_space, XK_Shift_L, XK_p };
    int waitingForAction = -1;

    int menuPage = 0;

    Camera rtCam(Vec3(GRID_X/2.0 - 8, 12, GRID_Z/2.0 - 20),
                 35, -12, fov, (double)RT_W / RT_H);

    // === 玩家物理 ===
    PlayerParams playerParams;
    playerParams.eyeHeight = 1.6;
    playerParams.height = 1.8;
    playerParams.radius = 0.3;
    playerParams.gravity = -25.0;
    playerParams.jumpSpeed = 9.0;
    PlayerCollider playerCollider(playerParams);
    Vec3 playerFeetPos = rtCam.position() - Vec3(0, playerParams.eyeHeight, 0);
    Vec3 playerVelocity(0, 0, 0);
    bool onGround = false;
    bool prevJump = false;
    const double WALK_SPEED = 6.0;
    const double MAX_DT = 0.05;

    // 初始位置卡体素时自动弹射
    if (playerCollider.collides(playerFeetPos, grid)) {
        for (int i = 0; i < 20; i++) {
            playerFeetPos = playerFeetPos + Vec3(0, 0.5, 0);
            if (!playerCollider.collides(playerFeetPos, grid)) break;
        }
    }
    rtCam.setPosition(playerFeetPos + Vec3(0, playerParams.eyeHeight, 0));

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

    double renderMS = 0;

    auto lastTime = Clock::now();
    auto lastFrameTime = Clock::now();
    char titleBuf[128];

    // ===== ImGui 初始化 =====
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    ImGui::StyleColorsDark();

    // 加载中文字体
    io.Fonts->AddFontFromFileTTF("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                                 18.0f, nullptr, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());

    ImGui_ImplX11_Init(&win);
    ImGui_ImplFB_Init(WIN_W, WIN_H, win.framebuffer());
    ImGui::GetStyle().WindowRounding = 4.0f;
    ImGui::GetStyle().WindowTitleAlign = ImVec2(0.5f, 0.5f);
    win.setEventCallback(ImGui_ImplX11_ProcessEvent);

    bool menuActive = false;
    bool prevEsc = false;
    bool quitRequested = false;

    while (win.isRunning() && !quitRequested) {
        win.processEvents();
        auto now = Clock::now();
        double dt = std::chrono::duration<double>(now - lastTime).count();
        lastTime = now;

        // === ESC 切换菜单 ===
        bool nowEsc = win.isKeyDown(XK_Escape);
        if (nowEsc && !prevEsc) {
            menuActive = !menuActive;
            if (menuActive) win.ungrabMouse();
        }
        prevEsc = nowEsc;
        ImGui_ImplX11_SetMenuActive(menuActive);

        // === 玩家物理 + 相机控制 (菜单关闭时) ===
        Vec3 fwd = rtCam.forward();
        Vec3 rgt = rtCam.right();

        if (!menuActive) {
            if (dt > MAX_DT) dt = MAX_DT;

            // === 水平移动输入 ===
            Vec3 walkDir(0, 0, 0);
            if (win.isKeyDown(actionKeys[ActForward]))
                walkDir = walkDir + Vec3(fwd.x, 0, fwd.z).normalized();
            if (win.isKeyDown(actionKeys[ActBackward]))
                walkDir = walkDir - Vec3(fwd.x, 0, fwd.z).normalized();
            if (win.isKeyDown(actionKeys[ActLeft]))
                walkDir = walkDir - rgt;
            if (win.isKeyDown(actionKeys[ActRight]))
                walkDir = walkDir + rgt;

            double walkLen = walkDir.length();
            if (walkLen > 1e-6)
                walkDir = walkDir * (1.0 / walkLen);

            // === 跳跃 ===
            bool nowJump = win.isKeyDown(actionKeys[ActUp]);
            if (nowJump && !prevJump && onGround) {
                playerVelocity.y = playerParams.jumpSpeed;
                onGround = false;
            }
            prevJump = nowJump;

            // === 重力 ===
            if (!onGround)
                playerVelocity.y += playerParams.gravity * dt;

            // === 总位移 ===
            Vec3 moveDelta = walkDir * WALK_SPEED * dt;
            moveDelta.y = playerVelocity.y * dt;

            // === 碰撞解析 ===
            Vec3 newFeetPos = playerCollider.resolveMove(playerFeetPos, moveDelta, grid);

            // === 地面检测 ===
            if (playerVelocity.y <= 0) {
                if (playerCollider.onGround(newFeetPos, grid)) {
                    playerVelocity.y = 0;
                    onGround = true;
                    int feetBlockY = (int)std::floor(newFeetPos.y);
                    if (grid.isSolid((int)std::floor(newFeetPos.x), feetBlockY,
                                     (int)std::floor(newFeetPos.z)) ||
                        grid.isSolid((int)std::floor(newFeetPos.x + playerParams.radius - 0.01), feetBlockY,
                                     (int)std::floor(newFeetPos.z + playerParams.radius - 0.01))) {
                        newFeetPos.y = feetBlockY + 1.0;
                    }
                }
            } else if (playerVelocity.y > 0) {
                Vec3 testUp = playerFeetPos + Vec3(0, moveDelta.y, 0);
                Vec3 resolvedUp = playerCollider.resolveMove(playerFeetPos, Vec3(0, moveDelta.y, 0), grid);
                if (resolvedUp.y < testUp.y - 0.001)
                    playerVelocity.y = 0;
                onGround = false;
            }

            // 更新玩家位置
            bool feetMoved = (newFeetPos - playerFeetPos).length2() > 1e-10;
            playerFeetPos = newFeetPos;
            Vec3 newCamPos = playerFeetPos + Vec3(0, playerParams.eyeHeight, 0);
            rtCam.setPosition(newCamPos);

            // 截图键
            static bool prevScreenshot = false;
            bool nowScreenshot = win.isKeyDown(actionKeys[ActScreenshot]);
            if (nowScreenshot && !prevScreenshot) {
                std::vector<CudaColor> ssBuf(WIN_W * WIN_H);
                cudaRender(h_gridData.data(), GRID_X, GRID_Y, GRID_Z,
                           rtCam.position().x, rtCam.position().y, rtCam.position().z,
                           rtCam.yaw(), rtCam.pitch(), fov, (double)WIN_W / WIN_H,
                           sunX, sunY, sunZ,
                           WIN_W, WIN_H, 16, 3, ssBuf.data());
                savePPM(ssBuf, WIN_W, WIN_H, "gpu_screenshot.ppm");
                printf("Saved gpu_screenshot.ppm (%dx%d 16spp 3b)\n", WIN_W, WIN_H);
            }
            prevScreenshot = nowScreenshot;

            // === 鼠标旋转 ===
            int mdx = win.mouseDX();
            int mdy = win.mouseDY();
            if (mdx != 0 || mdy != 0) {
                rtCam.rotate(mdx * mouseSens, -mdy * mouseSens);
                lastMoveTime = Clock::now();
            }

            if (feetMoved || walkLen > 1e-6) {
            }
        }

        // === 自适应渲染质量 ===
        double idleSec = std::chrono::duration<double>(Clock::now() - lastMoveTime).count();

        int spp, bounces;
        if (idleSec < 0.1) {
            spp = 1; bounces = 0;
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
                        rtCam.yaw(), rtCam.pitch(), fov, (double)RT_W / RT_H,
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

        // === ImGui 渲染 ===
        ImGui_ImplFB_UpdateFramebuffer(WIN_W, WIN_H, win.framebuffer());
        ImGui_ImplX11_NewFrame();
        ImGui_ImplFB_NewFrame();
        ImGui::NewFrame();

        if (menuActive) {
            // 键位绑定等待检测
            if (waitingForAction >= 0) {
                KeySym ks = win.lastKeyPressed();
                if (ks != 0) {
                    if (ks == XK_Escape) {
                        waitingForAction = -1;
                    } else {
                        actionKeys[waitingForAction] = ks;
                        waitingForAction = -1;
                    }
                }
            }

            ImGui::SetNextWindowPos(ImVec2(WIN_W * 0.5f, WIN_H * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            int winH = menuPage == 0 ? 320 : (menuPage == 1 ? 380 : 420);
            ImGui::SetNextWindowSize(ImVec2(360, winH), ImGuiCond_Always);
            ImGui::Begin("暂停", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

            float winW = ImGui::GetWindowWidth();
            float btnW = 200;

            if (menuPage == 0) {
                ImGui::SetCursorPosX((winW - 100) * 0.5f);
                ImGui::Text("=== 菜单 ===");
                ImGui::Spacing(); ImGui::Spacing(); ImGui::Spacing();

                ImGui::SetCursorPosX((winW - btnW) * 0.5f);
                if (ImGui::Button("回到场景", ImVec2(btnW, 40))) {
                    menuActive = false;
                    menuPage = 0;
                    ImGui_ImplX11_SetMenuActive(false);
                    win.grabMouse();
                    lastMoveTime = Clock::now() - std::chrono::seconds(10);
                }
                ImGui::Spacing();
                ImGui::SetCursorPosX((winW - btnW) * 0.5f);
                if (ImGui::Button("图形选项", ImVec2(btnW, 40))) {
                    menuPage = 1;
                }
                ImGui::Spacing();
                ImGui::SetCursorPosX((winW - btnW) * 0.5f);
                if (ImGui::Button("键位配置", ImVec2(btnW, 40))) {
                    menuPage = 2;
                }
                ImGui::Spacing();
                ImGui::SetCursorPosX((winW - btnW) * 0.5f);
                if (ImGui::Button("退出", ImVec2(btnW, 40))) {
                    quitRequested = true;
                }
            } else if (menuPage == 1) {
                ImGui::SetCursorPosX((winW - 100) * 0.5f);
                ImGui::Text("=== 图形选项 ===");
                ImGui::Spacing(); ImGui::Spacing();

                ImGui::SetCursorPosX(20);
                ImGui::SetNextItemWidth(winW - 40);
                if (ImGui::SliderFloat("视野 (FOV)", &fov, 30, 120, "%.0f°")) {
                    rtCam.setFov(fov);
                }

                ImGui::SetCursorPosX(20);
                ImGui::SetNextItemWidth(winW - 40);
                ImGui::SliderFloat("鼠标灵敏度", &mouseSens, 0.05f, 0.50f, "%.2f");

                ImGui::Spacing(); ImGui::Spacing();
                ImGui::SetCursorPosX((winW - btnW) * 0.5f);
                if (ImGui::Button("返回", ImVec2(btnW, 40))) {
                    menuPage = 0;
                }
            } else if (menuPage == 2) {
                ImGui::SetCursorPosX((winW - 100) * 0.5f);
                ImGui::Text("=== 键位配置 ===");
                ImGui::Spacing();

                for (int i = 0; i < ActCount; i++) {
                    ImGui::SetCursorPosX(20);
                    ImGui::Text("%s:", actionNames[i]);
                    ImGui::SameLine();
                    ImGui::SetCursorPosX(winW - 150);

                    char label[64];
                    if (waitingForAction == i) {
                        std::snprintf(label, sizeof(label), "...");
                    } else {
                        std::snprintf(label, sizeof(label), "%s", keyName(actionKeys[i]));
                    }
                    ImGui::PushID(i);
                    if (ImGui::Button(label, ImVec2(120, 30))) {
                        waitingForAction = i;
                    }
                    ImGui::PopID();
                }

                ImGui::Spacing(); ImGui::Spacing();
                ImGui::SetCursorPosX((winW - btnW) * 0.5f);
                if (ImGui::Button("返回", ImVec2(btnW, 40))) {
                    waitingForAction = -1;
                    menuPage = 0;
                }
            }

            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplFB_RenderDrawData(ImGui::GetDrawData());

        win.present();

        // Overlay
        int fh = win.fontHeight();
        char line1[128], line2[128];
        std::snprintf(line1, sizeof(line1),
                      "FPS: %.1f | %dx%d | %dspp %db | GPU",
                      fps, RT_W, RT_H, spp, bounces);
        std::snprintf(line2, sizeof(line2),
                      "IDLE: %.1fs | %.1fms/frame | P:Save PPM | ESC:Menu",
                      idleSec, renderMS);
        win.drawText(10, fh + 4, line1, 0x00CCFF66);
        win.drawText(10, fh * 2 + 8, line2, 0x00CCCCCC);
        win.flush();

        frameCount++;

        // FPS
        double elapsed = std::chrono::duration<double>(Clock::now() - lastFPSTime).count();
        if (elapsed > 0.5) {
            fps = frameCount / elapsed;
            std::snprintf(titleBuf, sizeof(titleBuf),
                          "Voxel RT (CUDA) | %dx%d %dspp %db | %.1f FPS",
                          RT_W, RT_H, spp, bounces, fps);
            win.setTitle(titleBuf);
            frameCount = 0;
            lastFPSTime = Clock::now();
        }

        lastFrameTime = Clock::now();
    }

    // ===== 清理 =====
    ImGui_ImplFB_Shutdown();
    ImGui_ImplX11_Shutdown();
    ImGui::DestroyContext();
    cudaDestroyRenderer(gpuRes);
    return 0;
}
