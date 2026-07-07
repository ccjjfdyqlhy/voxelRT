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
#include "collision.h"
#include "imgui/imgui.h"
#include "imgui_impl_x11.h"
#include "imgui_impl_fb.h"

// ============ 场景构建 ============

// 本地 RNG (renderer.h 的 FastRNG 也在同一目录，引入即可)
// 实际上 renderer.h 已定义 FastRNG, 这里直接引用

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

// ============ 主循环 ============

int main() {
    const int GRID_X = 96, GRID_Y = 64, GRID_Z = 96;

    // 场景
    VoxelGrid grid(GRID_X, GRID_Y, GRID_Z);
    buildScene(grid);

    // 测试：直接渲染一帧保存 PPM
    const int TEST_W = 320, TEST_H = 240;
    Camera testCam(Vec3(GRID_X/2.0 - 8, 12, GRID_Z/2.0 - 20),
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

    // === 图形选项 ===
    float fov = 60.0f;
    float mouseSens = 0.15f;
    int fullSpp = 4;
    int fullBounces = 2;

    // === 键位配置 ===
    enum Action {
        ActForward, ActBackward, ActLeft, ActRight,
        ActUp, ActDown, ActScreenshot, ActCount
    };
    static const char* actionNames[] = { "前进", "后退", "左移", "右移", "跳跃", "下降", "截图" };
    KeySym actionKeys[ActCount] = { XK_w, XK_s, XK_a, XK_d, XK_space, XK_Shift_L, XK_p };
    int waitingForAction = -1;

    int menuPage = 0;
    bool forceRender = false;

    // 相机 (初始在塔前方上空)
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
    const double WALK_SPEED = 6.0;   // m/s 行走速度
    const double MAX_DT = 0.05;      // 最大帧间隔 (防穿墙)

    // 把玩家从体素中弹出来（如果初始位置卡在体素里）
    if (playerCollider.collides(playerFeetPos, grid)) {
        // 尝试向上抬，直到不碰撞
        for (int i = 0; i < 20; i++) {
            playerFeetPos = playerFeetPos + Vec3(0, 0.5, 0);
            if (!playerCollider.collides(playerFeetPos, grid)) break;
        }
    }
    // 更新相机到玩家位置
    rtCam.setPosition(playerFeetPos + Vec3(0, playerParams.eyeHeight, 0));

    // 渲染器
    Renderer fullRenderer(WIN_W, WIN_H, fullSpp, fullBounces);
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

    auto lastTime = Clock::now();

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
    char titleBuf[128];

    bool sceneRendered = false;

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
        bool cameraMoved = false;
        Vec3 fwd = rtCam.forward();
        Vec3 rgt = rtCam.right();

        if (!menuActive) {
            // dt 上限防止穿墙
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
            if (walkLen > 1e-6) {
                walkDir = walkDir * (1.0 / walkLen); // 归一化
            }

            // === 跳跃 ===
            bool nowJump = win.isKeyDown(actionKeys[ActUp]);
            if (nowJump && !prevJump && onGround) {
                playerVelocity.y = playerParams.jumpSpeed;
                onGround = false;
            }
            prevJump = nowJump;

            // === 重力 ===
            if (!onGround) {
                playerVelocity.y += playerParams.gravity * dt;
            }

            // === 总位移 ===
            Vec3 moveDelta = walkDir * WALK_SPEED * dt;
            moveDelta.y = playerVelocity.y * dt;

            // === 碰撞解析 ===
            Vec3 newFeetPos = playerCollider.resolveMove(playerFeetPos, moveDelta, grid);

            // === 地面检测 ===
            // 如果 Y 速度向下 且 新位置踩在地面上
            if (playerVelocity.y <= 0) {
                if (playerCollider.onGround(newFeetPos, grid)) {
                    playerVelocity.y = 0;
                    onGround = true;
                    // 精确贴地：把脚底卡到地面体素顶面
                    // 找出脚下最近的固体体素顶面（脚底下方体素，非脚所在层）
                    int feetBlockY = (int)std::floor(newFeetPos.y - 0.001);
                    if (grid.isSolid((int)std::floor(newFeetPos.x), feetBlockY,
                                     (int)std::floor(newFeetPos.z)) ||
                        grid.isSolid((int)std::floor(newFeetPos.x + playerParams.radius - 0.01), feetBlockY,
                                     (int)std::floor(newFeetPos.z + playerParams.radius - 0.01)) ||
                        grid.isSolid((int)std::floor(newFeetPos.x - playerParams.radius + 0.01), feetBlockY,
                                     (int)std::floor(newFeetPos.z - playerParams.radius + 0.01))) {
                        newFeetPos.y = feetBlockY + 1.0; // 站在体素顶面
                    }
                }
            } else if (playerVelocity.y > 0) {
                // 上升时头顶撞到体素
                // resolveMove 已经处理了 Y 轴碰撞
                // 检查 Y 是否被阻挡
                Vec3 testUp = playerFeetPos + Vec3(0, moveDelta.y, 0);
                Vec3 resolvedUp = playerCollider.resolveMove(playerFeetPos, Vec3(0, moveDelta.y, 0), grid);
                if (resolvedUp.y < testUp.y - 0.001) {
                    playerVelocity.y = 0; // 撞头停止
                }
                onGround = false;
            }

            // 更新玩家位置
            bool feetMoved = (newFeetPos - playerFeetPos).length2() > 1e-10;
            playerFeetPos = newFeetPos;

            // 更新相机位置
            Vec3 newCamPos = playerFeetPos + Vec3(0, playerParams.eyeHeight, 0);
            rtCam.setPosition(newCamPos);

            // 截图键
            static bool prevScreenshot = false;
            bool nowScreenshot = win.isKeyDown(actionKeys[ActScreenshot]);
            if (nowScreenshot && !prevScreenshot) {
                std::vector<Color> saveBuf;
                Camera saveCam(rtCam.position(), rtCam.yaw(), rtCam.pitch(),
                               60, (double)WIN_W / WIN_H);
                Renderer saveRenderer(WIN_W, WIN_H, 8, 2);
                saveRenderer.render(saveCam, grid, saveBuf);
                savePPM(saveBuf, WIN_W, WIN_H, "screenshot.ppm");
                printf("Saved screenshot.ppm (%dx%d, 8spp)\n", WIN_W, WIN_H);
            }
            prevScreenshot = nowScreenshot;

            if (feetMoved || walkLen > 1e-6) {
                cameraMoved = true;
                lastMoveTime = Clock::now();
            }

            // === 鼠标旋转 (不受碰撞影响) ===
            int mdx = win.mouseDX();
            int mdy = win.mouseDY();
            if (mdx != 0 || mdy != 0) {
                rtCam.rotate(mdx * mouseSens, -mdy * mouseSens);
                cameraMoved = true;
                lastMoveTime = Clock::now();
            }
        }

        // === 强制重绘 (设置变更后) ===
        if (forceRender) {
            cameraMoved = true;
            forceRender = false;
        }

        // === 渲染决策 ===
        sceneRendered = false;
        double idleSec = std::chrono::duration<double>(Clock::now() - lastMoveTime).count();
        bool doFull = (firstFrame) || (idleSec > 0.3 && !cameraMoved);

        if (cameraMoved || doFull || firstFrame) {
            firstFrame = false;
            fullQuality = doFull;

            if (doFull) {
                Camera fullCam(rtCam.position(), rtCam.yaw(), rtCam.pitch(),
                               fov, (double)WIN_W / WIN_H);
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

            frameCount++;
            sceneRendered = true;
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
                    forceRender = true;
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
                    forceRender = true;
                }

                ImGui::SetCursorPosX(20);
                ImGui::SetNextItemWidth(winW - 40);
                ImGui::SliderFloat("鼠标灵敏度", &mouseSens, 0.05f, 0.50f, "%.2f");

                ImGui::SetCursorPosX(20);
                ImGui::SetNextItemWidth(winW - 40);
                if (ImGui::SliderInt("采样数 (SPP)", &fullSpp, 1, 8)) {
                    fullRenderer.setSamples(fullSpp);
                    forceRender = true;
                }

                ImGui::SetCursorPosX(20);
                ImGui::SetNextItemWidth(winW - 40);
                if (ImGui::SliderInt("最大弹射次数", &fullBounces, 0, 4)) {
                    fullRenderer.setMaxBounces(fullBounces);
                    forceRender = true;
                }

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

        // === 显示 ===
        if (sceneRendered || menuActive) {
            win.present();
        }

        // === FPS 更新 ===
        double elapsed = std::chrono::duration<double>(Clock::now() - lastFPSTime).count();
        if (elapsed > 1.0) {
            fps = frameCount / elapsed;
            std::snprintf(titleBuf, sizeof(titleBuf),
                          "Voxel RT | %s | %.1f FPS",
                          fullQuality ? "FULL 4spp" : "FAST 1spp", fps);
            win.setTitle(titleBuf);
            frameCount = 0;
            lastFPSTime = Clock::now();
        }

        if (!menuActive && !cameraMoved && !doFull && elapsed < 1.0) {
            struct timespec ts = {0, 8000000};
            nanosleep(&ts, nullptr);
        }
    }

    // ===== 清理 ImGui =====
    ImGui_ImplFB_Shutdown();
    ImGui_ImplX11_Shutdown();
    ImGui::DestroyContext();

    return 0;
}
