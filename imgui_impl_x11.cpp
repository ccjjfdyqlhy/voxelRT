#include "imgui_impl_x11.h"
#include "window.h"
#include <X11/keysym.h>
#include <cstdio>
#include <ctime>
#include <string>

struct X11BackendData {
    WindowX11* window = nullptr;
    double time = 0;
    bool menuActive = false;
};

static X11BackendData* bd = nullptr;

static ImGuiKey keysymToImGui(KeySym ks) {
    switch (ks) {
        case XK_Tab: return ImGuiKey_Tab;
        case XK_Left: return ImGuiKey_LeftArrow;
        case XK_Right: return ImGuiKey_RightArrow;
        case XK_Up: return ImGuiKey_UpArrow;
        case XK_Down: return ImGuiKey_DownArrow;
        case XK_Page_Up: return ImGuiKey_PageUp;
        case XK_Page_Down: return ImGuiKey_PageDown;
        case XK_Home: return ImGuiKey_Home;
        case XK_End: return ImGuiKey_End;
        case XK_Insert: return ImGuiKey_Insert;
        case XK_Delete: return ImGuiKey_Delete;
        case XK_BackSpace: return ImGuiKey_Backspace;
        case XK_space: return ImGuiKey_Space;
        case XK_Return: return ImGuiKey_Enter;
        case XK_Escape: return ImGuiKey_Escape;
        case XK_KP_Enter: return ImGuiKey_KeypadEnter;
        case XK_a: case XK_A: return ImGuiKey_A;
        case XK_b: case XK_B: return ImGuiKey_B;
        case XK_c: case XK_C: return ImGuiKey_C;
        case XK_d: case XK_D: return ImGuiKey_D;
        case XK_e: case XK_E: return ImGuiKey_E;
        case XK_f: case XK_F: return ImGuiKey_F;
        case XK_g: case XK_G: return ImGuiKey_G;
        case XK_h: case XK_H: return ImGuiKey_H;
        case XK_i: case XK_I: return ImGuiKey_I;
        case XK_j: case XK_J: return ImGuiKey_J;
        case XK_k: case XK_K: return ImGuiKey_K;
        case XK_l: case XK_L: return ImGuiKey_L;
        case XK_m: case XK_M: return ImGuiKey_M;
        case XK_n: case XK_N: return ImGuiKey_N;
        case XK_o: case XK_O: return ImGuiKey_O;
        case XK_p: case XK_P: return ImGuiKey_P;
        case XK_q: case XK_Q: return ImGuiKey_Q;
        case XK_r: case XK_R: return ImGuiKey_R;
        case XK_s: case XK_S: return ImGuiKey_S;
        case XK_t: case XK_T: return ImGuiKey_T;
        case XK_u: case XK_U: return ImGuiKey_U;
        case XK_v: case XK_V: return ImGuiKey_V;
        case XK_w: case XK_W: return ImGuiKey_W;
        case XK_x: case XK_X: return ImGuiKey_X;
        case XK_y: case XK_Y: return ImGuiKey_Y;
        case XK_z: case XK_Z: return ImGuiKey_Z;
        case XK_0: return ImGuiKey_0;
        case XK_1: return ImGuiKey_1;
        case XK_2: return ImGuiKey_2;
        case XK_3: return ImGuiKey_3;
        case XK_4: return ImGuiKey_4;
        case XK_5: return ImGuiKey_5;
        case XK_6: return ImGuiKey_6;
        case XK_7: return ImGuiKey_7;
        case XK_8: return ImGuiKey_8;
        case XK_9: return ImGuiKey_9;
        case XK_Shift_L: return ImGuiKey_LeftShift;
        case XK_Shift_R: return ImGuiKey_RightShift;
        case XK_Control_L: return ImGuiKey_LeftCtrl;
        case XK_Control_R: return ImGuiKey_RightCtrl;
        case XK_Alt_L: return ImGuiKey_LeftAlt;
        case XK_Alt_R: return ImGuiKey_RightAlt;
        case XK_Super_L: return ImGuiKey_LeftSuper;
        case XK_Super_R: return ImGuiKey_RightSuper;
        default: return ImGuiKey_None;
    }
}

static const char* ImGui_ImplX11_GetClipboardText(ImGuiContext*) {
    if (!bd) return nullptr;
    static std::string clip;
    char* cb = XFetchBytes(nullptr, nullptr);
    if (cb) {
        clip = cb;
        XFree(cb);
    }
    return clip.c_str();
}

static void ImGui_ImplX11_SetClipboardText(ImGuiContext*, const char* text) {
    if (bd) {
        XStoreBytes(nullptr, (const char*)text, (int)strlen(text));
    }
}

void ImGui_ImplX11_SetMenuActive(bool active) {
    if (bd) bd->menuActive = active;
}

bool ImGui_ImplX11_Init(WindowX11* window) {
    bd = new X11BackendData();
    bd->window = window;

    ImGuiIO& io = ImGui::GetIO();
    io.BackendPlatformName = "imgui_impl_x11";
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;

    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
    platform_io.Platform_GetClipboardTextFn = ImGui_ImplX11_GetClipboardText;
    platform_io.Platform_SetClipboardTextFn = ImGui_ImplX11_SetClipboardText;

    return true;
}

void ImGui_ImplX11_Shutdown() {
    ImGuiIO& io = ImGui::GetIO();
    io.BackendPlatformName = nullptr;
    delete bd;
    bd = nullptr;
}

void ImGui_ImplX11_NewFrame() {
    ImGuiIO& io = ImGui::GetIO();
    if (!bd) return;

    io.DisplaySize = ImVec2((float)bd->window->width(), (float)bd->window->height());

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double currentTime = ts.tv_sec + ts.tv_nsec * 1e-9;
    io.DeltaTime = bd->time > 0 ? (float)(currentTime - bd->time) : (float)(1.0f / 60.0f);
    bd->time = currentTime;

    // Sync keyboard state from WindowX11
    // Map of XK keysym -> ImGuiKey for keys that can be held
    struct { KeySym sym; ImGuiKey key; } keyMap[] = {
        {XK_w, ImGuiKey_W}, {XK_a, ImGuiKey_A}, {XK_s, ImGuiKey_S}, {XK_d, ImGuiKey_D},
        {XK_q, ImGuiKey_Q}, {XK_e, ImGuiKey_E}, {XK_p, ImGuiKey_P},
        {XK_space, ImGuiKey_Space},
        {XK_Shift_L, ImGuiKey_LeftShift}, {XK_Shift_R, ImGuiKey_RightShift},
        {XK_Control_L, ImGuiKey_LeftCtrl}, {XK_Control_R, ImGuiKey_RightCtrl},
        {XK_Escape, ImGuiKey_Escape}, {XK_Return, ImGuiKey_Enter},
        {XK_Up, ImGuiKey_UpArrow}, {XK_Down, ImGuiKey_DownArrow},
        {XK_Left, ImGuiKey_LeftArrow}, {XK_Right, ImGuiKey_RightArrow},
        {XK_Tab, ImGuiKey_Tab}, {XK_BackSpace, ImGuiKey_Backspace},
        {XK_0, ImGuiKey_0}, {XK_1, ImGuiKey_1}, {XK_2, ImGuiKey_2},
        {XK_3, ImGuiKey_3}, {XK_4, ImGuiKey_4}, {XK_5, ImGuiKey_5},
        {XK_6, ImGuiKey_6}, {XK_7, ImGuiKey_7}, {XK_8, ImGuiKey_8}, {XK_9, ImGuiKey_9},
    };
    for (const auto& km : keyMap) {
        if (bd->window->isKeyDown(km.sym))
            io.AddKeyEvent(km.key, true);
    }

    // Query mouse position from X11 (works even when not grabbed)
    ::Window rootRet, childRet;
    int rootX, rootY, winX, winY;
    unsigned int mask;
    if (XQueryPointer(bd->window->display(), bd->window->xwindow(),
                      &rootRet, &childRet, &rootX, &rootY, &winX, &winY, &mask)) {
        io.AddMousePosEvent((float)winX, (float)winY);
        io.AddMouseButtonEvent(0, mask & Button1Mask);
        io.AddMouseButtonEvent(1, mask & Button3Mask);
    }
}

bool ImGui_ImplX11_ProcessEvent(void* xev) {
    XEvent* ev = (XEvent*)xev;
    ImGuiIO& io = ImGui::GetIO();
    if (!bd) return false;

    // When menu is active, consume mouse/keyboard events so they don't
    // trigger mouse grab or interfere with game state.
    bool consume = bd->menuActive;

    switch (ev->type) {
        case KeyPress: {
            char buf[8] = {};
            KeySym ks;
            XLookupString(&ev->xkey, buf, sizeof(buf), &ks, nullptr);
            ImGuiKey key = keysymToImGui(ks);
            if (key != ImGuiKey_None)
                io.AddKeyEvent(key, true);
            if (buf[0])
                io.AddInputCharacter((unsigned int)(unsigned char)buf[0]);
            return consume;
        }
        case KeyRelease: {
            KeySym ks = XLookupKeysym(&ev->xkey, 0);
            ImGuiKey key = keysymToImGui(ks);
            if (key != ImGuiKey_None)
                io.AddKeyEvent(key, false);
            return consume;
        }
        case ButtonPress: {
            if (ev->xbutton.button == Button1) io.AddMouseButtonEvent(0, true);
            if (ev->xbutton.button == Button3) io.AddMouseButtonEvent(1, true);
            if (ev->xbutton.button == Button4) io.AddMouseWheelEvent(0.0f, 1.0f);
            if (ev->xbutton.button == Button5) io.AddMouseWheelEvent(0.0f, -1.0f);
            io.AddMousePosEvent((float)ev->xbutton.x, (float)ev->xbutton.y);
            return consume;
        }
        case ButtonRelease: {
            if (ev->xbutton.button == Button1) io.AddMouseButtonEvent(0, false);
            if (ev->xbutton.button == Button3) io.AddMouseButtonEvent(1, false);
            io.AddMousePosEvent((float)ev->xbutton.x, (float)ev->xbutton.y);
            return consume;
        }
        case MotionNotify: {
            io.AddMousePosEvent((float)ev->xmotion.x, (float)ev->xmotion.y);
            return consume;
        }
    }
    return false;
}
