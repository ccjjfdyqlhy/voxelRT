#include "imgui_impl_fb.h"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>

struct FBBackendData {
    int width = 0;
    int height = 0;
    uint8_t* framebuffer = nullptr;
    uint8_t* fontTex = nullptr;
    int fontTexW = 0;
    int fontTexH = 0;
    ImTextureID fontTexID = (ImTextureID)1;
};

static FBBackendData* fb = nullptr;

static inline void blendPixel(int x, int y, ImU32 col, uint8_t alpha) {
    if (x < 0 || x >= fb->width || y < 0 || y >= fb->height) return;
    uint8_t* dst = fb->framebuffer + (y * fb->width + x) * 4;
    uint8_t sr = (col >> 16) & 0xFF;
    uint8_t sg = (col >> 8) & 0xFF;
    uint8_t sb = col & 0xFF;
    uint8_t sa = ((col >> 24) & 0xFF) * alpha / 255;
    if (sa == 255) {
        dst[0] = sb;
        dst[1] = sg;
        dst[2] = sr;
    } else if (sa > 0) {
        uint8_t inv = 255 - sa;
        dst[0] = (sb * sa + dst[0] * inv + 128) >> 8;
        dst[1] = (sg * sa + dst[1] * inv + 128) >> 8;
        dst[2] = (sr * sa + dst[2] * inv + 128) >> 8;
    }
}

static float edge(const ImVec2& a, const ImVec2& b, const ImVec2& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

static void rasterizeTriangle(const ImVec2& v0, const ImVec2& v1, const ImVec2& v2,
                               const ImVec2& uv0, const ImVec2& uv1, const ImVec2& uv2,
                               ImU32 col0, ImU32 col1, ImU32 col2,
                               bool textured, const ImVec4& clip) {
    int minX = (int)std::max(clip.x, std::min({v0.x, v1.x, v2.x}));
    int minY = (int)std::max(clip.y, std::min({v0.y, v1.y, v2.y}));
    int maxX = (int)std::min(clip.z, std::ceil(std::max({v0.x, v1.x, v2.x})));
    int maxY = (int)std::min(clip.w, std::ceil(std::max({v0.y, v1.y, v2.y})));
    if (minX >= maxX || minY >= maxY) return;

    float area = edge(v0, v1, v2);
    if (fabsf(area) < 1e-8f) return;
    float invArea = 1.0f / area;

    for (int y = minY; y < maxY; y++) {
        for (int x = minX; x < maxX; x++) {
            ImVec2 p((float)x + 0.5f, (float)y + 0.5f);
            float w0 = edge(v1, v2, p);
            float w1 = edge(v2, v0, p);
            float w2 = edge(v0, v1, p);

            if ((area > 0 && w0 >= 0 && w1 >= 0 && w2 >= 0) ||
                (area < 0 && w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                w0 *= invArea;
                w1 *= invArea;
                w2 *= invArea;

                uint8_t r = (uint8_t)(((col0 >> 16) & 0xFF) * w0 + ((col1 >> 16) & 0xFF) * w1 + ((col2 >> 16) & 0xFF) * w2);
                uint8_t g = (uint8_t)(((col0 >> 8) & 0xFF) * w0 + ((col1 >> 8) & 0xFF) * w1 + ((col2 >> 8) & 0xFF) * w2);
                uint8_t b = (uint8_t)((col0 & 0xFF) * w0 + (col1 & 0xFF) * w1 + (col2 & 0xFF) * w2);
                uint8_t a = (uint8_t)(((col0 >> 24) & 0xFF) * w0 + ((col1 >> 24) & 0xFF) * w1 + ((col2 >> 24) & 0xFF) * w2);
                ImU32 col = IM_COL32(r, g, b, a);

                if (textured && fb->fontTex) {
                    float u = uv0.x * w0 + uv1.x * w1 + uv2.x * w2;
                    float v = uv0.y * w0 + uv1.y * w1 + uv2.y * w2;
                    int tx = (int)(u * fb->fontTexW);
                    int ty = (int)(v * fb->fontTexH);
                    tx = std::max(0, std::min(fb->fontTexW - 1, tx));
                    ty = std::max(0, std::min(fb->fontTexH - 1, ty));
                    uint8_t texel = fb->fontTex[ty * fb->fontTexW + tx];
                    uint8_t alpha = ((col >> 24) & 0xFF) * texel / 255;
                    if (alpha > 0) {
                        col = (col & 0x00FFFFFF) | (alpha << 24);
                        blendPixel(x, y, col, 255);
                    }
                } else {
                    blendPixel(x, y, col, 255);
                }
            }
        }
    }
}

bool ImGui_ImplFB_Init(int width, int height, void* framebuffer) {
    fb = new FBBackendData();
    fb->width = width;
    fb->height = height;
    fb->framebuffer = (uint8_t*)framebuffer;

    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "imgui_impl_fb";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

    // Build font atlas (legacy path)
    unsigned char* pixels;
    io.Fonts->GetTexDataAsAlpha8(&pixels, &fb->fontTexW, &fb->fontTexH);
    fb->fontTex = new uint8_t[fb->fontTexW * fb->fontTexH];
    memcpy(fb->fontTex, pixels, fb->fontTexW * fb->fontTexH);
    io.Fonts->SetTexID(fb->fontTexID);

    return true;
}

void ImGui_ImplFB_Shutdown() {
    if (fb) {
        ImGuiIO& io = ImGui::GetIO();
        io.Fonts->SetTexID(ImTextureID_Invalid);
        delete[] fb->fontTex;
        delete fb;
        fb = nullptr;
    }
}

void ImGui_ImplFB_NewFrame() {
}

void ImGui_ImplFB_UpdateFramebuffer(int width, int height, void* framebuffer) {
    if (!fb) return;
    fb->width = width;
    fb->height = height;
    fb->framebuffer = (uint8_t*)framebuffer;
}

void ImGui_ImplFB_RenderDrawData(ImDrawData* drawData) {
    if (!fb || !fb->framebuffer) return;
    if (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f) return;

    for (int n = 0; n < drawData->CmdListsCount; n++) {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        const ImDrawVert* vtx = cmdList->VtxBuffer.Data;
        const ImDrawIdx* idx = cmdList->IdxBuffer.Data;

        for (int cmdi = 0; cmdi < cmdList->CmdBuffer.Size; cmdi++) {
            const ImDrawCmd* pcmd = &cmdList->CmdBuffer[cmdi];

            ImVec4 clip;
            clip.x = pcmd->ClipRect.x - drawData->DisplayPos.x;
            clip.y = pcmd->ClipRect.y - drawData->DisplayPos.y;
            clip.z = pcmd->ClipRect.z - drawData->DisplayPos.x;
            clip.w = pcmd->ClipRect.w - drawData->DisplayPos.y;

            clip.x = std::max(clip.x, 0.0f);
            clip.y = std::max(clip.y, 0.0f);
            clip.z = std::min(clip.z, (float)fb->width);
            clip.w = std::min(clip.w, (float)fb->height);
            if (clip.x >= clip.z || clip.y >= clip.w) continue;

            ImTextureID texID = pcmd->GetTexID();
            bool textured = (texID != ImTextureID_Invalid);

            for (unsigned int i = 0; i + 3 <= pcmd->ElemCount; i += 3) {
                unsigned int i0 = idx[pcmd->IdxOffset + i + 0];
                unsigned int i1 = idx[pcmd->IdxOffset + i + 1];
                unsigned int i2 = idx[pcmd->IdxOffset + i + 2];

                const ImDrawVert& v0 = vtx[pcmd->VtxOffset + i0];
                const ImDrawVert& v1 = vtx[pcmd->VtxOffset + i1];
                const ImDrawVert& v2 = vtx[pcmd->VtxOffset + i2];

                rasterizeTriangle(
                    v0.pos, v1.pos, v2.pos,
                    v0.uv, v1.uv, v2.uv,
                    v0.col, v1.col, v2.col,
                    textured, clip);
            }
        }
    }
}
