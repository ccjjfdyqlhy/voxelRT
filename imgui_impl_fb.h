#pragma once
#include "imgui/imgui.h"

bool ImGui_ImplFB_Init(int width, int height, void* framebuffer);
void ImGui_ImplFB_Shutdown();
void ImGui_ImplFB_NewFrame();
void ImGui_ImplFB_RenderDrawData(ImDrawData* drawData);
void ImGui_ImplFB_UpdateFramebuffer(int width, int height, void* framebuffer);
