#pragma once
#include "imgui/imgui.h"

struct X11BackendData;
class WindowX11;

bool ImGui_ImplX11_Init(WindowX11* window);
void ImGui_ImplX11_Shutdown();
void ImGui_ImplX11_NewFrame();
bool ImGui_ImplX11_ProcessEvent(void* xev);
void ImGui_ImplX11_SetMenuActive(bool active);
