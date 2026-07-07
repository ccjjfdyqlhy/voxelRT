# Voxel Ray Tracer

Zero-dependency voxel ray tracer with CPU (OpenMP) and GPU (CUDA) backends.
Outputs live 1600×900 X11 window with real-time FPS overlay.

## Quick start

```bash
make          # build both backends
make run      # CPU version
make gpu      # GPU version
```

- **WASD** — move, **Mouse** — look, **Space/Shift** — up/down
- **P** — save screenshot (1600×900 16spp 3bounces)
- **ESC** — quit

## Backends

| Backend | File | Render | Notes |
|---------|------|--------|-------|
| CPU | `main.cpp` + `renderer.h` | 800×450 → bilinear upscale | OpenMP 12-thread |
| GPU | `main.cu` + `cuda_renderer.cuh` | 1600×900 native | Adaptive 1–8 spp, pre-allocated resources |

Both share `window.h`, `camera.h`, `voxel_grid.h`, `vec3.h`, `ray.h`.

## Architecture

```
main.cpp/.cu     → scene + main loop
renderer.h       → CPU path tracer (OpenMP)
cuda_renderer.cuh→ GPU path tracer (CUDA kernels)
window.h         → X11 window + keyboard/mouse + text overlay
camera.h         → perspective camera (yaw/pitch)
voxel_grid.h     → dense voxel grid + DDA raycast
shader_binding.h → BRDF / atmosphere / tonemap / GI
```

## Adaptive quality

| Idle | SPP×Bounces | FPS |
|------|-------------|-----|
| <0.1s (moving) | 1×1 | ~40 |
| 0.1–0.5s | 2×1 | ~24 |
| 0.5–2.0s | 4×2 | ~10 |
| >2.0s | 8×3 | ~5 |

## Requirements

- Linux with X11, OpenMP
- CUDA 12.x + NVIDIA GPU (sm_75+, e.g. RTX 2080 Ti)
