CXX      := /usr/bin/g++
CXXFLAGS := -std=c++17 -O3 -fopenmp -march=native -I/usr/include
LDFLAGS  := -fopenmp -lX11 -lXext

NVCC     := /usr/local/cuda-12.8/bin/nvcc
NVCCFLAGS:= -O3 -arch=sm_75 -std=c++17 --compiler-bindir=/usr/bin
# RTX 2080 Ti = sm_75 (Turing)

TARGET_CPU  := voxel_rt
TARGET_GPU  := voxel_rt_gpu

# Dear ImGui sources
IMGUI_DIR   := imgui
IMGUI_SRCS  := $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_draw.cpp \
               $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp

# Backend sources
BACKEND_SRCS := imgui_impl_x11.cpp imgui_impl_fb.cpp

# CPU objects
SRCS_CPU    := main.cpp $(IMGUI_SRCS) $(BACKEND_SRCS)
OBJS_CPU    := $(SRCS_CPU:.cpp=.o)

.PHONY: all clean run fast quality gpu gpuclean gpurun

all: $(TARGET_CPU) $(TARGET_GPU)

# ===== CPU 编译 =====
$(TARGET_CPU): $(OBJS_CPU)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(IMGUI_DIR)/%.o: $(IMGUI_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ===== GPU 编译 (CUDA) =====
$(TARGET_GPU): main.cu cuda_renderer.cuh cuda_voxel_grid.cuh cuda_vec3.cuh $(IMGUI_SRCS) $(BACKEND_SRCS)
	$(NVCC) $(NVCCFLAGS) --compiler-bindir=/usr/bin -o $@ \
	    main.cu $(IMGUI_SRCS) $(BACKEND_SRCS) -lX11 -lXext -lcudart

# ===== 快捷命令 =====
run: $(TARGET_CPU)
	./$(TARGET_CPU)

gpu: $(TARGET_GPU)
	./$(TARGET_GPU)

fast: CXXFLAGS += -DSAMPLES=4 -DMAX_BOUNCES=1
fast: clean all run

quality: CXXFLAGS += -DSAMPLES=16 -DMAX_BOUNCES=3
quality: clean all run

# ===== GPU 性能测试 =====
bench: $(TARGET_GPU)
	@echo "=== GPU Bench 800x600 8spp 3bounces ==="
	@CUDA_VISIBLE_DEVICES=0 ./$(TARGET_GPU) 2>&1 | head -5

clean:
	rm -f $(OBJS_CPU) $(TARGET_CPU) $(TARGET_GPU) *.ppm *.png

gpuclean:
	rm -f $(TARGET_GPU) *.ppm *.png
