TARGET = emnist_dataset

CROSS_COMPILE =
#CROSS_COMPILE = /opt/ivot/aarch64-ca53-linux-gnueabihf-8.4/bin/aarch64-ca53-linux-gnu-
CXX = $(CROSS_COMPILE)g++

###debug
CXXFLAGS    = -I/opt/libtorch/x64/debug/include -I/opt/libtorch/x64/debug/include/torch/csrc/api/include -I/opt/opencv/debug/include/opencv4 -DUSE_EXTERNAL_MZCRC -DUSE_PTHREADPOOL -DNDEBUG -DUSE_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER -DNNP_CONVOLUTION_ONLY=0 -DNNP_INFERENCE_ONLY=0 -DHAVE_MALLOC_USABLE_SIZE=1 -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DGFLAGS_IS_A_DLL=0 -O0 -g -fopenmp -fPIC -Wall -fmessage-length=0 -std=c++14 -pthread
LDFLAGS     = -L/opt/libtorch/x64/debug/lib -L/opt/opencv/debug/lib -Wl,-rpath,/opt/libtorch/x64/debug/lib:/opt/opencv/debug/lib -ltorch -ltorch_cpu -lc10 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio -lpthread

###release -O3
#CXXFLAGS    = -I/opt/libtorch/x64/include -I/opt/libtorch/x64/include/torch/csrc/api/include -I/opt/opencv/include/opencv4 -DUSE_EXTERNAL_MZCRC -DUSE_PTHREADPOOL -DNDEBUG -DUSE_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER -DNNP_CONVOLUTION_ONLY=0 -DNNP_INFERENCE_ONLY=0 -DHAVE_MALLOC_USABLE_SIZE=1 -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DGFLAGS_IS_A_DLL=0 -O3 -g -fopenmp -fPIC -Wall -fmessage-length=0 -std=c++14 -pthread
#LDFLAGS     = -L/opt/libtorch/x64/lib -L/opt/opencv/lib -Wl,-rpath,/opt/libtorch/x64/lib:/opt/opencv/lib -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio -lpthread

COMPILE_DIR	= ./objs

$(COMPILE_DIR)/%.o : ./%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

SRC  = $(notdir $(wildcard ./*.cpp))
OBJ  = $(patsubst %.cpp,$(COMPILE_DIR)/%.o,$(SRC))
	
$(TARGET) : $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	
all: $(TARGET)
		
clean:
	rm -rf $(COMPILE_DIR) $(TARGET)
	