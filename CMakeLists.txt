cmake_minimum_required(VERSION 3.14)

# Point CMake directly to nvcc on Linux/Jetson (from reference)
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# Set target CUDA architecture (87 is for Jetson Orin. Change to 72, 80, 86, etc. depending on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 87)

project(nanosam LANGUAGES CXX CUDA)

# C++ standard (nanosam.vcxproj explicitly required 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output exe into build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# QoL: compiler flags (from reference)
add_compile_options(-O3 -fno-omit-frame-pointer)

# Let the loader look next to the binary as well for shared libraries
set(CMAKE_BUILD_RPATH   "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV ${OpenCV_VERSION} at: ${OpenCV_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})

# ---- 1) CUDA ----
find_package(CUDAToolkit REQUIRED)
include_directories("/usr/local/cuda/include")

# ---- 2) TensorRT & cuDNN ----
# The reference file uses aarch64 paths, common for Jetson/ARM systems. 
# (If you are on x86_64, you can change this to /usr/include/x86_64-linux-gnu)
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")

include_directories(${TENSORRT_INCLUDE_DIR})

# Find all the specific TensorRT/NVIDIA libraries required by nanosam
find_library(NVINFER_LIB NAMES nvinfer HINTS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(NVINFER_PLUGIN_LIB NAMES nvinfer_plugin HINTS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(NVONNXPARSER_LIB NAMES nvonnxparser HINTS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
# find_library(NVPARSERS_LIB NAMES nvparsers HINTS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(CUDNN_LIB NAMES cudnn HINTS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)

message(STATUS "Found TensorRT core: ${NVINFER_LIB}")

# ---- 3) Sources ----
# Add current directory to includes so "nanosam/..." headers resolve correctly
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME}
    main.cpp
    nanosam/nanosam.cpp
    nanosam/trt_module.cpp
)

# ---- 4) Link ----
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${NVONNXPARSER_LIB}
    # ${NVPARSERS_LIB}
    ${CUDNN_LIB}
    CUDA::cublas
    CUDA::cuda_driver
    CUDA::cudart
)

# Apply NDEBUG definition for release build optimizations
target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)