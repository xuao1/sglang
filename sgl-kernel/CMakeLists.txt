cmake_minimum_required(VERSION 3.18)
project(sgl-kernel LANGUAGES CXX CUDA)

# Basic settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

find_package(Torch REQUIRED)

# Warp Reduce library
add_library(warp_reduce SHARED
    src/sgl-kernel/csrc/warp_reduce.cc
    src/sgl-kernel/csrc/warp_reduce_kernel.cu
)

target_include_directories(warp_reduce
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sgl-kernel/csrc
        ${CUDA_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(warp_reduce
    PRIVATE
        ${TORCH_LIBRARIES}
        Python3::Python
)

# TRT Reduce library
add_library(trt_reduce SHARED
    src/sgl-kernel/csrc/trt_reduce.cc
    src/sgl-kernel/csrc/trt_reduce_internal.cu
    src/sgl-kernel/csrc/trt_reduce_kernel.cu
)

target_include_directories(trt_reduce
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sgl-kernel/csrc
        ${CUDA_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(trt_reduce
    PRIVATE
        ${TORCH_LIBRARIES}
        Python3::Python
)

# Set common properties for both libraries
foreach(target warp_reduce trt_reduce)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        PREFIX ""
        SUFFIX ".so"
    )
endforeach()
