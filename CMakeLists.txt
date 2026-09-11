cmake_minimum_required(VERSION 3.22)

project(cuDepthFusion VERSION 0.1.0 LANGUAGES CXX)

option(CUDEPTHFUSION_ENABLE_CUDA "Build CUDA support" OFF)
option(CUDEPTHFUSION_BUILD_TESTS "Build the C++ test suite" ${PROJECT_IS_TOP_LEVEL})
option(CUDEPTHFUSION_BUILD_PYTHON "Build the pybind11 extension module" OFF)
option(CUDEPTHFUSION_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(CUDEPTHFUSION_SANITIZE "Build host code with AddressSanitizer and UBSan" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CUDEPTHFUSION_ENABLE_CUDA)
  # Default to the project's target matrix rather than one developer's GPU:
  # 87 = Jetson Orin (Nano Super), 89 = Ada laptop GPUs (RTX 4070/4090 Laptop).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "87;89")
    message(STATUS "cuDepthFusion: CMAKE_CUDA_ARCHITECTURES not set, using ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  find_package(CUDAToolkit REQUIRED)
endif()

if(CUDEPTHFUSION_SANITIZE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
                      $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
  add_link_options(-fsanitize=address,undefined)
endif()

function(cudepthfusion_set_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion>)
  if(CUDEPTHFUSION_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endfunction()

add_library(cudepthfusion STATIC
  src/build_info.cpp
  src/config.cpp
  src/engine.cpp
  src/geometry.cpp
  src/types.cpp
  src/cpu/measurement.cpp
)
add_library(cudepthfusion::cudepthfusion ALIAS cudepthfusion)
target_include_directories(cudepthfusion
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(cudepthfusion PUBLIC cxx_std_17)
set_target_properties(cudepthfusion PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(cudepthfusion PRIVATE
  CUDEPTHFUSION_VERSION="${PROJECT_VERSION}"
  CUDEPTHFUSION_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  CUDEPTHFUSION_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)
cudepthfusion_set_warnings(cudepthfusion)

if(CUDEPTHFUSION_ENABLE_CUDA)
  target_sources(cudepthfusion PRIVATE src/cuda/device_info.cu)
  # A ";" inside a compile definition would split it into two definitions.
  string(REPLACE ";" "," cudepthfusion_cuda_archs "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(cudepthfusion PRIVATE
    CUDEPTHFUSION_WITH_CUDA=1
    CUDEPTHFUSION_CUDA_COMPILER="${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}"
    CUDEPTHFUSION_CUDA_ARCHITECTURES="${cudepthfusion_cuda_archs}"
  )
  # Static runtime: the Python extension then needs no libcudart on the loader path.
  target_link_libraries(cudepthfusion PRIVATE CUDA::cudart_static)
endif()

if(CUDEPTHFUSION_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core MODULE bindings/module.cpp)
  target_link_libraries(_core PRIVATE cudepthfusion::cudepthfusion)
  cudepthfusion_set_warnings(_core)
  install(TARGETS _core LIBRARY DESTINATION cudepthfusion)
endif()

if(CUDEPTHFUSION_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
