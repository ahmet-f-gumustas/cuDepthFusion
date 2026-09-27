find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

function(cudepthfusion_add_test_suite target label)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(${target} PRIVATE cudepthfusion::cudepthfusion GTest::gtest_main)
  cudepthfusion_set_warnings(${target})
  # gtest_discover_tests already maps GTEST_SKIP to CTest "Skipped" (never "Passed");
  # overriding SKIP_REGULAR_EXPRESSION here breaks its escaping.
  gtest_discover_tests(${target} PROPERTIES LABELS ${label})
endfunction()

cudepthfusion_add_test_suite(cudepthfusion_cpu_tests cpu
  test_config.cpp
  test_geometry.cpp
  test_fuse.cpp
  test_measurement.cpp
  test_reproject.cpp
  test_spatial.cpp
  test_engine.cpp
)

# Needs a CUDA build and a CUDA device; skips otherwise.
set(cudepthfusion_gpu_test_sources test_gpu_device.cpp test_parity.cpp)
if(CUDEPTHFUSION_ENABLE_CUDA)
  # Needs the CUDA runtime headers, so it only exists in a CUDA build.
  list(APPEND cudepthfusion_gpu_test_sources test_cuda_memory.cu)
endif()

cudepthfusion_add_test_suite(cudepthfusion_gpu_tests gpu ${cudepthfusion_gpu_test_sources})
