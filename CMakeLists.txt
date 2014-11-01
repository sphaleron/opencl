cmake_minimum_required (VERSION 2.8)
project (OpenCL-query)

add_library(openclutils opencl_utils.c)
add_executable(query query.c)
add_executable(mandelbrot mandelbrot.c)

# Could try to check if we have the library, but for personal use this is fine.
# Maybe do this once CMake distribution has FindOpenCL module.
# This is not an CMake exercise, after all.
target_link_libraries(openclutils OpenCL)
target_link_libraries(query openclutils)
target_link_libraries(mandelbrot openclutils)

# Clang defaults to gnu11, do that with gcc as well
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11")
endif()