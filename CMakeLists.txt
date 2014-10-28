cmake_minimum_required (VERSION 2.8)
project (OpenCL-query)
add_executable(query query.c opencl_utils.c)

# Could try to check if we have the library, but for personal use this is fine.
# Maybe do this once CMake distribution has FindOpenCL module.
# This is not an CMake exercise, after all.
target_link_libraries(query OpenCL)

# Clang defaults to gnu11, do that with gcc as well
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11")
endif()