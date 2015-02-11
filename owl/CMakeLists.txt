include_directories(${CMAKE_CURRENT_BINARY_DIR})

# TODO make a script out if this sed magic. Add null terminator just in case.
# Would it be better to create a short binary replacing xxd?
add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/owl_fft.cl.hex
   COMMAND xxd -i ${CMAKE_CURRENT_SOURCE_DIR}/owl_fft.cl ${CMAKE_CURRENT_BINARY_DIR}/owl_fft.cl.hex
   COMMAND sed -i -e 's/\\w*owl_fft_cl/owl_fft_cl/' ${CMAKE_CURRENT_BINARY_DIR}/owl_fft.cl.hex
   COMMAND sed -i -e 's/unsigned char/static const char/' ${CMAKE_CURRENT_BINARY_DIR}/owl_fft.cl.hex
   COMMAND sed -i -e 's/unsigned int/static const size_t/' ${CMAKE_CURRENT_BINARY_DIR}/owl_fft.cl.hex
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/owl_fft.cl
)


add_library(owl owl_fft.c owl_opencl.c ${CMAKE_CURRENT_BINARY_DIR}/owl_fft.cl.hex)

target_link_libraries(owl openclutils OpenCL)
