# InferSSD-derived modules (Apache-2.0), imported from wici-ai/inferssd at the
# revision recorded in ../NOTICE. Paths below mirror InferSSD's engine/ tree so
# relative includes in the imported files keep working unchanged. Changes to
# imported files are listed in ../PATCHES.md.
#
# This stage builds the backend-independent core: platform layer, read rings,
# host expert cache and budget arbiter, placement policy, settings and the
# GGUF/pack source index. The CPU sink, the llama integration and the
# scheduler stream hooks arrive in later stages.

set(OH_SSD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-ssd")

add_library(ggml-ssd INTERFACE)
target_link_libraries(ggml-ssd INTERFACE ggml-base)
target_include_directories(ggml-ssd INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src" "${OH_SSD_SRC}")

add_subdirectory(${OH_SSD_SRC}/platform)
add_subdirectory(${OH_SSD_SRC}/host)
add_subdirectory(${OH_SSD_SRC}/settings)
add_subdirectory(${OH_SSD_SRC}/io)
add_subdirectory(${OH_SSD_SRC}/store)
target_link_libraries(ggml-ssd INTERFACE ggml-ssd-platform ggml-ssd-host ggml-ssd-settings ggml-ssd-c0-store)

# The CPU sink (routed experts read through the host expert cache and the read
# ring, computed with ggml-cpu's own vec_dot kernels). As in InferSSD it is
# compiled into ggml-cpu, so it shares that target's ISA flags and internal
# headers (vec.h, quants.h). It does nothing until a context creates a sink.
if (NOT TARGET ggml-cpu)
    message(FATAL_ERROR "openharness-engine: the SSD CPU sink needs the ggml-cpu target (GGML_BACKEND_DL is not supported)")
endif()
set(OH_SSD_CPU "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/ssd")
target_sources(ggml-cpu PRIVATE "${OH_SSD_CPU}/cpu-sink.cpp" "${OH_SSD_CPU}/cpu-prefill.cpp")
target_include_directories(ggml-cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu" "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src")
target_link_libraries(ggml-cpu PRIVATE ggml-ssd)

# The llama library side: the glue that gives a context an SSD expert runtime
# (settings, original-GGUF expert source, host budget and receipt) and the
# model adapter the graph uses to accept or refuse a layer's experts.
if (NOT TARGET llama)
    message(FATAL_ERROR "openharness-engine: add llama.cpp before engine/inferssd")
endif()
target_sources(llama PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ssd/oh-runtime.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ssd/model-adapter.cpp")
target_include_directories(llama PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu")
target_link_libraries(llama PRIVATE ggml-ssd)

# The CUDA VRAM expert tier, compiled into ggml-cuda so that it gets that target's CUDA flags and internal headers.
# The imported ggml-cuda/ssd/CMakeLists.txt is not used: it assumes GCC. Grouped prefill is not built; a stub reports it
# unavailable, so prefill stays on the CPU and decode uses the generic kernels (generic-compute-adapter.cu).
if (TARGET ggml-cuda)
    set(OH_SSD_CUDA "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cuda/ssd")
    target_sources(ggml-cuda PRIVATE
        "${OH_SSD_CUDA}/vram-tier.cpp"
        "${OH_SSD_CUDA}/generic-compute-adapter.cu"
        "${OH_SSD_CUDA}/provider.cpp"
        "${OH_SSD_CUDA}/transport.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ssd/vram-group-stub.cpp")
    target_include_directories(ggml-cuda PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src")
    target_link_libraries(ggml-cuda PRIVATE ggml-ssd)
endif()

if (OH_ENGINE_BUILD_TESTS)
    add_subdirectory(tests)
endif()
