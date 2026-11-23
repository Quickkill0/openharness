# OpenHarness engine: a llama.cpp derivative for running large MoE models with
# experts streamed from NVMe. This file is the build entry point; the vendored
# upstream tree lives in llama.cpp/ (see UPSTREAM.json and PATCHES.md).

cmake_minimum_required(VERSION 3.21)

# Remote RPC devices (ggml-rpc) open outbound TCP connections to other hosts,
# which the engine never does. Refuse a configure that asks for them, under the
# current name or upstream's deprecated LLAMA_RPC alias, before anything else
# runs; both are forced off below.
foreach (oh_rpc_opt GGML_RPC LLAMA_RPC)
    if (${oh_rpc_opt})
        message(FATAL_ERROR "openharness-engine: ${oh_rpc_opt}=ON is not supported. "
                            "Remote RPC devices connect to other hosts, and the engine makes no outbound connections.")
    endif()
endforeach()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/UPSTREAM.json" OH_UPSTREAM_JSON)
string(JSON OH_LLAMA_TAG    GET "${OH_UPSTREAM_JSON}" tag)
string(JSON OH_LLAMA_BUILD  GET "${OH_UPSTREAM_JSON}" build_number)
string(JSON OH_LLAMA_COMMIT GET "${OH_UPSTREAM_JSON}" commit)

set(OH_ENGINE_VERSION "0.1.0")

project(openharness-engine VERSION ${OH_ENGINE_VERSION} LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(OH_ENGINE_BUILD_TESTS "openharness-engine: build the engine tests" OFF)
# Upstream's own test suite. Off by default with MSVC: at b12192 two upstream
# test files (test-batch-alloc, test-server-component) do not compile with
# MSVC 14.51. The Linux CI job builds and runs them.
if (MSVC)
    set(OH_UPSTREAM_TESTS_DEFAULT OFF)
else()
    set(OH_UPSTREAM_TESTS_DEFAULT ON)
endif()
option(OH_ENGINE_BUILD_UPSTREAM_TESTS "openharness-engine: also build llama.cpp's tests" ${OH_UPSTREAM_TESTS_DEFAULT})

# Single self-contained executable by default (CUDA runtime DLLs still ship
# next to it). Must be decided before llama.cpp declares its own default.
option(BUILD_SHARED_LIBS "build shared libraries" OFF)

# The vendored tree has no .git, so tell llama.cpp which upstream build it is
# instead of letting build-info.cmake pick up the OpenHarness repository's
# commit.
set(LLAMA_BUILD_NUMBER ${OH_LLAMA_BUILD})
string(SUBSTRING "${OH_LLAMA_COMMIT}" 0 8 LLAMA_BUILD_COMMIT)

# Fixed product choices. These are forced so a stale cache cannot turn them
# back on: no embedded web UI (OPENHARNESS_NO_UI also stops a stale dist/
# from being embedded), no prebuilt-UI download at build time, no
# llama-connect tunnel, no remote RPC devices (GGML_RPC; --rpc is refused at
# argument parsing too), no ffmpeg helper (MTMD_VIDEO launches ffmpeg/ffprobe
# from PATH for video and some image input) and no TLS library. Turning off TLS
# alone does not stop plain-HTTP downloads; OPENHARNESS_NO_REMOTE below does
# that. The MCP proxy, MCP servers, built-in tools and --connect are refused at
# argument parsing in every build, and their routes are not registered.
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI        OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_CONNECT         OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_BORINGSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_LIBRESSL  OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL   OFF CACHE BOOL "" FORCE)
set(LLAMA_TESTS_INSTALL   OFF CACHE BOOL "" FORCE)
set(MTMD_VIDEO            OFF CACHE BOOL "" FORCE)
set(GGML_RPC              OFF CACHE BOOL "" FORCE)
set(LLAMA_RPC             OFF CACHE BOOL "" FORCE)
set(OPENHARNESS_NO_UI     ON  CACHE BOOL "" FORCE)
if (OH_ENGINE_BUILD_TESTS AND OH_ENGINE_BUILD_UPSTREAM_TESTS)
    set(LLAMA_BUILD_TESTS ON CACHE BOOL "" FORCE)
else()
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
endif()

# EXCLUDE_FROM_ALL keeps upstream's tools (llama-server included) out of the
# default build and drops upstream's install rules, so `cmake --install`
# installs only the files listed below. Targets we link still build as
# dependencies, and llama-server stays buildable by name for comparisons.
add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)

# Engine-wide no-download rule. common_models_handler_init/apply refuse model,
# draft and projector URLs and Hugging Face or Docker repos before resolving
# anything, and common_http_client refuses to open an outbound connection at
# all (covers router /models requests). Remote media URLs in chat requests are
# refused before any request is made. See PATCHES.md.
target_compile_definitions(llama-common PUBLIC OPENHARNESS_NO_REMOTE)

# Test builds only: lets server_smoke_test.py hold the model load so it can
# observe the "loading" 503 deterministically (OPENHARNESS_TEST_LOAD_GATE).
if (OH_ENGINE_BUILD_TESTS)
    target_compile_definitions(llama-server-impl PRIVATE OPENHARNESS_TEST_HOOKS)
endif()

if (LLAMA_BUILD_TESTS)
    # Upstream test executables are excluded from ALL with the rest of the
    # subdirectory; build them by default when their tests are enabled.
    get_property(OH_UPSTREAM_TEST_TARGETS DIRECTORY llama.cpp/tests PROPERTY BUILDSYSTEM_TARGETS)
    add_custom_target(oh-upstream-tests ALL)
    add_dependencies(oh-upstream-tests ${OH_UPSTREAM_TEST_TARGETS})
endif()

if (OH_ENGINE_BUILD_TESTS)
    enable_testing()
endif()

# InferSSD-derived SSD expert-streaming modules (Apache-2.0).
add_subdirectory(inferssd)

add_executable(openharness-engine-server server/main.cpp)
target_link_libraries(openharness-engine-server PRIVATE llama-server-impl)
target_compile_features(openharness-engine-server PRIVATE cxx_std_17)
target_compile_definitions(openharness-engine-server PRIVATE
    OH_ENGINE_VERSION="${OH_ENGINE_VERSION}"
    OH_LLAMA_TAG="${OH_LLAMA_TAG}"
    OH_LLAMA_COMMIT="${OH_LLAMA_COMMIT}"
    $<$<TARGET_EXISTS:ggml-cuda>:OH_VRAM_TIER> # inferssd/CMakeLists.txt compiles the tier into ggml-cuda
)
install(TARGETS openharness-engine-server RUNTIME)
install(FILES NOTICE
              LICENSES/llama.cpp-MIT.txt
              LICENSES/InferSSD-Apache-2.0.txt
              LICENSES/InferSSD-NOTICE.txt
              LICENSES/THIRD-PARTY-NOTICES.txt
        DESTINATION share/openharness-engine)

if (OH_ENGINE_BUILD_TESTS)
    add_subdirectory(tests)
endif()
