# Unit tests for the InferSSD core modules, imported from InferSSD's
# tests/ssd. Registered with CTest under the labels "engine" and "ssd".

find_package(Threads REQUIRED)
set(OH_SSD_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/ssd")
set(OH_SSD_GGML_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src")

# oh_ssd_test(<name> <source> [LIBS ...] [INCLUDES ...] [TIMEOUT n])
function(oh_ssd_test name source)
    cmake_parse_arguments(T "" "TIMEOUT" "LIBS;INCLUDES" ${ARGN})
    if (NOT T_TIMEOUT)
        set(T_TIMEOUT 60)
    endif()
    add_executable(${name} "${OH_SSD_TESTS}/${source}")
    target_compile_features(${name} PRIVATE cxx_std_17)
    target_include_directories(${name} PRIVATE "${OH_SSD_GGML_SRC}" ${T_INCLUDES})
    target_link_libraries(${name} PRIVATE Threads::Threads ${T_LIBS})
    # The tests use assert(); keep it active in Release builds.
    if (MSVC)
        target_compile_options(${name} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${name} PRIVATE -UNDEBUG -ffp-contract=off)
    endif()
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "engine;ssd" TIMEOUT ${T_TIMEOUT})
endfunction()

oh_ssd_test(ssd-test-resources       test-test-resources.cpp       LIBS ggml-ssd-platform ggml-ssd-host)
oh_ssd_test(ssd-platform-threads     test-ssd-platform-threads.cpp LIBS ggml-ssd-platform)
oh_ssd_test(ssd-platform-memory      test-ssd-platform-memory.cpp  LIBS ggml-ssd-platform)
oh_ssd_test(ssd-platform-file        test-ssd-platform-file.cpp    LIBS ggml-ssd-platform
            INCLUDES "${OH_SSD_GGML_SRC}/ggml-ssd")
oh_ssd_test(ssd-read-ring            test-ssd-read-ring.cpp        LIBS ggml-ssd-pool-ring ggml-ssd-platform
            INCLUDES "${OH_SSD_GGML_SRC}/ggml-ssd" TIMEOUT 120)
foreach(ring ggml-ssd-linux-ring ggml-ssd-win-ring)
    if (TARGET ${ring})
        target_link_libraries(ssd-read-ring PRIVATE ${ring})
    endif()
endforeach()
oh_ssd_test(ssd-vram-layer-slots     test-vram-layer-slots.cpp)
oh_ssd_test(ssd-cache-policy         test-kimi-cpu-cache-policy.cpp LIBS ggml-ssd-host)
oh_ssd_test(ssd-cache-lease          test-kimi-cache-lease.cpp     LIBS ggml-ssd-host ggml-base)
oh_ssd_test(ssd-host-lock            test-kimi-host-lock.cpp)
oh_ssd_test(ssd-runtime-policy       test-ssd-runtime-policy.cpp)
oh_ssd_test(ssd-prefill-cost         test-kimi-prefill-cost.cpp)
oh_ssd_test(ssd-pack-copy-plan       test-moe-pack-copy-plan.cpp)
oh_ssd_test(ssd-source-index         test-source-index.cpp         LIBS ggml-ssd-c0-store ggml-ssd-platform)
oh_ssd_test(ssd-gpu-contract         test-ssd-gpu-contract.cpp     LIBS ggml-base)

# The CPU sink, compiled into ggml-cpu.
oh_ssd_test(ssd-cpu-api              test-ssd-cpu-api.cpp          LIBS ggml-cpu ggml-base
            INCLUDES "${OH_SSD_GGML_SRC}/ggml-cpu")
oh_ssd_test(ssd-cpu-sink             test-kimi-cpu-sink.cpp        LIBS ggml-ssd-host ggml-ssd-settings ggml-cpu ggml-base
            INCLUDES "${OH_SSD_GGML_SRC}/ggml-cpu" "${PROJECT_SOURCE_DIR}/llama.cpp/vendor" TIMEOUT 300)

# POSIX-only fixtures (setrlimit, mkdtemp, fcntl descriptor scans), as in InferSSD.
if (UNIX)
    oh_ssd_test(ssd-settings-telemetry test-settings-telemetry.cpp LIBS ggml-ssd-settings
                INCLUDES "${OH_SSD_GGML_SRC}/ggml-ssd")
    oh_ssd_test(ssd-store-parser-failures test-store-parser-failures.cpp LIBS ggml-ssd-c0-store ggml-ssd-platform
                INCLUDES "${OH_SSD_GGML_SRC}/ggml-ssd/store")
endif()

# The arbiter over the Linux ledger (cgroup and /proc files injected), and over
# the native budget sources on Windows and macOS.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    oh_ssd_test(ssd-host-arbiter test-ssd-host-arbiter.cpp LIBS ggml-ssd-host TIMEOUT 30)
    oh_ssd_test(ssd-host-cgroup-root test-ssd-host-cgroup-root.cpp LIBS ggml-ssd-host TIMEOUT 30)
endif()
if (WIN32 OR APPLE)
    oh_ssd_test(ssd-host-budget  test-ssd-host-budget.cpp  LIBS ggml-ssd-host TIMEOUT 30)
endif()
