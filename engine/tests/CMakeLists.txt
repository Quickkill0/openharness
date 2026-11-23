# OpenHarness engine tests. Registered with CTest under the label "engine";
# run them with `ctest --test-dir <build> -L engine`.

find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)

set(OH_TEST_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/models")
file(MAKE_DIRECTORY "${OH_TEST_MODEL_DIR}")

# Tiny random-weight models with a dummy tokenizer, written by upstream's
# test-llama-archs. Fixed seed so the files are identical run to run. Build
# the tool ourselves when upstream's tests are not part of this build.
if (TARGET test-llama-archs)
    set(OH_LLAMA_ARCHS test-llama-archs)
else()
    set(OH_LLAMA_ARCHS oh-test-llama-archs)
    add_executable(oh-test-llama-archs "${PROJECT_SOURCE_DIR}/llama.cpp/tests/test-llama-archs.cpp")
    target_compile_features(oh-test-llama-archs PRIVATE cxx_std_17)
    target_link_libraries(oh-test-llama-archs PRIVATE llama-common)
endif()
add_test(NAME oh-generate-test-models
         COMMAND ${OH_LLAMA_ARCHS} -a llama -s 42 -o "${OH_TEST_MODEL_DIR}")
set_tests_properties(oh-generate-test-models PROPERTIES
    LABELS engine
    TIMEOUT 120
    FIXTURES_SETUP oh-test-models)

add_test(NAME oh-engine-info
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/engine_info_test.py"
                 --server "$<TARGET_FILE:openharness-engine-server>"
                 --vram-tier "$<TARGET_EXISTS:ggml-cuda>")
set_tests_properties(oh-engine-info PROPERTIES LABELS engine TIMEOUT 60)

add_test(NAME oh-server-smoke
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/server_smoke_test.py"
                 --server "$<TARGET_FILE:openharness-engine-server>"
                 --model "${OH_TEST_MODEL_DIR}/llama-dense.gguf")
set_tests_properties(oh-server-smoke PROPERTIES
    LABELS engine
    TIMEOUT 180
    FIXTURES_REQUIRED oh-test-models)

add_test(NAME oh-vendor-script
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/vendor_script_test.py")
set_tests_properties(oh-vendor-script PROPERTIES LABELS engine TIMEOUT 60)

add_test(NAME oh-no-remote
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/no_remote_test.py"
                 --server "$<TARGET_FILE:openharness-engine-server>"
                 --model "${OH_TEST_MODEL_DIR}/llama-dense.gguf")
set_tests_properties(oh-no-remote PROPERTIES
    LABELS engine
    TIMEOUT 300
    FIXTURES_REQUIRED oh-test-models)

add_test(NAME oh-ui-no-assets
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/ui_assets_test.py"
                 --cmake "${CMAKE_COMMAND}")
set_tests_properties(oh-ui-no-assets PROPERTIES LABELS engine TIMEOUT 120)

add_test(NAME oh-rpc-configure
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/rpc_configure_test.py"
                 --cmake "${CMAKE_COMMAND}" --generator "${CMAKE_GENERATOR}")
set_tests_properties(oh-rpc-configure PROPERTIES LABELS engine TIMEOUT 120)

add_test(NAME oh-install
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/install_test.py"
                 --cmake "${CMAKE_COMMAND}" --build-dir "${PROJECT_BINARY_DIR}" --config "$<CONFIG>")
set_tests_properties(oh-install PROPERTIES LABELS engine TIMEOUT 120)

# SSD expert streaming on a tiny synthetic MoE (tests/make_moe_fixture.py), CPU
# only: resident and streamed runs give identical tokens and logits, and the
# host budget reserves KV, compute buffers and headroom before the expert cache.
add_executable(oh-ssd-stream-test ssd_stream_test.cpp)
target_compile_features(oh-ssd-stream-test PRIVATE cxx_std_17)
target_link_libraries(oh-ssd-stream-test PRIVATE llama)
target_include_directories(oh-ssd-stream-test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/src" "${CMAKE_CURRENT_SOURCE_DIR}/../inferssd/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../inferssd/ggml/src")
if (WIN32)
    target_link_libraries(oh-ssd-stream-test PRIVATE psapi)
endif()
foreach(type f16 q8_0)
    add_test(NAME oh-ssd-fixture-${type}
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/make_moe_fixture.py"
                     --out "${OH_TEST_MODEL_DIR}" --expert-type ${type})
    set_tests_properties(oh-ssd-fixture-${type} PROPERTIES
        LABELS engine TIMEOUT 60 FIXTURES_SETUP oh-ssd-fixture-${type})
    add_test(NAME oh-ssd-stream-equal-${type}
             COMMAND oh-ssd-stream-test equal "${OH_TEST_MODEL_DIR}/ssd-moe-${type}.gguf")
    set_tests_properties(oh-ssd-stream-equal-${type} PROPERTIES
        LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-${type})
endforeach()
add_test(NAME oh-ssd-receipt
         COMMAND oh-ssd-stream-test receipt -)
set_tests_properties(oh-ssd-receipt PROPERTIES LABELS "engine;ssd" TIMEOUT 30)

add_test(NAME oh-ssd-vram-tier-pairs
         COMMAND oh-ssd-stream-test tier-pairs "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-vram-tier-pairs PROPERTIES
    LABELS "engine;ssd" TIMEOUT 60 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-ssd-context-budget
         COMMAND oh-ssd-stream-test context "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-context-budget PROPERTIES
    LABELS "engine;ssd" TIMEOUT 180 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-ssd-admit-before-alloc
         COMMAND oh-ssd-stream-test admit "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-admit-before-alloc PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-ssd-refuse-huge-context
         COMMAND oh-ssd-stream-test huge "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-refuse-huge-context PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-ssd-batch-budget
         COMMAND oh-ssd-stream-test batch "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-batch-budget PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-ssd-sampler-budget
         COMMAND oh-ssd-stream-test sampler "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-sampler-budget PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f16)

if (WIN32) # Linux counts physical pages, and an untouched scheduler has few
    add_test(NAME oh-ssd-sched-commit
             COMMAND oh-ssd-stream-test sched "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
    set_tests_properties(oh-ssd-sched-commit PROPERTIES
        LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f16)
endif()

add_test(NAME oh-ssd-output-charge
         COMMAND oh-ssd-stream-test outputs "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-output-charge PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-ssd-cell-charge
         COMMAND oh-ssd-stream-test cells "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-cell-charge PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-ssd-borrowed-cells
         COMMAND oh-ssd-stream-test borrow "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-borrowed-cells PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-ssd-reserve-max
         COMMAND oh-ssd-stream-test reserve "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-reserve-max PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-ssd-fixture-f32
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/make_moe_fixture.py"
                 --out "${OH_TEST_MODEL_DIR}" --expert-type f32)
set_tests_properties(oh-ssd-fixture-f32 PROPERTIES
    LABELS engine TIMEOUT 60 FIXTURES_SETUP oh-ssd-fixture-f32)
add_test(NAME oh-ssd-refuse-f32
         COMMAND oh-ssd-stream-test refuse "${OH_TEST_MODEL_DIR}/ssd-moe-f32.gguf")
set_tests_properties(oh-ssd-refuse-f32 PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-f32)

add_test(NAME oh-ssd-fixture-mtp
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/make_moe_fixture.py"
                 --out "${OH_TEST_MODEL_DIR}" --expert-type f16 --mtp)
set_tests_properties(oh-ssd-fixture-mtp PROPERTIES
    LABELS engine TIMEOUT 60 FIXTURES_SETUP oh-ssd-fixture-mtp)
add_test(NAME oh-ssd-mtp-unloaded
         COMMAND oh-ssd-stream-test mtp "${OH_TEST_MODEL_DIR}/ssd-moe-mtp-f16.gguf")
set_tests_properties(oh-ssd-mtp-unloaded PROPERTIES
    LABELS "engine;ssd" TIMEOUT 180 FIXTURES_REQUIRED oh-ssd-fixture-mtp)

# GLM-5.3-Flash's shape and quant mix, small enough for CI. Python writes the F32 base and placeholders; the test
# program fills the experts and quantizes them, because the IQ types need ggml.
add_test(NAME oh-ssd-fixture-glm-base
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/make_moe_fixture.py"
                 --out "${OH_TEST_MODEL_DIR}" --glm)
set_tests_properties(oh-ssd-fixture-glm-base PROPERTIES
    LABELS engine TIMEOUT 60 FIXTURES_SETUP oh-ssd-fixture-glm-base)
add_test(NAME oh-ssd-fixture-glm
         COMMAND oh-ssd-stream-test glm-fixture "${OH_TEST_MODEL_DIR}/ssd-glm-base.gguf")
set_tests_properties(oh-ssd-fixture-glm PROPERTIES
    LABELS engine TIMEOUT 180
    FIXTURES_REQUIRED oh-ssd-fixture-glm-base FIXTURES_SETUP oh-ssd-fixture-glm)
foreach(check glm glm-admit)
    add_test(NAME oh-ssd-${check}
             COMMAND oh-ssd-stream-test ${check} "${OH_TEST_MODEL_DIR}/ssd-glm-iq.gguf")
    set_tests_properties(oh-ssd-${check} PROPERTIES
        LABELS "engine;ssd" TIMEOUT 240 FIXTURES_REQUIRED oh-ssd-fixture-glm)
endforeach()
# Touches the GPU briefly, on a tiny fixture: labelled apart so that "ctest -L engine" stays CPU-only.
if (TARGET ggml-cuda)
    foreach(fixture iq f16)
        add_test(NAME oh-ssd-vram-tier-${fixture}
                 COMMAND oh-ssd-stream-test vram "${OH_TEST_MODEL_DIR}/ssd-glm-${fixture}.gguf")
        set_tests_properties(oh-ssd-vram-tier-${fixture} PROPERTIES
            LABELS "cuda" TIMEOUT 180 FIXTURES_REQUIRED oh-ssd-fixture-glm)
    endforeach()
endif()
add_test(NAME oh-ssd-glm-mixed
         COMMAND oh-ssd-stream-test glm-mixed "${OH_TEST_MODEL_DIR}/ssd-glm-iq-mixed.gguf")
set_tests_properties(oh-ssd-glm-mixed PROPERTIES
    LABELS "engine;ssd" TIMEOUT 120 FIXTURES_REQUIRED oh-ssd-fixture-glm)

add_test(NAME oh-ssd-router-settings
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/ssd_router_test.py"
                 --server "$<TARGET_FILE:openharness-engine-server>"
                 --model "${OH_TEST_MODEL_DIR}/ssd-moe-f16.gguf")
set_tests_properties(oh-ssd-router-settings PROPERTIES
    LABELS "engine;ssd" TIMEOUT 180 FIXTURES_REQUIRED oh-ssd-fixture-f16)

add_test(NAME oh-inferssd-import
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/inferssd_import_test.py")
set_tests_properties(oh-inferssd-import PROPERTIES LABELS engine TIMEOUT 60)
