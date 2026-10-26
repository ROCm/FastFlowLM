# d3 compose tests (assets, box, composed requests): no NPU, no XRT, so this builds and runs on Linux CI and Windows.
#   cmake -S src/test/d3_compose -B build-d3cmp && cmake --build build-d3cmp && ctest --test-dir build-d3cmp
# -DFLM_D3_TEST_TOKENIZER=ON -DFLM_D3_TURBO_PACKAGE=<D3-Turbo-NPU2 dir> also runs the prompts through FastFlowLM's own
# Tokenizer (tokenizers-cpp, needs cargo) instead of the recorded token ids.
# -DFLM_D3_TEST_ADAPTER=ON builds test_d3_adapter (the engine adapter against a mock engine; needs the device runtime headers).
cmake_minimum_required(VERSION 3.22)
project(d3_compose_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(FLM_D3_TEST_TOKENIZER "Run with FastFlowLM's tokenizer (tokenizers-cpp)" OFF)
option(FLM_D3_TEST_ADAPTER "Build the engine adapter test (device runtime headers needed)" OFF)
set(FLM_D3_TURBO_PACKAGE "" CACHE PATH "D3-Turbo-NPU2 package directory (tokenizer.json) for the tokenizer run")

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
set(D3_COMPOSE_SOURCES
    "${FLM_SOURCE_DIR}/common/decision/py_json.cpp"
    "${FLM_SOURCE_DIR}/common/decision/d3_contract.cpp"
    "${FLM_SOURCE_DIR}/common/decision/d3_compose.cpp"
    "${FLM_SOURCE_DIR}/common/decision/d3_composed.cpp"
    # d3_contract.cpp reads the video plan (src/common/video); decoding itself is not used here
    "${FLM_SOURCE_DIR}/common/video/video_sampling.cpp"
    "${FLM_SOURCE_DIR}/common/video/video_preprocess.cpp")

add_executable(test_d3_compose test_d3_compose.cpp ${D3_COMPOSE_SOURCES})
target_include_directories(test_d3_compose PRIVATE "${FLM_SOURCE_DIR}/include")
# numpy rounds every operation: keep the compiler from fusing a * b + c (matters on aarch64)
if(NOT MSVC)
    target_compile_options(test_d3_compose PRIVATE -ffp-contract=off)
endif()

# M4 of the composed requests (box at a video moment, track, videos): the window loop through a scripted engine, and
# load_video_windows over a fake decoder (video_decode.cpp / video_input.cpp; FFmpeg itself is not loaded here)
add_executable(test_d3_video_composed test_d3_video_composed.cpp ${D3_COMPOSE_SOURCES}
    "${FLM_SOURCE_DIR}/common/video/video_decode.cpp"
    "${FLM_SOURCE_DIR}/common/video/video_input.cpp")
target_include_directories(test_d3_video_composed PRIVATE "${FLM_SOURCE_DIR}/include")
if(NOT MSVC)
    target_compile_options(test_d3_video_composed PRIVATE -ffp-contract=off)
endif()

# The committed fixtures are a compact subset (every 5th case, subset_fixtures.py); the full set is regenerated with
# gen_compose_fixtures.py (see README.md) and used with  -DFLM_D3_COMPOSE_FIXTURES=<dir of the full set>.
set(FLM_D3_COMPOSE_FIXTURES "${CMAKE_CURRENT_LIST_DIR}/fixtures" CACHE PATH "Directory of the compose fixtures (default: the committed subset)")

enable_testing()
add_test(NAME test_d3_video_composed COMMAND test_d3_video_composed "${FLM_D3_COMPOSE_FIXTURES}")
add_test(NAME test_d3_compose
         COMMAND test_d3_compose "${FLM_D3_COMPOSE_FIXTURES}" "-"
                 "${CMAKE_CURRENT_LIST_DIR}/../d3_contract/fixtures/fuzz.jsonl")

if(FLM_D3_TEST_TOKENIZER)
    add_subdirectory("${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp" tokenizers EXCLUDE_FROM_ALL)
    add_executable(test_d3_compose_tokenizer test_d3_compose.cpp "${FLM_SOURCE_DIR}/common/tokenizer/tokenizer.cpp" ${D3_COMPOSE_SOURCES})
    target_include_directories(test_d3_compose_tokenizer PRIVATE "${FLM_SOURCE_DIR}/include"
                               "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include")
    target_compile_definitions(test_d3_compose_tokenizer PRIVATE FLM_D3_TEST_TOKENIZER)
    target_link_libraries(test_d3_compose_tokenizer PRIVATE tokenizers_cpp)
    if(NOT MSVC)
        target_compile_options(test_d3_compose_tokenizer PRIVATE -ffp-contract=off)
    endif()
    if(FLM_D3_TURBO_PACKAGE)
        add_test(NAME test_d3_compose_tokenizer
                 COMMAND test_d3_compose_tokenizer "${FLM_D3_COMPOSE_FIXTURES}" "${FLM_D3_TURBO_PACKAGE}"
                         "${CMAKE_CURRENT_LIST_DIR}/../d3_contract/fixtures/fuzz.jsonl")
    endif()
endif()

if(FLM_D3_TEST_ADAPTER)
    add_executable(test_d3_adapter test_d3_adapter.cpp "${FLM_SOURCE_DIR}/common/AutoDecisionModel/d3_engine_adapter.cpp")
    target_include_directories(test_d3_adapter PRIVATE "${FLM_SOURCE_DIR}/include" "${FLM_SOURCE_DIR}/include/aiebu")
    add_test(NAME test_d3_adapter COMMAND test_d3_adapter)
endif()
