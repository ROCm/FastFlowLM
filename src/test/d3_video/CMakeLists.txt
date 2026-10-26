# d3 video input tests: no NPU, no XRT, no FFmpeg headers needed (FFmpeg is loaded at run time and the
# backend tests skip when the libraries are missing).
#   cmake -S src/test/d3_video -B build-d3v && cmake --build build-d3v && ctest --test-dir build-d3v --output-on-failure
# The committed fixtures (fixtures/) use a small frame budget; `gen_fixtures.py --big` makes the real 0.2 MP set:
#   ctest ... -DFLM_D3_VIDEO_BIG=<dir>   or   test_d3_video <fixtures> <big dir>
cmake_minimum_required(VERSION 3.22)
project(d3_video_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
find_package(OpenMP QUIET)

add_executable(test_d3_video
    test_d3_video.cpp
    "${FLM_SOURCE_DIR}/common/decision/py_json.cpp"
    "${FLM_SOURCE_DIR}/common/decision/d3_contract.cpp"
    "${FLM_SOURCE_DIR}/common/video/video_sampling.cpp"
    "${FLM_SOURCE_DIR}/common/video/video_preprocess.cpp"
    "${FLM_SOURCE_DIR}/common/video/video_decode.cpp"
    "${FLM_SOURCE_DIR}/common/video/video_input.cpp"
    "${FLM_SOURCE_DIR}/common/video/ffmpeg_dl.cpp")
target_include_directories(test_d3_video PRIVATE "${FLM_SOURCE_DIR}/include")
target_link_libraries(test_d3_video PRIVATE ${CMAKE_DL_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_d3_video PRIVATE OpenMP::OpenMP_CXX)
endif()

# The hand-written FFmpeg struct offsets of ffmpeg_dl.cpp against the real headers (skipped without them).
find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h)
if(FFMPEG_INCLUDE_DIR)
    add_executable(test_ffmpeg_layout test_ffmpeg_layout.cpp)
    target_include_directories(test_ffmpeg_layout PRIVATE "${FFMPEG_INCLUDE_DIR}")
endif()

enable_testing()
add_test(NAME test_d3_video COMMAND test_d3_video "${CMAKE_CURRENT_LIST_DIR}/fixtures")
if(FFMPEG_INCLUDE_DIR)
    add_test(NAME test_ffmpeg_layout COMMAND test_ffmpeg_layout)
endif()
