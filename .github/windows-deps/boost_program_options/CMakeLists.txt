# Builds Boost.ProgramOptions as the static library the self-hosted Windows
# runner has under C:/dev/boost_1_88_0/stage/lib (a b2 link=static
# runtime-link=shared release build), with the same file name, because
# src/CMakeLists.txt links it by that raw name. Used by
# .github/script/setup_windows_deps.ps1 on GitHub-hosted runners.
cmake_minimum_required(VERSION 3.20)
project(boost_program_options CXX)

if(NOT BOOST_ROOT)
    message(FATAL_ERROR "Set BOOST_ROOT to the extracted Boost source tree")
endif()

file(GLOB _po_sources ${BOOST_ROOT}/libs/program_options/src/*.cpp)
add_library(boost_program_options STATIC ${_po_sources})
target_include_directories(boost_program_options PRIVATE ${BOOST_ROOT})
target_compile_definitions(boost_program_options PRIVATE BOOST_ALL_NO_LIB=1)
set_target_properties(boost_program_options PROPERTIES
    OUTPUT_NAME libboost_program_options-vc143-mt-x64-1_88
    MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
