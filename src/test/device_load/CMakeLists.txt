cmake_minimum_required(VERSION 3.22)
project(device_load_tests LANGUAGES CXX)

add_executable(test_device_load test_device_load.cpp ../../common/device_load.cpp)
target_compile_features(test_device_load PRIVATE cxx_std_20)
target_include_directories(test_device_load PRIVATE ../../include)

enable_testing()
add_test(NAME test_device_load COMMAND test_device_load)
