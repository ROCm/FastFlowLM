cmake_minimum_required(VERSION 3.20)
project(gemma4_core_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

# FLM_SOURCE_DIR is passed in by the configure command, e.g.:
#   cmake -B build-g4 -S src/test/gemma4_core -DFLM_SOURCE_DIR=<repo>/src
if(NOT DEFINED FLM_SOURCE_DIR)
    set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()

include_directories("${FLM_SOURCE_DIR}/include" "${FLM_SOURCE_DIR}/test/phi4_core")

# The Gemma 4 test targets:
#   test_gemma4_frontend  the Gemma4e frontend, over a stub engine
#   test_gemma4_tokenizer the shipped tokenizer stack on the real packages
#   test_gemma4_registry  the core backend, registered under gemma4e

# --- test_gemma4_frontend (Task A3) --------------------------------------
# Exercises Gemma4e::engine_config() through the real AutoModel/Gemma4e
# machinery, with a stub causal_lm engine standing in for the core engine
# library this build does not link -- the same pattern test_phi4_frontend
# uses in src/test/phi4_core/. It needs XRT and boost headers (xrt/detail/
# any.h pulls in boost/any.hpp) but not the corelib header itself, since the
# frontend never includes core/corelib_api.hpp directly.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_frontend.cpp")
    # Not REQUIRED: a configure run that only wants another target here
    # must not be blocked by this target's
    # probes. Absence is reported below -- loudly, and as a ctest FAILURE --
    # rather than failing the whole configure.
    #
    # THE FOURTH HINT IS FOR A GIT WORKTREE, and its absence made this the
    # last live instance in this file of the bug the shape-plan and engine
    # blocks were already hardened against. The third hint resolves to
    # <workspace>/xrt_package only when the checkout sits directly in the
    # workspace; a worktree under <repo>/.worktrees/<name> is two levels
    # deeper. Without it this target was silently dropped on a FIRST
    # configure in a worktree -- ctest then reported "100% tests passed ...
    # out of 7" without ever naming it -- and reappeared on a SECOND
    # configure, because the engine block below does have the deeper hint and
    # writes XRT_INCLUDE_DIR into the CMake cache, which this block then
    # picks up on the next pass. Every test count taken from this directory
    # was therefore a function of configure history. Measured both ways
    # before this line was added: 7 targets / 99 PASS on a first configure,
    # 8 / 101 on a second.
    find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
        HINTS "$ENV{XRT_INCLUDE_DIR}"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../xrt_package/xrt/include"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../../../xrt_package/xrt/include"
              "C:/dev/XRT/src/runtime_src/core/include")
    find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp
        HINTS "$ENV{CONDA_PREFIX}/Library/include"
              "$ENV{USERPROFILE}/anaconda3/Library/include")
    if(XRT_INCLUDE_DIR AND BOOST_INCLUDE_DIR)
        # automodel.cpp parses output and constrains decoding with the chat library
        include("${FLM_SOURCE_DIR}/common/chat/chat_sources.cmake")
        add_executable(test_gemma4_frontend
            test_gemma4_frontend.cpp
            "${FLM_SOURCE_DIR}/common/AutoModel/automodel.cpp"
            "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
            "${FLM_SOURCE_DIR}/common/AutoModel/modeling_gemma4e.cpp"
            "${FLM_SOURCE_DIR}/common/prefix_slot_cache.cpp"
            ${FLM_CHAT_SOURCES})
        target_include_directories(test_gemma4_frontend PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FLM_SOURCE_DIR}/include"
            "${XRT_INCLUDE_DIR}"
            "${BOOST_INCLUDE_DIR}")
        target_compile_definitions(test_gemma4_frontend PRIVATE
            DEV_BUILD=1 __WINDOWS__ USEAVX2=1 DISABLE_ABI_CHECK=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX)
        target_compile_options(test_gemma4_frontend PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
        target_link_directories(test_gemma4_frontend PRIVATE "${XRT_INCLUDE_DIR}/../lib")
        target_link_libraries(test_gemma4_frontend PRIVATE xrt_coreutil flm_xgrammar)
        add_test(NAME test_gemma4_frontend COMMAND test_gemma4_frontend)
    else()
        # LOUD AT CONFIGURE, AND RED AT CTEST -- the convention every block
        # in this file follows. message(STATUS) alone is visible while
        # configuring and invisible afterwards: the ctest run that follows
        # says "100% tests passed" and nothing in it names the target that
        # was never compiled.
        # A green run that covers less than it appears to is the same species
        # as the vacuous test, the self-confirming fixture and the stale .exe,
        # and this project has now been bitten by all four.
        message(WARNING "gemma4_core: test_gemma4_frontend will NOT be built -- "
            "XRT_INCLUDE_DIR=${XRT_INCLUDE_DIR}, BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR} "
            "(need XRT's xrt/xrt_bo.h and Boost's boost/any.hpp). "
            "ctest will report test_gemma4_frontend_WAS_NOT_BUILT as FAILED so "
            "this run cannot be mistaken for a complete one.")
        add_test(NAME test_gemma4_frontend_WAS_NOT_BUILT
            COMMAND "${CMAKE_COMMAND}" -E echo
                "test_gemma4_frontend WAS NOT BUILT: xrt/xrt_bo.h or "
                "boost/any.hpp was not found at configure time, so this ctest "
                "run does not cover it.")
        set_tests_properties(test_gemma4_frontend_WAS_NOT_BUILT PROPERTIES
            FAIL_REGULAR_EXPRESSION "WAS NOT BUILT")
    endif()
endif()

# --- test_gemma4_tokenizer (Task R5) -------------------------------------
# THE BINARY THAT TAKES A STEP OFF THE HARDWARE CHECKLIST. review-R4.md §5
# settled the doubled-BOS question on the host and recommended landing it as
# a test; this is it. It links the PRODUCTION Tokenizer class against the
# same tokenizers-cpp libraries the shipped binary links, and renders through
# the production minja chat template, so the answer is the shipped stack's
# answer and not a reimplementation of it.
#
# This is the only target here that needs a LIBRARY rather than a header.
# tokenizers-cpp is a git submodule plus a Rust staticlib, and building it
# from source here would add cargo to this suite's dependencies and minutes
# to its build, so the three libraries are located instead. They are produced
# by any build of src/CMakeLists.txt, under <build>/tokenizers-cpp:
#
#   tokenizers_c.lib          the Rust staticlib (cargo)          <dir>/
#   tokenizers_cpp.lib        the C++ binding                     <dir>/Release/
#   sentencepiece.lib         the other tokenizer backend         <dir>/sentencepiece/src/Release/
#
# Point FLM_TOKENIZERS_BUILD_DIR (cache or environment) at that directory if
# the hints below miss it.
#
# The header is IN-TREE (src/include/tokenizer/tokenizers_cpp.h), so the
# submodule does not need to be checked out.
#
# XRT and Boost are needed for a reason that has nothing to do with
# tokenizing: tokenizer.hpp includes typedef.hpp, which includes the FROZEN
# buffer.hpp, which includes device_runtime.hpp, which includes xrt/xrt_bo.h
# (and through xrt/detail/any.h, boost/any.hpp). Same probes as the frontend
# target above, including the fifth hint for a git worktree.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_tokenizer.cpp")
    find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
        HINTS "$ENV{XRT_INCLUDE_DIR}"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../xrt_package/xrt/include"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../../../xrt_package/xrt/include"
              "C:/dev/XRT/src/runtime_src/core/include")
    find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp
        HINTS "$ENV{CONDA_PREFIX}/Library/include"
              "$ENV{USERPROFILE}/anaconda3/Library/include")
    find_path(FLM_TOKENIZERS_BUILD_DIR NAMES tokenizers_c.lib
        HINTS "$ENV{FLM_TOKENIZERS_BUILD_DIR}"
              "${FLM_SOURCE_DIR}/build/tokenizers-cpp"
              "${FLM_SOURCE_DIR}/../build/tokenizers-cpp"
              # A sibling worktree: this port is developed in
              # <repo>/.worktrees/<name>, and the product has usually been
              # built in one of the others rather than in this one.
              "${CMAKE_CURRENT_LIST_DIR}/../../../../src/build/tokenizers-cpp")
    set(FLM_TOKENIZERS_CPP_LIB "${FLM_TOKENIZERS_BUILD_DIR}/Release/tokenizers_cpp.lib")
    set(FLM_SENTENCEPIECE_LIB "${FLM_TOKENIZERS_BUILD_DIR}/sentencepiece/src/Release/sentencepiece.lib")
    if(XRT_INCLUDE_DIR AND BOOST_INCLUDE_DIR AND FLM_TOKENIZERS_BUILD_DIR
       AND EXISTS "${FLM_TOKENIZERS_CPP_LIB}" AND EXISTS "${FLM_SENTENCEPIECE_LIB}")
        add_executable(test_gemma4_tokenizer
            test_gemma4_tokenizer.cpp
            "${FLM_SOURCE_DIR}/common/tokenizer/tokenizer.cpp")
        target_include_directories(test_gemma4_tokenizer PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FLM_SOURCE_DIR}/include"
            "${FLM_SOURCE_DIR}/include/tokenizer"
            "${XRT_INCLUDE_DIR}"
            "${BOOST_INCLUDE_DIR}")
        target_compile_definitions(test_gemma4_tokenizer PRIVATE
            DEV_BUILD=1 __WINDOWS__ USEAVX2=1 DISABLE_ABI_CHECK=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX)
        target_compile_options(test_gemma4_tokenizer PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244 /bigobj>)
        target_link_directories(test_gemma4_tokenizer PRIVATE "${XRT_INCLUDE_DIR}/../lib")
        target_link_libraries(test_gemma4_tokenizer PRIVATE
            xrt_coreutil
            "${FLM_TOKENIZERS_CPP_LIB}"
            "${FLM_TOKENIZERS_BUILD_DIR}/tokenizers_c.lib"
            "${FLM_SENTENCEPIECE_LIB}"
            # What the Rust staticlib needs on Windows -- copied from
            # third_party/tokenizers-cpp/CMakeLists.txt's own list.
            ntdll wsock32 ws2_32 Bcrypt iphlpapi userenv psapi)
        add_test(NAME test_gemma4_tokenizer COMMAND test_gemma4_tokenizer)
        # Opt-in on the real packages: without them it exits 77, a skip.
        set_tests_properties(test_gemma4_tokenizer PROPERTIES SKIP_RETURN_CODE 77)
    else()
        # LOUD AT CONFIGURE, AND RED AT CTEST -- the convention every block in
        # this file follows, and it matters more here than anywhere else:
        # silently dropping THIS target silently puts checklist step 0b back
        # on the hardware list, and nothing in a green ctest line would say
        # so. That is the exact failure -- a run that covers less than it
        # appears to -- this suite has been bitten by four times.
        message(WARNING "gemma4_core: test_gemma4_tokenizer will NOT be built -- "
            "FLM_TOKENIZERS_BUILD_DIR=${FLM_TOKENIZERS_BUILD_DIR}, "
            "XRT_INCLUDE_DIR=${XRT_INCLUDE_DIR}, BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR} "
            "(need tokenizers_c.lib, Release/tokenizers_cpp.lib and "
            "sentencepiece/src/Release/sentencepiece.lib, as produced by any "
            "build of src/CMakeLists.txt under <build>/tokenizers-cpp). "
            "Pass -DFLM_TOKENIZERS_BUILD_DIR=<dir> or set it in the "
            "environment. ctest will report test_gemma4_tokenizer_WAS_NOT_BUILT "
            "as FAILED so this run cannot be mistaken for a complete one.")
        add_test(NAME test_gemma4_tokenizer_WAS_NOT_BUILT
            COMMAND "${CMAKE_COMMAND}" -E echo
                "test_gemma4_tokenizer WAS NOT BUILT: the tokenizers-cpp "
                "libraries were not found at configure time, so the "
                "doubled-BOS question is NOT answered by this run and "
                "checklist step 0b is back on the hardware list. "
                "Re-configure with -DFLM_TOKENIZERS_BUILD_DIR=<dir>.")
        set_tests_properties(test_gemma4_tokenizer_WAS_NOT_BUILT PROPERTIES
            FAIL_REGULAR_EXPRESSION "WAS NOT BUILT")
    endif()
endif()

# --- test_gemma4_registry ----------------------------------------------------
# The core backend's traits, its registration under gemma4e, and the shipped
# catalog and server gates. The engine is corelib's own model behind phi4_core's
# fake, so nothing here says the model is right.
find_path(RYZENAI_CORELIB_INCLUDE_DIR NAMES ryzenai/corelib.h
    HINTS "$ENV{RYZENAI_CORELIB_INCLUDE_DIR}")
find_path(BOOST_INCLUDE_DIR NAMES boost/program_options.hpp
    HINTS "$ENV{CONDA_PREFIX}/Library/include"
          "$ENV{USERPROFILE}/anaconda3/Library/include")
find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
    HINTS "$ENV{XRT_INCLUDE_DIR}"
          "${CMAKE_CURRENT_LIST_DIR}/../../../../xrt_package/xrt/include"
          "${CMAKE_CURRENT_LIST_DIR}/../../../../../../xrt_package/xrt/include"
          "C:/dev/XRT/src/runtime_src/core/include")
if(RYZENAI_CORELIB_INCLUDE_DIR AND BOOST_INCLUDE_DIR AND XRT_INCLUDE_DIR)
    add_executable(test_gemma4_registry
        test_gemma4_registry.cpp
        "${FLM_SOURCE_DIR}/test/phi4_core/fake_corelib.cpp"
        "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
        "${FLM_SOURCE_DIR}/common/core/corelib_api.cpp"
        "${FLM_SOURCE_DIR}/common/core/corelib_runtime.cpp"
        "${FLM_SOURCE_DIR}/common/core/host_ops.cpp"
        "${FLM_SOURCE_DIR}/common/core/corelib_model.cpp"
        "${FLM_SOURCE_DIR}/common/core/corelib_model_backend.cpp"
        "${FLM_SOURCE_DIR}/common/models/gemma4/core/dev04/gemma4_core_backend.cpp")
    target_include_directories(test_gemma4_registry PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${FLM_SOURCE_DIR}/include"
        "${RYZENAI_CORELIB_INCLUDE_DIR}"
        "${BOOST_INCLUDE_DIR}"
        "${XRT_INCLUDE_DIR}")
    target_compile_definitions(test_gemma4_registry PRIVATE
        RYZENAI_CORELIB_STATIC=1 USEAVX2=1 DISABLE_ABI_CHECK=1
        _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX
        FLM_ENABLE_CORE=1 FLM_CORELIB_TESTING=1
        "FLM_REST_HANDLER_SOURCE=\"${FLM_SOURCE_DIR}/server/rest_handler.cpp\""
        "FLM_ALL_MODELS_SOURCE=\"${FLM_SOURCE_DIR}/include/AutoModel/all_models.hpp\""
        "FLM_TEST_MODEL_LIST_PATH=\"${FLM_SOURCE_DIR}/model_list.json\"")
    target_compile_options(test_gemma4_registry PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
    target_link_directories(test_gemma4_registry PRIVATE "${XRT_INCLUDE_DIR}/../lib")
    target_link_libraries(test_gemma4_registry PRIVATE xrt_coreutil)
    add_test(NAME test_gemma4_registry COMMAND test_gemma4_registry)
else()
    message(WARNING "gemma4_core: test_gemma4_registry will NOT be built -- "
        "RYZENAI_CORELIB_INCLUDE_DIR=${RYZENAI_CORELIB_INCLUDE_DIR}, "
        "BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR}, XRT_INCLUDE_DIR=${XRT_INCLUDE_DIR}")
    add_test(NAME test_gemma4_registry_WAS_NOT_BUILT
        COMMAND ${CMAKE_COMMAND} -E echo "test_gemma4_registry WAS NOT BUILT")
    set_tests_properties(test_gemma4_registry_WAS_NOT_BUILT PROPERTIES
        FAIL_REGULAR_EXPRESSION "WAS NOT BUILT")
endif()
