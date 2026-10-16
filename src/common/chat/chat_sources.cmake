# The shared chat library and the grammar engine under it, for every target that
# compiles automodel.cpp: flm itself and the standalone test harnesses.
#
# Sets FLM_CHAT_SOURCES and defines the flm_xgrammar static library once per
# build tree; link it alongside the sources.

set(FLM_CHAT_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/chat_args.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chat_constraint.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chat_format.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chat_grammar.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chat_msg.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chat_parser.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chat_stream.cpp")

# ———————————————————————————————————————————————
# xgrammar: the grammar engine behind constrained decoding (tool_choice,
# response_format). Built from its sources as a plain static library rather than
# through its own CMakeLists, which would add install rules and -Werror.
# ———————————————————————————————————————————————
if(NOT TARGET flm_xgrammar)
    get_filename_component(XGRAMMAR_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../third_party/xgrammar" ABSOLUTE)
    file(GLOB_RECURSE XGRAMMAR_SOURCES ${XGRAMMAR_DIR}/cpp/*.cc)
    list(FILTER XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
    add_library(flm_xgrammar STATIC ${XGRAMMAR_SOURCES})
    target_compile_features(flm_xgrammar PUBLIC cxx_std_17)
    target_include_directories(flm_xgrammar PUBLIC ${XGRAMMAR_DIR}/include)
    target_include_directories(flm_xgrammar SYSTEM PUBLIC
        ${XGRAMMAR_DIR}/3rdparty/picojson
        ${XGRAMMAR_DIR}/3rdparty/dlpack/include)
    target_compile_definitions(flm_xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)
    set_target_properties(flm_xgrammar PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(MSVC)
        # Its sources hold UTF-8 literals (the SentencePiece "▁" that tokenizer
        # metadata detection compares against, DeepSeek's "｜" markers). MSVC reads
        # source in the system code page unless told otherwise, so without this the
        # bytes survive only on machines whose code page happens to pass them through.
        target_compile_options(flm_xgrammar PRIVATE /utf-8)
    endif()
endif()
