# Audio codecs for AudioReader (src/common/audio/audio_decoder.cpp).
#
#   - libopus 1.6.1, libogg 1.3.6 and libvorbis 1.3.7 (all BSD-3-Clause), built
#     as static libraries through ExternalProject from pinned release tarballs.
#     libvorbis finds libogg through the shared install prefix.
#   - dr_wav/dr_flac/dr_mp3 (public domain or MIT-0), vendored header-only in
#     include/dr_libs and compiled into audio_decoder.cpp.
#
# Defines the INTERFACE target flm_audio_codecs. Link it from anything that
# compiles audio_decoder.cpp.
include_guard(GLOBAL)
include(ExternalProject)

set(_flm_audio_prefix ${CMAKE_BINARY_DIR}/audio-codecs)
if(MSVC)
    set(_flm_lib_prefix "")
    set(_flm_lib_suffix ".lib")
else()
    set(_flm_lib_prefix "lib")
    set(_flm_lib_suffix ".a")
endif()
foreach(_lib opus ogg vorbis vorbisfile)
    set(_flm_${_lib}_lib ${_flm_audio_prefix}/lib/${_flm_lib_prefix}${_lib}${_flm_lib_suffix})
endforeach()

set(_flm_audio_args
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_PREFIX_PATH=${_flm_audio_prefix}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_TESTING=OFF)
if(CMAKE_C_COMPILER)
    list(APPEND _flm_audio_args -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
endif()

# Short working directories keep MSBuild under Windows' 260-char MAX_PATH on
# CI runners with deep work directories (see FlmImageCodecs.cmake).
function(_flm_audio_project name url sha256 byproduct)
    set(_work ${CMAKE_BINARY_DIR}/ac/${name})
    ExternalProject_Add(flm_${name}
        URL ${url}
        URL_HASH SHA256=${sha256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        EXCLUDE_FROM_ALL TRUE
        PREFIX ${_work}
        DOWNLOAD_DIR ${_work}/dl
        SOURCE_DIR ${_work}/s
        BINARY_DIR ${_work}/b
        STAMP_DIR ${_work}/st
        TMP_DIR ${_work}/t
        INSTALL_DIR ${_flm_audio_prefix}
        CMAKE_ARGS ${_flm_audio_args} ${ARGN}
        BUILD_BYPRODUCTS ${byproduct})
endfunction()

_flm_audio_project(opus
    https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz
    6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1
    ${_flm_opus_lib}
    -DOPUS_BUILD_PROGRAMS=OFF
    -DOPUS_INSTALL_PKG_CONFIG_MODULE=OFF
    -DOPUS_INSTALL_CMAKE_CONFIG_MODULE=OFF)
_flm_audio_project(ogg
    https://downloads.xiph.org/releases/ogg/libogg-1.3.6.tar.gz
    83e6704730683d004d20e21b8f7f55dcb3383cdf84c0daedf30bde175f774638
    ${_flm_ogg_lib}
    -DINSTALL_DOCS=OFF)
_flm_audio_project(vorbis
    https://downloads.xiph.org/releases/vorbis/libvorbis-1.3.7.tar.gz
    0e982409a9c3fc82ee06e08205b1355e5c6aa4c36bca58146ef399621b0ce5ab
    "${_flm_vorbis_lib};${_flm_vorbisfile_lib}")
ExternalProject_Add_StepDependencies(flm_vorbis configure flm_ogg)

# The include directory must exist at configure time for the imported targets.
file(MAKE_DIRECTORY ${_flm_audio_prefix}/include)
add_library(flm_audio_codecs INTERFACE)
foreach(_lib vorbisfile vorbis ogg opus)
    add_library(flm_${_lib}_imported STATIC IMPORTED GLOBAL)
    set_target_properties(flm_${_lib}_imported PROPERTIES
        IMPORTED_LOCATION ${_flm_${_lib}_lib}
        INTERFACE_INCLUDE_DIRECTORIES ${_flm_audio_prefix}/include)
    target_link_libraries(flm_audio_codecs INTERFACE flm_${_lib}_imported)
endforeach()
add_dependencies(flm_vorbisfile_imported flm_vorbis)
add_dependencies(flm_vorbis_imported flm_vorbis)
add_dependencies(flm_ogg_imported flm_ogg)
add_dependencies(flm_opus_imported flm_opus)

# The decoders do wrapping integer arithmetic on corrupt input (e.g. FLAC LPC
# prediction); make signed overflow defined there instead of undefined.
get_filename_component(_flm_audio_decoder_src ${CMAKE_CURRENT_LIST_DIR}/../common/audio/audio_decoder.cpp ABSOLUTE)
set_source_files_properties(${_flm_audio_decoder_src} PROPERTIES
    COMPILE_OPTIONS "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fwrapv>")
