# Fetches one commit of a git repository into SOURCE_DIR, for use as an
# ExternalProject DOWNLOAD_COMMAND:
#
#   cmake -DREPO=<url> -DCOMMIT=<full sha> -DSOURCE_DIR=<dir>
#         [-DPACKAGE_VERSION=<version>] -P FlmFetchGitCommit.cmake
#
# Fetching the commit by SHA with --depth 1 transfers that commit only, where
# ExternalProject's own GIT_SHALLOW clones every branch tip and then cannot
# reach a tag that is not on one. HEAD is checked against COMMIT, then .git is
# removed so the tree matches a release tarball. If PACKAGE_VERSION is given,
# it is written to a package_version file the way release tarballs ship it.
foreach(_var REPO COMMIT SOURCE_DIR)
    if(NOT ${_var})
        message(FATAL_ERROR "FlmFetchGitCommit.cmake: ${_var} is not set")
    endif()
endforeach()

find_package(Git REQUIRED)

function(_flm_git)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C ${SOURCE_DIR} ${ARGN}
        RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "git ${ARGN} failed with exit code ${_rc}")
    endif()
endfunction()

# CI runner networks reset transfers now and then, so retry the fetch.
foreach(_attempt RANGE 1 3)
    file(REMOVE_RECURSE ${SOURCE_DIR})
    file(MAKE_DIRECTORY ${SOURCE_DIR})
    _flm_git(init --quiet)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${SOURCE_DIR} fetch --depth 1 ${REPO} ${COMMIT}
        RESULT_VARIABLE _rc)
    if(_rc EQUAL 0)
        break()
    endif()
    if(_attempt EQUAL 3)
        message(FATAL_ERROR "Fetching ${COMMIT} from ${REPO} failed 3 times")
    endif()
    message(STATUS "Fetching ${COMMIT} from ${REPO} failed (attempt ${_attempt}), retrying...")
    execute_process(COMMAND ${CMAKE_COMMAND} -E sleep 15)
endforeach()

_flm_git(-c core.autocrlf=false checkout --quiet FETCH_HEAD)
execute_process(COMMAND ${GIT_EXECUTABLE} -C ${SOURCE_DIR} rev-parse HEAD
    OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0 OR NOT _head STREQUAL COMMIT)
    message(FATAL_ERROR "Fetched ${REPO} at '${_head}', expected ${COMMIT}")
endif()

file(REMOVE_RECURSE ${SOURCE_DIR}/.git)
if(PACKAGE_VERSION)
    file(WRITE ${SOURCE_DIR}/package_version "PACKAGE_VERSION=\"${PACKAGE_VERSION}\"\n")
endif()
message(STATUS "Fetched ${REPO} at ${COMMIT}")
