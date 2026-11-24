// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file xclbin_path.hpp
/// \brief locate a bitstream that ships alongside the binary
/// \author FastFlowLM Team, Alfred
/// \date 2026-07-31
/// \note The xclbins are build artifacts of this repo, not model data: they
///       change when the IRON kernels change, not when the checkpoint does.
///       They are therefore shipped with the binary (next to run.exe in the
///       source tree, inside the wheel for the python module) and located
///       relative to it, never relative to a model directory.
#ifndef __XCLBIN__PATH__HPP__
#define __XCLBIN__PATH__HPP__

#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#endif

#include "utils.hpp"

namespace utils {

/// \brief directory holding the binary this code was linked into
/// \return the directory, or "" if it could not be determined
/// \note Deliberately not /proc/self/exe: under the python extension the
///       executable is the interpreter, so that would resolve to the venv's
///       python3 rather than to site-packages. Asking the loader which module
///       a local address belongs to returns the .so for the module build and
///       the executable for run.exe, which is exactly the distinction needed.
/// \note The result is made absolute. dladdr hands back the path as the loader
///       saw it, which for a directly-invoked executable is argv[0] -- "./run.exe"
///       or a bare "run.exe" found on PATH. Resolving it means the lookup does
///       not silently follow the process's working directory.
inline std::string binary_directory() {
    // The address handed to the loader must live in this binary. A function
    // -local static gives one without depending on how the compiler emitted
    // the surrounding inline function.
    static const char anchor = 0;
    std::string full;
#ifdef _WIN32
    HMODULE mod = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                                | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            &anchor, &mod)) {
        return "";
    }
    char buffer[MAX_PATH];
    DWORD len = GetModuleFileNameA(mod, buffer, MAX_PATH);
    if (len == 0 || len == MAX_PATH) return "";
    full.assign(buffer, len);
    const char sep = '\\';
#else
    Dl_info info;
    if (dladdr(&anchor, &info) == 0 || info.dli_fname == nullptr) return "";
    char resolved[PATH_MAX];
    full = realpath(info.dli_fname, resolved) ? resolved : info.dli_fname;
    const char sep = '/';
#endif
    const size_t cut = full.find_last_of(sep);
    return (cut == std::string::npos) ? "" : full.substr(0, cut);
}

/// \brief absolute path to a bitstream shipped alongside the binary
/// \param name the bitstream file name, e.g. "vision.xclbin"
/// \return the first candidate that exists
/// \throws std::runtime_error naming every candidate tried, if none exists.
///         Throwing rather than exiting keeps a missing bitstream a catchable
///         error for the python binding instead of killing the interpreter.
/// \note Search order:
///         1. $SMOLVLA_XCLBIN_DIR/<name>  -- test a rebuilt bitstream in place
///         2. <binary dir>/XCLBINS/<name> -- the normal layout, source tree and
///                                           installed wheel alike
///         3. <binary dir>/<name>         -- flat, hand-assembled install
inline std::string xclbin_path(const std::string& name) {
    std::vector<std::string> tried;

    if (const char* env = std::getenv("SMOLVLA_XCLBIN_DIR")) {
        if (env[0] != '\0') tried.push_back(path_join(env, name));
    }
    const std::string bin_dir = binary_directory();
    if (!bin_dir.empty()) {
        tried.push_back(path_join(bin_dir, "XCLBINS", name));
        tried.push_back(path_join(bin_dir, name));
    }

    for (const std::string& candidate : tried) {
        if (check_file_exists(candidate)) return candidate;
    }

    std::string msg = "could not find the bitstream " + name + ". Tried:";
    for (const std::string& candidate : tried) msg += "\n    " + candidate;
    if (tried.empty()) {
        msg += "\n    (nothing -- the binary's own location is unknown)";
    }
    msg += "\nThe xclbins ship with the binary; point SMOLVLA_XCLBIN_DIR at a "
           "directory holding them to override.";
    throw std::runtime_error(msg);
}

} // end of namespace utils

#endif // __XCLBIN__PATH__HPP__
