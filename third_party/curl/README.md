# curl for the Windows build

Windows packages ship `src/lib/libcurl.dll` (curl 8.15.0). The default
Windows build compiles and links against that exact DLL, using:

- `include/curl/`: the curl 8.15.0 public headers.
- `libcurl.def`: the export list of `src/lib/libcurl.dll`. `src/CMakeLists.txt`
  runs `lib /def` on it at configure time to make `libcurl.lib` in the build
  directory.

The vcpkg-toolchain and `FLM_WIN_DEPS_PREFIX` routes use their own curl
instead, and CMake warns if its version differs from `FLM_CURL_VERSION`.

curl is licensed under the curl license; see `THIRD_PARTY_NOTICES.txt` at the
repository root.

## Updating the DLL

Replace these together, in one commit:

1. `src/lib/libcurl.dll`.
2. `include/curl/*.h`, with the headers of the same curl release.
3. `libcurl.def`. From a Visual Studio developer prompt at the repository
   root, run `dumpbin /exports src\lib\libcurl.dll`. Keep the header lines of
   the current file and list every exported name under `EXPORTS`.
4. `FLM_CURL_VERSION` in `src/CMakeLists.txt`. Configure fails if it does not
   match `include/curl/curlver.h`.
5. The version under libcurl in `THIRD_PARTY_NOTICES.txt`.
