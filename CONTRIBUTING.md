# Contributing to FastFlowLM

## Build

Clone **recursively** — the `third_party/tokenizers-cpp` submodule is required and a
non-recursive clone will not build.

```bash
git clone --recursive https://github.com/ROCm/FastFlowLM.git
cd FastFlowLM/src
cmake --preset linux-default     # or windows-default / windows-vs18
cmake --build build
```

Needs CMake ≥ 3.22, a C++20 compiler, and Ninja (recommended). Linux system packages are
listed in [docs/linux-getting-started.md](docs/linux-getting-started.md). Always configure
through a preset — `FLM_VERSION` and `NPU_VERSION` are required and the presets supply them.

**Rebuild fully after pulling.** Commits routinely refresh `.xclbin` kernels and prebuilt
engine libraries without touching a single source line, so a `git diff` that looks empty can
still change behavior. Delete `src/build` when in doubt.

## Pull requests

Open a PR against `main`. Use Conventional Commits — `feat:`, `fix:`, `docs:`, `chore:`,
`ci:`, `refactor:` — with an optional scope like `chore(hrx):`. No sign-off needed.

Rebase before pushing, and match the surrounding code (files use `/// \file`, `/// \brief`
Doxygen headers). Don't edit vendored code: `src/include/nlohmann/`, `src/include/minja/`,
`third_party/`.

CI must pass: Windows portable + MSI + HRX, Linux portable + HRX, and a four-distro `.deb`
matrix.

## Gotchas

**Adding a model?** Both NPU backends need its engine library. The link list in
`CMakeLists.txt` is unconditional, so an engine that lands only in `src/lib/xrt/` breaks the
`FLM_USE_HRX=ON` build at link time while the default build stays green. Also register it in
`src/model_list.json` and add `src/xclbins/<MODEL>-NPU2/`.

**Adding a docs page?** It's invisible until you add it to the hardcoded `docs_nav` list in
`docs/_config.yml` — the sidebar is not generated from front matter. Give the URL a trailing
slash; the active-page highlight is an exact match on `page.url`.

## Bug reports

Include your OS, NPU driver version, `flm version` and `flm validate` output, and the command
that failed. Most startup failures are driver-version related — the docs recommend a newer
driver than the binary enforces, so report the version you actually have.

## Licensing

Source code is MIT ([LICENSE_RUNTIME.txt](LICENSE_RUNTIME.txt)). The NPU kernel binaries in
`src/xclbins/` and `src/lib/` are **not** open source — see [TERMS.md](TERMS.md). They're
built from a separate toolchain and refreshed by maintainers, so if your change needs new
kernels, open an issue first; a source-only PR won't be enough.
