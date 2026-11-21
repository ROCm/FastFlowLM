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

Working in `flm-dev-kit/`? Read [flm-dev-kit/AGENTS.md](flm-dev-kit/AGENTS.md) first.

Rebase before pushing, and match the surrounding code (files use `/// \file`, `/// \brief`
Doxygen headers). Don't edit vendored code: `src/include/nlohmann/`, `src/include/minja/`,
`third_party/`.

CI must pass: Windows portable + MSI + HRX, Linux portable + HRX, and a four-distro `.deb`
matrix.

## Gotchas

**Never edit `src/lib/xrt/`, `src/lib/hrx/` or `src/xclbins/` by hand.** The engine libraries
and NPU kernels there come from the dev kit (`flm-dev-kit/`), and a pull request commits exactly
the files its own CI run built, staged into its product builds and tested on the NPU: change the
dev-kit source and push; CI builds it, stages it into every product build and runs the product
NPU tests on those packages; while the branch lacks those files, the required check fails with
"Dev-kit swap pending" and the command to run: `python .github/script/ci/devkit_swap.py fetch
--run <id>` on a clean checkout of the tested commit, then push the swap commit it makes; CI
checks its hashes against that run and skips the NPU tests for it. Steps, conditions and the full
refresh: "Prebuilt binaries" in [flm-dev-kit/AGENTS.md](flm-dev-kit/AGENTS.md).

**Adding a model?** Add its dev-kit source (engine under `flm-dev-kit/FLM_DLL/detail/`,
kernels under `flm-dev-kit/FLM_Xclbin/`) with its `engines.json` and `kernel_assignment.json`
entries, its `src/model_list.json` entry, the engine's line in `FLM_ENGINE_LINK_LIBS`
(`src/CMakeLists.txt`) and the CI tables (`.github/script/ci/changes.py`,
`.github/script/npu_test/plan.py`), but nothing under `src/lib/` or `src/xclbins/`. CI builds the
engine for both NPU backends and the kernels, stages them into the product builds (which then
link the new engine), tests the model, and says "Dev-kit swap pending"; fetch and push as above.
The walkthrough: "Adding a model" in [flm-dev-kit/AGENTS.md](flm-dev-kit/AGENTS.md).

**Adding a docs page?** It's invisible until you add it to the hardcoded `docs_nav` list in
`docs/_config.yml` — the sidebar is not generated from front matter. Give the URL a trailing
slash; the active-page highlight is an exact match on `page.url`.

## Bug reports

Include your OS, NPU driver version, `flm version` and `flm validate` output, and the command
that failed. Most startup failures are driver-version related — the docs recommend a newer
driver than the binary enforces, so report the version you actually have.

## Licensing

Source code is MIT ([LICENSE_RUNTIME.txt](LICENSE_RUNTIME.txt)). The NPU kernel binaries in
`src/xclbins/` and `src/lib/hrx/` are **not** open source, but are free to use and redistribute
unmodified — see [LICENSE_BINARY.txt](LICENSE_BINARY.txt). CI builds them from the dev kit
(`flm-dev-kit/`), and they are committed only through the swap flow under Gotchas.
