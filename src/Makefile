BUILD_DIR := build
OUT_DIR   := out

.PHONY: all clean run run-hrx serve stage install hrx

# Which flm to build and run: the native Windows one or the Linux one.
#   * make on Windows (cmd, PowerShell, Git Bash, conda) -> windows
#   * make inside WSL on a Windows-drive checkout (/mnt/c/...) -> windows, driven
#     through WSL interop (powershell.exe), since the NPU is only reachable from
#     the Windows side
#   * anything else (Linux, or WSL on its own filesystem) -> linux
# Override with `make FLM_PLATFORM=linux` (or windows).
ifeq ($(OS),Windows_NT)
FLM_PLATFORM ?= windows
else ifneq ($(shell grep -qis microsoft /proc/sys/kernel/osrelease && echo wsl),)
FLM_PLATFORM ?= $(if $(filter /mnt/%,$(CURDIR)),windows,linux)
else
FLM_PLATFORM ?= linux
endif

ifeq ($(FLM_PLATFORM),windows)

# Everything Windows-side lives in make-windows.ps1. Knobs:
#   RUN_ARGS   what `make run` passes after `flm run`   (make run RUN_ARGS="qwen3:8b")
#   SERVE_ARGS what `make serve` passes after `flm serve`
#   JOBS       compilers at once, across all projects (cl, cargo); default every core
#   PRESET     configure preset; default picks windows-vs18 or windows-default
#              (windows-core-on: also ryzenai-corelib)
#   BUILD_DIR  build tree; default the preset's binaryDir
#   FLM_DEVENV script setting up MSVC + Windows SDK when cl.exe isn't on PATH
#              (default: C:\dev\flm-devenv.ps1 when it exists)
RUN_ARGS   ?= llama3.2:1b -a 0
SERVE_ARGS ?= qwen2.5vl-it -c 32768

WIN_MAKE = powershell.exe -NoProfile -ExecutionPolicy Bypass -File make-windows.ps1
WIN_ARGS = -RunArgs "$(RUN_ARGS)" -ServeArgs "$(SERVE_ARGS)" $(if $(JOBS),-Jobs $(JOBS)) \
	$(if $(filter command line environment,$(origin BUILD_DIR)),-BuildDir "$(BUILD_DIR)") \
	-OutDir "$(OUT_DIR)" \
	$(if $(PRESET),-Preset "$(PRESET)") \
	$(if $(FLM_DEVENV),-DevEnv "$(FLM_DEVENV)") \
	$(if $(PREFIX),-Prefix "$(PREFIX)")

all hrx run run-hrx serve stage clean install:
	@$(WIN_MAKE) $@ $(WIN_ARGS)

else

all:
	@mkdir -p $(OUT_DIR)
	@cmake --preset linux-default
	@cmake --build build
	@cp build/flm $(OUT_DIR)

hrx:
	@mkdir -p $(OUT_DIR)
	@out="$$(../hrx-integration/fetch-hrx-release.sh)"; \
	echo "$$out"; \
	prefix="$$(echo "$$out" | sed -n 's/^HRX_CMAKE_PREFIX=//p')"; \
	if [ -z "$$prefix" ]; then echo "failed to determine HRX_CMAKE_PREFIX from fetch-hrx-release.sh output" >&2; exit 1; fi; \
	cmake --preset linux-default -DFLM_USE_HRX=ON -DCMAKE_PREFIX_PATH="$$prefix"
	@cmake --build build
	@cp build/flm $(OUT_DIR)

run:
	@cp lib/*.so $(OUT_DIR)
	@cp model_list.json $(OUT_DIR)
	@cp model_info.json $(OUT_DIR)
	@cd $(OUT_DIR) && ./flm run gpt-oss -a 0

run-hrx:
	@cp lib/hrx/*.so $(OUT_DIR)
	@cp lib/*.so $(OUT_DIR) 2>/dev/null || true
	@hrxlib="$$(find ../hrx-integration/.hrx-release -iname 'libhrx.so*' 2>/dev/null | head -n1)"; \
	if [ -n "$$hrxlib" ]; then cp -P "$$hrxlib" $(OUT_DIR); else echo "libhrx.so not found under hrx-integration/.hrx-release; run make hrx first" >&2; fi
	@cp model_list.json $(OUT_DIR)
	@cp model_info.json $(OUT_DIR)
	@cd $(OUT_DIR) && ./flm run gpt-oss -a 0

serve:
	@cp lib/*.so $(OUT_DIR)
	@cp model_list.json $(OUT_DIR)
	@cd $(OUT_DIR) && ./flm serve -a 1 -e 1

clean:
	@rm -rf $(BUILD_DIR)
	@rm -rf $(OUT_DIR)

install:
	@echo "Installing using CMake..."
	@cd $(BUILD_DIR) && sudo cmake --install . 

endif
