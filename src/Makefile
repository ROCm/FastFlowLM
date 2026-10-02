BUILD_DIR := build
OUT_DIR   := out
PWSH      := powershell.exe

.PHONY: all clean run run-hrx serve install hrx

ifeq ($(OS),Windows_NT)

all:
	@$(PWSH) "if (!(Test-Path '$(BUILD_DIR)')) { New-Item -ItemType Directory -Path '$(BUILD_DIR)' }"
	@$(PWSH) "if (!(Test-Path '$(OUT_DIR)'))   { New-Item -ItemType Directory -Path '$(OUT_DIR)' }"
	@$(PWSH) "cmake --preset windows-vs18"
	@$(PWSH) "cmake --build build --preset windows-default"

hrx:
	@$(PWSH) "if (!(Test-Path '$(BUILD_DIR)')) { New-Item -ItemType Directory -Path '$(BUILD_DIR)' }"
	@$(PWSH) "if (!(Test-Path '$(OUT_DIR)'))   { New-Item -ItemType Directory -Path '$(OUT_DIR)' }"
	@$(PWSH) "$$lines = & '..\\hrx-integration\\fetch-hrx-release.ps1' 6>&1 | ForEach-Object { $$_.ToString() }; $$lines | ForEach-Object { Write-Host $$_ }; $$prefix = ($$lines | Where-Object { $$_ -like 'HRX_CMAKE_PREFIX=*' } | Select-Object -First 1) -replace 'HRX_CMAKE_PREFIX=', ''; if (!$$prefix) { throw 'failed to determine HRX_CMAKE_PREFIX from fetch-hrx-release.ps1 output' }; $$prefixArg = '-DCMAKE_PREFIX_PATH=' + $$prefix; cmake --preset windows-vs18 -DFLM_USE_HRX=ON $$prefixArg; cmake --build build --preset windows-default"

run:
	@$(PWSH) "Copy-Item 'lib\\xrt\\*.dll' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'lib\\*.dll' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'model_list.json' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'model_info.json' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'xclbins' '$(OUT_DIR)' -Recurse -Force"
	@$(PWSH) "cd '$(OUT_DIR)'; .\\flm.exe run llama3.2:1b -a 0"

run-hrx:
	@$(PWSH) "Copy-Item 'lib\\hrx\\*.dll' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'lib\\*.dll' '$(OUT_DIR)' -Force"
	@$(PWSH) "$$hrxDll = Get-ChildItem -Path '..\\hrx-integration\\.hrx-release' -Recurse -Filter 'hrx.dll' -ErrorAction SilentlyContinue | Select-Object -First 1; if ($$hrxDll) { Copy-Item $$hrxDll.FullName '$(OUT_DIR)' -Force } else { Write-Warning 'hrx.dll not found under hrx-integration\\.hrx-release; run make hrx first' }"
	@$(PWSH) "Copy-Item 'model_list.json' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'model_info.json' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'xclbins' '$(OUT_DIR)' -Recurse -Force"
	@$(PWSH) "cd '$(OUT_DIR)'; .\\flm.exe run llama3.2:1b -a 0"

serve:
	@$(PWSH) "Copy-Item 'lib\\*.dll' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'model_list.json' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'xclbins' '$(OUT_DIR)' -Recurse -Force"
	@$(PWSH) "cd '$(OUT_DIR)'; .\\flm.exe serve qwen2.5vl-it -c 32768"

clean:
	@$(PWSH) "Remove-Item -Recurse -Force '$(BUILD_DIR)'"
	@$(PWSH) "Remove-Item -Recurse -Force '$(OUT_DIR)'"

install:
	@echo "Installing using CMake..."
	@cd $(BUILD_DIR) && cmake --install . --prefix $(PREFIX)

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
