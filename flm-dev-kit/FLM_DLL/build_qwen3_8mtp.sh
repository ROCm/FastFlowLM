#!/usr/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# Phase 1 is CPU only: no mha / gemm / dequant / lm_head xclbin modules are
# linked, so only q4nx (SafeTensors + Q4NX, which causal_lm's header needs)
# and the engine itself get built.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
failed_dirs=()
target_dirs=(q4nx qwen3_8mtp_npu)

for name in "${target_dirs[@]}"; do
	dir="detail/${name}"
	if [ -f "${dir}/Makefile" ]; then
		if ! (cd "$dir" && make clean && make -j4); then
			failed_dirs+=("$dir")
		fi
	fi
done

if [ ${#failed_dirs[@]} -ne 0 ]; then
	echo "Build failures in:"
	for dir in "${failed_dirs[@]}"; do
		echo "  - $dir"
	done
	exit 1
else
	echo "All builds succeeded."
fi
