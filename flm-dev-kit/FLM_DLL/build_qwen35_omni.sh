#!/usr/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
failed_dirs=()
target_dirs=(q4nx mha gemm dequant dequant_new lm_head qwen3_5_omni)

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
