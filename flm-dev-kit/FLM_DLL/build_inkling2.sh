#!/usr/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Inkling-2-Mini. The engine carries its own host model, so no shared host compute
# modules (mha / gemm / dequant / lm_head) are linked; q4nx is the weight reader.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
failed_dirs=()
target_dirs=(q4nx inkling2_npu)

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
