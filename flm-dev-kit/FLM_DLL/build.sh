#!/usr/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
if ! ls detail/*/Makefile >/dev/null 2>&1; then
	echo "build.sh: no detail/*/Makefile under $PWD" >&2
	exit 1
fi

rm -rf build

failed_dirs=()
priority_dirs=(q4nx mha gemm dequant lm_head)

for name in "${priority_dirs[@]}"; do
	dir="detail/${name}"
	if [ -f "${dir}/Makefile" ]; then
		if ! (cd "$dir" && make clean && make -j4); then
			failed_dirs+=("$dir")
		fi
	fi
done

for dir in detail/*/; do
	name="${dir%/}"
	name="${name##*/}"
	case "$name" in
		q4nx|mha|gemm|dequant|lm_head)
			continue
			;;
	esac
	if [ -f "${dir}Makefile" ]; then
		if ! (cd "$dir" && make clean && make -j4); then
			failed_dirs+=("${dir%/}")
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