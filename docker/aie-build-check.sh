#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# aie-build-check.sh <image>
#
# Builds each kernel docker/aie-build.pdi.sha256 lists in <image> ("make
# bitstream [MODEL_TYPE=...]", no setup.sh: the image's environment must be
# enough), dumps the PDI from the xclbin with the image's xclbinutil, and
# compares its sha256 and size with the pinned ones. Exits 1 if a build fails
# or a PDI differs, after checking every kernel. Run from the repository root;
# it builds in flm-dev-kit/FLM_Xclbin/<unit>/build, which it removes first.
# Appends a table to $GITHUB_STEP_SUMMARY when that is set.
set -euo pipefail

image="${1:?usage: aie-build-check.sh <image>}"
pins="docker/aie-build.pdi.sha256"
src="flm-dev-kit/FLM_Xclbin"
[[ -f "$pins" && -d "$src" ]] || { echo "run from the repository root" >&2; exit 2; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
run() { docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD/$src:/src" "$@"; }

failed=0
rows=()
while read -r want_sha want_size unit mt xclbin; do
  [[ -z "$want_sha" || "$want_sha" == \#* ]] && continue
  label="${unit}${mt:+ $mt}"
  [[ "$mt" == - ]] && { mt=""; label="$unit"; }
  echo "::group::make bitstream in ${label}"
  rc=0
  run -w "/src/${unit}" -e MT="$mt" "$image" \
    bash -c 'rm -rf build && make -j"$(nproc)" bitstream ${MT:+MODEL_TYPE="$MT"}' || rc=$?
  echo "::endgroup::"
  if (( rc != 0 )); then
    echo "::error::${label}: make bitstream failed (exit ${rc})"
    rows+=("| \`${label}\` | build failed | \`${want_sha:0:12}\` |")
    failed=1
    continue
  fi
  out="${work}/${unit//\//_}${mt:+_$mt}"
  mkdir -p "$out"
  run -v "${out}:/out" -w /out "$image" \
    xclbinutil -q -i "/src/${unit}/${xclbin}" --dump-section AIE_PARTITION:JSON:/out/AIE_PARTITION.json --force
  mapfile -t pdis < <(find "$out" -maxdepth 1 -name '*.pdi')
  if (( ${#pdis[@]} != 1 )); then
    echo "::error::${label}: ${xclbin} has ${#pdis[@]} PDIs in AIE_PARTITION, expected 1"
    rows+=("| \`${label}\` | ${#pdis[@]} PDIs | \`${want_sha:0:12}\` |")
    failed=1
    continue
  fi
  got_sha="$(sha256sum "${pdis[0]}" | cut -d' ' -f1)"
  got_size="$(stat -c %s "${pdis[0]}")"
  if [[ "$got_sha" == "$want_sha" && "$got_size" == "$want_size" ]]; then
    echo "${label}: PDI ${got_sha} (${got_size} B) matches"
    rows+=("| \`${label}\` | \`${got_sha:0:12}\` ${got_size} B | match |")
  else
    echo "::error::${label}: PDI ${got_sha} (${got_size} B), ${pins} pins ${want_sha} (${want_size} B)"
    rows+=("| \`${label}\` | \`${got_sha:0:12}\` ${got_size} B | **differs** from \`${want_sha:0:12}\` ${want_size} B |")
    failed=1
  fi
done < "$pins"

if (( ${#rows[@]} == 0 )); then
  echo "::error::${pins} lists no kernel"
  exit 1
fi
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "Kernels built in \`${image}\` (PDI against \`${pins}\`):"
    echo
    echo "| Kernel | PDI | Result |"
    echo "|---|---|---|"
    printf '%s\n' "${rows[@]}"
  } >> "$GITHUB_STEP_SUMMARY"
fi
exit "$failed"
