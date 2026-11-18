#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Assert that Linux engine libraries stay inside the manylinux_2_28 ABI.
#
# The engines are linked into flm by the SDK's manylinux_2_28 job (glibc 2.28, system libstdc++ from
# GCC 8). A library built on a newer host (e.g. Ubuntu 24.04 + GCC 13) links fine there only until the
# linker meets a symbol version the 2.28 sysroot lacks, e.g. __libc_single_threaded@GLIBC_2.32 or
# std::ios_base_library_init()@GLIBCXX_3.4.32, and then the whole FLM link fails. The same holds for
# the GCC 8 runtime libraries next to it: libgcc_s (GCC_*) and, since every engine uses -fopenmp,
# libgomp (GOMP_* for compiler-emitted calls, OMP_* for the omp_* API).
#
# usage: check_abi.sh [lib-or-dir ...]     (default: build/lib)
# Ceilings can be overridden with MAX_GLIBC / MAX_GLIBCXX / MAX_CXXABI / MAX_GCC / MAX_GOMP / MAX_OMP.
set -uo pipefail

MAX_GLIBC="${MAX_GLIBC:-2.28}"
MAX_GLIBCXX="${MAX_GLIBCXX:-3.4.25}"
MAX_CXXABI="${MAX_CXXABI:-1.3.11}"
MAX_GCC="${MAX_GCC:-7.0.0}"
MAX_GOMP="${MAX_GOMP:-4.5}"
MAX_OMP="${MAX_OMP:-4.5}"

command -v objdump >/dev/null 2>&1 || { echo "check_abi: objdump (binutils) is required" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
(( $# )) || set -- "$here/build/lib"

libs=()
for p in "$@"; do
  if [[ -d $p ]]; then
    while IFS= read -r f; do libs+=("$f"); done < <(find "$p" -maxdepth 1 -name 'lib*.so' | sort)
  else
    libs+=("$p")
  fi
done
(( ${#libs[@]} )) || { echo "check_abi: no libraries found in: $*" >&2; exit 2; }

# Highest version of a family (GLIBC, GLIBCXX, CXXABI, GCC, GOMP, OMP) required by a library, and the first symbol
# that needs it. Only undefined symbols count; versions the library itself defines are not a requirement.
need() { # lib family -> "version symbol" of the highest requirement, or empty
  objdump -T "$1" 2>/dev/null |
    awk -v fam="$2" '$0 ~ /\*UND\*/ { for (i = 1; i <= NF; i++) if ($i ~ "^\\(" fam "_[0-9.]+\\)$") { v = $i; gsub(/[()]/, "", v); sub("^" fam "_", "", v); print v, $NF } }' |
    sort -k1,1V | tail -1
}

exceeds() { # version ceiling -> true when version > ceiling
  [[ $1 != "$2" && "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" == "$1" ]]
}

rc=0
for lib in "${libs[@]}"; do
  if [[ ! -f $lib ]] || ! objdump -T "$lib" >/dev/null 2>&1; then
    echo "ERROR: $lib is missing or not a readable shared object" >&2
    rc=1
    continue
  fi
  for spec in "GLIBC:$MAX_GLIBC" "GLIBCXX:$MAX_GLIBCXX" "CXXABI:$MAX_CXXABI" \
              "GCC:$MAX_GCC" "GOMP:$MAX_GOMP" "OMP:$MAX_OMP"; do
    fam=${spec%%:*}; max=${spec#*:}
    read -r ver sym <<<"$(need "$lib" "$fam")"
    if [[ -n ${ver:-} ]] && exceeds "$ver" "$max"; then
      echo "ERROR: $(basename "$lib") needs ${fam}_${ver} (> ${fam}_${max}): $sym" >&2
      rc=1
    fi
  done
done

ceilings="glibc ${MAX_GLIBC}, GLIBCXX ${MAX_GLIBCXX}, CXXABI ${MAX_CXXABI}, GCC ${MAX_GCC}, GOMP ${MAX_GOMP}, OMP ${MAX_OMP}"
if (( rc )); then
  echo "check_abi: FAILED (${#libs[@]} libraries checked; ceiling ${ceilings}). Libraries above the" \
       "ceiling must be built in the manylinux_2_28 container (FLM_DLL/build_manylinux.sh), not on a" \
       "newer host." >&2
else
  echo "check_abi: ${#libs[@]} libraries within ${ceilings}"
fi
exit $rc
