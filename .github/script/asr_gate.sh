#!/usr/bin/env bash
# Decides whether a change needs the ASR test (.github/workflows/asr-test.yml).
#
#   asr_gate.sh <diff range>        e.g.  asr_gate.sh origin/main...HEAD   or   asr_gate.sh <before>..<after>
#
# The workflow's `paths:` filter is deliberately wide: src/server/rest_handler.cpp and src/server/server.cpp host
# /v1/audio/transcriptions, but they also hold every other endpoint, so every server change triggers the workflow. This
# script narrows that down on a hosted runner, before an NPU machine is reserved:
#
#   * a changed file that only serves ASR (the Whisper engine, its kernels and libraries, the audio decoders and front
#     ends, the multipart parser, the ASR test and its data, this workflow) always runs the test;
#   * a change to the shared server files runs the test only if a changed line mentions audio or speech handling
#     (audio, transcription, whisper, multipart, speech, asr, wav, mp3, flac, ogg, opus, m4a, pcm, resample);
#   * anything else (e.g. a model-family change that adds a tag to the REST handler) skips it.
#
# When in doubt it runs the test: an unreadable range, or no files at all, means "run". Prints "run=true|false" on
# stdout (the workflow appends it to $GITHUB_OUTPUT) and the reason on stderr. Keep asr_only below in step with the
# `paths:` lists in asr-test.yml: everything there except the two shared server files.
set -euo pipefail

range=${1:-}
shared='^src/server/(rest_handler|server)\.cpp$'
asr_only='^(src/common/whisper/|src/include/whisper/|src/xclbins/Whisper-V3-Turbo-NPU2/|src/lib/(xrt|hrx)/[^/]*whisper_npu|src/common/audio_process_utils/|src/include/audio_process_utils/|src/common/audio/|src/include/audio/|src/common/AutoModel/[^/]*_audio\.cpp$|src/detail/gemma4e_npu/gemma4e_audio|src/include/pocketfft/|src/include/dr_libs/|src/cmake/FlmAudioCodecs\.cmake$|src/server/multipart\.(cpp|hpp)$|src/test/whisper_npu/|\.github/script/(asr_test\.py|asr_summary\.py|asr_gate\.sh|sweep_common\.py)$|\.github/script/assets/asr_eval/|\.github/workflows/asr-test\.yml$)'
asr_words='audio|transcri|whisper|multipart|speech|\basr\b|\bwav\b|mp3|flac|\bogg\b|opus|m4a|\bpcm\b|resampl'

say() { echo "run=$1"; echo "asr_gate: $2" >&2; exit 0; }

[ -n "$range" ] || say true "no diff range given: running"
files=$(git diff --name-only "$range" 2>/dev/null) || say true "cannot diff $range (shallow clone or force-push): running"
[ -n "$files" ] || say true "no changed files found in $range: running"

hit=$(printf '%s\n' "$files" | grep -E "$asr_only" || true)
[ -z "$hit" ] || say true "ASR-only files changed: $(printf '%s' "$hit" | head -3 | tr '\n' ' ')"

touched=$(printf '%s\n' "$files" | grep -E "$shared" || true)
[ -n "$touched" ] || say false "no ASR-related files changed"

# Only the shared server files changed: look at the changed lines themselves.
touched_list=$(echo $touched)
# shellcheck disable=SC2086
lines=$(git diff -U0 "$range" -- $touched_list | grep -E '^[+-]' | grep -v -E '^(\+\+\+|---)' || true)
if printf '%s\n' "$lines" | grep -q -i -E "$asr_words"; then
  say true "changed lines in $touched_list mention audio or speech: $(printf '%s\n' "$lines" | grep -i -E "$asr_words" | head -1 | cut -c1-100)"
fi
say false "only $touched_list changed, and no changed line is about audio or speech"
