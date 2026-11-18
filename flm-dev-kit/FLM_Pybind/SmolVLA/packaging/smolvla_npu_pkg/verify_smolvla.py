#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Compare the NPU SmolVLA against the PyTorch reference over several timesteps.

This replaces an older internal script, which drove an
instrumented copy of lerobot (enable_capture / enable_compact_prefix). Those
hooks do not exist in a clean lerobot install, so the reference here comes from
the public API only: policy.predict_action_chunk(batch, noise=...).

Image preprocessing stays on the python side. prepare_images() does the
resize_with_pad to 512x512 and the [0,1] -> [-1,1] SigLIP rescale, and the
resulting tensors are handed to both paths, so the two sides see bit-identical
pixels and only the transformer stack differs.

Both paths are driven with the same per-step noise, so the denoise loop is
reproducible and the residual is attributable to the NPU implementation.

The two paths alternate per step by default (--order interleaved): torch then
NPU, one frame at a time, so each frame's comparison is printed as it is
computed. --order phased instead runs every torch step first, then every NPU
step, so the NPU occupies one contiguous window that an external meter can
isolate; the torch policy is freed before the NPU phase starts, so nothing but
the NPU path is resident during it. --npu_repeat N replays the NPU pass to
widen that window (accuracy comes from the first pass; the rest are timing
only), and --cooldown S idles between the phases.

--power selects phased on its own, since that is the only order it can measure.
Passing --power --order interleaved is an error rather than a silent override.

--power adds mean/peak watts per frame, sampled around each individual forward:

    python verify_smolvla.py --steps 16 --power

     frame |     torch   mean    peak |       npu   mean    peak | speedup
    ------------------------------------------------------------------------
         0 |    1.983s  49.7W   59.0W |    0.374s  36.0W   47.0W |   5.30x

The rail is amdgpu hwmon "PPT" -- the same package power htop, btop and
`sensors` display. It covers the whole APU: CPU, iGPU and NPU together. This
platform exposes no NPU-only rail (xrt-smi reports "Estimated Power : N/A", and
RAPL's energy_uj is root-only), so read the two columns against each other
rather than as either backend's own draw.

Because that rail is thermal/boost governed it does not step to its steady value
the instant a phase starts: from cold it ramps over ~10 s of load, and after a
hot phase it decays over a comparable window. Measuring a phase across that
transient reports the ramp, not the backend -- a cold-started torch phase reads
low early and a hot-started NPU phase reads high early, and the two means drift
towards each other for reasons that have nothing to do with either backend. So
with --power both phases are warmed up and separated by an idle cooldown:

    --warmup_torch S  discarded torch forwards before any measured torch step
                      (default 15s with --power, 0 without)
    --warmup_npu S    the same for the npu phase (default 45s with --power).
                      Longer because the two phases settle for different
                      reasons: torch is the thermal ramp, which clears in
                      ~10 s, while the npu plateau is set by the cpu frequency
                      governor ramping the host work the npu path drives, which
                      takes 35-40 s. A 15 s npu warmup lands mid-ramp and the
                      phase reads 23 W instead of its true ~46 W.
    --warmup S        set both at once, overriding both defaults
    --cooldown S      idle before and after the npu phase so the package sheds
                      the previous phase's heat (default 20s with --power)

The steady-state summary drops each phase's first --power_skip frames as well,
so the reported watts come only from frames after the rail has settled. Set
--warmup 0 --cooldown 0 --power_skip 0 to recover the old raw behaviour.

The summary also reports the package idle floor, sampled from the settled tail
of the cooldown windows (or from a short dedicated --idle_probe when those are
disabled), and a "net" column with that floor subtracted. Both backend figures
are package-wide and therefore include the machine simply being powered on --
around 3 W here, which is an eighth of a 24 W reading -- so the net column is
what running the model actually costs.

Either order holds the prepared inputs for every step in memory (~6 MB per step
for a 2-camera checkpoint): staging happens up front so that neither the dataset
decode nor prepare_* lands inside a measurement window. A few hundred steps is
fine, a few thousand is not.

On the residual: run.exe only reproduces the reference action after overwriting
the freshly computed kv cache with reference values (main.cpp:187-196), because
the NPU prefill disagrees with the torch prefill -- cosine similarity of the
cached keys/values degrades from ~0.999 at layer 0 to ~0.964 at layer 15. This
script never skips the prefill, so that discrepancy shows up in the output. The
default --tol is set above the level it produces; a step that exceeds it means
something changed beyond the known prefill gap.

Run from FLM_Pybind/SmolVLA (the venv there has a clean lerobot):

    source activate.sh          # sets HF_HOME
    python verify_smolvla.py --steps 8

Build the module with `make python PROFILE=0` first for readable output --
PROFILE=1 (the default) makes every forward print its per-stage timing table.

The NPU allows one model per process and cannot be re-initialised after a
forward, so the instance is built once and reused across every timestep.

Any lerobot dataset works: its cameras are paired positionally with the
checkpoint's camera1..cameraN slots (--cameras overrides the pairing), and a
state/action width that differs from the checkpoint's is padded/sliced with a
warning. Known datasets (star marks best accuracy detected):

    1 camera : 
        lerobot/pusht
        lerobot/berkeley_mvp
        lerobot/aloha_sim_insertion_human *
    2 cameras: 
        lerobot/libero
        lerobot/libero_plus
        lerobot/austin_buds_dataset *
    3 cameras: 
        lerobot/droid_100
        lerobot/abc_130k_v3_smoke
        lerobot/aloha_mobile_wipe_wine *

Known bad accuracy model:
    lerobot/svla_so100_pickplace
    lerobot/austin_sailor_dataset
"""

import argparse
import contextlib
import gc
import glob
import os
import sys
import threading
import time

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

import smolvla_npu


# Defaults for the thermal controls, applied only under --power (see parse_args).
# All measured on this Ryzen AI host with --power_period 0.1.
#
# The two phases settle on different timescales because they settle for
# different reasons, so they get different warmups:
#
#   torch  climbs ~27 W -> ~58 W over roughly 8-10 s and is flat after. That is
#          the boost/thermal ramp: the cores go to work immediately and the
#          package power follows within seconds. 15 s clears it with margin.
#
#   npu    sits on a 23 W plateau for 35-40 s of sustained load.
DEFAULT_WARMUP_TORCH = 15.0
DEFAULT_WARMUP_NPU = 20.0

# After a phase stops, the rail decays back under ~25 W in roughly 15 s; the
# cooldown is set one step past that so the next phase starts from a floor.
DEFAULT_COOLDOWN = 20.0
DEFAULT_POWER_SKIP = 2

# Seconds of idle used to establish the package floor. The rail reaches it a few
# seconds after load stops (measured: ~3 W here), so this only has to be long
# enough to sample the settled tail, not to wait out the whole decay.
DEFAULT_IDLE_PROBE = 6.0
# Leading seconds of an idle window to discard: the rail is still decaying from
# the phase that just ended, and averaging that decay would report a floor
# several watts above the real one.
IDLE_SETTLE_DISCARD = 4.0


def flm_models_dir():
    """Where flm keeps its models: $FLM_MODEL_PATH/models, else ~/.config/flm/models."""
    root = os.environ.get("FLM_MODEL_PATH")
    if root:
        return os.path.join(root, "models")
    return os.path.join(os.path.expanduser("~"), ".config", "flm", "models")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path",
                    default=os.path.join(flm_models_dir(), "SmolVLA"),
                    help="NPU model directory (config.json, model.safetensors)")
    ap.add_argument("--model_id", default="lerobot/smolvla_base",
                    help="huggingface id of the reference torch checkpoint")
    ap.add_argument("--dataset", default="lerobot/libero")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--cameras", default=None,
                    help="comma separated dataset camera keys, in the order they "
                         "should fill the policy's camera slots (e.g. "
                         "'top,left_wrist'). The 'observation.images.' prefix is "
                         "optional. Default: the dataset's own camera order")
    ap.add_argument("--steps", type=int, default=16,
                    help="number of consecutive frames to compare")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; step t uses seed+t for its noise")
    ap.add_argument("--tol", type=float, default=0.8,
                    help="per-step error threshold for the pass/fail summary; the "
                         "default sits above the known prefill gap")
    ap.add_argument("--torch_threads", type=int, default=8,
                    help="intra-op thread count for the torch reference; the default "
                         "matches cpu_func::max_prefill_thread so the timings compare "
                         "like for like")
    # default=None, not "interleaved": parse_args has to tell "the user asked for
    # interleaved" from "nobody said", because --power promotes the second and
    # must reject the first.
    ap.add_argument("--order", choices=("phased", "interleaved"), default=None,
                    help="interleaved (default): alternate torch/npu per step, "
                         "printing each frame's comparison as it is computed. "
                         "phased: run every torch step, then every npu step, so "
                         "the npu occupies one contiguous window a power meter "
                         "can isolate. --power selects phased on its own")
    ap.add_argument("--npu_repeat", type=int, default=1,
                    help="phased only: replay the npu pass this many times to widen "
                         "the measurement window. Accuracy comes from the first "
                         "pass; later passes are timing only")
    ap.add_argument("--cooldown", "--settle", type=float, default=None,
                    dest="cooldown", metavar="S",
                    help="phased only: seconds to idle before and after the npu "
                         "phase, so the package sheds the torch phase's heat and "
                         "the meter sees a clean idle baseline around it. "
                         f"Default: {DEFAULT_COOLDOWN:g}s with --power, 0 without")
    ap.add_argument("--warmup", type=float, default=None, metavar="S",
                    help="seconds of discarded forwards to run at the head of "
                         "BOTH phases, so their measured steps start from a "
                         "settled power rail instead of climbing through the "
                         "ramp. Overridden per phase by --warmup_torch / "
                         "--warmup_npu. Default: those two flags' defaults")
    ap.add_argument("--warmup_torch", type=float, default=None, metavar="S",
                    help="--warmup for the torch phase only. Default: "
                         f"{DEFAULT_WARMUP_TORCH:g}s with --power, 0 without")
    ap.add_argument("--warmup_npu", type=float, default=None, metavar="S",
                    help="--warmup for the npu phase only. Longer than the torch "
                         "default because the npu plateau is governed by the cpu "
                         "frequency governor, which needs 35-40s of load to ramp, "
                         "not by the faster thermal ramp. Default: "
                         f"{DEFAULT_WARMUP_NPU:g}s with --power, 0 without")
    ap.add_argument("--skip_torch", action="store_true",
                    help="phased only: skip the torch phase entirely and report npu "
                         "timings without an accuracy comparison. Still loads the "
                         "policy, which prepare_images/prepare_state need")
    ap.add_argument("--power", action="store_true",
                    help="report SoC package watts (amdgpu hwmon PPT, the rail "
                         "htop/sensors show) alongside each frame's time. The rail "
                         "is whole-package, not NPU-only")
    ap.add_argument("--power_period", type=float, default=0.1,
                    help="seconds between power samples")
    ap.add_argument("--power_skip", type=int, default=None, metavar="N",
                    help="drop each phase's first N frames from the steady-state "
                         "power summary (they are still measured and printed). "
                         "Backstop for whatever ramp --warmup did not absorb. "
                         f"Default: {DEFAULT_POWER_SKIP} with --power, 0 without")
    ap.add_argument("--idle_probe", type=float, default=DEFAULT_IDLE_PROBE,
                    metavar="S",
                    help="seconds to idle at the end of the run to measure the "
                         "package's idle floor, which both backend figures sit on "
                         "top of. Skipped for free when a --cooldown window "
                         "already provides one. 0 disables the baseline")
    args = ap.parse_args()
    if args.npu_repeat < 1:
        raise SystemExit("--npu_repeat must be >= 1")

    # --order defaults to interleaved, except that --power promotes it to phased.
    # Interleaved is the better default for a plain accuracy run: it holds one
    # step's inputs instead of all of them, and alternating the backends costs
    # nothing when nobody is metering the rail. But a power run is *only*
    # meaningful phased -- interleaving re-heats each backend from the other
    # every step, so neither one ever gets a settled rail to be measured on.
    # An explicit --order interleaved is therefore an error under --power rather
    # than something to silently override: the user asked for a measurement this
    # script cannot make.
    if args.power and args.order == "interleaved":
        raise SystemExit("--power requires --order phased (which is what --power "
                         "selects by default); interleaving the backends re-heats "
                         "each from the other every step, so neither is ever "
                         "measured on a settled rail")
    if args.order is None:
        args.order = "phased" if args.power else "interleaved"

    # Reject only what the user actually typed. These are checked before the
    # defaults are filled in, so --power (which implies a warmup/cooldown) does
    # not get blamed for a phased-only flag nobody passed.
    phased_only = [name for name, val in (("--cooldown", args.cooldown),
                                          ("--warmup", args.warmup),
                                          ("--warmup_torch", args.warmup_torch),
                                          ("--warmup_npu", args.warmup_npu))
                   if val is not None] + \
                  (["--npu_repeat"] if args.npu_repeat != 1 else []) + \
                  (["--skip_torch"] if args.skip_torch else [])
    if args.order == "interleaved" and phased_only:
        # Alternating backends re-heat each other every step, so no amount of
        # warmup or cooldown gives either one a settled rail; only the phase
        # split can. The other two are phased-only for their own reasons.
        verb = "requires" if len(phased_only) == 1 else "require"
        raise SystemExit(f"{'/'.join(phased_only)} {verb} --order phased")

    # The ramp defaults only make sense when something is metering the rail, and
    # they cost 35s+ of wall clock, so a run without --power keeps its old speed.
    # --power implies phased by the resolution above, so this is just --power;
    # the order term stays as an assertion of that coupling.
    ramped = args.power and args.order == "phased"
    if args.cooldown is None:
        args.cooldown = DEFAULT_COOLDOWN if ramped else 0.0

    # Three-level resolution, most specific first: the per-phase flag wins, then
    # the both-phases --warmup, then the per-phase default. So --warmup 30 sets
    # both, --warmup 30 --warmup_npu 60 sets torch to 30 and npu to 60, and
    # passing nothing gives each phase the value its own ramp needs.
    for phase, both, dflt in (("torch", args.warmup, DEFAULT_WARMUP_TORCH),
                              ("npu", args.warmup, DEFAULT_WARMUP_NPU)):
        if getattr(args, f"warmup_{phase}") is None:
            setattr(args, f"warmup_{phase}",
                    both if both is not None else (dflt if ramped else 0.0))

    if args.power_skip is None:
        args.power_skip = DEFAULT_POWER_SKIP if args.power else 0

    if args.power_period <= 0:
        raise SystemExit("--power_period must be > 0")
    if args.warmup_torch < 0 or args.warmup_npu < 0 or args.cooldown < 0:
        raise SystemExit("--warmup/--warmup_torch/--warmup_npu/--cooldown "
                         "must be >= 0")
    if args.power_skip < 0:
        raise SystemExit("--power_skip must be >= 0")
    return args


IMAGE_PREFIX = "observation.images."


def resolve_camera_map(dataset, policy_cfg, requested=None):
    """Map the dataset's camera keys onto the ones the policy config declares.

    Every dataset names its cameras differently -- libero has image/image2,
    abc_130k has top/left_wrist/right_wrist -- while the checkpoint's config
    fixes its own slots (camera1..cameraN). prepare_images() looks up
    config.image_features by name, so a dataset whose names do not happen to
    match produces no images at all. Pair them up positionally instead: the
    dataset's cameras, in declaration order, fill the policy's slots in
    declaration order. --cameras overrides the source order (and can select a
    subset) when the positional guess pairs the wrong views.

    Returns {dataset_key: policy_key}, plus the policy slots left unfilled.
    """
    slots = list(policy_cfg.image_features)
    available = list(dataset.meta.camera_keys)

    if requested is not None:
        src_keys = []
        for name in requested.split(","):
            name = name.strip()
            if not name:
                continue
            key = name if name.startswith(IMAGE_PREFIX) else IMAGE_PREFIX + name
            if key not in available:
                raise SystemExit(f"camera {key!r} not in dataset (has: {', '.join(available)})")
            src_keys.append(key)
    else:
        src_keys = available

    if len(src_keys) > len(slots):
        print(f"WARNING: dataset has {len(src_keys)} cameras, policy declares "
              f"{len(slots)}; dropping {', '.join(src_keys[len(slots):])}")
        src_keys = src_keys[:len(slots)]

    return dict(zip(src_keys, slots)), slots[len(src_keys):]


def rename_cameras(frame, camera_map):
    """Rename the dataset's camera keys to the policy's slot names.

    Unmapped camera keys are dropped: they would be normalized by preprocess()
    for nothing, since prepare_images() only reads config.image_features. The
    renamed views are collected before any is written back, so a dataset key
    that collides with a slot name (top -> camera1 while camera1 -> camera2)
    cannot clobber a view that has not been read yet.
    """
    renamed = {dst: frame[src] for src, dst in camera_map.items() if src in frame}
    kept = {k: v for k, v in frame.items() if not k.startswith(IMAGE_PREFIX)}
    kept.update(renamed)
    return kept


def find_ppt_sensor():
    """Locate the amdgpu PPT hwmon sensor, or None if this host has no readable one.

    Same rail htop/btop and `sensors` read. On a Ryzen AI APU the NPU shares the
    package with the CPU and iGPU, and the amdgpu driver exposes package power
    (label "PPT") as a plain hwmon file readable without root. It is SoC-wide,
    not per-IP: this platform has no NPU-only rail (xrt-smi reports
    "Estimated Power : N/A", RAPL's energy_uj is root-only). So the watts below
    are the whole package while that backend runs.
    """
    for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(os.path.join(hwmon, "name")) as f:
                if f.read().strip() != "amdgpu":
                    continue
        except OSError:
            continue
        for label_path in sorted(glob.glob(os.path.join(hwmon, "power*_label"))):
            try:
                with open(label_path) as f:
                    if f.read().strip() != "PPT":
                        continue
            except OSError:
                continue
            path = label_path.replace("_label", "_input")
            try:
                with open(path) as f:
                    int(f.read())
            except (OSError, ValueError):
                continue
            return path
    return None


class PowerSampler:
    """Samples a hwmon power*_input file on a daemon thread, in microwatts.

    A thread is needed because npu.forward() blocks; the read is a ~10 us sysfs
    open/read/close, so at the default 100 ms period it costs nothing and stays
    off the thread doing the work. Start/stop around one forward to get that
    forward's mean and peak watts.
    """

    def __init__(self, path, period=0.1):
        self.path, self.period = path, period
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                with open(self.path) as f:
                    self.samples.append(int(f.read()))
            except (OSError, ValueError):
                pass  # a transient sysfs read failure should not kill the run
            self._stop.wait(self.period)

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)
        return False

    def read_now(self):
        """One immediate synchronous reading, in watts (or None)."""
        try:
            with open(self.path) as f:
                return int(f.read()) / 1e6
        except (OSError, ValueError):
            return None

    def stats(self):
        """(mean, peak) in watts, or (None, None) if nothing was sampled.

        A forward shorter than one sample period can end with an empty list, so
        the caller has to tolerate None rather than assume a reading landed.
        """
        if not self.samples:
            return None, None
        w = [s / 1e6 for s in self.samples]
        return sum(w) / len(w), max(w)


def clone_batch(batch):
    """predict_action_chunk mutates the batch (queues get stacked back into it)."""
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def summarize(a, b):
    """max-abs, mean-abs, cosine similarity and normalized MSE.

    Absolute error alone is hard to read: the action values themselves span a
    couple of units, so cosine similarity says whether the NPU is producing the
    same chunk with a small perturbation or something structurally different.
    nMSE is ||a-b||^2 / ||b||^2 -- error energy over reference energy. The 1/N
    in each mean cancels, so it is exactly MSE(a,b) / MSE(b,0): dimensionless,
    and comparable across steps with different scales. This is the same
    definition the C++ harness prints (metrices.hpp NormalizedMSE), so numbers
    from run.exe and from here are in the same units; note it is a *squared*
    quantity, so a 3% relative error reads as 9e-4, not 0.03.
    Cosine similarity is blind to a uniform gain error, nMSE is not; reporting
    both separates "rotated" from "rescaled".
    """
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
    ref_energy = float((b.float() ** 2).sum())
    err_energy = float((d.float() ** 2).sum())
    nmse = err_energy / ref_energy if ref_energy > 0 else float("nan")
    return float(d.max()), float(d.mean()), float(cos), nmse


def main():
    args = parse_args()
    device = torch.device("cpu")  # the reference runs alongside the NPU on this host

    # The NPU path runs its CPU kernels on an 8 thread OpenMP team
    # (cpu_func::max_prefill_thread), so give torch the same budget. Left at the
    # default torch would spread over every core and the wall clock comparison
    # below would be measuring core count, not the two implementations.
    torch.set_num_threads(args.torch_threads)

    print(f"npu model:  {args.model_path}")
    print(f"torch ref:  {args.model_id}")
    print(f"torch cpu:  {torch.get_num_threads()} threads")
    print(f"dataset:    {args.dataset} episode {args.episode}, {args.steps} steps")

    # --- torch reference ---------------------------------------------------
    policy = SmolVLAPolicy.from_pretrained(args.model_id).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim
    action_dim = policy.config.action_feature.shape[0]

    # pyav ships its own ffmpeg, so this works without a system libav install
    dataset = LeRobotDataset(args.dataset, video_backend="pyav")
    if not 0 <= args.episode < dataset.meta.total_episodes:
        raise SystemExit(f"episode {args.episode} out of range "
                         f"(dataset has {dataset.meta.total_episodes})")
    from_idx = int(dataset.meta.episodes["dataset_from_index"][args.episode])
    # Clamp to this episode, not to the dataset: running off its end would feed
    # frames from the next episode, which is a different scene and task.
    to_idx = int(dataset.meta.episodes["dataset_to_index"][args.episode])
    end_idx = min(from_idx + args.steps, to_idx)
    frame_indices = list(range(from_idx, end_idx))
    if len(frame_indices) < args.steps:
        print(f"WARNING: only {len(frame_indices)} frames in episode {args.episode} "
              f"(wanted {args.steps})")

    # The dataset's camera names are its own; the checkpoint's slots are fixed.
    camera_map, empty_slots = resolve_camera_map(dataset, policy.config, args.cameras)
    print("cameras:   " + ", ".join(f"{src.removeprefix(IMAGE_PREFIX)} -> "
                                    f"{dst.removeprefix(IMAGE_PREFIX)}"
                                    for src, dst in camera_map.items()))
    if empty_slots:
        # prepare_images only pads missing slots up to config.empty_cameras, so
        # beyond that the prefix is simply shorter than the checkpoint expects.
        filled = min(len(empty_slots), policy.config.empty_cameras)
        print(f"WARNING: {len(empty_slots)} policy camera slot(s) unfilled "
              f"({', '.join(s.removeprefix(IMAGE_PREFIX) for s in empty_slots)}); "
              f"{filled} padded with blank images, {len(empty_slots) - filled} dropped")

    # The checkpoint declares its own state/action widths. A dataset that
    # disagrees still runs -- prepare_state pads to max_state_dim and the chunk
    # is sliced to the config's action width -- but the comparison is then
    # against a reference that is not modelling this robot, so say so.
    ds_state = dataset.meta.features.get("observation.state", {}).get("shape")
    ds_action = dataset.meta.features.get("action", {}).get("shape")
    cfg_state = policy.config.robot_state_feature.shape[0]
    if ds_state and ds_state[0] != cfg_state:
        print(f"WARNING: dataset state is {ds_state[0]}-dim, checkpoint expects "
              f"{cfg_state}; both paths see the same padded vector, so the "
              f"npu/torch comparison is still valid")
    if ds_action and ds_action[0] != action_dim:
        print(f"WARNING: dataset action is {ds_action[0]}-dim, checkpoint emits "
              f"{action_dim}; actions are compared at the checkpoint's width")

    # --- npu ---------------------------------------------------------------
    npu = smolvla_npu.SmolVLA(args.model_path)
    print(f"npu:        {npu!r}")
    if npu.chunk_size != chunk_size or npu.action_dim != max_action_dim:
        print(f"WARNING: shape mismatch -- npu [{npu.chunk_size}, {npu.action_dim}] vs "
              f"torch [{chunk_size}, {max_action_dim}]")

    # --- stage every step's inputs -----------------------------------------
    # Both phases read from this list, so the dataset decode, preprocess and
    # prepare_* run exactly once per step regardless of --order. Doing them here
    # also keeps them out of the NPU measurement window: by the time that phase
    # starts, the only work left is npu.forward().
    staged = []
    for step, frame_index in enumerate(frame_indices):
        # Dataset read and camera renaming are harness cost -- a real deployment
        # gets these frames from a camera -- so they sit outside every timer.
        frame = rename_cameras(dataset[frame_index], camera_map)

        # preprocess() is charged to BOTH paths: normalization and device
        # placement are unavoidable for either backend.
        t0 = time.perf_counter()
        batch = preprocess(frame)
        t_pre = time.perf_counter() - t0

        # prepare_images (resize_with_pad + SigLIP rescale) and prepare_state are
        # charged to the NPU path ONLY. predict_action_chunk runs both itself
        # inside _get_action_chunk, so billing torch for them here as well would
        # count the same resize twice against it.
        t0 = time.perf_counter()
        with torch.inference_mode():
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
        t_prep = time.perf_counter() - t0
        tokens = batch[OBS_LANGUAGE_TOKENS]

        if step == 0:
            print(f"\nimages: {len(images)} x {tuple(images[0].shape)} {images[0].dtype}"
                  f" | tokens {tuple(tokens.shape)} | state {tuple(state.shape)}")
            if len(images) > smolvla_npu.MAX_IMAGES:
                print(f"WARNING: policy produced {len(images)} images, npu takes "
                      f"{smolvla_npu.MAX_IMAGES}; extra ones are dropped")

        # Same noise on both sides: sample_actions seeds its denoise loop with it,
        # so without pinning it the two outputs would be incomparable.
        gen = torch.Generator(device=device).manual_seed(args.seed + step)
        noise = torch.normal(mean=0.0, std=1.0, size=(1, chunk_size, max_action_dim),
                             generator=gen, dtype=torch.float32, device=device)

        staged.append({
            "step": step,
            "frame_index": frame_index,
            "batch": batch,
            "noise": noise,
            "t_pre": t_pre,
            "t_prep": t_prep,
            # flattened exactly as npu.forward wants them, so the NPU phase does
            # no tensor reshaping of its own
            "npu_args": (
                [img.reshape(-1).contiguous() for img in images[:smolvla_npu.MAX_IMAGES]],
                tokens.reshape(-1).contiguous(),
                state.reshape(-1).contiguous(),
                noise.reshape(-1).contiguous(),
            ),
        })

    power_path = find_ppt_sensor() if args.power else None
    if args.power and not power_path:
        print("WARNING: --power requested but no readable amdgpu PPT hwmon sensor "
              "was found; continuing without power measurement")

    def sampled(power_key, s):
        """Context manager that measures one forward's watts into s[power_key].

        Without --power it is a no-op. Callers start their clock INSIDE this
        context, so the thread start/join (~1.7 ms) stays out of the reported
        time; what remains inside is the sampler thread's periodic sysfs read,
        a ~10 us open/read/close every --power_period.
        """
        if not power_path:
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def _cm():
            with PowerSampler(power_path, args.power_period) as ps:
                yield
            s[power_key] = ps.stats()
        return _cm()

    def run_torch(s):
        """One torch reference forward + postprocess. Fills t_ref/t_post_ref/ref_a0."""
        with sampled("w_ref", s):
            # Inside the sampler, not around it: starting the thread and joining
            # it costs ~1.7 ms, which has nothing to do with the forward and
            # would otherwise make every --power run look slower than it is.
            t0 = time.perf_counter()
            policy.reset()  # drop the action queue so this frame drives a fresh chunk
            with torch.inference_mode():
                ref_chunk = policy.predict_action_chunk(clone_batch(s["batch"]),
                                                        noise=s["noise"])
            s["t_ref"] = time.perf_counter() - t0
        ref_chunk = ref_chunk[0].float().cpu()          # [chunk_size, action_dim]

        # First action of the chunk, in real units -- the row select_action would
        # actually hand to the robot, and the number worth reading per step.
        # Timed per side; unnormalization is part of what it costs to produce a
        # usable action, even though it happens to be a no-op for this checkpoint.
        t0 = time.perf_counter()
        with torch.inference_mode():
            s["ref_a0"] = postprocess(ref_chunk[:1].clone()).reshape(-1)
        s["t_post_ref"] = time.perf_counter() - t0

    def run_npu(s, keep):
        """One NPU forward. Appends its time; only the first pass keeps the output.

        Everything outside npu.forward() is deliberately minimal here -- the
        postprocess and the finiteness check are hoisted out to the compare pass
        so the sampled region is npu.forward() and nothing else.
        """
        with sampled("w_npu", s):
            # See run_torch: the clock starts inside the sampler so the thread
            # start/join never lands in a reported forward time.
            t0 = time.perf_counter()
            npu_chunk = npu.forward(*s["npu_args"])
            s.setdefault("t_npu", []).append(time.perf_counter() - t0)
        if keep:
            s["npu_chunk"] = npu_chunk[:, :action_dim]  # drop the action padding

    def rail_now():
        """Current package watts as a printable suffix, or '' without a sensor."""
        if not power_path:
            return ""
        w = PowerSampler(power_path).read_now()
        return f" (rail {w:.1f}W)" if w is not None else ""

    def warm_up(seconds, run_one, label):
        """Run discarded forwards for `seconds` so the phase starts on a settled rail.

        The PPT rail is boost/thermal governed: it does not jump to the steady
        value when a phase begins, it ramps there over several seconds. Whichever
        phase runs first therefore reads low early, and whichever runs second
        starts hot and reads high early -- exactly the artefact where torch
        appeared to climb 27->58 W while the npu appeared to fall 44->23 W. The
        two curves are the package's thermal response, not the backends.

        Loading the rail here, with the output thrown away, moves that ramp
        before the first measured frame. Each iteration is one whole forward --
        stopping mid-forward is not possible -- so the actual warmup overshoots
        `seconds` by up to one forward (~0.5s), which is the intent: it should
        end on a settled rail, not exactly on the clock.

        Results go to a scratch dict so nothing here reaches the staged steps:
        run_npu would otherwise append warmup times to s["t_npu"] and drag the
        reported per-frame mean towards the ramp it exists to exclude.
        """
        if seconds <= 0:
            return
        print(f"warming up {label} for {seconds:.0f}s{rail_now()}", flush=True)
        # Rotate over the staged steps rather than repeating one: a single frame
        # would let a cache stay hot in a way the measured loop never sees.
        scratch = [{k: s[k] for k in ("batch", "noise", "npu_args")} for s in staged]
        deadline = time.perf_counter() + seconds
        n = 0
        while time.perf_counter() < deadline:
            run_one(scratch[n % len(scratch)])
            n += 1
        print(f"  {label} warmup done, {n} discarded forward(s){rail_now()}", flush=True)

    idle_samples = []      # watts, from the settled tail of every idle window

    def measure_idle(seconds, why):
        """Idle for `seconds`, recording the settled tail as the package floor.

        Both backend figures are package-wide, so both include whatever the
        machine draws doing nothing. Sampling that floor here lets the summary
        report each backend net of it, instead of leaving the reader to guess
        how much of "24 W npu" is just the box being on.

        The first IDLE_SETTLE_DISCARD seconds are dropped: the rail is still
        falling from the phase that just ended (measured decay from ~47 W to
        ~3 W takes a few seconds), and averaging that tail-off would report a
        floor well above the real one.
        """
        if seconds <= 0:
            return
        print(f"idling {seconds:.0f}s {why}{rail_now()}", flush=True)
        if not power_path:
            time.sleep(seconds)
            return
        with PowerSampler(power_path, args.power_period) as ps:
            time.sleep(seconds)
        # Keep only the settled tail. If the window is too short to have one,
        # take the last sample rather than nothing -- a slightly high floor is
        # still more informative than no floor, and it is reported as a count.
        drop = int(min(IDLE_SETTLE_DISCARD, seconds * 0.5) / args.power_period)
        tail = ps.samples[drop:] or ps.samples[-1:]
        idle_samples.extend(s / 1e6 for s in tail)
        print(f"  idle done{rail_now()}", flush=True)

    def cool_down(seconds, why):
        """Idle so the package sheds the previous phase's heat before the next one.

        Warmup alone cannot fix a hot start: it would spend the whole window
        measuring the npu on a rail still elevated by torch, and report the decay
        of someone else's heat as the npu's own draw. The idle has to come first,
        then the npu's own warmup brings the rail up to *its* steady state.
        """
        measure_idle(seconds, why)

    if args.order == "interleaved":
        npu_window = None
        for s in staged:
            run_torch(s)
            run_npu(s, keep=True)
    else:
        if args.skip_torch:
            print(f"\n--- torch phase skipped ---")
        else:
            print(f"\n--- phase 1/2: torch, {len(staged)} steps ---", flush=True)
            # The staging loop above already ran preprocess/prepare_* on the CPU,
            # so the rail is part-way up rather than at idle. Warm up regardless:
            # the measured frames need the *torch phase's* steady state, and a
            # partial ramp is still a ramp.
            warm_up(args.warmup_torch, run_torch, "torch")
            t_phase0 = time.perf_counter()
            for s in staged:
                run_torch(s)
            print(f"--- torch phase done in {time.perf_counter() - t_phase0:.1f}s"
                  f"{rail_now()} ---")

        # Free the reference policy before the measurement: its weights are
        # ~2 GB of resident float32 that the NPU phase has no use for, and
        # dropping them keeps the phase from competing for memory bandwidth.
        # preprocess/postprocess are separate objects and stay alive.
        del policy
        gc.collect()

        # Cool down BEFORE the npu warmup, not instead of it: the idle sheds
        # torch's heat, then the warmup takes the rail back up to whatever the
        # npu itself sustains. Doing only one of the two leaves the npu measured
        # on a rail that torch, not the npu, put where it is.
        cool_down(args.cooldown, "before the npu phase, to shed the torch phase's heat")

        reps = f" x{args.npu_repeat}" if args.npu_repeat > 1 else ""
        print(f"\n--- phase 2/2: NPU, {len(staged)} steps{reps} ---", flush=True)
        warm_up(args.warmup_npu, lambda s: run_npu(s, keep=False), "npu")
        t_phase1 = time.perf_counter()
        for rep in range(args.npu_repeat):
            for s in staged:
                run_npu(s, keep=(rep == 0))
        npu_window = time.perf_counter() - t_phase1
        print(f"--- NPU phase done in {npu_window:.1f}s{rail_now()} ---", flush=True)

        cool_down(args.cooldown, "after the npu phase, for a clean idle baseline")

    # The cooldown windows above already sat idle with the sampler running, so
    # normally the floor is free. Only pay for a dedicated probe when they were
    # disabled (--cooldown 0, or --order interleaved, which has no phase gap).
    if power_path and args.idle_probe > 0 and not idle_samples:
        measure_idle(args.idle_probe, "to measure the package idle floor")

    # --- compare -----------------------------------------------------------
    # One line per frame: how long each backend took on it and, with --power,
    # what the package drew while it ran.
    def w(cell):
        """Format a (mean, peak) watt pair, or blanks when nothing was sampled."""
        if not power_path:
            return ""
        mean, peak = cell if cell else (None, None)
        if mean is None:
            return f"{'  --':>7}{'':>8}"   # forward shorter than one sample period
        return f"{mean:6.1f}W{peak:7.1f}W"

    hdr_pow = f"{'mean':>7}{'peak':>8}" if power_path else ""
    print()
    print(f"{'frame':>6} | {'torch':>9}{hdr_pow} | {'npu':>9}{hdr_pow} | speedup"
          + ("" if args.skip_torch else " |      nMSE      cos"))
    print("-" * (78 + (30 if power_path else 0)))

    rows = []
    for s in staged:
        step = s["step"]
        npu_chunk = s["npu_chunk"]
        t0 = time.perf_counter()
        with torch.inference_mode():
            npu_a0 = postprocess(npu_chunk[:1].clone()).reshape(-1)
        t_post_npu = time.perf_counter() - t0

        # mean over the repeats; with --npu_repeat 1 this is just that one time
        t_npu = sum(s["t_npu"]) / len(s["t_npu"])

        # End to end: observation in, executable action out. torch does its own
        # prepare_images inside predict_action_chunk, so t_prep belongs to the
        # NPU column only -- that resize is real work either way, just billed
        # where it is actually performed.
        e2e_npu = s["t_pre"] + s["t_prep"] + t_npu + t_post_npu
        finite = bool(torch.isfinite(npu_chunk).all())

        if args.skip_torch:
            rows.append((step, s["frame_index"], float("nan"), float("nan"),
                         finite, float("nan"), e2e_npu,
                         s["t_pre"], s["t_prep"], float("nan"), t_npu))
            print(f"{s['frame_index']:6d} | {'':>9}{'' if not power_path else ' ' * 15}"
                  f" | {t_npu:8.3f}s{w(s.get('w_npu'))} |")
            continue

        ref_a0 = s["ref_a0"]
        e2e_ref = s["t_pre"] + s["t_ref"] + s["t_post_ref"]
        _, _, a0_cos, a0_nmse = summarize(npu_a0, ref_a0)

        rows.append((step, s["frame_index"], a0_cos, a0_nmse, finite,
                     e2e_ref, e2e_npu, s["t_pre"], s["t_prep"], s["t_ref"], t_npu))
        print(f"{s['frame_index']:6d} | {s['t_ref']:8.3f}s{w(s.get('w_ref'))}"
              f" | {t_npu:8.3f}s{w(s.get('w_npu'))}"
              f" | {s['t_ref'] / t_npu:6.2f}x | {a0_nmse:9.3e} {a0_cos:8.5f}")
        if step == 0:
            print("       ref action[0]: [" + ", ".join(f"{v:.5f}" for v in ref_a0.tolist()) + "]")
            print("       npu action[0]: [" + ", ".join(f"{v:.5f}" for v in npu_a0.tolist()) + "]")

    # --- summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    n = len(rows)
    if args.skip_torch:
        print(f"{n} steps | no accuracy comparison (--skip_torch)")
    else:
        worst_a0_cos = min(r[2] for r in rows)
        worst_a0_nmse = max(r[3] for r in rows)
        print(f"{n} steps | action[0]: worst nMSE {worst_a0_nmse:.3e},"
              f" worst cos {worst_a0_cos:.5f}")

    if not all(r[4] for r in rows):
        bad = [str(r[0]) for r in rows if not r[4]]
        print(f"FAILED: non-finite output at step(s) {', '.join(bad)}")
        return 1

    # Step 0 pays for lazily-built caches on both sides, so it is reported but
    # excluded from the mean.
    timed = rows[1:] if n > 1 else rows
    avg = lambda i: sum(r[i] for r in timed) / len(timed)
    print(f"\ntiming, mean over {len(timed)} step(s)"
          f"{' (step 0 excluded as warmup)' if n > 1 else ''}:")
    print(f"  preprocess   {avg(7) * 1e3:7.1f} ms   (both paths)")
    print(f"  prepare_*    {avg(8) * 1e3:7.1f} ms   (npu path; torch does this internally)")
    if args.skip_torch:
        print(f"  model        {avg(10) * 1e3:7.1f} ms npu")
        print(f"  end to end   {avg(6) * 1e3:7.1f} ms npu")
    else:
        print(f"  model        {avg(9) * 1e3:7.1f} ms torch / {avg(10) * 1e3:7.1f} ms npu")
        print(f"  end to end   {avg(5) * 1e3:7.1f} ms torch / {avg(6) * 1e3:7.1f} ms npu"
              f"   ({avg(5) / avg(6):.2f}x)")

    if power_path:
        def phase_w(key, skip):
            """(mean, peak, n) over the frames kept for the steady-state number.

            `skip` drops the head of the phase. Averaging a phase whole averages
            its boost ramp too, which is what made the raw means unreadable: the
            same frames that made torch look cheap early made the npu look
            expensive early, and neither had anything to do with the backend.
            """
            vals = [s[key] for s in staged[skip:]
                    if s.get(key) and s[key][0] is not None]
            if not vals:
                return None, None, 0
            return (sum(v[0] for v in vals) / len(vals),
                    max(v[1] for v in vals), len(vals))

        # Never drop so much that nothing is left: on a short --steps run the
        # skip can exceed the frame count, and a summary computed over zero
        # frames is worse than one computed over a slightly ramped frame.
        skip = args.power_skip if args.power_skip < len(staged) else 0
        if skip != args.power_skip:
            print(f"\nNOTE: --power_skip {args.power_skip} >= {len(staged)} measured "
                  f"frames; reporting all of them instead")

        idle_w = sum(idle_samples) / len(idle_samples) if idle_samples else None

        print(f"\nSoC package power ({power_path}, the rail htop/sensors read):")
        means = {}
        for name, key in (("torch", "w_ref"), ("npu", "w_npu")):
            m, pk, n = phase_w(key, skip)
            means[name] = m
            if m is None:
                continue
            # The net column is the point of measuring the floor: a package
            # figure includes the machine simply being powered on, and at a
            # ~3 W floor against a 24 W reading that is an eighth of the number.
            net = f"  net {m - idle_w:6.1f} W" if idle_w is not None else ""
            print(f"  {name:<6} {m:6.1f} W mean  {pk:6.1f} W peak{net}   "
                  f"(over {n} frame(s))")

        if idle_w is not None:
            print(f"  {'idle':<6} {idle_w:6.1f} W mean                     "
                  f"(over {len(idle_samples)} sample(s), package doing nothing)")

        if means.get("torch") and means.get("npu"):
            line = (f"  ratio  {means['torch'] / means['npu']:.2f}x  "
                    f"(torch / npu, package while each backend runs)")
            if idle_w is not None and means["npu"] > idle_w:
                # Subtracting a floor common to both makes the ratio larger; the
                # two numbers answer different questions ("what does the wall
                # see" vs "what did the work cost"), so print both rather than
                # picking one.
                net_ratio = (means["torch"] - idle_w) / (means["npu"] - idle_w)
                line += f",  {net_ratio:.2f}x net of idle"
            print(line)

        conditioned = []
        if args.warmup_torch or args.warmup_npu:
            # Name them separately when they differ -- which is the default --
            # so the summary does not imply both phases got the same treatment.
            if args.warmup_torch == args.warmup_npu:
                conditioned.append(f"{args.warmup_torch:.0f}s warmup per phase")
            else:
                conditioned.append(f"{args.warmup_torch:.0f}s torch / "
                                   f"{args.warmup_npu:.0f}s npu warmup")
        if args.cooldown:
            conditioned.append(f"{args.cooldown:.0f}s cooldown between phases")
        if skip:
            conditioned.append(f"first {skip} frame(s) dropped")
        if conditioned:
            print(f"  Measured after: {', '.join(conditioned)} -- so these are\n"
                  "  steady-state figures, not the boost ramp.")
        else:
            # Worth saying plainly: without any of the three, the two means are
            # a cold phase and a hot phase and are not comparable.
            print("  WARNING: no warmup/cooldown/skip applied, so each phase's mean\n"
                  "  includes its power ramp -- the first phase reads low and the\n"
                  "  second reads high. Re-run with --warmup/--cooldown for a fair\n"
                  "  comparison.")
        print("  This is the whole APU -- CPU, iGPU and NPU share it, and there is\n"
              "  no NPU-only rail on this platform. Compare the two numbers against\n"
              "  each other rather than reading either as the NPU's own draw.")
        if idle_w is not None:
            print(f"  'net' subtracts the {idle_w:.1f} W idle floor, so it is the cost of\n"
                  f"  running the model rather than of the machine being on. It is\n"
                  f"  still package-wide: the npu path also drives host CPU work,\n"
                  f"  which lands in the same rail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
