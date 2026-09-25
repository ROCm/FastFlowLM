# Building the `--rai` backend on Linux from scratch

How to get from a bare checkout to `flm run phi4-mini-it-rai` on a machine with
an AMD NPU. The `--rai` backend reaches kernels through **ryzenai-corelib**
instead of FastFlowLM's own kernel flow, and corelib lives outside this
repository, so most of the work is standing up that prefix.

Written against the working install on this machine. Paths use `$P` for the
dependency prefix (here `/home/alfxu/ddprefix`); substitute your own.

> **Status:** the backend builds, loads a model and dispatches, but a corelib
> bug makes it very slow — see [Known issues](#known-issues) before you start,
> so the performance is not a surprise.

---

## 0. What you need first

| Thing | This machine | Check |
|---|---|---|
| AMD NPU + `amdxdna` driver | `/dev/accel/accel0` | `ls /dev/accel/` |
| XRT | `/opt/xilinx/xrt` | `source /opt/xilinx/xrt/setup.sh && xrt-smi examine` |
| CMake ≥ 3.21, Ninja, GCC with C++17 | GCC 15 | `cmake --version` |
| Raised `memlock` limit | see step 4 | `ulimit -l` |

No sudo is needed for anything below **except** the `memlock` limits file in
step 4 and installing XRT itself.

---

## 1. Dependency prefix: DynamicDispatch, then corelib

This is the hard part and it is already written up in **`$P/BUILD-NOTES.md`** —
read that first. It records the two things that actually matter (the required
DD branch, and `BUILD_SHARED_LIBS=ON` for DD), the sudo-free dependency prefix,
the non-PIC link traps, and the exact `cmake` command lines for both projects.

Do not re-derive it. The short version:

```bash
P=/home/alfxu/ddprefix      # your dependency prefix

# DynamicDispatch -> $P/install-dds     (see BUILD-NOTES.md for the full line)
cmake -S DynamicDispatch -B build-dds ... -DCMAKE_INSTALL_PREFIX=$P/install-dds
cmake --build build-dds -j"$(nproc)" --target install

# ryzenai-corelib -> $P/install-corelib
cmake -S ryzenai-corelib -B build-corelib -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$P/install-corelib \
  -DRYZENAI_CORELIB_STAGE_RUNTIME_DLLS=OFF \
  -DCMAKE_PREFIX_PATH="$P/install-dds;$P/shim;$P/root/usr;$P/root/usr/lib/x86_64-linux-gnu;/opt/xilinx/xrt"
cmake --build build-corelib -j"$(nproc)"
```

**Version must match exactly.** `src/include/rai/corelib_api.hpp` has a
`#error` for anything other than corelib **0.5.0**:

```bash
grep -E 'CORELIB_VERSION_(MAJOR|MINOR|PATCH)' \
  $P/install-corelib/include/ryzenai/corelib.h
```

Must print `0`, `5`, `0`. (`BUILD-NOTES.md` still says 0.4.0 in its opening
line — that text is stale; the installed headers are 0.5.0.)

### Verify corelib before going further

```bash
source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=$P/install-dds/lib:$P/install-corelib/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
$P/build-corelib/tests/ryzenai_corelib_tests
```

All tests should pass. Do this now — if corelib cannot reach the NPU, every
failure later in the FastFlowLM build will be reported as something else.

Note the suite only dispatches **one** op on hardware, **once**. A pass means
device setup and the build are sound; it does not exercise repeated dispatch.

---

## 2. Build FastFlowLM with `--rai`

The installer does configure, build and relocate in one step:

```bash
cd /home/alfxu/FastFlowLM/src
RYZENAI_CORELIB_ROOT=$P/install-corelib ./home_install.sh --rai
```

`RYZENAI_CORELIB_ROOT` is needed **once** — CMake caches it. If you ever delete
`src/build/`, you must pass it again, because the cache goes with it.

What the flags do:

- `--rai` selects the `linux-rai-on` preset (`-DFLM_ENABLE_RAI=ON`) and installs
  to `/scratch/$USER/flm_exe_rai` rather than `~/flm_exe`. A rai prefix stages
  corelib and DynamicDispatch, which are far too large for a home directory, and
  a rai build and a stock one would otherwise overwrite each other.
- `FLM_PREFIX=/path` overrides the install location.
- `--no-build` installs an existing `build/` tree without rebuilding.

### How the corelib lookup resolves

`CMakeLists.txt:57-85`, in order:

| | Headers (`ryzenai/corelib.h`) | Library (`libryzenai_corelib`) |
|---|---|---|
| 1 | `src/include/` (vendored) | `src/lib/` (vendored) |
| 2 | `$RYZENAI_CORELIB_ROOT/include` | `$RYZENAI_CORELIB_ROOT/lib` |
| 3 | `CMAKE_PREFIX_PATH` | `CMAKE_PREFIX_PATH` |

So dropping the headers into `src/include/` and `libryzenai_corelib.so` into
`src/lib/` is an alternative to the variable entirely.

**If you put a `.so` in `src/lib/`, it must keep its RUNPATH.** Copy the one
from the corelib *build tree*, not a stripped or patched copy:

```bash
objdump -p src/lib/libryzenai_corelib.so | grep RUNPATH
# -> /home/alfxu/ddprefix/install-dds/lib:/opt/xilinx/xrt/lib:
```

Without that RUNPATH it will not find DynamicDispatch at run time.

### Rebuilding after a source change

```bash
cd /home/alfxu/FastFlowLM/src
cmake --build build -j"$(nproc)"
cp build/flm /scratch/$USER/flm_exe_rai/flm
```

Faster than re-running the installer, and it skips the XRT-shadowing problem in
step 3.

---

## 3. Fix the shadowed XRT library — every time you run the installer

`home_install.sh` stages `libxrt_coreutil.so.2` into the install prefix, where
it shadows the real XRT and breaks the NPU with confusing errors. Delete it
after **every** installer run:

```bash
rm -f /scratch/$USER/flm_exe_rai/lib/libxrt_coreutil.so.2
```

This is a known bug in the installer's `ldd` staging walk
(`home_install.sh:251-261`), not yet fixed. A rai prefix deliberately bundles no
XRT — `flm_env.sh` pins `XILINX_XRT` to the real root instead — so the staged
copy is wrong by construction.

---

## 4. Raise the `memlock` limit

A Phi-4 model pins several GB. The default 8 MB fails at load with a pinning
error that does not name the limit.

```bash
# /etc/security/limits.d/99-memlock.conf   (needs sudo, once)
alfxu hard memlock 16777216
```

That is the **hard** limit in KB (16 GiB). Then, in the shell that runs `flm`:

```bash
ulimit -l unlimited     # or: ulimit -l 16777216
ulimit -l               # confirm
```

**The limits file applies only to sessions started after it was written.** A
long-running shell, a tmux server, or any process whose chain predates the file
keeps the old 8 MB cap and cannot raise it. If `ulimit -l` still reports `8192`,
log out and back in, and `tmux kill-server` if you use tmux.

---

## 5. Run

```bash
source /opt/xilinx/xrt/setup.sh
source /scratch/$USER/flm_exe_rai/flm_env.sh
/scratch/$USER/flm_exe_rai/flm run phi4-mini-it-rai -c 4096
```

`flm_env.sh` sets `FLM_CONFIG_PATH`, `FLM_XCLBIN_PATH`, `FLM_MODEL_PATH`
(`/scratch/alfxu`) and pins `XILINX_XRT`. It does **not** set `ulimit` — step 4
is on you.

Models live under `$FLM_MODEL_PATH/<name>`; `phi4-mini-it-rai` is the one set up
here, and it must also have an entry in
`share/flm/model_list.json` for `flm run` to resolve the name.

---

## Known issues

### Stream reuse — the big one

A corelib stream faults on a later dispatch: the array raises stream switch port
parity errors, the driver tears the hardware context down, and XRT reports
`ERT_CMD_STATE_TIMEOUT` against whichever op was in flight. It is not a
FastFlowLM bug — there is a ~90-line standalone reproducer, with a full write-up
and a report note, in **`~/corelib-stream-repro/`**.

FastFlowLM works around it on Linux by synchronizing after every dispatch and
replacing the stream, guarded by `#if defined(__linux__)` in
`src/common/models/phi4/rai/aie_next/phi4_rai.cpp`. That costs all the
pipelining: expect roughly **1.6 s per token**, under 1 tok/s. The workaround is
annotated for removal once corelib no longer needs it.

### Misleading diagnostics

When a dispatch fails, ignore these — they are printed unconditionally on that
XRT error path and describe nothing about real device state:

- `CTX_STATUS_UNASSIGNED`
- `ctx_error_type = NPU_ASYNC_EVENT_CTX_ERR_HWSCH_FAILURE`
- `number of uC reported = 0`
- the "N dispatch(es) outstanding" count — it is the size of a list that is
  never pruned, not a failure count
- `amdxdna_ubuf_get_pages: Failed to pin pages ret -14` in `dmesg` — a red
  herring alongside the stream fault, confirmed by testing with
  `RYZENAI_CORELIB_SAFE_MODE=1`, which bypasses host-pointer import entirely

Also: the failing op named in an error is usually just whichever op was in
flight when the context died, not the op at fault. Chasing a specific kernel or
shape is a dead end.

### Useful corelib switches when debugging

| Variable | Effect |
|---|---|
| `RYZENAI_CORELIB_TRACE_XRT` | trace XRT calls |
| `RYZENAI_CORELIB_SAFE_MODE` | force staging buffers, no host-pointer import |
| `RYZENAI_CORELIB_NO_RUN_POOL` | disable run-object reuse |
| `RYZENAI_CORELIB_NO_CRASH_HANDLER` | leave crashes to the debugger |

### Other rough edges

- `home_install.sh`'s `needs_configure` does not check that the cached
  `CMAKE_HOME_DIRECTORY` matches `$SRC_DIR`, so a build tree configured from a
  different source dir is reused rather than reconfigured.
- `src/src/main.cpp` is **CRLF**. Editing it with a tool that writes LF rewrites
  every line and produces a ~1800-line diff. Check `git diff --stat -w` if a
  diff looks impossibly large.
