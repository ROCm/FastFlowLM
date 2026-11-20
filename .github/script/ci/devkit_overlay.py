#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Overlays a pull request's dev-kit engines and kernels on a product package.

    devkit_overlay.py --root <package root> --os linux|windows
                      [--engines flm-dev-kit/FLM_DLL/engines.json]
                      [--families <csv>] [--libs <dir>] [--xclbins <dir>]
                      [--xclbin-models <json list, default $DEVKIT_XCLBIN_MODELS>]
                      [--new-models <csv>]
                      [--github-output]

<root> is an unpacked portable package: the directory of flm (Linux, engines
in <root>/lib) or flm.exe (Windows, engines beside it), with xclbins/ and
model_list.json. FastFlowLM links every engine library at build time, by
file name (no SONAME, no plugin loader), so an engine is replaced by
overwriting its file; and it reads its kernels from one xclbins/ root, never
merging two, so the rebuilt kernels are copied over the package's own tree,
which leaves a complete one.

Engines: from --libs (the devkit-libs-<os>-<runtime> artifact), those whose
engines.json families meet --families; when --families holds every engine's
families (shared engine code, the dev-kit CI or an unknown change), every
engine, the shared units (q4nx, mha, gemm, dequant, lm_head) and the engines
that serve no family too, so no prebuilt is left to mix with dev-kit code
built against other shared headers. Each library replaces the package's file
of the same name; one the package lacks is an error (flm would not load it).

ABI check, before anything is copied, on the package as it will be: on
Windows, every name an .exe or .dll of the package imports from another one
must be exported by it, where either side is a dev-kit build (import and
export tables); on Linux, which names no provider per symbol, every global
undefined symbol of flm-real and the package's libraries that a replaced
library defined, and no other package file defines, must still be defined by
its dev-kit build, and every C++ symbol a dev-kit build needs that its
prebuilt did not must be defined by some library of the overlaid package
(std:: and C++ runtime symbols aside: libstdc++ has them; C symbols aside:
they come from the system). flm links every
engine at start-up, so one missing symbol would break every model, not only
the overlaid family's: that fails here, naming the symbols, instead. Header
drift that keeps the names but changes a class layout is not caught; the
dev-kit headers match the product's (additive members on a few engines) as
of the import.

Kernels: every file under --xclbins (the devkit-xclbins artifact,
<model folder>/<slot>.xclbin) is copied to <root>/xclbins/<same path>.

Test selection, for .github/actions/npu-test with the pr tier:
  models     every NPU catalog model (<root>/model_list.json, the entries an
             aie2p runner can run) of --families and of the other families
             of the engines replaced for them, and the --xclbin-models in
             that catalog; empty when every engine's families are overlaid,
             so the pr tier picks one model per family
  families   --families, or empty with every engine's families (as above)
  suites     the suites that exercise those families and those of the
             required models (changes.family_suites)
  required   --new-models and the --xclbin-models in the catalog: downloaded
             whatever their size, and a failure if they do not
  test       "false" when nothing is left to test (an error when some model
             is required: it would go untested)
--xclbin-models outside the catalog (smolvla:500m, a pybind module) are
reported and left out.

Writes a Markdown summary to $GITHUB_STEP_SUMMARY when that is set. Exit 1 on
an error (nothing is copied then), 2 on bad arguments. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import changes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINES_JSON = REPO_ROOT / changes.DEVKIT_ENGINES

# detail/<dir> -> library stem, where the two differ. The other engines and
# shared units build lib<dir>.so and <dir>.dll.
LIB_STEMS = {
    "q4nx": "q4_npu_eXpress",
    "qwen3_5_omni": "qwen3_5_omni_npu",
}
# Shared detail/ directories that build no library of their own.
NO_LIBRARY = {"vision_common"}

# libstdc++ defines these too, so a package library that stops exporting one
# loses nothing (mangled std::, with the St/Sa/Sb/Ss/Si/So/Sd abbreviations,
# its typeinfo, vtables, VTTs and guards).
STD_SYMBOL = re.compile(r"^_Z(?:T[ISVT]|GV)?N?[KVRO]*S[tabsiod]")
# The system's, not the package's: operators new and delete, __cxxabiv1 and
# __gnu_cxx (libstdc++, libsupc++) and the vector math variants the
# compiler calls (_ZGV<isa><mask><length>..., glibc's libmvec).
RUNTIME_SYMBOL = re.compile(r"^_Z(?:n[wa]|d[la])|^_Z(?:T[ISVT])?N\d+__(?:cxxabiv1|gnu_cxx)"
                            r"|^_ZGV[a-z][MN]\d")


def lib_stem(unit: str) -> str:
    return LIB_STEMS.get(unit, unit)


def lib_name(unit: str, os_name: str) -> str:
    stem = lib_stem(unit)
    return f"{stem}.dll" if os_name == "windows" else f"lib{stem}.so"


def lib_dir(root: Path, os_name: str) -> Path:
    return root if os_name == "windows" else root / "lib"


def units_to_overlay(engines: changes.DevkitEngines, families: set) -> tuple[list[str], bool]:
    """(detail/ directories whose libraries to overlay, whether that is every one)."""
    every = engines.all_families()
    if families and families >= every:
        units = sorted(set(engines.engines) | set(engines.shared))
        return [u for u in units if u not in NO_LIBRARY], True
    return sorted(e for e in engines.engines if engines.families(e) & families), False


# ── Symbol tables ────────────────────────────────────────────────────────────

class BinaryError(ValueError):
    pass


def _elf_sections(data: bytes) -> list[tuple]:
    if data[:4] != b"\x7fELF":
        raise BinaryError("not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise BinaryError("not a 64-bit little-endian ELF file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    out = []
    for i in range(shnum):
        # name, type, flags, addr, offset, size, link, info, align, entsize
        out.append(struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize))
    return out


def elf_symbols(data: bytes) -> tuple[set, set]:
    """(defined, undefined) global dynamic symbols of an ELF shared object or
    executable. Defined: global, weak or unique binding, default or protected
    visibility. Undefined: global binding only (a weak reference may stay
    unresolved)."""
    sections = _elf_sections(data)
    defined, undefined = set(), set()
    for sec in sections:
        if sec[1] != 11:  # SHT_DYNSYM
            continue
        strtab = sections[sec[6]]
        str_off = strtab[4]
        off, size, entsize = sec[4], sec[5], sec[9] or 24
        for pos in range(off + entsize, off + size, entsize):  # entry 0 is null
            name_off, info, other, shndx = struct.unpack_from("<IBBH", data, pos)
            end = data.index(b"\0", str_off + name_off)
            name = data[str_off + name_off:end].decode("utf-8", "replace")
            if not name:
                continue
            bind = info >> 4
            if shndx == 0:
                if bind == 1:
                    undefined.add(name)
            elif bind in (1, 2, 10) and (other & 3) in (0, 3):
                defined.add(name)
    return defined, undefined


class _Pe:
    def __init__(self, data: bytes):
        if data[:2] != b"MZ":
            raise BinaryError("not a PE file")
        pe, = struct.unpack_from("<I", data, 0x3C)
        if data[pe:pe + 4] != b"PE\0\0":
            raise BinaryError("no PE signature")
        nsections, = struct.unpack_from("<H", data, pe + 6)
        opt_size, = struct.unpack_from("<H", data, pe + 20)
        opt = pe + 24
        magic, = struct.unpack_from("<H", data, opt)
        if magic not in (0x10B, 0x20B):
            raise BinaryError("unknown PE optional header")
        self.pe64 = magic == 0x20B
        dirs = opt + (112 if self.pe64 else 96)
        self.dirs = [struct.unpack_from("<II", data, dirs + 8 * i) for i in range(2)]
        sec = opt + opt_size
        self.sections = [struct.unpack_from("<IIII", data, sec + 40 * i + 8) for i in range(nsections)]
        self.data = data

    def offset(self, rva: int) -> int:
        for vsize, vaddr, raw_size, raw_ptr in self.sections:
            if vaddr <= rva < vaddr + max(vsize, raw_size):
                return raw_ptr + rva - vaddr
        raise BinaryError(f"RVA {rva:#x} is in no section")

    def cstr(self, rva: int) -> str:
        off = self.offset(rva)
        return self.data[off:self.data.index(b"\0", off)].decode("ascii", "replace")


def pe_exports(data: bytes) -> set:
    """The names a PE file exports."""
    pe = _Pe(data)
    rva, size = pe.dirs[0]
    if not rva or not size:
        return set()
    off = pe.offset(rva)
    count, = struct.unpack_from("<I", data, off + 24)
    names_rva, = struct.unpack_from("<I", data, off + 32)
    names_off = pe.offset(names_rva) if count else 0
    return {pe.cstr(struct.unpack_from("<I", data, names_off + 4 * i)[0]) for i in range(count)}


def pe_imports(data: bytes) -> dict[str, set]:
    """DLL name (lower case) -> the names a PE file imports from it (by
    ordinal ones left out)."""
    pe = _Pe(data)
    rva, size = pe.dirs[1]
    out: dict[str, set] = {}
    if not rva or not size:
        return out
    desc = pe.offset(rva)
    width, flag = (8, 1 << 63) if pe.pe64 else (4, 1 << 31)
    while True:
        lookup, _, _, name_rva, first = struct.unpack_from("<IIIII", data, desc)
        if not (lookup or name_rva or first):
            break
        names = out.setdefault(pe.cstr(name_rva).lower(), set())
        thunk = pe.offset(lookup or first)
        while True:
            entry, = struct.unpack_from("<Q" if pe.pe64 else "<I", data, thunk)
            if not entry:
                break
            if not entry & flag:
                names.add(pe.cstr((entry & 0x7FFFFFFF) + 2))  # skip the hint
            thunk += width
        desc += 20
    return out


def _short(names: list[str]) -> str:
    return ", ".join(names[:10]) + (f" and {len(names) - 10} more" if len(names) > 10 else "")


def abi_errors(root: Path, os_name: str, replace: dict[Path, Path]) -> list[str]:
    """The imports the package would leave unresolved once each key of
    replace (a package library) is replaced by its value (a dev-kit build)."""
    final = {p: replace.get(p, p) for p in package_binaries(root, os_name)}
    errors = []
    if os_name == "windows":
        # Exact: who imports what from whom, after the overlay. Only pairs
        # with a replaced side are checked; the package's own are its own.
        by_name = {old.name.lower(): old for old in final}
        exports: dict[Path, set] = {}
        for consumer_old, consumer in final.items():
            for dll, names in sorted(pe_imports(consumer.read_bytes()).items()):
                provider_old = by_name.get(dll)
                if provider_old is None or provider_old == consumer_old:
                    continue
                if consumer_old not in replace and provider_old not in replace:
                    continue
                provider = final[provider_old]
                if provider not in exports:
                    exports[provider] = pe_exports(provider.read_bytes())
                missing = sorted(names - exports[provider])
                if missing:
                    errors.append(f"{consumer_old.name} imports {len(missing)} name(s) the "
                                  f"{'dev-kit' if provider_old in replace else 'package'} "
                                  f"{provider_old.name} does not export: {_short(missing)}")
        return errors
    # ELF names no provider per symbol: take the symbols the rest of the
    # package needs, that the replaced library defined and nothing else does.
    tables = {old: elf_symbols(new.read_bytes()) for old, new in final.items()}
    for old, new in replace.items():
        needed, elsewhere = set(), set()
        for other, (defined, undefined) in tables.items():
            if other != old:
                needed |= undefined
                elsewhere |= defined
        old_defined, _ = elf_symbols(old.read_bytes())
        lost = (needed & old_defined) - elsewhere - tables[old][0]
        missing = sorted(s for s in lost if not STD_SYMBOL.match(s))
        if missing:
            errors.append(f"the dev-kit {old.name} lacks {len(missing)} symbol(s) the package "
                          f"takes from it: {_short(missing)}")
        # The other way: what the dev-kit build needs that its prebuilt did
        # not, which only the package (or the system, for C names) can give.
        new_undef = tables[old][1] - elf_symbols(old.read_bytes())[1]
        provided = set().union(*(d for d, _ in tables.values()))
        unresolved = sorted(s for s in new_undef - provided if s.startswith("_Z")
                            and not STD_SYMBOL.match(s) and not RUNTIME_SYMBOL.match(s))
        if unresolved:
            errors.append(f"the dev-kit {old.name} needs {len(unresolved)} symbol(s) no package "
                          f"library defines: {_short(unresolved)}")
    return errors


def package_binaries(root: Path, os_name: str) -> list[Path]:
    """The package's executable and libraries, which import from its engines
    (files of those names that are not binaries, such as a linker script
    named lib*.so, left out)."""
    if os_name == "windows":
        found = [p for p in root.iterdir() if p.suffix.lower() in (".exe", ".dll")]
    else:
        found = [root / "flm-real"]
        if (root / "lib").is_dir():
            found += [p for p in (root / "lib").rglob("*.so*") if not p.is_symlink()]
    return sorted(p for p in found if p.is_file() and readable(p, os_name))


def readable(path: Path, os_name: str) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(4)
    return head[:2] == b"MZ" if os_name == "windows" else head == b"\x7fELF"


# ── Test selection ───────────────────────────────────────────────────────────

def npu_catalog(root: Path) -> dict[str, dict]:
    """The package's catalog entries an aie2p runner's NPU can run."""
    try:
        text = (root / "model_list.json").read_text(encoding="utf-8")
    except OSError:
        return {}
    out = {}
    for mid, entry in changes.load_catalog(text).items():
        if changes.runs_on_gpu(mid, entry):
            continue
        platforms = entry.get("supported_platforms")
        if not platforms or changes.RUNNER_PLATFORMS & set(platforms):
            out[mid] = entry
    return out


def select_tests(catalog: dict[str, dict], families: set, every: bool,
                 xclbin_models: list[str], new_models: list[str],
                 test_families: Optional[set] = None) -> tuple[dict, list[str]]:
    """(npu-test inputs, notes). test_families (default families) are the
    families whose models and suites run: those of every replaced engine,
    which can be more than the touched ones."""
    if test_families is None:
        test_families = families
    notes = []
    kernels = [m for m in xclbin_models if m in catalog]
    outside = [m for m in xclbin_models if m not in catalog]
    if outside:
        notes.append(f"rebuilt kernels for models this package's NPU catalog lacks, not tested: "
                     f"{', '.join(outside)}")
    tested = set(test_families) | {changes.family_of(catalog[m])
                                    for m in kernels + [n for n in new_models if n in catalog]}
    tested.discard("")
    required = list(dict.fromkeys(new_models + kernels))
    if every:
        models = []
    else:
        models = sorted(m for m, e in catalog.items() if changes.family_of(e) in test_families)
        models = list(dict.fromkeys(models + kernels))
    suites = changes.family_suites(tested, catalog)
    out = {
        "models": ",".join(models),
        "families": "" if every else ",".join(sorted(families)),
        "suites": ",".join(s for s in changes.SUITES if s in suites),
        "required": ",".join(required),
        "test": str(bool(suites) and bool(every or models or required)).lower(),
    }
    return out, notes


# ── Main ─────────────────────────────────────────────────────────────────────

def csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def overlay(args: argparse.Namespace) -> tuple[dict, list[str], list[str], list[str]]:
    """(outputs, copied, notes, errors); copies nothing when there are errors."""
    errors: list[str] = []
    notes: list[str] = []
    root, os_name = args.root, args.os
    engines = changes.load_devkit_engines(args.engines.read_text(encoding="utf-8")
                                          if args.engines.is_file() else "")
    if engines is None:
        return {}, [], notes, [f"{args.engines} is missing or unreadable"]
    families = set(csv(args.families))
    unknown = families - engines.all_families()
    if unknown:
        notes.append(f"families no dev-kit engine serves, tested as they are: {', '.join(sorted(unknown))}")
    units, every = units_to_overlay(engines, families)
    # An engine replaced for one touched family runs its other families'
    # models too, so those are tested as well.
    test_families = set(families) if every else families | {
        f for u in units for f in engines.families(u)}

    plan: list[tuple[Path, Path]] = []
    if units and args.libs is None:
        errors.append(f"engines to overlay ({', '.join(units)}) but no --libs")
        units = []
    replace: dict[Path, Path] = {}
    for unit in units:
        name = lib_name(unit, os_name)
        new, old = args.libs / name, lib_dir(root, os_name) / name
        if not new.is_file():
            errors.append(f"{unit}: {name} is not in the dev-kit libraries ({args.libs})")
        elif not old.is_file():
            errors.append(f"{unit}: the package has no {name} to replace, so flm would not load it")
        elif not readable(new, os_name):
            errors.append(f"{unit}: the dev-kit {name} is not a {os_name} library")
        else:
            replace[old] = new
    if replace and not errors:
        try:
            errors += abi_errors(root, os_name, replace)
        except (BinaryError, struct.error, ValueError, IndexError) as exc:
            errors.append(f"cannot read the symbol tables of the package or the dev-kit libraries: {exc}")
    plan += [(new, old) for old, new in replace.items()]

    if args.xclbins is not None and args.xclbins.is_dir():
        dest_root = root / "xclbins"
        if not dest_root.is_dir():
            errors.append(f"the package has no xclbins/ directory under {root}")
        for src in sorted(p for p in args.xclbins.rglob("*") if p.is_file()):
            rel = src.relative_to(args.xclbins)
            if not (dest_root / rel.parts[0]).is_dir():
                notes.append(f"xclbins/{rel.parts[0]} is new to the package")
            plan.append((src, dest_root / rel))

    outputs, more = select_tests(npu_catalog(root), families, every,
                                 json.loads(args.xclbin_models or "[]"), csv(args.new_models),
                                 test_families)
    notes += more
    if outputs.get("required") and outputs.get("test") != "true":
        errors.append(f"required models {outputs['required']} select no suite or model in this "
                      f"package's NPU catalog, so they would go untested")
    if errors:
        return outputs, [], notes, errors
    copied = []
    for src, dest in plan:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        copied.append(str(dest.relative_to(root)).replace("\\", "/"))
    return outputs, copied, notes, errors


def summary(outputs: dict, copied: list[str], notes: list[str], errors: list[str]) -> str:
    libs = [c for c in copied if not c.startswith("xclbins/")]
    kernels = [c for c in copied if c.startswith("xclbins/")]
    lines = ["### Dev-kit overlay", ""]
    lines += [f"- ❌ {e}" for e in errors]
    lines.append(f"- Engine libraries replaced ({len(libs)}): "
                 + (", ".join(f"`{Path(c).name}`" for c in libs) or "none"))
    folders = sorted({c.split("/")[1] for c in kernels})
    lines.append(f"- Kernels replaced: {len(kernels)} in "
                 + (", ".join(f"`{f}`" for f in folders) or "no model folder"))
    lines += [f"- {n}" for n in notes]
    if outputs:
        lines.append(f"- Tests: suites `{outputs['suites'] or 'none'}`; models "
                     f"`{outputs['models'] or 'one per family (pr tier)'}`; required "
                     f"`{outputs['required'] or 'none'}`")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--os", choices=("linux", "windows"), required=True)
    parser.add_argument("--engines", type=Path, default=ENGINES_JSON)
    parser.add_argument("--families", default="")
    parser.add_argument("--libs", type=Path, default=None)
    parser.add_argument("--xclbins", type=Path, default=None)
    parser.add_argument("--xclbin-models", default=os.environ.get("DEVKIT_XCLBIN_MODELS", "[]"),
                        help="JSON list; default $DEVKIT_XCLBIN_MODELS (PowerShell 5.1 mangles "
                             "quotes in a native program's arguments)")
    parser.add_argument("--new-models", default="")
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not isinstance(json.loads(args.xclbin_models or "[]"), list):
            raise ValueError
    except ValueError:
        parser.error("--xclbin-models must be a JSON list")
    if not args.root.is_dir():
        parser.error(f"--root {args.root} is not a directory")

    outputs, copied, notes, errors = overlay(args)
    text = summary(outputs, copied, notes, errors)
    print(text, end="")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    for e in errors:
        print(f"::error::devkit overlay: {e}")
    if errors:
        return 1
    if outputs.get("test") != "true":
        print("::notice::The dev-kit change serves no model in this package's NPU catalog: nothing to test.")
    for key, value in outputs.items():
        print(f"{key}={value}")
    if args.github_output:
        target = os.environ.get("GITHUB_OUTPUT")
        if not target:
            print("--github-output given but $GITHUB_OUTPUT is not set", file=sys.stderr)
            return 2
        with open(target, "a", encoding="utf-8") as fh:
            fh.writelines(f"{k}={v}\n" for k, v in outputs.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
