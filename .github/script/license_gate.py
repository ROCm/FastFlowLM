#!/usr/bin/env python3
"""License gate for FastFlowLM release packages.

FastFlowLM ships an MIT runtime together with closed NPU engine libraries, so no
GPL (or non-free) code may be bundled in, or linked by, anything we package.
This script unpacks build artifacts and fails if it finds any of:

  * a bundled or linked library on the GPL/non-free denylist below,
  * a binary that reports its own license as GPL or non-free (FFmpeg embeds
    "libavcodec license: GPL version 2 or later" in GPL builds),
  * a Linux binary that links a *system* FFmpeg (Debian/Ubuntu configure theirs
    with --enable-gpl), or
  * a .deb whose Depends names a GPL package.

Findings matching a waiver in license-waivers.txt are reported as warnings.

Usage: license_gate.py [--waivers FILE] PATH...
  PATH may be a directory (scanned recursively), .tar.gz/.tgz, .zip, .deb or
  .msi. Archives found inside directories are unpacked and scanned too.
Needs objdump (binutils) and, for .deb/.msi, dpkg-deb and msiextract.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

# Library basenames (lower-cased) that must never be bundled or linked.
DENY_LIBS = [
    (r"^lib(readline|history)\.so", "GNU readline (GPL-3)"),
    (r"^(lib)?fftw3[flq]?(_threads|_omp|_mpi)?(-3)?\.(so|dll)", "FFTW (GPL-2+)"),
    (r"^libx264", "x264 (GPL-2+)"),
    (r"^libx265", "x265 (GPL-2+)"),
    (r"^libxvidcore", "Xvid (GPL-2+)"),
    (r"^(lib)?postproc", "libpostproc (GPL-2+)"),
    (r"^libdvd(read|nav)", "libdvdread/libdvdnav (GPL-2+)"),
    (r"^libfaad", "FAAD2 (GPL-2+)"),
    (r"^libfdk-aac", "fdk-aac (non-free)"),
    (r"^librubberband", "Rubber Band (GPL-2+)"),
    (r"^libvidstab", "vid.stab (GPL-2+)"),
    (r"^libcdio", "libcdio (GPL-3+)"),
    (r"^libsmbclient", "Samba client (GPL-3+)"),
    (r"^libpoppler", "Poppler (GPL-2+)"),
    (r"^libmysqlclient", "MySQL client (GPL-2)"),
]

# FFmpeg shared libraries. Fine when we ship an LGPL build ourselves (checked
# via its embedded license string); not fine when a Linux binary picks up the
# distro copy, which Debian and Ubuntu configure with --enable-gpl.
FFMPEG_LIB = re.compile(r"^lib(av(codec|format|util|filter|device)|sw(scale|resample)|postproc)\.so")

# .deb Depends package names that pull in GPL code.
DENY_DEB_PACKAGES = [
    (r"^libreadline\d", "GNU readline (GPL-3)"),
    (r"^libfftw3", "FFTW (GPL-2+)"),
    (r"^libx26[45]", "x264/x265 (GPL-2+)"),
    (r"^libxvidcore", "Xvid (GPL-2+)"),
    (r"^libpostproc", "libpostproc (GPL-2+)"),
    (r"^libdvd(read|nav)", "libdvdread/libdvdnav (GPL-2+)"),
    (r"^libfaad", "FAAD2 (GPL-2+)"),
    (r"^libfdk-aac", "fdk-aac (non-free)"),
    (r"^lib(av(codec|format|util|filter|device)|sw(scale|resample))\d",
     "distro FFmpeg (Debian/Ubuntu build it with --enable-gpl)"),
]

# "<lib> license: GPL version 2 or later" / "... license: nonfree and unredistributable".
# LGPL builds say "license: LGPL ...", which this deliberately does not match.
SELF_REPORT = re.compile(rb"(lib[a-z0-9_]+) license: (GPL|nonfree)[ a-zA-Z0-9.]*")

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".zip", ".deb", ".msi")


def deny_reason(name, table):
    low = name.lower()
    for pattern, reason in table:
        if re.search(pattern, low):
            return reason
    return None


def binary_kind(path):
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError:
        return None
    if head == b"\x7fELF":
        return "elf"
    if head[:2] == b"MZ":
        return "pe"
    return None


def linked_libs(path, kind):
    """NEEDED entries (ELF) or imported DLL names (PE), via objdump -p."""
    out = subprocess.run(["objdump", "-p", path], capture_output=True, text=True, errors="replace").stdout
    if kind == "elf":
        return re.findall(r"^\s*NEEDED\s+(\S+)", out, re.M)
    return re.findall(r"DLL Name:\s*(\S+)", out)


def self_reports(path):
    with open(path, "rb") as f:
        data = f.read()
    return sorted({m.group(0).decode("ascii", "replace").strip() for m in SELF_REPORT.finditer(data)})


def unpack(path, dest):
    """Unpack an archive into dest. Returns the .deb control Depends string, if any."""
    low = path.lower()
    if low.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path) as t:
            t.extractall(dest, filter="tar") if hasattr(tarfile, "data_filter") else t.extractall(dest)
    elif low.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    elif low.endswith(".deb"):
        subprocess.run(["dpkg-deb", "-x", path, dest], check=True)
        return subprocess.run(["dpkg-deb", "-f", path, "Depends"], capture_output=True, text=True,
                              check=True).stdout.strip()
    elif low.endswith(".msi"):
        if not shutil.which("msiextract"):
            raise RuntimeError("msiextract (msitools) not installed")
        subprocess.run(["msiextract", "-C", dest, path], check=True, capture_output=True)
    return None


class Gate:
    def __init__(self, waivers):
        self.waivers = waivers
        self.findings = []  # (text, waiver_reason or None)
        self.scanned = 0
        self.skipped = []

    def add(self, text):
        for pattern, reason in self.waivers:
            if pattern.search(text):
                self.findings.append((text, reason))
                return
        self.findings.append((text, None))

    def scan_package(self, label, root, depends=None):
        """Scan one unpacked package (a directory tree) as a unit."""
        files = []
        nested = []
        for dirpath, _, names in os.walk(root):
            for n in names:
                p = os.path.join(dirpath, n)
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                if n.lower().endswith(ARCHIVE_SUFFIXES):
                    nested.append(p)
                else:
                    files.append(p)
        shipped = {os.path.basename(p) for p in files}
        # Symlink names count as shipped too (libfoo.so.1 -> libfoo.so.1.2.3).
        for dirpath, _, names in os.walk(root):
            shipped.update(n for n in names if os.path.islink(os.path.join(dirpath, n)))

        for p in files:
            kind = binary_kind(p)
            if not kind:
                continue
            self.scanned += 1
            rel = os.path.relpath(p, root)
            name = os.path.basename(p)
            reason = deny_reason(name, DENY_LIBS)
            if reason:
                self.add(f"{label}: {rel}: bundles {reason}")
            for report in self_reports(p):
                self.add(f"{label}: {rel}: reports '{report}'")
            for dep in linked_libs(p, kind):
                reason = deny_reason(dep, DENY_LIBS)
                if reason:
                    self.add(f"{label}: {rel}: links {dep} ({reason})")
                elif kind == "elf" and FFMPEG_LIB.search(dep.lower()) and dep not in shipped:
                    self.add(f"{label}: {rel}: links system FFmpeg {dep} "
                             "(Debian/Ubuntu build it with --enable-gpl)")

        if depends is not None:
            for item in re.split(r"[,|]", depends):
                pkg = item.strip().split(" ")[0]
                reason = deny_reason(pkg, DENY_DEB_PACKAGES) if pkg else None
                if reason:
                    self.add(f"{label}: Depends {pkg}: {reason}")

        for p in nested:
            self.scan_archive(p, f"{label}/{os.path.relpath(p, root)}")

    def scan_archive(self, path, label):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                depends = unpack(path, tmp)
            except (RuntimeError, OSError, subprocess.CalledProcessError, tarfile.TarError,
                    zipfile.BadZipFile) as e:
                self.skipped.append(f"{label}: {e}")
                return
            self.scan_package(label, tmp, depends)

    def scan(self, path):
        label = os.path.basename(os.path.normpath(path))
        if os.path.isdir(path):
            # Each artifact directory is its own package (download-artifact
            # puts every artifact in a subdirectory named after it).
            subdirs = [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
            loose = [d for d in os.listdir(path) if not os.path.isdir(os.path.join(path, d))]
            if subdirs and not loose:
                for d in subdirs:
                    self.scan_package(d, os.path.join(path, d))
            else:
                self.scan_package(label, path)
        else:
            self.scan_archive(path, label)


def load_waivers(path):
    waivers = []
    if not path or not os.path.exists(path):
        return waivers
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pattern, _, reason = line.partition("\t")
            waivers.append((re.compile(pattern.strip()), reason.strip() or "waived"))
    return waivers


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waivers", default=os.path.join(here, "license-waivers.txt"))
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    if not shutil.which("objdump"):
        sys.exit("license_gate: objdump not found (install binutils)")

    gate = Gate(load_waivers(args.waivers))
    for p in args.paths:
        gate.scan(p)

    violations = [t for t, w in gate.findings if w is None]
    waived = [(t, w) for t, w in gate.findings if w is not None]
    gh = os.environ.get("GITHUB_ACTIONS") == "true"

    print(f"license_gate: scanned {gate.scanned} binaries")
    for s in gate.skipped:
        print(f"::warning::license_gate skipped {s}" if gh else f"SKIPPED  {s}")
    for t, w in waived:
        print(f"::warning::{t} [waived: {w}]" if gh else f"WAIVED   {t} [{w}]")
    for t in violations:
        print(f"::error::{t}" if gh else f"VIOLATION {t}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## License gate\n\n")
            f.write(f"Scanned {gate.scanned} binaries: {len(violations)} violation(s), "
                    f"{len(waived)} waived, {len(gate.skipped)} skipped.\n\n")
            for t in violations:
                f.write(f"- ❌ {t}\n")
            for t, w in waived:
                f.write(f"- ⚠️ {t} — waived: {w}\n")
            for s in gate.skipped:
                f.write(f"- ⏭️ skipped {s}\n")

    if gate.scanned == 0:
        print("::error::license_gate found no binaries to scan" if gh else "ERROR no binaries scanned")
        return 1
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
