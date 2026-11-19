"""Remembers the GPU-backend models a runner has verified, so that later runs
need not verify them again.

`flm pull` of a model it already has hashes every file against the pinned
hashes: minutes for each oga-hipep package (14 for the 124 GB one), on every
run. Once a pull succeeds, a stamp in $FLM_MODEL_PATH/npu-test-verified
records what it verified: the model's pins in the catalog flm reads (its
Hugging Face revision, its files and where each comes from) and the size and
modification time of each of those files. A later run skips the pull while
`flm list` still calls the model installed, its pins are the same and none of
its files changed; anything else pulls again, which downloads what changed
and verifies the rest. Deleting a stamp, or the folder, makes the next run
verify again.

An entry that names a branch instead of a revision can change on Hugging Face
with the catalog unchanged, so it is pulled every time.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

STAMP_DIR = "npu-test-verified"
# The catalog fields that decide which files a model has.
PIN_FIELDS = ("name", "url", "files", "file_sources")
REVISION = re.compile(r"^[0-9a-f]{40}$")


def find_model_list(flm_bin: str) -> Path | None:
    """The model_list.json flm reads (utils::find_model_list), bar its last resorts."""
    configured = os.environ.get("FLM_CONFIG_PATH")
    if configured and Path(configured).is_file():
        return Path(configured)
    binary = shutil.which(flm_bin)
    if not binary:
        return None
    exe_dir = Path(binary).resolve().parent
    for candidate in (exe_dir / "model_list.json", exe_dir.parent / "share" / "flm" / "model_list.json"):
        if candidate.is_file():
            return candidate
    return None


def pins(entry: dict) -> dict | None:
    """The fields that fix a catalog entry's files, or None when the entry
    does not pin them all to a revision."""
    revision = re.search(r"/resolve/([^/]+)", str(entry.get("url") or ""))
    sources = entry.get("file_sources") or {}
    if (not revision or not REVISION.match(revision.group(1))
            or not entry.get("name") or not entry.get("files") or not isinstance(sources, dict)
            or not all(isinstance(s, dict) and REVISION.match(str(s.get("revision", "")))
                       for s in sources.values())):
        return None
    return {k: entry.get(k) for k in PIN_FIELDS}


class ModelCache:
    def __init__(self, root: Path, models_dir: Path, entries: dict[str, dict]):
        self.dir = root / STAMP_DIR
        self.models_dir = models_dir
        self.entries = entries  # model id -> its model_list.json entry

    @classmethod
    def open(cls, flm_bin: str) -> "ModelCache | None":
        """The stamps for this binary's catalog; None without $FLM_MODEL_PATH
        or a model_list.json to read the pins from."""
        root = os.environ.get("FLM_MODEL_PATH")
        path = find_model_list(flm_bin)
        if not root or path is None:
            return None
        try:
            catalog = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        entries = {f"{name}:{tag}": entry
                   for name, tags in (catalog.get("models") or {}).items() if isinstance(tags, dict)
                   for tag, entry in tags.items() if isinstance(entry, dict)}
        return cls(Path(root), Path(root) / str(catalog.get("model_path") or "models"), entries)

    def _path(self, model_id: str) -> Path:
        return self.dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", model_id) + ".json")

    def _state(self, model_id: str) -> dict | None:
        """The model's pins, and the size and modification time of each of its
        files; None when it has no pins or lacks a file."""
        entry = self.entries.get(model_id)
        pinned = pins(entry) if entry else None
        if pinned is None:
            return None
        folder = self.models_dir / entry["name"]
        files = {}
        for name in entry["files"]:
            try:
                stat = (folder / name).stat()
            except OSError:
                return None
            files[name] = [stat.st_size, stat.st_mtime_ns]
        return {"pins": pinned, "files": files}

    def verified(self, model_id: str) -> str | None:
        """When the model was verified, if nothing about it changed since."""
        state = self._state(model_id)
        if state is None:
            return None
        try:
            stamp = json.loads(self._path(model_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(stamp, dict) or {k: stamp.get(k) for k in state} != state:
            return None
        return str(stamp.get("verified") or "earlier")

    def stamp(self, model_id: str) -> str:
        """Records the model as verified now; returns why not, else ""."""
        state = self._state(model_id)
        if state is None:
            return "not pinned to a revision, or a file is missing"
        path = self._path(model_id)
        stamp = {"model": model_id, "verified": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                 **state}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            partial.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
            os.replace(partial, path)
        except OSError as exc:
            return f"{type(exc).__name__}: {exc}"
        return ""
