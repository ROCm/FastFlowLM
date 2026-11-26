"""Run with `python test_pull_dry_run_cli.py <built flm executable>`; uses only loopback."""
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


def check_preview(exe, root):
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def log_request(self, *args):
            requests.append((self.command, self.path.split("?")[0]))

    files = {"config.json": b'{"flm_version":"1.0.0"}', "weight.bin": b"weights"}
    source = root / "resolve"
    source.mkdir()
    for name, data in files.items():
        (source / name).write_bytes(data)
    handler = functools.partial(Handler, directory=str(root))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            entry = {"name": "tiny", "url": f"http://127.0.0.1:{server.server_port}/resolve",
                     "details": {"family": "qwen3"}, "flm_min_version": "1.0.0", "files": list(files),
                     "file_records": [{"path": n, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                                      for n, b in files.items()]}
            catalog = root / "model_list.json"
            catalog.write_text(json.dumps({"model_path": "models", "models": {"tiny": {"1b": entry}}}))
            store = root / "store"
            env = {**os.environ, "FLM_CONFIG_PATH": str(catalog), "FLM_MODEL_PATH": str(store),
                   "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
            args = [str(exe), "pull", "tiny"]

            def preview():
                return subprocess.check_output([*args, "--dry-run"], env=env, cwd=root, text=True, timeout=30)

            assert f"Total download: {sum(map(len, files.values()))} B" in preview()
            assert not store.exists() and not requests
            installed = store / "models/tiny"
            installed.mkdir(parents=True)
            (installed / "config.json").write_bytes(files["config.json"])
            plan = preview()
            assert "reuse config.json" in plan and "download weight.bin (7 B)" in plan
            assert "Total download: 7 B" in plan and not requests
            subprocess.run(args, env=env, cwd=root, check=True, timeout=30)
            assert requests == [("GET", "/resolve/weight.bin")]
            assert (installed / "weight.bin").read_bytes() == files["weight.bin"]
            assert "Total download: 0 B" in preview() and len(requests) == 1
        finally:
            server.shutdown()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        check_preview(Path(sys.argv[1]).resolve(), Path(tmp))
