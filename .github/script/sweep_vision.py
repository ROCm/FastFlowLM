#!/usr/bin/env python3
"""Sweep every vision-capable model with a two-image conversation.

Both test images go in the first message, so multi-image prompts are covered
as well as basic image understanding. The follow-up asks the model to relate
the two, which only works if the image context survived the first turn.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

from sweep_common import SweepTask, run_task

PROMPT = "Describe these two images in detail."
FOLLOWUP = "Write a short story that connects the two images together."

MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


class VisionSweep(SweepTask):
    name = "vision"
    csv_header = [
        "Model", "Round", "Input",
        "Reasoning Content", "Output Content",
        "Latency (s)", "Status", "Error",
    ]

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--images",
            nargs="+",
            default=None,
            metavar="PATH",
            help="Images to send. Defaults to the two bundled test images.",
        )

    def image_paths(self) -> list[Path]:
        if self.args.images:
            paths = [Path(p) for p in self.args.images]
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Image(s) not found: {', '.join(missing)}")
            return paths
        return [self.asset("test_image1.jpg"), self.asset("test_image2.jpg")]

    @staticmethod
    def _as_data_url(path: Path) -> str:
        mime = MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def select_models(self, catalog: list[dict]) -> list[str]:
        vlms = [entry["id"] for entry in catalog if entry["vlm"]]
        if not vlms:
            # The fallback catalog from /v1/models carries no vlm flag, so an
            # empty result there means "unknown", not "none exist".
            print(
                "[vision] No vision-capable models found. If the flm binary was "
                "unavailable, pass --models explicitly."
            )
        return vlms

    def _round(self, writer, model: str, round_no: int,
               messages: list, prompt: str) -> str | None:
        print(f"  round {round_no}: {prompt}")
        started = time.monotonic()
        try:
            reasoning, output = self.chat(model, messages, stream=True)
        except Exception as exc:
            elapsed = f"{time.monotonic() - started:.2f}"
            print(f"    ERROR: {exc}")
            self.record(
                writer,
                [model, round_no, prompt, "", "", elapsed, "error", str(exc)],
                model=model,
                error=f"round {round_no}: {exc}",
            )
            return None

        elapsed = f"{time.monotonic() - started:.2f}"
        self.record(
            writer,
            [model, round_no, prompt, reasoning, output, elapsed, "ok", ""],
            model=model,
        )
        print(f"    done in {elapsed}s, {len(output)} chars")
        breakdown = self.format_usage()
        if breakdown:
            print(f"      {breakdown}")
        return output

    def run(self) -> None:
        paths = self.image_paths()
        print(f"[vision] Images: {', '.join(str(p) for p in paths)}")
        image_parts = [
            {"type": "image_url", "image_url": {"url": self._as_data_url(p)}}
            for p in paths
        ]

        models = self.resolve_models()
        handle, writer = self.open_csv()
        try:
            for model in self.iter_models(models):
                print(f"\n--- Vision: {model} ---")
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": PROMPT}] + image_parts,
                    }
                ]

                output = self._round(writer, model, 1, messages, PROMPT)
                if output is None:
                    continue

                if self.is_single_turn(model):
                    # Expected behaviour for these models, not a defect; see
                    # SweepTask.is_single_turn.
                    self.record(
                        writer,
                        [model, 2, FOLLOWUP, "", "", "", "skipped",
                         "single-turn model: follow-up not supported"],
                        model=model,
                    )
                    print("    round 2 skipped: model is labelled single-turn")
                    time.sleep(1)
                    continue

                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": FOLLOWUP})
                self._round(writer, model, 2, messages, FOLLOWUP)
                time.sleep(1)
        finally:
            handle.close()
        print(f"\nVision sweep complete. Saved to {self.csv_path}")
        self.write_summary(models)


if __name__ == "__main__":
    sys.exit(run_task(VisionSweep))
