#!/usr/bin/env python3
"""Sweep the audio path through /v1/audio/transcriptions.

Whisper is absent from /v1/models, so the server needs `-a 1` and the model
set comes from `flm list`. A transcript is only accepted if it comes back
non-empty, since the server answers with an empty body when no ASR model is
loaded rather than reporting an error.

Some chat models accept audio inline via `input_audio` instead of going
through the transcription endpoint. That path is off by default because it is
a different feature with different model support. Turn it on with
--include-chat-audio.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

from sweep_common import SweepTask, run_task

CHAT_AUDIO_PROMPT = "Describe what you hear in this audio."

AUDIO_FORMAT_BY_SUFFIX = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".m4a": "m4a",
    ".flac": "flac",
    ".ogg": "ogg",
}


class AudioSweep(SweepTask):
    name = "audio"
    needs_asr = True
    csv_header = [
        "Model", "Mode", "Input", "Output Text", "Characters",
        "Latency (s)", "Status", "Error",
    ]

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--audio",
            default=None,
            metavar="PATH",
            help="Audio clip to transcribe. Defaults to the bundled test clip.",
        )
        parser.add_argument(
            "--include-chat-audio",
            action="store_true",
            help="Also send the clip inline to audio-capable chat models.",
        )

    def audio_path(self) -> Path:
        if self.args.audio:
            path = Path(self.args.audio)
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")
            return path
        return self.asset("test_audio.mp3")

    def select_models(self, catalog: list[dict]) -> list[str]:
        models = [e["id"] for e in catalog if e["id"].startswith("whisper-v3")]
        if self.args.include_chat_audio:
            models += [
                e["id"] for e in catalog
                if e["asr"] and not e["id"].startswith("whisper-v3")
            ]
        if not models:
            print(
                "[audio] No audio models found. The /v1/models fallback hides "
                "Whisper, so pass --models explicitly if 'flm list' failed."
            )
        return models

    def _is_whisper(self, model: str) -> bool:
        return model.startswith("whisper-v3")

    def _transcribe(self, writer, model: str, path: Path) -> None:
        print(f"  transcription: {path.name}")
        started = time.monotonic()
        try:
            with open(path, "rb") as handle:
                response = self.client.audio.transcriptions.create(
                    model=model, file=handle
                )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                # The server returns an empty body when ASR is not loaded, so
                # an empty transcript means a broken run, not a silent clip.
                raise RuntimeError(
                    "empty transcript; the ASR model may not have loaded"
                )
        except Exception as exc:
            elapsed = f"{time.monotonic() - started:.2f}"
            print(f"    ERROR: {exc}")
            self.record(
                writer,
                [model, "transcription", path.name, "", 0, elapsed, "error", str(exc)],
                model=model,
                error=f"transcription: {exc}",
            )
            return

        elapsed = f"{time.monotonic() - started:.2f}"
        self.record(
            writer,
            [model, "transcription", path.name, text, len(text), elapsed, "ok", ""],
            model=model,
        )
        print(f"    {len(text)} chars in {elapsed}s: {text[:80]!r}")

    def _chat_audio(self, writer, model: str, path: Path) -> None:
        audio_format = AUDIO_FORMAT_BY_SUFFIX.get(path.suffix.lower(), "mp3")
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        print(f"  chat audio: {path.name}")
        started = time.monotonic()
        try:
            _, output = self.chat(
                model,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CHAT_AUDIO_PROMPT},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded,
                                    "format": audio_format,
                                },
                            },
                        ],
                    }
                ],
                stream=True,
            )
            output = (output or "").strip()
            if not output:
                raise RuntimeError("empty response")
        except Exception as exc:
            elapsed = f"{time.monotonic() - started:.2f}"
            print(f"    ERROR: {exc}")
            self.record(
                writer,
                [model, "chat-audio", path.name, "", 0, elapsed, "error", str(exc)],
                model=model,
                error=f"chat-audio: {exc}",
            )
            return

        elapsed = f"{time.monotonic() - started:.2f}"
        self.record(
            writer,
            [model, "chat-audio", path.name, output, len(output), elapsed, "ok", ""],
            model=model,
        )
        print(f"    {len(output)} chars in {elapsed}s")

    def run(self) -> None:
        path = self.audio_path()
        print(f"[audio] Clip: {path}")

        models = self.resolve_models()
        handle, writer = self.open_csv()
        try:
            for model in self.iter_models(models):
                print(f"\n--- Audio: {model} ---")
                if self._is_whisper(model):
                    self._transcribe(writer, model, path)
                else:
                    self._chat_audio(writer, model, path)
                time.sleep(1)
        finally:
            handle.close()
        print(f"\nAudio sweep complete. Saved to {self.csv_path}")
        self.write_summary(models)


if __name__ == "__main__":
    sys.exit(run_task(AudioSweep))
