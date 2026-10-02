#!/usr/bin/env python3
"""Sweep every chat model with a two-round text conversation.

Each model gets a first prompt and a follow-up that depends on the first
answer, so multi-turn context handling is covered as well as single-shot
generation. Runs in streaming mode by default; pass --modes to add or swap in
the non-streaming path.

Models the catalog labels "single-turn" only get the first round: refusing a
follow-up is correct behaviour for them, not a failure.
"""

from __future__ import annotations

import sys
import time

from sweep_common import SweepTask, run_task

PROMPTS = {
    "stream": ("Tell me a joke.", "Explain why it is funny."),
    "non-stream": ("Teach me Maxwell's equations.", "Summarize your answer."),
}


class LLMSweep(SweepTask):
    name = "llm"
    csv_header = [
        "Model", "Mode", "Round", "Input",
        "Reasoning Content", "Output Content",
        "Latency (s)", "Status", "Error",
    ]

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--modes",
            nargs="+",
            default=["stream"],
            choices=["stream", "non-stream"],
            help="Which completion modes to exercise (default: stream).",
        )

    def select_models(self, catalog: list[dict]) -> list[str]:
        # Every chat model, vision-capable ones included: they all accept a
        # plain text conversation. The embedding and Whisper models are not
        # chat models and are handled by their own sweeps.
        return [
            entry["id"]
            for entry in catalog
            if entry["family"] != "embed-gemma"
            and not entry["id"].startswith("whisper-v3")
        ]

    def _round(self, writer, model: str, mode: str, round_no: int,
               messages: list, prompt: str) -> str | None:
        """Runs one turn. Returns the assistant text, or None on failure."""
        print(f"  [{mode}] round {round_no}: {prompt}")
        started = time.monotonic()
        try:
            reasoning, output = self.chat(model, messages, stream=(mode == "stream"))
        except Exception as exc:
            elapsed = f"{time.monotonic() - started:.2f}"
            print(f"    ERROR: {exc}")
            self.record(
                writer,
                [model, mode, round_no, prompt, "", "", elapsed, "error", str(exc)],
                model=model,
                error=f"{mode} round {round_no}: {exc}",
            )
            return None

        elapsed = f"{time.monotonic() - started:.2f}"
        self.record(
            writer,
            [model, mode, round_no, prompt, reasoning, output, elapsed, "ok", ""],
            model=model,
        )
        print(f"    done in {elapsed}s, {len(output)} chars")
        # Empty unless the server reported usage, so this adds a line only
        # when there is something in it.
        breakdown = self.format_usage()
        if breakdown:
            print(f"      {breakdown}")
        return output

    def run(self) -> None:
        models = self.resolve_models()
        handle, writer = self.open_csv()
        try:
            for model in self.iter_models(models):
                print(f"\n--- LLM: {model} ---")
                for mode in self.args.modes:
                    prompt, followup = PROMPTS[mode]
                    messages = [{"role": "user", "content": prompt}]

                    output = self._round(writer, model, mode, 1, messages, prompt)
                    if output is None:
                        # The follow-up needs the first answer for context, so
                        # skip it rather than send a conversation with a hole.
                        continue

                    if self.is_single_turn(model):
                        # Rejecting a second turn is what these models are
                        # supposed to do, so the round is recorded as skipped
                        # rather than sent and counted as a failure.
                        self.record(
                            writer,
                            [model, mode, 2, followup, "", "", "", "skipped",
                             "single-turn model: follow-up not supported"],
                            model=model,
                        )
                        print("    round 2 skipped: model is labelled single-turn")
                        time.sleep(1)
                        continue

                    messages.append({"role": "assistant", "content": output})
                    messages.append({"role": "user", "content": followup})
                    self._round(writer, model, mode, 2, messages, followup)
                    time.sleep(1)
        finally:
            handle.close()
        print(f"\nLLM sweep complete. Saved to {self.csv_path}")
        self.write_summary(models)


if __name__ == "__main__":
    sys.exit(run_task(LLMSweep))
