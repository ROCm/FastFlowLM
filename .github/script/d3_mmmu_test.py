#!/usr/bin/env python3
"""Ask d3 the MMMU-Pro vision questions that build_d3_mmmu_eval.py picked.

Each question is one POST /v1/systemone: the screenshot as the request's only
image and a choice question whose options are the letters A, B, ... (the
vision setting shows the question and the option texts only in the image).
Checks:

    answered      every question gets a 200 reply with a choice answer over
                  exactly its letters, probabilities summing to 1
    image tokens  every question's input is longer than the same question
                  without the image, so the image reached the prompt

With --baseline, the results of another build (normally main's, run on the
same machine just before) become the reference. The run fails if fewer than
--min-agreement of the questions get the same answer as there, or if more
than --tolerance fewer questions are answered correctly. d3-lite is close to
chance on MMMU-Pro vision (docs/docs/benchmarks/d3_results.md has 15.0% for
the NPU and 16.7% for upstream BF16 on the Decision Index's 60 questions), so
accuracy alone says little; agreement with main's build is what catches a
broken vision path. --min-accuracy adds a fixed floor for runs without a
baseline.

Every reply, and a results JSON that --baseline can read back, are written to
--output-dir.
"""

from __future__ import annotations

import base64
import json
import statistics
import sys
from pathlib import Path

from asr_summary import label as platform_label
from d3_mmmu_summary import render
from sweep_common import run_task
from sweep_decision import DecisionSweep, answer_shape_ok

INSTRUCTIONS = ("The image shows a multiple-choice question and its options, labelled with letters. "
                "Which option answers the question correctly?")


def letters(count: int) -> list[str]:
    return [chr(65 + i) for i in range(count)]


def question(count: int) -> dict:
    return {"answer": {"type": "choice", "instructions": INSTRUCTIONS,
                       "criteria": {letter: None for letter in letters(count)}}}


class D3MmmuTest(DecisionSweep):
    name = "d3_mmmu"
    csv_header = ["Model", "Question", "Answer", "Choice", "Confidence", "Input tokens",
                  "Latency (s)", "Status", "Error"]

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--eval-dir", required=True,
                            help="Directory holding manifest.json and the images.")
        parser.add_argument("--baseline", default=None, metavar="RESULTS_JSON",
                            help="Results of a reference build to compare against.")
        parser.add_argument("--min-agreement", type=float, default=0.9,
                            help="Share of questions that must get the baseline's answer (default: 0.9).")
        parser.add_argument("--tolerance", type=int, default=3,
                            help="How many fewer questions than the baseline may be answered "
                                 "correctly (default: 3).")
        parser.add_argument("--min-accuracy", type=float, default=None,
                            help="Fail below this accuracy, as a fraction. Unset: not checked.")

    def __init__(self, args):
        super().__init__(args)
        self.rows: list[dict] = []
        self.baseline = (json.loads(Path(args.baseline).read_text(encoding="utf-8"))
                         if args.baseline else None)
        self.replies_dir = self.output_dir / "replies"
        self.replies_dir.mkdir(exist_ok=True)

    def select_models(self, catalog: list[dict]) -> list[str]:
        return [self.needs_decision]

    def check(self, writer, model: str, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append({"check": name, "status": "ok" if ok else "FAIL", "detail": detail})
        self.record(writer, [model, "", "", "", "", "", "", "ok" if ok else "FAIL", f"{name}: {detail}"],
                    model=model, error=None if ok else f"{name} ({detail})")
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    # -- questions ---------------------------------------------------------

    def ask(self, writer, model: str, item: dict, eval_dir: Path, text_tokens: dict) -> dict:
        data = base64.b64encode((eval_dir / item["file"]).read_bytes()).decode()
        try:
            status, body, seconds = self.post({"questions": question(item["options"]),
                                               "images": [f"data:{item['mime']};base64,{data}"]})
        except OSError as exc:   # no reply at all: timed out, or the server is gone
            status, body, seconds = 0, {"_raw": str(exc)}, 0.0
        (self.replies_dir / f"{Path(item['file']).stem}.json").write_text(json.dumps(body), encoding="utf-8")
        answer = (body.get("answers") or {}).get("answer", {})
        probabilities = answer.get("probabilities") or {}
        tokens = (body.get("usage") or {}).get("input_tokens", 0)
        error = ""
        if status != 200:
            error = f"status {status}: {json.dumps(body)[:160]}"
        elif "error" in answer or not answer_shape_ok(answer):
            error = json.dumps(answer)[:160]
        elif list(probabilities) != letters(item["options"]) or abs(sum(probabilities.values()) - 1) > 1e-3:
            error = f"probabilities over {list(probabilities)}, sum {sum(probabilities.values()):.4f}"
        elif tokens <= text_tokens[item["options"]]:
            error = f"{tokens} input tokens, {text_tokens[item['options']]} without the image"
        result = {"answer": item["answer"], "choice": answer.get("choice"),
                  "correct": not error and answer.get("choice") == item["answer"],
                  "probabilities": probabilities, "confidence": answer.get("confidence"),
                  "input_tokens": tokens, "seconds": seconds, "error": error}
        self.record(writer, [model, item["id"], item["answer"], result["choice"], result["confidence"],
                             tokens, f"{seconds:.2f}", "error" if error else "ok", error], model=model)
        mark = "error" if error else ("right" if result["correct"] else "wrong")
        print(f"  {item['id']}: {result['choice']} (answer {item['answer']}) {mark}, "
              f"{tokens} tokens, {seconds:.2f} s" + (f"  {error}" if error else ""))
        return result

    def text_only_tokens(self, counts: set[int]) -> dict[int, int]:
        """Input tokens of each question shape without its image."""
        tokens = {}
        for count in sorted(counts):
            status, body, _ = self.post({"questions": question(count)})
            tokens[count] = (body.get("usage") or {}).get("input_tokens", 0) if status == 200 else 0
        return tokens

    # -- comparison --------------------------------------------------------

    def compare(self, writer, model: str, items: dict) -> dict | None:
        if not self.baseline:
            return None
        base_items = self.baseline.get("items", {})
        if self.baseline.get("eval", {}).get("revision") != self.results["eval"]["revision"] \
                or self.baseline.get("eval", {}).get("seed") != self.results["eval"]["seed"]:
            self.check(writer, model, "baseline asked the same questions", False,
                       "different dataset revision or seed")
            return None
        shared = [q for q in items if q in base_items and not items[q]["error"] and not base_items[q]["error"]]
        agree = sum(items[q]["choice"] == base_items[q]["choice"] for q in shared)
        correct = sum(items[q]["correct"] for q in shared)
        base_correct = sum(base_items[q]["correct"] for q in shared)
        dps = [sum(abs(items[q]["probabilities"][k] - v) for k, v in base_items[q]["probabilities"].items())
               / len(base_items[q]["probabilities"]) for q in shared]
        seconds = [base_items[q]["seconds"] for q in base_items]
        if shared:
            self.check(writer, model, "answers match main's build",
                       agree >= self.args.min_agreement * len(shared),
                       f"{agree}/{len(shared)}, at least {self.args.min_agreement:.0%} required")
            self.check(writer, model, "accuracy no lower than main's build",
                       correct >= base_correct - self.args.tolerance,
                       f"{correct} vs {base_correct} of {len(shared)}, {self.args.tolerance} fewer allowed")
        else:
            self.check(writer, model, "answers match main's build", False,
                       "no question was answered by both builds")
        return {"version": self.baseline.get("version"), "questions": len(shared), "agree": agree,
                "correct": base_correct, "mean_dp": statistics.fmean(dps) if dps else None,
                "median_seconds": statistics.median(seconds) if seconds else None}

    # -- entry point -------------------------------------------------------

    def run(self) -> None:
        eval_dir = Path(self.args.eval_dir)
        manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest_items = manifest["items"]
        model = self.resolve_models()[0]
        self.results = {"platform": self.args.platform, "version": self.version, "model": model,
                        "eval": {k: manifest.get(k) for k in ("source", "revision", "seed", "count")},
                        "items": {}, "checks": self.rows}
        print(f"\n--- d3 MMMU-Pro vision: {model}, {len(manifest_items)} questions from {manifest['source']} "
              f"at {manifest['revision'][:12]}, seed {manifest['seed']} ---")
        handle, writer = self.open_csv()
        try:
            text_tokens = self.text_only_tokens({i["options"] for i in manifest_items})
            items = {i["id"]: self.ask(writer, model, i, eval_dir, text_tokens) for i in manifest_items}
            self.results["items"] = items
            failed = [q for q, r in items.items() if r["error"]]
            self.check(writer, model, "every question answered", not failed,
                       f"{len(items) - len(failed)}/{len(items)}"
                       + (f"; {failed[0]}: {items[failed[0]]['error']}" if failed else ""))
            correct = sum(r["correct"] for r in items.values())
            self.results.update({
                "questions": len(items), "correct": correct,
                "chance": statistics.fmean(1 / i["options"] for i in manifest_items),
                "median_seconds": statistics.median(r["seconds"] for r in items.values()),
                "median_input_tokens": int(statistics.median(r["input_tokens"] for r in items.values())),
            })
            print(f"  accuracy {correct}/{len(items)}")
            if self.args.min_accuracy is not None:
                self.check(writer, model, "accuracy at least --min-accuracy",
                           correct >= self.args.min_accuracy * len(items),
                           f"{correct}/{len(items)}, at least {self.args.min_accuracy:.0%} required")
            self.results["baseline"] = self.compare(writer, model, items)
        finally:
            handle.close()
            # Failures that are not a check, such as the server crashing, still belong in the report.
            checked = {f"{model}: {c['check']} ({c['detail']})" for c in self.rows}
            self.rows += [{"check": f, "status": "FAIL", "detail": ""} for f in self.failures if f not in checked]
            results_path = self.output_dir / f"{self.name}_{self.args.platform}_v{self.version}_results.json"
            results_path.write_text(json.dumps(self.results, indent=1), encoding="utf-8")
            (self.output_dir / f"{self.name}_{self.args.platform}_v{self.version}.md").write_text(
                render([self.results], title=f"d3 MMMU-Pro vision test: {platform_label(self.args.platform)}")
                + "\n", encoding="utf-8")
        self.write_summary([model])


if __name__ == "__main__":
    sys.exit(run_task(D3MmmuTest))
