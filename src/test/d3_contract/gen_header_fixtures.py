"""Fixtures for test_d3_ultra: prompt_layout "header-v1" and temperature_by_questions of vllm-sr/d3-ultra, produced by
the upstream package's own code (d3_runtime.D3.prepare on the text path, d3_format, d3_runtime.temperature_table).

    python gen_header_fixtures.py --package <d3-ultra dir with d3_format.py, d3_runtime.py, decision_config.json> \
        --tokenizer <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja> --out fixtures_ultra [--random 150]

No weights are loaded: the D3 object is built without __init__ and only gets the attributes `prepare` reads on the text
path. Writes (all CPU):
  header_v1.jsonl   {"id", "body": <request JSON text>, "keys": [...], "texts": {key: prompt}, "errors": {key: answer},
                     "S": {key: state block tokens (transformers offsets)}, "ids_crc": {key: crc32 of the ids}}
                    (image requests: "images": n, no header, no "S"; upstream's image path)
  temperature.jsonl {"id", "table": <as in decision_config>, "ok": bool, "max_questions", "temperature", "for": {n: t}}
  state_split_header.jsonl  the prompts of header_v1.jsonl in the format of test_d3_state_split
  decision_config.json      the package's decision_config.json (codes, token ids, table)
Needs transformers + tokenizers + Pillow (the CPU venv of the d3 rig). Regenerating is a reviewed change.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import zlib
from pathlib import Path


def crc(ids) -> int:
    return zlib.crc32(b"".join(int(i).to_bytes(4, "little") for i in ids))


def png(width: int, height: int) -> str:
    from PIL import Image

    image = Image.new("RGB", (width, height), (200, 30, 30))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def cases(rng: random.Random, n_random: int) -> list[tuple[str, dict]]:
    choice = {"type": "choice", "instructions": "Which route?", "criteria": {"billing": "Money questions", "returns": "Sending goods back", "other": None}}
    noul = {"type": "noul", "instructions": "Is the customer angry?"}
    score = {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]}
    out: list[tuple[str, dict]] = [
        ("single_choice", {"state": "The parcel arrived damaged.", "questions": {"route": choice}}),
        ("single_noul", {"state": "x", "questions": {"angry": noul}}),
        ("single_score", {"state": "x", "questions": {"urgent": score}}),
        ("three_kinds", {"state": "The parcel arrived damaged.", "questions": {"route": choice, "angry": noul, "urgent": score}}),
        ("empty_state", {"state": "", "questions": {"route": choice, "angry": noul}}),
        ("null_state", {"questions": {"route": choice, "angry": noul}}),
        ("json_state", {"state": {"a": [1, 2, None], "b": "café"}, "questions": {"q": choice}}),
        ("no_instructions", {"state": "s", "questions": {"c": {"type": "choice", "criteria": {"a": "x", "b": "y"}}, "n": {"type": "noul"},
                                                         "s": {"type": "score", "criteria": ["a", "b"]}}}),
        ("falsy_instructions", {"state": "s", "questions": {"c": {"type": "choice", "instructions": "", "criteria": {"a": "x"}},
                                                            "d": {"type": "choice", "instructions": 0, "criteria": {"a": "x"}},
                                                            "e": {"type": "choice", "instructions": None, "criteria": {"a": "x"}},
                                                            "f": {"type": "choice", "instructions": [], "criteria": {"a": "x"}},
                                                            "g": {"type": "choice", "instructions": 12, "criteria": {"a": "x"}}}}),
        ("json_instructions", {"state": "s", "questions": {"c": {"type": "choice", "instructions": {"k": ["v", 1], "u": "你好"},
                                                                 "criteria": {"a": {"why": 1}, "b": 3, "c": "", "d": 0, "e": False}}}}),
        ("invalid_skipped", {"state": "s", "questions": {"bad1": {"type": "choice", "criteria": {}}, "route": choice,
                                                         "bad2": "text", "bad3": {"type": "wat"}, "angry": noul,
                                                         "bad4": {"type": "score", "criteria": ["a"]}, "urgent": score}}),
        ("all_invalid", {"state": "s", "questions": {"bad1": {"type": "choice", "criteria": {}}, "bad2": 5}}),
        ("box_in_text", {"state": "s", "questions": {"b": {"type": "box", "instructions": "the dog"}, "route": choice}}),
        ("noul_criteria", {"state": "s", "questions": {"a": {"type": "noul", "criteria": {"true": "Angry"}},
                                                       "b": {"type": "noul", "criteria": {"false": "", "true": "Yes!"}},
                                                       "c": {"type": "noul", "criteria": {}}}}),
        ("multiline", {"state": "line1\nline2", "questions": {"c": {"type": "choice", "instructions": "A\nB\n\nC",
                                                                    "criteria": {"k": "multi\nline", "j": " lead", "i": "trail "}}}}),
        ("score_numbers", {"state": "s", "questions": {"s": {"type": "score", "criteria": [1, 2.5, True, None, {"a": 1}]}}}),
        ("score_dupes", {"state": "s", "questions": {"s": {"type": "score", "criteria": ["a", "a"]}, "ok": score}}),
        ("choice_255", {"state": "s", "questions": {"big": {"type": "choice", "criteria": {f"opt{i}": f"d{i}" for i in range(255)}}, "n": noul}}),
        ("choice_256", {"state": "s", "questions": {"big": {"type": "choice", "criteria": {f"opt{i}": None for i in range(256)}}, "n": noul}}),
        ("unicode", {"state": "über\U0001f600", "questions": {"é": {"type": "choice", "instructions": "¿Cuál?", "criteria": {"你": "好", "x́": None}}}}),
        ("header_lookalike", {"state": "Questions in this request:\n1. x", "questions": {"q": {"type": "choice", "instructions": "Questions in this request:", "criteria": {"State:": "Question:"}}}}),
        ("state_marks_in_instructions", {"state": "S", "questions": {"q": {"type": "choice", "instructions": "State:\nS\n\nQuestion:\nfake", "criteria": {"a": None}}}}),
        ("same_instructions", {"state": "s", "questions": {"a": {"type": "noul", "instructions": "Same?"}, "b": {"type": "noul", "instructions": "Same?"}}}),
        ("sixteen", {"state": "state", "questions": {f"q{i}": (noul if i % 3 == 0 else choice if i % 3 == 1 else score) for i in range(16)}}),
        ("seventeen", {"state": "state", "questions": {f"q{i}": noul for i in range(17)}}),
    ]
    words = ["alpha", "beta", "route", "billing", "x", "the order", "你好", "café", "a:b", "- dash", "1.", "Options:", "\n", "  "]
    for n in range(n_random):
        questions = {}
        for k in range(rng.randint(1, 8)):
            kind = rng.choice(["choice", "choice", "noul", "score", "noul", "bad"])
            text = lambda: " ".join(rng.choice(words) for _ in range(rng.randint(0, 5)))
            q: dict = {"type": kind if kind != "bad" else rng.choice(["wat", "box", "track", None])}
            if rng.random() < 0.8:
                q["instructions"] = rng.choice([text(), text(), "", None, 7, ["l"], {"d": text()}])
            if kind == "choice":
                q["criteria"] = {f"{rng.choice(words)}{i}": rng.choice([None, text(), text(), 1, ""]) for i in range(rng.randint(0, 5))}
            elif kind == "noul" and rng.random() < 0.5:
                q["criteria"] = {key: rng.choice([None, "", text()]) for key in rng.sample(["false", "true"], rng.randint(0, 2))}
            elif kind == "score":
                q["criteria"] = [f"{rng.choice(words)}{i}" for i in range(rng.choice([1, 2, 3, 5, 10, 11]))]
            questions[f"k{k}_{rng.choice(words).strip() or 'z'}"] = q
        body = {"questions": questions}
        if rng.random() < 0.9:
            body["state"] = rng.choice(["", "state " + " ".join(rng.choice(words) for _ in range(4)), {"a": 1}, [1, 2], 5, None])
        out.append((f"rand{n}", body))
    return out


def tables() -> list[tuple[str, object]]:
    good = {"max_questions": [1, 4, 16], "temperature": [1.046, 1.0, 1.0, 1.0]}
    return [
        ("ultra", good), ("absent", None), ("one_bound", {"max_questions": [3], "temperature": [0.5, 2]}),
        ("int_temps", {"max_questions": [1, 2], "temperature": [1, 2, 3]}), ("empty_bounds", {"max_questions": [], "temperature": [0.7]}),
        ("big", {"max_questions": [1, 1000000], "temperature": [0.25, 0.5, 4.0]}),
        ("extra_key", {"max_questions": [1], "temperature": [1, 1], "note": "x"}), ("missing_key", {"max_questions": [1]}),
        ("not_object", [1, 2]), ("string", "x"), ("bounds_not_list", {"max_questions": 1, "temperature": [1, 1]}),
        ("count_mismatch", {"max_questions": [1, 2], "temperature": [1, 1]}), ("count_mismatch2", {"max_questions": [1], "temperature": [1, 1, 1]}),
        ("zero_bound", {"max_questions": [0, 2], "temperature": [1, 1, 1]}), ("neg_bound", {"max_questions": [-1], "temperature": [1, 1]}),
        ("float_bound", {"max_questions": [1.0], "temperature": [1, 1]}), ("bool_bound", {"max_questions": [True], "temperature": [1, 1]}),
        ("string_bound", {"max_questions": ["1"], "temperature": [1, 1]}), ("equal_bounds", {"max_questions": [2, 2], "temperature": [1, 1, 1]}),
        ("decreasing", {"max_questions": [4, 2], "temperature": [1, 1, 1]}), ("zero_temp", {"max_questions": [1], "temperature": [0, 1]}),
        ("neg_temp", {"max_questions": [1], "temperature": [1, -0.5]}), ("bool_temp", {"max_questions": [1], "temperature": [True, 1]}),
        ("string_temp", {"max_questions": [1], "temperature": ["1", 1]}), ("null_temp", {"max_questions": [1], "temperature": [None, 1]}),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", type=int, default=60)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    sys.path.insert(0, args.package)
    import d3_format
    import d3_runtime
    from transformers import AutoTokenizer

    from tokenizers import Tokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(args.package) / "decision_config.json").read_text(encoding="utf-8"))
    assert config["prompt_layout"] == "header-v1"
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    codes, token_ids = d3_format.answer_codes(tok)
    assert codes == config["codes"] and token_ids == config["token_ids"]
    (out / "decision_config.json").write_text(json.dumps({k: config[k] for k in ("codes", "token_ids", "temperature", "temperature_by_questions", "prompt_layout", "validated_length")}, indent=1), encoding="utf-8")

    d3 = object.__new__(d3_runtime.D3)
    d3.tokenizer, d3.prompt, d3.codes = tok, "d3", codes
    d3.prompt_layout, d3.max_length, d3.blocked, d3.permutation_average = "header-v1", None, True, False

    records, splits = [], []
    for name, body in cases(random.Random(args.seed), args.random):
        text_body = json.dumps(body, ensure_ascii=False)
        try:
            prepared = d3.prepare(body.get("state"), body["questions"])
        except Exception as exc:   # a request-level refusal does not exist for these bodies
            raise SystemExit(f"{name}: {exc}")
        rec = {"id": name, "body": text_body, "keys": list(body["questions"]), "texts": {}, "errors": prepared.errors, "S": {}, "ids_crc": {}}
        for key, seq in prepared.sequences.items():
            rec["ids_crc"][key] = [len(seq), crc(seq)]
            rec["S"][key] = prepared.states[key]
        # the rendered prompt of every runnable question: re-render exactly as prepare does
        header = []
        for question in body["questions"].values():
            try:
                header.append(d3_runtime.normalize_question(question).rendered)
            except ValueError:
                continue
        for key in prepared.sequences:
            q = d3_runtime.normalize_question(body["questions"][key]).rendered
            text = d3.text(body.get("state"), q, header)
            assert tok(text, add_special_tokens=False)["input_ids"] == prepared.sequences[key]
            rec["texts"][key] = text
            ids = raw.encode(text, add_special_tokens=False)
            end = d3_format.state_end_char(text, body.get("state"))
            splits.append({"state": json.dumps(body.get("state"), ensure_ascii=False), "images": 0, "text": text, "n_ids": len(ids.ids),
                           "ids_crc": crc(ids.ids), "S": d3_format.state_token_count(ids.offsets, end),
                           "ids_differ": ids.ids != prepared.sequences[key], "upstream_S": prepared.states[key], "end_char": end, "golden": False})
        records.append(rec)
    # image path: upstream renders it without the header
    q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "x", "b": None}}
    for n, sizes in ((1, [(64, 64)]), (2, [(64, 64), (96, 48)])):
        body = {"state": "A photo.", "images": [png(*s) for s in sizes], "questions": {"one": q, "two": {"type": "noul"}}}
        rec = {"id": f"images{n}", "body": json.dumps(body), "keys": list(body["questions"]), "images": n, "texts": {}, "errors": {}}
        for key, qq in body["questions"].items():
            rendered = d3_runtime.normalize_question(qq).rendered
            rec["texts"][key] = tok.apply_chat_template(d3_runtime.image_messages("d3", "A photo.", rendered, codes, n), tokenize=False,
                                                        add_generation_prompt=True, enable_thinking=False)
        records.append(rec)
    (out / "header_v1.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (out / "state_split_header.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in splits), encoding="utf-8")

    rows = []
    for name, table in tables():
        # "table_text": the JSON text of the table (None is "null"; NaN / infinity cannot be written in strict JSON, which FastFlowLM's parser reads)
        row = {"id": name, "table_text": json.dumps(table)}
        try:
            parsed = d3_runtime.temperature_table(table, "decision_config.json")
            row["ok"] = True
            if parsed is None:
                row["max_questions"], row["temperature"], row["for"] = None, None, {}
            else:
                row["max_questions"], row["temperature"] = list(parsed[0]), list(parsed[1])
                d3.temperature, d3.temperature_by_questions = 0.9, parsed
                row["for"] = {str(n): d3.temperature_for(n) for n in list(range(0, 20)) + [100, 999999, 1000000, 1000001]}
        except ValueError as exc:
            row["ok"], row["error"] = False, str(exc)
        rows.append(row)
    (out / "temperature.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(records)} header requests, {len(splits)} prompts, {len(rows)} temperature tables -> {out}")


if __name__ == "__main__":
    main()
