"""Fixtures for test_d3_state_split: the prefix_bidir state block length S of many rendered prompts, from the
upstream package's own code (d3_format.encode_split: tokenizer offset mapping + state_end_char).

    python gen_state_split.py --package <vllm-sr/d3-turbo dir with d3_format.py, d3_runtime.py> \
        --tokenizer <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja> \
        --goldens fixtures/goldens.jsonl --out fixtures/state_split.jsonl [--random 400] [--seed 11]

Each line: {"state": <the state as JSON text>, "images": n, "text": rendered prompt, "n_ids", "ids_crc", "S",
"end_char"}. The C++ test re-tokenizes `text` with FastFlowLM's tokenizer, which has no offsets API, and must
get the same ids (crc) and the same S. Needs transformers + tokenizers (the CPU venv of the d3 rig).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path


def edge_states() -> list:
    """States built to hit byte-level merges at the state/question boundary and unicode corners."""
    out: list = [None, "", "(empty)", "State:", " ", "  ", "\t", "\n", "\n\n", "\n\n\n", " \n", "  \n", "\r\n", "\r", "a",
                 "0", "12345", "-1", 0, 1, 3.5, True, False, [], {}, [1, 2, 3], {"a": 1, "b": [True, None, "x"]},
                 {"k": "v\n"}, ["line\n"], [" "], "<|im_end|>", "x<|im_end|>", "<|im_start|>user", "<think>", "</think>",
                 "<|vision_start|>", "<tool_call>", "<|endoftext|>", "a<|endoftext|>"]
    tails = [".", "..", "...", "....", "!", "!!", "!!!", "?", "?!", "!?", ",", ";", ":", "::", ")", "))", "]", "}", ")}",
             '"', "'", "`", "``", "```", "-", "--", "---", "=", "==", "*", "**", "#", "##", "/", "//", "\\", "|", "||",
             "_", "__", "~", "^", "&", "%", "$", "@", "+", "<", ">", "</", "\u3002", "\u3001", "\u2026", "\u2014",
             "\u00a0", "\u200b", "\u200d", "\ufeff", "\u00ab", "\u00bb", "\u201c", "\u201d", "\u2019"]
    for t in tails:
        out += ["word" + t, "The order arrived damaged" + t, "12" + t, t, " " + t, "x " + t, "x" + t + " ", "x" + t + "  ",
                "x" + t + "\n", "x" + t + "\n\n", "x" + t + " \n"]
    for ws in [" ", "  ", "   ", "\t", " \t", "\n", "\n\n", "\n\n\n", " \n", "  \n\n", "\r\n", "\r\n\r\n", "\u3000", "\u2003"]:
        out += ["word" + ws, "word." + ws, "word 1" + ws, ws + "word", "\u4f60\u597d" + ws]
    out += [
        "caf\u00e9", "cafe\u0301", "a\u0323\u0301", "a\u0301\u0323", "\u1100\u1161\u11a8", "\uac01", "\u212b", "A\u030a",
        "\u00e9\u00e9\u00e9", "e\u0301e\u0301", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629", "\u05e9\u05dc\u05d5\u05dd",
        "\u4f60\u597d\u4e16\u754c", "\u3053\u3093\u306b\u3061\u306f\u3002", "\ud55c\uad6d\uc5b4", "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35",
        "\U0001f600", "\U0001f600\U0001f600", "ok \U0001f44d", "\U0001f468\u200d\U0001f469\u200d\U0001f467", "\U0001f1fa\U0001f1f8",
        "\U00010348", "\U0001d11e", "x\u0301", "x\u0301\u0301\u0301", "\u0e01\u0e34", "\u0915\u094d\u0937", "\u0301",
        "1\u0301", "\u00b2", "\u0663", "\u2460", "\u00bd", "\u2155", "\u0966\u0967", "1,234.56", "3.14159", "1e-5",
        "2026-10-25T03:12:44Z", "user@example.com", "https://example.com/a?b=c&d=e#f", "C:\\temp\\file.txt",
        "line1\nline2", "line1\nline2\n", "# Title\n\nbody\n\n", "- a\n- b\n", "a\tb\tc", "  indented", "trailing  ",
        "'s", "it's", "they've", "I'm", "x's", "'S", "dogs'", "'", "''", "\"quoted\"", "(paren)", "[bracket]", "{brace}",
        "a" * 300, " " * 40, "\n" * 5, "!" * 20, "." * 37, "word " * 60, "\u4f60" * 50, "\U0001f600" * 30,
        "Question:", "Options:", "A: x", "\n\nQuestion:\nfoo", "State:\nnested", "x\n\nQuestion:\n",
    ]
    return out


def random_states(rng: random.Random, n: int) -> list:
    pieces = ["the", "order", "a", "I", "x", "1", "23", "4567", ".", ",", "!", "?", "...", ")", "(", "\"", "'", ":", ";", "-",
              "--", "*", "#", " ", "  ", "\t", "\n", "\n\n", "\r\n", "\u00e9", "e\u0301", "\u4f60\u597d", "\u3002", "\U0001f600",
              "\u0627\u0644", "\u05e9", "$", "%", "&", "/", "\\", "_", "~", "=", "+", "<|im_end|>", "<think>", "\u200b", "\u00a0",
              "'s", "n't", "http://a.b/c", "{", "}", "[", "]", "|", "`", "@", "^"]
    out = []
    for _ in range(n):
        out.append("".join(rng.choice(pieces) for _ in range(rng.randint(1, 14))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-chars", type=int, default=8000, help="skip longer prompts (keeps the fixture small)")
    ap.add_argument("--images", type=int, default=60, help="how many states also get an image prompt (1 and 2 images)")
    args = ap.parse_args()
    sys.path.insert(0, args.package)
    import d3_format
    from transformers import AutoTokenizer

    from tokenizers import Tokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # tokenizer.json as the tokenizers library (and FastFlowLM's copy of it) reads it. transformers' Qwen2Tokenizer
    # rebuilds the pre-tokenizer (no \p{M} in the word class, trim_offsets on), so it differs on combining marks
    # (Thai, Devanagari ...): "ids_differ" marks those prompts; S is then checked against the raw tokenizer's offsets
    raw = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    codes, _ = d3_format.answer_codes(tok)
    question = {"type": "choice", "instructions": "Which one?", "criteria": {"a": "first", "b": None}}

    states: list = []
    for line in open(args.goldens, encoding="utf-8"):
        body = json.loads(line)["body"]
        body = json.loads(body) if isinstance(body, str) else body
        if isinstance(body, dict) and "state" in body:
            states.append(body["state"])
    n_golden = len(states)
    states += edge_states()
    rng = random.Random(args.seed)
    states += random_states(rng, args.random)

    seen = set()
    records = []
    for index, state in enumerate(states):
        key = json.dumps(state, ensure_ascii=False, sort_keys=False)
        if key in seen:
            continue
        seen.add(key)
        try:
            d3_format.user_prompt(state, question, codes)
            json.dumps(state, ensure_ascii=False)
        except Exception:
            continue
        variants = [(0, d3_format.render(tok, state, question, codes))]
        if len(records) % max(1, len(states) // max(1, args.images)) == 0 or index < 8:
            import d3_runtime

            for n in (1, 2):
                text = tok.apply_chat_template(
                    d3_runtime.image_messages("d3", state, question, codes, n),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                variants.append((n, text))
        for n_images, text in variants:
            try:
                ids, S = d3_format.encode_split(tok, text, state)
            except ValueError:
                continue   # a state that makes the prompt unsplittable (upstream refuses it too)
            if len(text) > args.max_chars:
                continue
            end_char = d3_format.state_end_char(text, state)
            enc = raw.encode(text, add_special_tokens=False)
            raw_ids = enc.ids
            raw_S = d3_format.state_token_count(enc.offsets, end_char)
            records.append({
                "state": json.dumps(state, ensure_ascii=False),
                "images": n_images,
                "text": text,
                "n_ids": len(raw_ids),
                "ids_crc": zlib.crc32(b"".join(i.to_bytes(4, "little") for i in raw_ids)),
                "S": raw_S,
                "ids_differ": raw_ids != ids,
                "upstream_S": S,
                "end_char": end_char,
                "golden": index < n_golden,
            })
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(f"{len(records)} prompts ({sum(r['images'] > 0 for r in records)} with images) from {len(seen)} states -> {args.out}")


if __name__ == "__main__":
    main()
