# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Package the non-weight files of MiniCPM-OCR-1.0-NPU2 next to model.q4nx.

    config.json            FLM_DLL/detail/minicpm_ocr_npu/model_config.template.json,
                           top-level "_comment*" keys stripped, values untouched
    tokenizer.json         copied from the HF folder (the MiniCPM converter does
    tokenizer_config.json  not extract a tokenizer from the GGUF; see
    chat_template.jinja    q4nx/models/minicpm_ocr.py)

Order does not matter relative to convert.py; neither step writes the other's files.

    python package_minicpm_ocr.py [--hf DIR] [--out DIR]
"""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(os.path.dirname(HERE), "FLM_DLL", "detail", "minicpm_ocr_npu",
                        "model_config.template.json")
HF_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", default="/path/to/models/MiniCPM-OCR-1.0")
    ap.add_argument("--out", default="/path/to/models/MiniCPM-OCR-1.0-NPU2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(TEMPLATE))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_comment")}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    print(f"[INFO] config.json <- {os.path.relpath(TEMPLATE, os.path.dirname(HERE))}")

    for fn in HF_FILES:
        shutil.copyfile(os.path.join(args.hf, fn), os.path.join(args.out, fn))
        print(f"[INFO] {fn} <- {args.hf}")

    _patch_tokenizer_config(args.hf, args.out)


def _patch_tokenizer_config(hf_dir, out_dir):
    """Add the two integer id fields FastFlowLM requires and HF does not ship.

    _shared_setup_tokenizer() hard-exits with "bos_token is set in
    tokenizer_config.json but bos_token_id is missing or not an integer", and
    then again on a missing eos_token_id array. MiniCPM's HF tokenizer_config
    carries the token *strings* but keeps the ids in generation_config.json, so
    a verbatim copy makes `flm run` die before the model ever loads.

    Both values are read from generation_config.json rather than hardcoded, and
    nothing else in the file is touched. S6 patched the packaged copy by hand to
    get Gate A green; doing it here means re-running packaging cannot silently
    revert it.
    """
    gen = json.load(open(os.path.join(hf_dir, "generation_config.json")))
    bos = gen["bos_token_id"]
    eos = gen["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]

    path = os.path.join(out_dir, "tokenizer_config.json")
    cfg = json.load(open(path))
    cfg["bos_token_id"] = bos
    cfg["eos_token_id"] = eos
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    # 248046 <|im_end|> is what actually ends a chat turn; 248044 <|endoftext|>
    # is the config.json eos and was never observed terminating a turn.
    print(f"[INFO] tokenizer_config.json += bos_token_id {bos}, eos_token_id {eos}")


if __name__ == "__main__":
    main()
