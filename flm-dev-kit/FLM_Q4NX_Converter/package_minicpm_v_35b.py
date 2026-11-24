# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Package the non-weight files of MiniCPM-V-4.7-35B-A3B-NPU2 next to model.q4nx and vision_weight.q4nx.

    config.json            FLM_DLL/detail/minicpm_v_4_7_moe_npu/model_config.template.json,
                           top-level "_comment*" keys stripped, values untouched
    tokenizer.json         copied from the HF folder (the GGUF-extracted tokenizer.json is lossy,
    tokenizer_config.json  it drops information the HF file has)
    chat_template.jinja    (thinking is off unless enable_thinking is true, as the model card recommends)
    LICENSE / NOTICE       not shipped by the HF repo: the licence is Apache-2.0 (see its card)

    python package_minicpm_v_35b.py --hf DIR --out DIR

Weights are separate steps (see FLM_Q4NX_Converter/README.md):
    python convert.py -i MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf -o OUT -f minicpm-v-35b-a3b            (model.q4nx)
    python convert.py -i mmproj-model-f16.gguf              -o OUT -f minicpm-v-35b-a3b -t vision  (vision_weight.q4nx)
"""
import argparse
import json
import os
import shutil

from package_minicpm_v import _patch_tokenizer_config

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(os.path.dirname(HERE), "FLM_DLL", "detail", "minicpm_v_4_7_moe_npu",
                        "model_config.template.json")
HF_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", required=True, help="folder with the upstream openbmb/MiniCPM-V-4.7-35B-A3B files")
    ap.add_argument("--out", required=True)
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


if __name__ == "__main__":
    main()
