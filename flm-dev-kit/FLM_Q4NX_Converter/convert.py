#!./venv/bin/python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import argparse
import os
import subprocess
import sys
from pathlib import Path
from q4nx import create_converter


def convert_gguf_to_q4nx(gguf_path: str, q4nx_path: str, override_model_arch:str, weights_type: str = 'language',
                         embed_bf16: str = None, head_bf16: str = None,
                         allow_f16_vision: bool = False, require_bf16_vision: bool = False):
    model = create_converter(gguf_path, override_model_arch)
    # Optional: take token_embd.weight from a BF16 checkpoint instead of the GGUF.
    # Set before convert() so it does not change the abstract convert() signature.
    model.embed_bf16_path = embed_bf16
    model.head_bf16_path = head_bf16
    model.allow_f16_vision = allow_f16_vision
    model.require_bf16_vision = require_bf16_vision
    model.convert(q4nx_path=q4nx_path, weights_type=weights_type)
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Convert GGUF model files to Q4NX format (output always named model.q4nx)',
        epilog='Examples:\n'
               '  python convert.py -i model.gguf\n'
               '  python convert.py -i model.gguf -o output_folder\n'
               '  python convert.py model.gguf output_folder\n'
               '  python convert.py model.gguf .\n'
               '  python convert.py -i vision_model.gguf -o output_folder -t vision\n'
               '  python convert.py -i model.gguf -o out -e bf16_weights.safetensors',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add support for both flag-based and positional arguments
    parser.add_argument('input_file', nargs='?', help='Input GGUF file (positional)')
    parser.add_argument('output_folder', nargs='?', help='Output folder (positional, optional)')
    parser.add_argument('-i', '--input', dest='input_flag', help='Input GGUF file')
    parser.add_argument('-o', '--output', dest='output_flag', help='Output folder (optional, defaults to input file directory)')
    parser.add_argument('-t', '--type', dest='weights_type', default='language', help='Type of weights to convert (default: language)',
                        choices=['language', 'vision', 'audio', 'npu_q41'])
    parser.add_argument('-f', '--force', dest='force_model_type', default="", help="Model type. Empty string for automatic recognition from gguf file")
    parser.add_argument('-e', '--embed-bf16', dest='embed_bf16', default=None,
                        help='Safetensors file to read the BF16 token embedding table from, '
                             'instead of dequantizing the GGUF copy. q4nx stores this tensor as '
                             'BF16 anyway, so this costs no file size and avoids inheriting the '
                             "GGUF's quantization loss in the embedding.")

    parser.add_argument('--allow-f16-vision', dest='allow_f16_vision', action='store_true',
                        help='Vision only: convert even if the mmproj stores the tower as F16/quantised. The tower is shipped '
                             'as bf16, so such a source is rounded to bf16 and can differ from the checkpoint (a stale '
                             'export); the default is to refuse it for the models whose conversion is checked on BF16 sources '
                             'and to warn for the others. The rounded tensor count is recorded in the file metadata.')
    parser.add_argument('--require-bf16-vision', dest='require_bf16_vision', action='store_true',
                        help='Vision only: refuse a non-BF16 source for every model (also those that only warn by default).')
    parser.add_argument('--verify-hf', dest='verify_hf', default=None, metavar='MODEL_SAFETENSORS',
                        help='MiniCPM-V-4.7-1B vision only: after converting, compare every tensor of the '
                             'written vision_weight.q4nx with this HF model.safetensors '
                             '(verify_minicpm_v_vision_q4nx.py) and exit non-zero unless 0 elements differ. '
                             'Without it the conversion is NOT checked against the checkpoint.')

    parser.add_argument('--head-bf16', dest='head_bf16', default=None,
                        help='Safetensors shard holding lm_head.weight in BF16: the output head is quantized to Q8_0 from it '
                             'instead of from the GGUF copy (MiniCPM-V-35B-A3B only, where the GGUF has it at Q6_K).')

    args = parser.parse_args()

    # Determine input file (prioritize flag, then positional)
    input_path = args.input_flag or args.input_file

    if not input_path:
        parser.error('Input file is required. Use -i <file> or provide as positional argument.')

    # Determine output folder (prioritize flag, then positional)
    output_folder = args.output_flag or args.output_folder

    # Check if input file exists
    if not os.path.exists(input_path):
        parser.error(f'Input file does not exist: {input_path}')

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_folder)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Converting {input_path} to {output_folder}...")
    if args.embed_bf16 and not os.path.exists(args.embed_bf16):
        parser.error(f'--embed-bf16 file does not exist: {args.embed_bf16}')

    model = convert_gguf_to_q4nx(input_path, output_folder, args.force_model_type, weights_type=args.weights_type,
                         embed_bf16=args.embed_bf16, head_bf16=args.head_bf16,
                         allow_f16_vision=args.allow_f16_vision, require_bf16_vision=args.require_bf16_vision)
    verifier_name = getattr(model, 'vision_hf_verifier', None)
    if args.weights_type == 'vision' and args.verify_hf:
        if not verifier_name:
            print(f"[ERROR] --verify-hf is not available for {type(model).__name__}: no HF comparison is written "
                  "for this model's vision tower yet.", file=sys.stderr)
            sys.exit(1)
        verifier = os.path.join(os.path.dirname(os.path.abspath(__file__)), verifier_name)
        cmd = [sys.executable, verifier, '--gguf', input_path, '--out', output_folder,
               '--hf', args.verify_hf, '--max-hf-diff', '0']
        print(f"[INFO] Gate: {' '.join(cmd)}")
        if subprocess.call(cmd) != 0:
            print(f"[ERROR] {output_folder}/vision_weight.q4nx does NOT match {args.verify_hf}; "
                  "do not ship it.", file=sys.stderr)
            sys.exit(1)
    elif args.weights_type == 'vision' and verifier_name:
        print("[WARN] vision_weight.q4nx was not compared with the HF checkpoint (--verify-hf).")
    print(f"[INFO] Conversion complete! Output saved to {output_folder}")



if __name__ == "__main__":
    # for debug, give the path and ouptut path here by directly set the command line args
    import sys
    # sys.argv = ['convert.py', '-i', 'unsloth_gpt-oss-20b-Q4_0.gguf', '-o', 'unsloth-gotoss20b-q40']

    # sys.argv = ['convert.py', '-i', 'unsloth_gpt-oss-20b-Q4_1.gguf', '-o', 'unsloth-gotoss20b-q41']

    # import sys
    # # sys.argv = ['convert.py', '-i', 'gemma-3-4b-it-Q4_1.gguf', '-o', 'unsloth-gemma3-q41']
    # # main()


    # # sys.argv = ['convert.py', '-i', 'gemma3-mmproj-BF16.gguf', '-o', 'unsloth-gemma3-vision', '-t', 'vision']
    # # main()



    # # sys.argv = ['convert.py', '-i'
    # , 'medgemma3-mmproj-BF16.gguf', '-o', 'unsloth-medgemma3-vision', '-t', 'vision']
    # # main()
    # sys.argv = ['convert.py', '-i', 'Qwen3-VL-4B-Instruct-Q4_1.gguf', '-o', 'unsloth-qwen3vl-4b-q41' ]
    # main()

    # sys.argv = ['convert.py', '-i', 'Qwen3-4B-Q4_1.gguf', '-o', 'unsloth-qwen3-4b-q41' ]
    # main()
    # sys.argv = ['convert.py', '-i', 'qwen3vl-4b-mmproj-BF16.gguf', '-o', 'unsloth-qwen3vl-vision', '-t', 'vision']
    # main()



    #sys.argv = ['convert.py', '-i', 'qwen3_5vl-4b-mmproj-BF16.gguf', '-o', 'unsloth-qwen3_5vl-vision', '-t', 'vision']


    #sys.argv = ['convert.py', '-i', 'qwen3_5vl-9bmmproj-BF16.gguf', '-o', 'unsloth-qwen3_5_9bvl-vision', '-t', 'vision']

    #sys.argv = ['convert.py', '-i', 'Qwen3.5-0.8B-Q4_1.gguf', '-o', 'output/unsloth-qwen3_5_0.8bq41']
    #sys.argv = ['convert.py', '-i', 'Qwen3.5-4B-Q4_1.gguf', '-o', 'output/unsloth-qwen3_5_4bq41']
    # sys.argv = ['convert.py', '-i', "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", '-o', 'output/unsloth-qwen3_6_a3bq4k_s']
    #sys.argv = ['convert.py', '-i', 'Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf', '-o', 'unsloth-qwen3_59b_uncensored', "-f", "qwen3.5-9B"]
    # sys.argv = ['convert.py', '-i', 'Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf', '-o', 'unsloth-qwen3_59b_uncensored', "-f", "qwen3.5-9B"]

    # sys.argv = ['convert.py', '-i', 'Qwen3.5-9B-Q4_1.gguf', '-o', 'unsloth-qwen3_5_9bq41']



    #sys.argv = ['convert.py', '-i', 'gemma-4-E2B-it-Q4_1.gguf', '-o', 'unsloth-gemma4-2b-it-q41']

    #sys.argv = ['convert.py', '-i', 'gemma4-2b-mmproj.gguf', '-o', 'unsloth-gemma4-2b-vision', '-t', 'vision']

    #sys.argv = ['convert.py', '-i', 'gemma4-2b-mmproj.gguf', '-o', 'unsloth-gemma4-2b-audio', '-t', 'audio']
    # sys.argv = ['convert.py', '-i', 'debug_gemma4e2b_model.gguf', '-o', 'debug-gemma4-2b-audio', '-t', 'audio', '-f', 'gemma4']
    # sys.argv = ['convert.py', '-i', 'gguf_files/mmproj-BF16.gguf', '-o', 'q4nx_files/qwen3.6-moe-vision', "-f", "qwen35moe", "-t", "vision"]
    main()
