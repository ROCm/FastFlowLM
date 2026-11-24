# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import argparse

from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(
        description="Load and print a specific tensor from a safetensors file."
    )
    parser.add_argument("file", help="Path to the .safetensors file")
    parser.add_argument(
        "tensor",
        nargs="?",
        help="Name of the tensor to load. If omitted, lists available tensor names.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to load the tensor onto (default: cpu)"
    )
    args = parser.parse_args()

    with safe_open(args.file, framework="pt", device=args.device) as f:
        keys = list(f.keys())

        if args.tensor is None:
            print(f"{len(keys)} tensors in {args.file}:")
            for name in keys:
                print(f"  {name}")
            return

        if args.tensor not in keys:
            raise SystemExit(
                f"Tensor '{args.tensor}' not found. Available: {', '.join(keys)}"
            )

        tensor = f.get_tensor(args.tensor)
        print(f"name:  {args.tensor}")
        print(f"shape: {tuple(tensor.shape)}")
        print(f"dtype: {tensor.dtype}")
        print(tensor)
        print(tensor[0])        
        # print(tensor[0][0][0: 100])


if __name__ == "__main__":
    main()


#python3 load_tensor.py  /path/to/models/Qwen3.6-35B-A3B-NPU2/gated_attention_35B_A3B.safetensors selected_experts