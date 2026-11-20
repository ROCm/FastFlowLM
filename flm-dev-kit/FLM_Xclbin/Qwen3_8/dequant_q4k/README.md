<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Qwen3.8-27B Q4_K dequant test

`run.exe` dequantizes every Q4_K tensor of layers 0-4 of `model.q4nx` on the NPU and checks it twice
(details at the top of `host/host.cpp`):

1. **Against a CPU decode of the same bytes.** Always runs. Needs only the q4nx model.
2. **Against the bf16 weights of the Hugging Face checkpoint.** Runs only with `--reference`. This is
   the check that catches a wrong reorder in the converter. Expect cos about 0.997 (Q4_K's own error).

```bash
make bitstream host
./run.exe -m <Qwen3.8-27B-NPU2 folder>                                  # check 1 only
./run.exe -m <Qwen3.8-27B-NPU2 folder> -r <Qwen/Qwen3.8-27B HF folder>  # checks 1 and 2
```

## How `--reference` finds a tensor

`host/hf_checkpoint.hpp` takes either one `.safetensors` file or an HF checkpoint folder. For a folder
it reads `model.safetensors.index.json`, which maps every tensor name to the shard file that holds it,
and opens a shard only when one of its tensors is first needed. Tensor names are the HF ones
(`model.language_model.layers.<L>.<name>`). Layers 0-4 live in the first two shards, so a partial
download is enough:

```bash
hf download Qwen/Qwen3.8-27B model.safetensors.index.json \
    model-00001-of-00018.safetensors model-00002-of-00018.safetensors --local-dir <folder>
```
