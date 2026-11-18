<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# FLM Q4NX Converter

A utility for converting GGUF model files into the Q4NX format. This tool supports converting language, vision, and audio model weights.

## Supported Models
Based on the configuration, the converter supports several model architectures, including:
- Gemma 3
- GPT-OSS
- Hunyuan (dense, e.g. Hy-MT2-1.8B)
- LFM 2
- LLaMA
- Phi-4
- Qwen 2 / Qwen 2.5
- Qwen 2 VL
- Qwen 3
- Qwen 3 VL
- Qwen 3.5
- DFlash speculative-decoding drafts (`-f dflash`, e.g. the Nemotron-3.5-Bolt-E4B draft)

### Weight Type Support
- `language`: supported for model families in `configs/`
- `vision`: supported for vision-capable architectures (for example, Gemma 4 and Qwen3-VL)
- `audio`: currently supported for **Gemma 4** (`-t audio`)

## Setup

The project includes a setup script that automatically creates a Python virtual environment and installs the pinned dependencies from `requirements.txt`.

`requirements.txt` pins every package the converter and its check scripts import, at exact versions:
- `torch` is the CPU build (`+cpu`, from the PyTorch CPU wheel index named in the file), so no CUDA packages are installed; the converter does not use a GPU.
- `gguf` is llama.cpp's `gguf-py` as released on PyPI, instead of a clone of the latest llama.cpp.
- `transformers` is a released version instead of the latest git code.
- `mpmath` and `huggingface-hub` would come in anyway as dependencies; they are pinned because `q4nx/gguf_tensor.py` imports `mpmath` directly and `huggingface-hub` provides the `hf download` command.

To move a pin, change it in `requirements.txt`, rebuild the venv, and re-run the end-to-end test in [Verifying a Qwen3.5-family conversion](#verifying-a-qwen35-family-conversion-qwen35--36--38) (`make_tiny_qwen35_hf.py` -> `convert.py` -> `check_qwen35_family.py`). Also check that `convert.py` writes the same `model.q4nx` bytes as before (`cmp`).

1. Make the setup script executable (if it isn't already) and run it:
   ```bash
   chmod +x setup_venv.sh
   ./setup_venv.sh
   ```

2. Activate the environment using the provided activation script. **Note:** Always use `activate.sh` instead of directly sourcing the venv, as it also sets up necessary environment variables like `HF_HOME` and CUDA paths.
   ```bash
   source activate.sh
   ```

## Usage

The main entry point for conversion is `convert.py`.

### Help
```bash
python convert.py -h
```

### Basic Syntax
```bash
python convert.py [input_file] [output_folder] [-t TYPE]
```

### Arguments
- **`input_file`**  
  Also available as **`-i`** or **`--input`**.  
  Specifies the path to the input `.gguf` file that you want to convert.

- **`output_folder`**  
  Also available as **`-o`** or **`--output`**.  
   Specifies the destination folder for the converted output. The converter writes the generated tensors as **`model.q4nx`** into this folder.  
  If not provided, the converter uses the same directory as the input file.

- **`-t`, `--type`**  
  Specifies which weights to convert.  
  Available options are:
  - **`language`** — converts the language model weights
   - **`vision`** — converts the vision model weights
   - **`audio`** — converts audio model weights (currently Gemma 4)  
  If this option is not specified, the default is **`language`**.

- **`-f`, `--force`**  
  Forces the converter to use a specific model architecture instead of detecting it automatically from the GGUF metadata.  
  This can be useful when automatic detection is incorrect or when you want to explicitly select an architecture such as **`qwen2`**, **`llama`**, or **`gemma3`**.  
  Leave this option out if you want the converter to detect the architecture automatically.

### Examples

**1. Convert a language model (positional arguments):**
```bash
python convert.py model.gguf output_folder
```

**2. Convert a language model (flag arguments):**
```bash
python convert.py -i unsloth_gpt-oss-20b-Q4_0.gguf -o unsloth-gotoss20b-q40
```

**3. Convert a vision model:**
```bash
python convert.py -i qwen3vl-4b-mmproj-BF16.gguf -o unsloth-qwen3vl-vision -t vision
```

The vision tower is shipped as bf16, so convert it from a BF16 mmproj GGUF (or the checkpoint's own weights). A source that stores the
tower as F16 or quantised is rounded to bf16 and can differ from the checkpoint: the Qwen3.5 family and MiniCPM-V refuse it, other
models print a warning; `--require-bf16-vision` refuses for every model, `--allow-f16-vision` accepts it (the rounded tensor count is
recorded in the file metadata). For MiniCPM-V 1B, `--verify-hf model.safetensors` compares every written tensor with the checkpoint and
fails unless 0 elements differ. `python check_vision_source_gate.py` checks the guard itself.

**4. Force a specific model architecture:**
```bash
python convert.py -i model.gguf -o output_folder -f qwen2
```
This is useful when the GGUF file metadata doesn't correctly identify the architecture or when you want to override the automatic detection.

**5. Convert an audio model (Gemma 4):**
```bash
python convert.py -i gemma4-2b-mmproj.gguf -o unsloth-gemma4-2b-audio -t audio -f gemma4
```

**6. Convert from a different quantization format (e.g. Q4_K_M):**
```bash
python convert.py -i model.gguf -o output_folder
```
The converter automatically dequantizes non-Q4_0/Q4_1 weights and re-quantizes them into Q4NX. See the [Converting from Other Quantization Formats](#converting-from-other-quantization-formats) section for details.

**7. Convert a DFlash draft model (speculative decoding):**
```bash
python convert.py -i Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf -o dflash_pkg -f dflash
python check_dflash_q4nx.py Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf dflash_pkg/dflash_weight.q4nx   # round trip of every tensor
```
Writes `dflash_weight.q4nx`: the draft's matmul weights as Q4_1 except the tensors listed in `configs/dflash.json` ->
`dflash_config.bf16_tensors` (today `model.fc.weight`, the projection of the target taps: Q4_1 there cost 1.7% acceptance
length), which stay BF16 in the 32-row block layout; norms are BF16 vectors. The draft has no embedding or lm_head (it uses
the target's). The package of `nemotron3.5-bolt-dflash:e4b` adds a `config.json` (draft shape, `target_layer_ids`,
`block_size`, `mask_token_id`, `base_model`); see `FLM_DLL/detail/gemma4e_dflash/README.md`.

## Converting from Other Quantization Formats

The converter supports GGUF models that use quantization formats other than Q4_0 or Q4_1, such as **Q4_K_M**, **Q8_0**, **Q5_K_M**, and others. When the converter encounters weights in a non-native format, it automatically dequantizes them back to floating point and then re-quantizes them into the Q4NX target format.

This means you are not limited to sourcing GGUF files that are already in Q4_0 or Q4_1 — you can use a wider variety of community-quantized models as input.

The conversion process works as follows:

1. The converter reads the quantization type of each tensor from the GGUF file.
2. If the tensor is not already in the expected format (Q4_0 or Q4_1, depending on the model config), it is dequantized to FP32.
3. The dequantized weights are then re-quantized into the target format required by FLM.

This process is fully automatic and requires no additional flags beyond **`-f`** if the architecture cannot be auto-detected.

> **Note:** Dequantizing from a lossy format and re-quantizing introduces additional quantization error compared to converting directly from full-precision weights. For best quality, prefer starting from BF16, FP16, or Q8_0 sources when available.

When converting community-quantized models, the GGUF metadata may not always match the architecture names expected by the converter. Use the **`-f`** flag to explicitly specify the model architecture in those cases.

### Example

Convert a [Q4_K_M quantized model](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive) from the community:

```bash
python convert.py -i Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf -o unsloth-qwen3_59b_uncensored -f qwen3.5-9B
```

Here, **`-f qwen3.5-9B`** tells the converter to use the Qwen 3.5 9B architecture, since the GGUF metadata from the community model may not be detected correctly.


## Step-by-Step Converting and Registering Custom FLM Models

This guide walks you through how to convert a supported GGUF model into FLM's Q4NX format, then either:

- **replace an existing installed model**, or
- **register your converted weights as a separate custom model**

The examples below use **`qwen3vl-it:4b`**, but the same overall workflow applies to other supported models.

---

### Before You Begin

Make sure you have:

- **FLM** installed on your system
- a **supported GGUF model file**
- the **FLM converter**
- the correct **conversion type** for your model family

#### Important compatibility note

Different model families expect different quantization formats. The expected format for each model family is defined in the configuration files under `configs/`.

For example:

- **`lfm2`** uses **`q4_0`**
- **`qwen3vl`** uses **`q4_1`**

If the source GGUF file uses a different quantization format (such as Q4_K_M or Q8_0), the converter will automatically dequantize and re-quantize the weights into the expected format. See [Converting from Other Quantization Formats](#converting-from-other-quantization-formats) for details.

---

### Option 1: Replace an Existing Installed Model

Choose this option if you want to **swap the original FLM model weights** with your own converted weights while keeping the same model name and launch command.

#### Step 1: Download a compatible GGUF model

Download a supported GGUF model from Hugging Face or another trusted source.

Before continuing, verify that:

- the model is compatible with the converter
- the quantization matches the expected format for the model family
- you know which FLM model you plan to replace

For the `qwen3vl` family, make sure the GGUF model matches **Q4_1** expectations.

---

#### Step 2: Convert the GGUF model to Q4NX

Run the converter using the instructions from the **Usage** section of the converter project.

Provide:

- the input GGUF file
- an output folder
- the correct conversion type

The converter will generate one or more `.q4nx` files, depending on the model architecture.

For vision-language models such as Qwen3-VL, this may include files such as:

- `model.q4nx`
- `vision_weights.q4nx`

> Keep the output folder handy. You will need these converted files in the next step.

---

#### Step 3: Locate the installed FLM model directory

Find the installed FLM model directory for the model you want to replace.

**Default model paths**

**Windows**
```text
C:\Users\<username>\Documents\flm\models\Qwen3-VL-4B-Instruct-NPU2
```

**Linux**
```text
/home/<username>/.config/flm/models/Qwen3-VL-4B-Instruct-NPU2
```

> Replace `<username>` with your actual system username.

---

#### Step 4: Replace the existing Q4NX file or files

Copy the newly converted files from your output folder into the installed FLM model directory.

Replace the corresponding existing file(s), such as:

- `model.q4nx`
- `vision_weights.q4nx`

Be careful to preserve the original filenames expected by FLM.

**Recommended best practice**

Before replacing anything, create a backup of the original model folder or at least the original `.q4nx` files. This makes it easy to restore the original model later if needed.

---

#### Step 5: Start the model

Once the replacement files are in place, start the model using either of the following commands:

```bash
flm run qwen3vl:4b
```

or

```bash
flm serve qwen3vl:4b
```

If the conversion and replacement were successful, FLM should now load your custom-converted weights under the original model name.

---

### Option 2: Add a Custom Model Configuration

Choose this option if you want to **keep the original FLM model intact** and register your converted model as a **separate custom model**.

This is the safer and more flexible option, especially if you want to compare the original model with your custom version.

Before starting, complete:

- **Step 1: Download a compatible GGUF model**
- **Step 2: Convert the GGUF model to Q4NX**

Then continue below.

---

#### Step 1: Open the main FLM models directory

Locate the main FLM models directory.

**Default model paths**

**Windows**
```text
C:\Users\<username>\Documents\flm\models\
```

**Linux**
```text
/home/<username>/.config/flm/models/
```

---

#### Step 2: Duplicate an existing model folder

Copy the existing model folder:

```text
Qwen3-VL-4B-Instruct-NPU2
```

Rename the copied folder to something new, for example:

```text
Qwen3-VL-4B-Custom
```

This duplicated folder becomes the base for your custom model.

---

#### Step 3: Replace the Q4NX files in the copied folder

Inside your newly copied model folder, replace the relevant `.q4nx` files with the files generated by the converter.

Typical files include:

- `model.q4nx`
- `vision_weights.q4nx`

Make sure the filenames match what FLM expects for that model.

---

#### Step 4: Edit `model_list.json`

Open your FLM installation directory and locate `model_list.json`.

**Default installation paths**

**Windows**
```text
C:\Program Files\flm
```

**Linux**
```text
/opt/fastflowlm/share/flm
```

You now have two ways to register your custom model:

- **Standalone model**  

- **Submodel under an existing family**  

Both approaches work. Choose the one that best fits your organization and naming preference.

**Option 1: Add a standalone custom model entry**

Add a new entry under `models`:

```json
"qwen3vl-it-custom": {
   "4b": {
      "name": "Qwen3-VL-4B-Custom",
      "url": "https://huggingface.co/FastFlowLM/Qwen3-VL-4B-Custom/resolve/v0.9.22-faster-q4-1",
      "file_url": "https://huggingface.co/api/models/FastFlowLM/Qwen3-VL-4B-Custom/tree/v0.9.22-faster-q4-1",
      "size": 4000000000,
      "flm_min_version": "0.9.22",
      "files": [
         "config.json",
         "model.q4nx",
         "tokenizer.json",
         "tokenizer_config.json",
         "vision_weights.q4nx"
      ],
      "vlm": true,
      "default_context_length": 32768,
      "details": {
         "format": "NPU2",
         "family": "qwen3vl",
         "think": false,
         "parameter_size": "4B",
         "quantization_level": "Q4_1"
      },
      "label": [
         "vision"
      ],
      "footprint": 3.9
   }
}
   ```

---

**Option B: Add a custom submodel under `qwen3vl-it`**

Add a new sub-entry under the existing `qwen3vl-it` model family.

```json
"qwen3vl-it": {
   "4b-custom": {
      "name": "Qwen3-VL-4B-Custom",
      "url": "https://huggingface.co/FastFlowLM/Qwen3-VL-4B-Custom/resolve/v0.9.22-faster-q4-1",
      "file_url": "https://huggingface.co/api/models/FastFlowLM/Qwen3-VL-4B-Custom/tree/v0.9.22-faster-q4-1",
      "size": 4000000000,
      "flm_min_version": "0.9.22",
      "files": [
         "config.json",
         "model.q4nx",
         "tokenizer.json",
         "tokenizer_config.json",
         "vision_weights.q4nx"
      ],
      "vlm": true,
      "default_context_length": 32768,
      "details": {
         "format": "NPU2",
         "family": "qwen3vl",
         "think": false,
         "parameter_size": "4B",
         "quantization_level": "Q4_1"
      },
      "label": [
         "vision"
      ],
      "footprint": 3.9
   }
}
```

**Note on `url` and `file_url`**

The `url` and `file_url` fields only matter when FLM needs to fetch the model from a remote source, for example when you run:

```bash
flm pull <custom-model-name>
```

If you want that workflow to work, make sure:

- the model files are already uploaded and reachable online
- the `files` list matches what is actually hosted

In this guide, you have already copied all required model files into the local model directory manually, FLM can load them directly from disk. In that case, `url` and `file_url` can be dummy placeholder values and do not need to point to real hosted files.


---

#### Step 5: Copy the matching `xclbins/` folder

In the FLM installation directory, open the `xclbins/` folder.

If you created a **standalone custom model**, copy:

```text
Qwen3-VL-4B-Instruct-NPU2
```

Then rename the copied folder to:

```text
Qwen3-VL-4B-Custom
```

This folder name should match the `"name"` value used in your custom model entry.

> If you are using the **submodel style**, you can usually skip this step.

---

#### Step 6: Confirm that FLM recognizes the custom model

Run:

```bash
flm list
```

You should see one of the following in the output:

- `qwen3vl-it-custom:4b` for a **standalone model**
- `qwen3vl-it:4b-custom` for a **submodel** of qwen3vl-it family

If the new model appears in the list, FLM has recognized your configuration successfully.

---

#### Step 7: Start the custom model

Use the command that matches the registration style you chose.

**Standalone model**

```bash
flm run qwen3vl-it-custom:4b
```

or

```bash
flm serve qwen3vl-it-custom:4b
```

**Submodel**

```bash
flm run qwen3vl-it:4b-custom
```

or

```bash
flm serve qwen3vl-it:4b-custom
```



## Adding a New Model Architecture

The sections above cover *using* the converter. This one covers *extending* it:
what to touch when a model family the converter has never seen needs support.

`FLM_Xclbin/generic_decoding_layer` is the reference implementation of the
consumer side. When this document and that kernel disagree, the kernel wins ---
its `kernels/model_spec.h`, `kernels/aie_array_layout.h` and
`models/all_models.h` are the authoritative description of what the hardware
actually reads.

### The five things to touch

1. **`q4nx/constants.py`** --- add a `ModelArch` enum member, one or more
   detection strings in `ModelArchNames` (matched against the GGUF's
   `general.architecture` / name metadata, and against the `-f` flag), and the
   config filename in `ModelArchConfigs`.
2. **`configs/<arch>.json`** --- the layout and the tensor name map (below).
3. **`q4nx/models/<arch>.py`** --- a converter subclass.
4. **`q4nx/models/__init__.py`** --- import it and add it to `__all__`.
   Registration happens as an import side effect, so a converter that is never
   imported is never found.
5. **A kernel-side model header** if the target decoding layer needs one ---
   see `FLM_Xclbin/generic_decoding_layer/models/`.

### The config file

```jsonc
{
    "default_tensor_type": "Q4_1",     // Q4_0 | Q4_1 | Q8_0 | Q4_K
    "q4nx_config": {
        "row_block_size": 32,          // must match Q4NX_ROW_BLOCK_SIZE
        "col_block_size": 256,         // must match Q4NX_COL_BLOCK_SIZE
        "parallel_size": 16,           // must match the kernel's PARALLEL_ROWS
        "keep_block_in_2D": false,     // keep (p, q, block) vs flatten to (p*q, block)
        "q4_0_keep_min_plane": false   // Q4_0 only; see "Two shapes of Q4_0"
    },
    "name_map": {
        "blk.{bid}.attn_q.weight": "model.layers.{bid}.self_attn.q_proj.weight",
        "output.weight": {
            "name": "lm_head.weight",
            "default_tensor_type": "Q8_0"   // per-tensor override
        }
    }
}
```

`{bid}` is the block index and is substituted on both sides. A `name_map` entry
is either a plain string or an object that also overrides the quant type for
that one tensor.

### The converter subclass

```python
from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize
import torch

class MyArch(__Q4NX_Converter, model_arch=ModelArch.MY_ARCH):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        for key, gguf_tensor in self.gguf_tensors.items():
            if "token_embd.weight" in gguf_tensor.name:      # embeddings stay bf16
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = \
                    torch.from_numpy(w).contiguous().to(torch.bfloat16)
                continue
            target = self.get_used_quantization_type(gguf_tensor)
            unpacked = gguf_tensor.unpack(target)
            self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = \
                self._pack(*unpacked, tensor_type=target)
        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
```

`__init_subclass__(model_arch=...)` performs the registration, so the keyword
argument is required --- omitting it is a `TypeError`, not a silent no-op.

Copy the nearest existing converter rather than starting from this skeleton:
most of the real work in any of them is model-specific tensor surgery (fused
QKV splits, gate/up interleaving, MoE expert stacking, vision towers), and
`q4nx/models/qwen3.py` is the smallest complete example.

### RoPE channel order: does the architecture need a q/k reorder?

`q4nx/models/llama.py` reorders the rows of `q_proj` and `k_proj` before
packing, and no other converter does:

```python
DH = self.gguf_reader.fields["llama.rope.dimension_count"].contents()
d, m, qw = unpacked
qw = rearrange(qw, '(g p q) c -> (g q p) c', p = DH // 2, q = 2).contiguous()
```

That is not a quirk of the LLaMA weights --- it undoes a permutation that
llama.cpp's HF -> GGUF converter applied on the way in. GGML has two RoPE
conventions:

| llama.cpp rope type | pairs channels | what the HF converter does |
| --- | --- | --- |
| `LLAMA_ROPE_TYPE_NORM` | `(2i, 2i+1)` --- adjacent | permutes `q_proj` / `k_proj` rows |
| `LLAMA_ROPE_TYPE_NEOX` | `(i, i + DH/2)` --- half offset | ships HF row order unchanged |

The FLM kernels rotate `(i, i + DH/2)`, i.e. HF's `rotate_half`. So a NEOX-arch
GGUF already has the rows in the order the kernel wants, while a NORM-arch GGUF
is interleaved and has to be un-permuted --- `LlamaModel.permute` in llama.cpp
is exactly the inverse of the `rearrange` above.

**How to check a new architecture** --- two greps in a llama.cpp checkout:

1. **Rope type.** `llama_model_rope_type()` in `src/llama-model.cpp`. Find the
   `case LLM_ARCH_<YOURS>:` and see which `return` it falls through to. Only
   the `NORM` group needs a reorder.
2. **Tensor surgery.** The arch's model class in `conversion/<arch>.py` (older
   trees keep every class in `convert_hf_to_gguf.py`). Grep its
   `modify_tensors` for `permute`; absent means the rows were never touched.

The two must agree --- the rope type is the reason, `modify_tensors` is the
proof. If the checkout is missing the file on disk, it is still in the object
store: `git show HEAD:conversion/hunyuan.py`.

Checked so far:

- `LLM_ARCH_LLAMA` is `NORM` and `LlamaModel.modify_tensors` calls `permute`
  --- hence the reorder in `llama.py`.
- `LLM_ARCH_QWEN3` and `LLM_ARCH_HUNYUAN_DENSE` are `NEOX`, and
  `HunYuanModel.modify_tensors` (`conversion/hunyuan.py`) only drops a tied
  `lm_head.weight`. No reorder, so `qwen3.py` and `hunyuan_dense.py` pack the
  rows as they come.

Getting this wrong does not crash anything: RoPE is simply applied to the wrong
channel pairs, and the model emits fluent but wrong text --- more visibly the
longer the context.

### Two entry points: `_pack` vs `_pack_q4nx`

- **`self._pack(*unpacked, tensor_type=t)`** --- preferred. Dispatches on the
  quant type and is the only path that reaches the Q4_K packer.
- **`self._pack_q4nx(*unpacked)`** --- legacy. Always emits the 5 bpw
  `q41_block_t`, whatever the source type. Most existing converters still call
  it, which is fine as long as they never see Q4_K (see Known Issues).

### Supported quantization formats

One packed struct per `row_block_size x col_block_size` (32x256) chunk. The
`static_assert`s in `generic_decoding_layer/kernels/model_spec.h` pin every
size below; a mismatch is a silently misread weight, not a fault.

| GGUF type | Packed block    | Layout                                                     | bpw   | Bytes/block |
|-----------|-----------------|------------------------------------------------------------|-------|-------------|
| Q4_0      | `q40_block_t`   | bf16 scales + signed `int4`, no min plane                    | 4.5   | 4608        |
| Q4_0      | `q41_block_t`   | legacy: 5 bpw struct, all-zero min plane, signed nibbles     | 5     | 5120        |
| Q4_1      | `q41_block_t`   | bf16 scales + bf16 mins + `uint4`, `w = s*q + m`             | 5     | 5120        |
| Q4_K      | `q4k_block_t`   | uint8 s/m per group + `uint4` + bf16 S/M per 256-superblock  | 4.625 | 4736        |
| Q8_0      | q8nx block      | bf16 scales + `int8`                                         | 8     | 8704        |

Scale and min planes are **always** group major over the full row block
(`scales[g * ROW_BLOCK + r]`), never strided --- the kernel walks them with
`it_scale = w->scales + row; it_scale += Q4NX_ROW_BLOCK_SIZE`. Only the quant
plane is strided.

Q4_K is **read-only**: ggml has no Q4_K encoder, so `quantize()` raises
`NotImplementedError` and `get_used_quantization_type` substitutes Q4_1 when
Q4_K is requested as an *encode* target. A Q4_K GGUF read straight through is
fine; dequantizing something else *into* Q4_K is not possible.

### Two shapes of Q4_0

The same GGUF type ships in two different packed blocks, and the config picks
between them:

- `q4_0_keep_min_plane: false` (default) --- the pure 4.5 bpw `q40_block_t`
  with no min plane, which is what `generic_decoding_layer`'s `__Q4_0__` reads.
  `QUANT_HAS_MIN` is 0, so the kernel has no min accumulator at all.
- `q4_0_keep_min_plane: true` --- the legacy Q4_0-in-Q4_1 block: the 5 bpw
  `q41_block_t` struct carrying an all-zero min plane and **signed** nibbles.
  This is what the older per-model kernels declare under `#ifdef Q4_0` (see
  `FLM_Xclbin/Qwen2_5/decoding_3b/kernels/model_spec.h`). `configs/qwen2.json`
  and `configs/lfm2.json` use it.

The flag only affects `_pack`. `_pack_q4nx` called directly always produces the
legacy block.

### `parallel_size` is the kernel's `PARALLEL_ROWS`

This is the single easiest thing to get wrong, because a mismatch produces
plausible-looking garbage rather than an error.

The quant plane of a block is packed as `parallel_size`-row planes, each laid
out column major: nibble `(g, c, r)` is row `g * parallel_size + r` of column
`c`. So:

- `parallel_size == row_block_size` (32) --- plain column major, one pass.
  `Q4NX_ROW_STRIDE` is the whole row block.
- `parallel_size == 16` --- two 16-row planes over one shared quant cursor, and
  the cores take `ROW_PASSES == 2` passes.

`aie_array_layout.h` derives `PARALLEL_ROWS` the same way, and it is *not*
uniform across quant types:

```c
#if QUANT_TYPE == __Q4_K__
constexpr int PARALLEL_ROWS = Q4NX_ROW_BLOCK_SIZE;   // Q4_K is never strided
#else
constexpr int PARALLEL_ROWS = Q4NX_ROW_STRIDE;       // 16 for Q4_0 / Q4_1
#endif
```

Hence `_pack_q4k` asserts `parallel_size == row_block_size`. Q8_0 *is* strided
--- `FLM_DLL/detail/qwen3_5_omni/qwen3_5_omni_thinker/embedding_q8_0.hpp`
documents its reader as matching `_pack_q8nx` at `PARALLEL_SIZE=16`, while
Qwen3_5 / Qwen3_6's `q8_core.h` `static_assert`s `PARALLEL_ROWS ==
QXNX_ROW_BLOCK_SIZE`.

Current settings: `qwen3.5_*` use 32, everything else uses 16.

### Verifying a new architecture

1. **Block size.** Packed bytes per block must equal the table above exactly.
   `FLM_DLL/include/weight_desc.hpp`'s `get_quantization_byte_size` is the
   host-side copy of the same numbers.
2. **Round trip.** Decode a packed block in the order the kernel walks it and
   compare against `gguf.dequantize`. Expect a relative error around `2e-3`
   from bf16 scale rounding --- that is the floor, not a bug. Anything near
   `2.0` is a sign convention error; anything near `1.0` is a layout error.
3. **Regression.** Converting an already-supported model must stay byte-for-byte
   identical:
   ```bash
   python convert.py -i gguf_files/qwen3.5-0.8b-q4k.gguf -o /tmp/after
   cmp /tmp/before/model.q4nx /tmp/after/model.q4nx
   ```
4. **Both strides.** Exercise `parallel_size` 16 *and* 32. A packer bug that
   only shows at stride 16 is invisible if every test config uses 32.


## Project Structure
- `convert.py`: The main CLI script for running conversions.
- `setup_venv.sh`: Initializes the Python virtual environment and installs dependencies.
- `activate.sh`: Activates the virtual environment and sets up environment variables.
- `q4nx/`: Core package containing the conversion logic, gguf tensor parsing, and model-specific implementations.
- `configs/`: JSON configuration files for supported model architectures.


## Verifying a Qwen3.5-family conversion (Qwen3.5 / 3.6 / 3.8)

The Qwen3.5 architecture (gated delta-net + gated full attention, optional MTP head, optional MoE) is shared by
Qwen3.5 0.8B/2B/4B/9B/27B, Qwen3.6 (27B, 35B-A3B) and Qwen3.8-27B.  `convert.py` picks the family from the GGUF
geometry (`-f` is only needed for fine-tunes with a custom name).  To check a conversion against the original
HF checkpoint -- independent of how llama.cpp laid the GGUF out:

```bash
python convert.py -i Qwen3.5-2B-Q4_K.gguf -o out/                      # language weights
python check_qwen35_family.py --hf <HF dir> --q4nx out/model.q4nx --layers all
python convert.py -i mmproj-Qwen3.5-2B-BF16.gguf -o out-vision/ -t vision
python check_qwen35_vision.py --hf <HF dir> --q4nx out-vision/model.q4nx
python check_arch_detect.py qwen35-2B=Qwen3.5-2B-Q4_K.gguf             # auto-detection
python check_family_regression.py --base-ref <ref> --gguf X.gguf -f <family>   # byte-identical vs a ref
```

`check_qwen35_family.py` decodes the q4nx blocks and compares every tensor with HF (head order, q/gate layout,
norm convention, alpha/beta padding, tied/untied head, MoE experts, MTP placement, no leaked decoder layers).
`make_tiny_qwen35_hf.py` writes a small random-weight HF checkpoint with a chosen geometry (delta-net ratio 1/2/3,
untied head, MTP block, irregular layer pattern, `--moe`), so the 27B / Qwen3.8 / A3B paths can be exercised end to
end (HF -> llama.cpp `convert_hf_to_gguf.py` -> `llama-quantize` -> `convert.py` -> checker) without the real weights.

Notes: a GGUF that carries an MTP block (`nextn_predict_layers` > 0) has it dropped by the Qwen3.5/3.6 converters
(only the Qwen3.8 converter keeps it, under `model.mtp.*`).  The Qwen3.5/3.6 vision output is written as `model.q4nx`;
convert it into its own folder (the engine reads it as `vision_weight.q4nx`).

## Known Issues
- The converter outputs either Q4_0 or Q4_1 quantization format based on the setting in config files for each model. Input GGUF files can use other quantization formats — the converter will dequantize and re-quantize automatically, though this may introduce additional quantization error.
- For GPT-OSS:20B models, the converter currently uses the original `model.embed_tokens.weight` from the safetensors from OpenAI due to issues with Q4_1 quantization (from our experience, Q4_1 quantization messes up the quantization of the embedding layer for this model). 
  - **Workaround:** Place the [`model-00001-of-00001.safetensors`](https://huggingface.co/openai/gpt-oss-20b/tree/main) file in the root directory of this project before running the conversion for GPT-OSS. If the file is not found, the converter will print a warning and skip replacing the embedding weights. 
