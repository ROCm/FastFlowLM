# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from enum import IntEnum, auto
from typing import Tuple

from gguf.constants import GGMLQuantizationType
import numpy as np

class ModelArch(IntEnum):
    QWEN3VL = auto()
    QWEN3   = auto()
    QWEN35_08B = auto()
    QWEN35_2B = auto()
    QWEN35_4B = auto()
    QWEN35_9B = auto()
    QWEN35_27B = auto()
    QWEN35_OMNI = auto()
    QWEN35_A3B=auto()
    QWEN38_27B = auto()
    QWEN2   = auto()
    QWEN2VL = auto()
    GEMMA3  = auto()
    GEMMA4  = auto()
    GEMMA4_12B  = auto()
    LLAMA   = auto()
    LFM2    = auto()
    LFM3    = auto()
    PHI4    = auto()
    GPT_OSS = auto()
    NANBEIGE = auto()
    HUNYUAN_DENSE = auto()
    MINICPM_V_1B = auto()
    MINICPM_OCR_1B = auto()
    HY_MEMORY_30B_A3B = auto()
    MINICPM_V_35B_A3B = auto()
    DFLASH = auto()
    QWEN38_DFLASH2 = auto()
    QWEN3_ASR = auto()
    INKLING2_LOOP = auto()


ModelArchNames: dict[ModelArch, list[str]] = {
    # Confucius4-R2T2 (Qwen3-ASR): Youdao's GGUF reports architecture qwen3vl and a meaningless general.name, so select it
    # with -f qwen3-asr. It must stay ahead of the "qwen3" entries: -f matching is by prefix.
    ModelArch.QWEN3_ASR: ["qwen3-asr", "qwen3asr", "confucius4-r2t2", "r2t2"],
    # Qwen3.8 reports general.architecture = "qwen35", so it is recognised by
    # general.basename ("Qwen3.8") instead -- keep it ahead of the "qwen3" entry,
    # which "qwen3.8" also prefix-matches.
    ModelArch.QWEN38_27B: ["qwen3.8-27B", "qwen38-27B", "qwen3.8", "qwen38"],
    ModelArch.QWEN35_08B: ["qwen35-0.8B","qwen3.5-0.8B"],
    ModelArch.QWEN35_4B:  ["qwen35-4B","qwen3.5-4B"],    
    ModelArch.QWEN35_9B:  ["qwen35-9B","qwen3.5-9B"],       
    ModelArch.QWEN35_27B: ["qwen35-27B","qwen3.5-27B"],
    ModelArch.QWEN35_2B:  ["qwen35-2B","qwen3.5-2B"],
    ModelArch.QWEN35_OMNI: ["qwen35-omni","qwen3.5-omni"],
    ModelArch.QWEN35_A3B: ["qwen35moe"],
    ModelArch.QWEN3VL: ["qwen3vl", "qwen3-vl"],
    ModelArch.QWEN2VL: ["qwen2.5-Vl"],    
    ModelArch.QWEN2:   ["qwen2"],
    ModelArch.QWEN3:   ["qwen3"],
    ModelArch.GEMMA3:  ["gemma3", "Medgemma", "Gemma-3"],
    ModelArch.GEMMA4_12B:  ["gemma4-12B", "gemma4_12b", "Gemma-4-12B"],
    ModelArch.GEMMA4:  ["gemma4", "Gemma-4"],
    ModelArch.LLAMA:   ["llama"],
    ModelArch.LFM2:    ["lfm2"],
    ModelArch.LFM3:    ["lfm3"],
    ModelArch.PHI4:    ["phi3"],
    ModelArch.GPT_OSS: ["gpt-oss"],
    ModelArch.NANBEIGE: ["nanbeige"],
    ModelArch.HUNYUAN_DENSE: ["hunyuan-dense", "hunyuan"],
    # Hy-Memory-30B-A3B (tencent) reports general.architecture = "hy_v3".
    ModelArch.HY_MEMORY_30B_A3B: ["hy_v3", "hy-memory", "hy_mem"],
    # Also reports general.architecture = "qwen35" (its text tower is MiniCPM-V-4.7's), but
    # unlike MiniCPM-V it does carry a general.basename, "MiniCPM-OCR", which is what matches
    # here. No prefix collision with the "minicpm-v" entries below.
    ModelArch.MINICPM_OCR_1B: ["minicpm-ocr-1.0", "minicpm-ocr"],
    # The llama.cpp GGUF reports general.architecture = "qwen35" and carries no
    # general.basename, so it is recognised by general.name ("MiniCPM-V-4.7-1B").
    # the 35B entry must precede the 1B's: -f matching is by prefix and "minicpm-v" is a prefix of it.
    # (Its GGUFs carry general.architecture = qwen35moe / clip and an unrelated general.name: use -f minicpm-v-35b-a3b.)
    ModelArch.MINICPM_V_35B_A3B: ["minicpm-v-4.7-35b-a3b", "minicpm-v-35b-a3b"],
    ModelArch.MINICPM_V_1B: ["minicpm-v-4.7-1b", "minicpm-v-1b", "minicpm-v"],
    # DFlash speculative-decoding draft (general.architecture = "dflash" in the llama.cpp GGUF).
    ModelArch.DFLASH: ["dflash"],
    # Read from the HF safetensors by convert_dflash2.py, never detected from a GGUF.
    ModelArch.QWEN38_DFLASH2: [],
    ModelArch.INKLING2_LOOP: ["inkling2_loop", "inkling-2", "inkling2"],
}

ModelArchConfigs: dict[ModelArch, str] = {
    ModelArch.QWEN3VL: "qwen3vl.json",
    ModelArch.QWEN2:   "qwen2.json",
    ModelArch.QWEN2VL: "qwen2vl.json",
    ModelArch.QWEN3:   "qwen3.json",
    ModelArch.QWEN35_4B: "qwen3.5_4b.json",
    ModelArch.QWEN35_9B: "qwen3.5_9b.json",
    ModelArch.QWEN35_27B: "qwen3.5_27b.json",
    ModelArch.QWEN35_2B: "qwen3.5_2b.json",
    ModelArch.QWEN35_OMNI: "qwen3.5_omni.json",
    ModelArch.QWEN35_A3B: "qwen3.5_a3b.json",
    ModelArch.QWEN35_08B: "qwen3.5_0.8b.json",
    ModelArch.QWEN38_27B: "qwen3.8_27b.json",
    ModelArch.GEMMA3:  "gemma3.json",
    ModelArch.GEMMA4:  "gemma4.json",
    ModelArch.GEMMA4_12B:  "gemma4_12b.json",
    ModelArch.LLAMA:   "llama.json",
    ModelArch.LFM2:    "lfm2.json",
    ModelArch.LFM3:    "lfm3.json",
    ModelArch.PHI4:    "phi4.json",
    ModelArch.GPT_OSS: "gpt-oss.json",
    ModelArch.NANBEIGE: "nanbeige.json",
    ModelArch.HUNYUAN_DENSE: "hunyuan_dense.json",
    ModelArch.MINICPM_V_1B: "minicpm_v_1b.json",
    ModelArch.MINICPM_OCR_1B: "minicpm_ocr_1b.json",
    ModelArch.HY_MEMORY_30B_A3B: "hy_mem_30b_a3b.json",
    ModelArch.MINICPM_V_35B_A3B: "minicpm_v_35b_a3b.json",
    ModelArch.DFLASH: "dflash.json",
    ModelArch.QWEN38_DFLASH2: "qwen3.8_dflash2.json",
    ModelArch.QWEN3_ASR: "qwen3_asr.json",
    ModelArch.INKLING2_LOOP: "inkling2_loop.json",
}