# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import IntEnum, auto
from safetensors.torch import save_file

import sys
import os
import torch

class model_id(IntEnum):
    LLAMA_3_2_1B = auto()
    LLAMA_3_2_3B = auto()
    QWEN3_0_6B = auto()
    QWEN3_4B = auto()
    PHI4_4B      = auto()
    QWEN2_3B     = auto()
    NANBEIGE_3B  = auto()

MODEL_NAMES: dict[str, model_id] = {
    "LLAMA_3_2_1B": model_id.LLAMA_3_2_1B,
    "LLAMA_3_2_3B": model_id.LLAMA_3_2_3B,
    "QWEN3_0_6B":   model_id.QWEN3_0_6B,
    "QWEN3_4B":   model_id.QWEN3_4B,
    "PHI4_4B":      model_id.PHI4_4B,
    "QWEN2_3B":    model_id.QWEN2_3B,  # alias for QWEN3_0_6B
    "NANBEIGE_3B": model_id.NANBEIGE_3B,
}

MODEL_HF_IDS: dict[model_id, str] = {
    model_id.LLAMA_3_2_1B: "unsloth/Llama-3.2-1B-Instruct-GGUF",
    model_id.LLAMA_3_2_3B: "unsloth/Llama-3.2-3B-Instruct-GGUF",
    model_id.QWEN3_0_6B:    "unsloth/Qwen3-0.6B-GGUF",
    model_id.QWEN3_4B:    "unsloth/Qwen3-4B-GGUF",
    model_id.PHI4_4B:      "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
    model_id.QWEN2_3B:    "mradermacher/Qwen2.5-3B-Instruct-i1-GGUF",
    model_id.NANBEIGE_3B: "Open4bits/Nanbeige4.1-3B-GGUF",
}

MODEL_FILENAMES: dict[model_id, str] = {
    model_id.LLAMA_3_2_1B: "Llama-3.2-1B-Instruct-Q4_1.gguf",
    model_id.LLAMA_3_2_3B: "Llama-3.2-3B-Instruct-Q4_1.gguf",
    model_id.QWEN3_0_6B:   "Qwen3-0.6B-Q4_1.gguf",
    model_id.QWEN3_4B:     "Qwen3-4B-Q4_1.gguf",
    model_id.PHI4_4B:      "microsoft_Phi-4-mini-instruct-Q4_1.gguf",
    model_id.QWEN2_3B:     "Qwen2.5-3B-Instruct.i1-Q4_1.gguf",
    model_id.NANBEIGE_3B:  "nanbeige4.1-3b-Q4_0.gguf",
}



def rope_one(x, cos, sin):
    rot_dim = cos.shape[-1]
    half = rot_dim // 2
    x_rot = x[..., :rot_dim]
    x_rest = x[..., rot_dim:]
    x_swapped = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
    x_out = (x_rot * cos) + (x_swapped * sin)
    return torch.cat([x_out, x_rest], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[:, None]
    sin = sin[:, None]
    return rope_one(q, cos, sin), rope_one(k, cos, sin)

def main(model_name:str, prompt: str = "Hello, tell me about yourself."):
    try:
        hf_model_id = MODEL_HF_IDS[MODEL_NAMES[model_name]]
        hf_model_filename = MODEL_FILENAMES[MODEL_NAMES[model_name]]
    except KeyError:
        raise ValueError(f"Model {model_name} not supported.")

    if hf_model_filename == "":
        hf_model_filename = None
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, gguf_file=hf_model_filename)
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, gguf_file=hf_model_filename)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    templated = tokenizer.apply_chat_template(messages, tokenize=False)
    print(model)
    print("Templated input:", templated)
    inputs = tokenizer(templated, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # since we only want to test the layer, we do embeds and the first layer only
    embed = model.model.embed_tokens(input_ids)
    seq_length = input_ids.shape[1]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    position_embeds = model.model.rotary_emb(embed, position_ids = position_ids)
    outputs = model.model.layers[0](embed, position_embeddings=position_embeds)

    inv_freq = model.model.rotary_emb.inv_freq.detach().cpu().to(torch.float32)

    torch.set_printoptions(sci_mode = True, precision = 4, linewidth = 200, edgeitems = 10)
    input_shape = embed.shape[:-1]
    hidden_shape = (*input_shape, -1, 128)
    query_states = model.model.layers[0].self_attn.q_proj(model.model.layers[0].input_layernorm(embed))
    if hasattr(model.model.layers[0].self_attn, 'q_norm') and model.model.layers[0].self_attn.q_norm is not None:
        query_states = model.model.layers[0].self_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    else:
        query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = model.model.layers[0].self_attn.k_proj(model.model.layers[0].input_layernorm(embed))
    if hasattr(model.model.layers[0].self_attn, 'k_norm') and model.model.layers[0].self_attn.k_norm is not None:
        key_states = model.model.layers[0].self_attn.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
    else:
        key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = model.model.layers[0].self_attn.v_proj(model.model.layers[0].input_layernorm(embed))
    key_states_before_rope = key_states.clone()

    value_states = value_states.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeds
    query_states, key_states = apply_rope(query_states, key_states, cos, sin)

    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    print("Inv freq:\n", inv_freq)
    print("Inputs:\n", embed)
    print("Outputs:\n", outputs)
    tensors = {
        "cos": position_embeds[0].contiguous().to(torch.bfloat16),
        "sin": position_embeds[1].contiguous().to(torch.bfloat16),
        "k_before_rope": key_states_before_rope.contiguous().to(torch.bfloat16),
        "k_cache": key_states.contiguous().to(torch.bfloat16),
        "v_cache": value_states.contiguous().to(torch.bfloat16),
        "inv_freq": inv_freq.contiguous().to(torch.float32),
        "input_ids": input_ids.to(torch.int32),
        "embed": embed.contiguous().to(torch.bfloat16),
        "layer_0_output": outputs[0].contiguous().to(torch.bfloat16),
    }

    save_file(tensors, f"{model_name}_ref.safetensors")


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False, precision=4, linewidth=200, edgeitems=10)
    if len(sys.argv) < 2:
        print("Usage: python ref_gen.py <model_name> [prompt]")
        sys.exit(1)

    model_name = sys.argv[1]
    if len(sys.argv) > 2:
        prompt = sys.argv[2]
    else:
        prompt = "Please explain what are \"KMG\", \"KCMG\", and \"KCMG\" in \"Yes, Minister\" TV series."

    main(model_name, prompt)