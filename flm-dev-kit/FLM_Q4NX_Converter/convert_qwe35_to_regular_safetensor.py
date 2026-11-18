# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import torch
from einops import rearrange
from safetensors.torch import load_file, save_file
from concurrent.futures import ThreadPoolExecutor, as_completed


INPUT_Q4NX_FILE_PATH = "/path/to/FastFlowLM/FLM_Q4NX_Converter/output/unsloth-qwen3_6_a3bq4k_s/model.safetensors"
OUPUT_SAFETENSOR_PATH = "/path/to/FastFlowLM/FLM_Q4NX_Converter/merged_safetensors/model.safetensors"

# Block sizes matching qwen3.5_a3b.json config
ROW_BLOCK_SIZE = 32
COL_BLOCK_SIZE = 256
PARALLEL_SIZE = 16
Q4_GROUP_SIZE = 32
Q8_GROUP_SIZE = 32



NUM_EXPERT = 256
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512



def dequant_q4_1(packed: torch.Tensor, orig_rows: int, orig_cols: int) -> torch.Tensor:
    """
    Undo _pack_q4nx (keep_block_in_2D=True) and dequantize Q4_1 to bf16.

    Packed layout per block [p, q] along last dim:
        [ d_bf16 (c*r*2 bytes) | m_bf16 (c*r*2 bytes) | qw (col_block_size//2 * row_block_size / 2 bytes) ]
    where:
        c = COL_BLOCK_SIZE // Q4_GROUP_SIZE   (number of scale groups per column block)
        r = ROW_BLOCK_SIZE
        p = rows // ROW_BLOCK_SIZE
        q = padded_cols // COL_BLOCK_SIZE
    """
    sc = COL_BLOCK_SIZE // Q4_GROUP_SIZE  # 8  — number of scale groups per col block
    r = ROW_BLOCK_SIZE                    # 32
    g = ROW_BLOCK_SIZE // PARALLEL_SIZE   # 2  (groups of rows per parallel slice)
    NUM_int4_in_byte = 2
    r_half = PARALLEL_SIZE // NUM_int4_in_byte  # 8 — packed rows per parallel group

    # Bytes per block for each section
    d_bytes = sc * r * 2          # bf16 = 2 bytes each
    m_bytes = sc * r * 2
    # qw packed layout: p q (g * COL_BLOCK_SIZE * r_half)
    # from: 'p q g c r b -> p q g c r b[0]' then 'p q g c r -> p q (g c r)'
    # where c=COL_BLOCK_SIZE, r=r_half
    qw_bytes_per_block = g * COL_BLOCK_SIZE * r_half

    assert packed.shape[-1] == d_bytes + m_bytes + qw_bytes_per_block, \
        f"Unexpected packed block size {packed.shape[-1]}, expected {d_bytes + m_bytes + qw_bytes_per_block}"

    p, q = packed.shape[0], packed.shape[1]

    # Split the last dimension into d, m, qw sections
    d_raw = packed[..., :d_bytes].view(torch.int8)
    m_raw = packed[..., d_bytes:d_bytes + m_bytes].view(torch.int8)
    qw_raw = packed[..., d_bytes + m_bytes:].view(torch.int8)

    # Reinterpret as bf16 and restore column-major -> row-major scale layout
    # Packed: p q (sc r) -> unpack to p q sc r -> back to (p r) (q sc) i.e. row-major
    d = d_raw.view(torch.bfloat16)        # p q (sc*r)
    d = d.reshape(p, q, sc, r)            # p q sc r
    d = rearrange(d, 'p q sc r -> (p r) (q sc)')   # padded_rows x (padded_cols // Q4_GROUP_SIZE)

    m = m_raw.view(torch.bfloat16)
    m = m.reshape(p, q, sc, r)
    m = rearrange(m, 'p q sc r -> (p r) (q sc)')

    # Unpack qw nibbles
    # Packed layout: 'p q g (r b) c -> p q g c r b' then take [..0] -> 'p q g c r_half'
    # then flatten: 'p q g c r -> p q (g c r)' where c=COL_BLOCK_SIZE, r=r_half
    qw_packed = qw_raw.reshape(p, q, g, COL_BLOCK_SIZE, r_half)  # p q g col r_half

    # Each byte: low nibble = first element, high nibble = second element
    low = (qw_packed & 0x0F).to(torch.float32)    # p q g col r_half
    high = ((qw_packed >> 4) & 0x0F).to(torch.float32)

    # Undo: 'p q g (r b) c -> p q g c r b' with b=2, r=r_half
    # stack low/high back into b dim: p q g col r_half 2 -> p q g col (r_half*2=PARALLEL_SIZE)
    qw_int = torch.stack([low, high], dim=-1)          # p q g col r_half 2
    qw_int = qw_int.reshape(p, q, g, COL_BLOCK_SIZE, PARALLEL_SIZE)  # p q g col PARALLEL_SIZE

    # Undo 'p q g (r b) c -> p q g c r b': col was originally the c dim, PARALLEL_SIZE was r dim
    # Restore to p q g PARALLEL_SIZE col, then flatten rows: p q (g PARALLEL_SIZE) col = p q ROW_BLOCK_SIZE COL_BLOCK_SIZE
    qw_int = rearrange(qw_int, 'p q g col par -> p q (g par) col')  # p q ROW_BLOCK_SIZE COL_BLOCK_SIZE

    # Dequantize: w[row_in_block, col] = d[row, col//Q4_GROUP_SIZE] * qw + m[row, col//Q4_GROUP_SIZE]
    # d and m shape: (p*r) x (q*c) = padded_rows x (padded_cols // Q4_GROUP_SIZE)
    # qw_int shape: p q ROW_BLOCK_SIZE COL_BLOCK_SIZE
    # Reshape qw to: padded_rows x padded_cols
    qw_2d = rearrange(qw_int, 'p q r c -> (p r) (q c)')  # padded_rows x padded_cols

    # Expand d and m to full column dimension for dequant
    # d[row, g] covers Q4_GROUP_SIZE columns starting at g*Q4_GROUP_SIZE
    d_exp = d.repeat_interleave(Q4_GROUP_SIZE, dim=1)   # padded_rows x padded_cols
    m_exp = m.repeat_interleave(Q4_GROUP_SIZE, dim=1)

    result = d_exp * qw_2d + m_exp   # bf16 arithmetic
    result = result.to(torch.bfloat16)

    # Trim padding
    return result[:orig_rows, :orig_cols].contiguous()


def dequant_q8_0(packed: torch.Tensor, orig_rows: int, orig_cols: int) -> torch.Tensor:
    """
    Undo _pack_q8nx (keep_block_in_2D=True, Q8_0 so no bias m) and dequantize to bf16.

    Packed layout per block [row_div, col_div] along last dim:
        [ scales_bf16 (c*r*2 bytes) | data_int8 (r*COL_BLOCK_SIZE bytes) ]
    where:
        c = COL_BLOCK_SIZE // Q8_GROUP_SIZE
        r = ROW_BLOCK_SIZE
    Data reorder within block: '(r_div_parallel parallel) c -> (r_div_parallel c parallel)'
    """
    c = COL_BLOCK_SIZE // Q8_GROUP_SIZE   # 8
    r = ROW_BLOCK_SIZE                    # 32

    scale_bytes = c * r * 2              # bf16
    data_bytes = r * COL_BLOCK_SIZE      # int8

    assert packed.shape[-1] == scale_bytes + data_bytes, \
        f"Unexpected packed block size {packed.shape[-1]}, expected {scale_bytes + data_bytes}"

    row_div, col_div = packed.shape[0], packed.shape[1]

    scales_raw = packed[..., :scale_bytes].view(torch.int8)
    data_raw = packed[..., scale_bytes:].view(torch.int8)

    # Restore scales: packed as (c r) column-major -> original (r c) row-major
    scales = scales_raw.view(torch.bfloat16)      # row_div col_div (c*r)
    scales = scales.reshape(row_div, col_div, c, r)
    scales = rearrange(scales, 'rd cd c r -> (rd r) (cd c)')   # padded_rows x (padded_cols//Q8_GROUP_SIZE)

    # Restore data: packed as '(r_div_parallel c parallel)' -> original '(r_div_parallel parallel) c'
    # where c=COL_BLOCK_SIZE (full column dimension), not scale groups
    r_div_parallel = r // PARALLEL_SIZE
    data = data_raw.reshape(row_div, col_div, r_div_parallel, COL_BLOCK_SIZE, PARALLEL_SIZE)
    data = rearrange(data, 'rd cd rp col p -> rd cd (rp p) col')  # rd cd ROW_BLOCK_SIZE COL_BLOCK_SIZE
    data = rearrange(data, 'rd cd r col -> (rd r) (cd col)')       # padded_rows x padded_cols

    # Dequantize: w = scale[row, col//Q8_GROUP_SIZE] * data[row, col]
    scales_exp = scales.repeat_interleave(Q8_GROUP_SIZE, dim=1)  # padded_rows x padded_cols
    result = scales_exp.to(torch.float32) * data.to(torch.float32)
    result = result.to(torch.bfloat16)

    return result[:orig_rows, :orig_cols].contiguous()




def convert_to_bf16(tensor:torch.Tensor):


    # Heuristic: bf16 tensors are already dequantized (norms, embeddings, etc.)
    if tensor.dtype == torch.bfloat16 or tensor.dtype == torch.float32:
        return tensor.to(torch.bfloat16)
        

    # keep_block_in_2D tensors: [p, q, block_bytes] where block_bytes identifies quant type
    if len(tensor.shape) == 3:
        p, q, block_bytes = tensor.shape
        orig_rows = p * ROW_BLOCK_SIZE
        orig_cols = q * COL_BLOCK_SIZE

        sc = COL_BLOCK_SIZE // Q4_GROUP_SIZE      # 8 scale groups
        r = ROW_BLOCK_SIZE                        # 32
        g = ROW_BLOCK_SIZE // PARALLEL_SIZE       # 2
        r_half = PARALLEL_SIZE // 2               # 8
        Q4_1_BLOCK_BYTES = sc * r * 2 * 2 + g * COL_BLOCK_SIZE * r_half  # d + m + qw
        Q8_0_BLOCK_BYTES = sc * r * 2 + r * COL_BLOCK_SIZE                # scales + data

        if block_bytes == Q4_1_BLOCK_BYTES:
            return dequant_q4_1(tensor, orig_rows, orig_cols).to(torch.bfloat16)
        elif block_bytes == Q8_0_BLOCK_BYTES:
            return dequant_q8_0(tensor, orig_rows, orig_cols).to(torch.bfloat16)
        else:
            raise ValueError("unknown block for now")
            print(f"    WARNING: unknown block size {block_bytes}, skipping")
            #out_tensors[name] = tensor.to(torch.bfloat16)
    else:
        raise ValueError(" block of shape != 3")

def replace_tensor(
    output_tensor_map: dict[str, torch.Tensor],
    old_tensor_name: str,
    new_tensor: torch.Tensor) -> None:
    
    
    
    old_tensor: torch.Tensor = output_tensor_map[old_tensor_name]
    
    old_bytes = old_tensor.numel() * old_tensor.element_size()
    new_bytes = new_tensor.numel() * new_tensor.element_size()

    if old_bytes != new_bytes:
        raise ValueError(f"Byte size mismatch: old={old_bytes}, new={new_bytes}")

    if new_tensor.shape != old_tensor.shape:
        new_tensor = new_tensor.contiguous().view(old_tensor.shape)

    output_tensor_map[old_tensor_name] = new_tensor
    


def _layer_tasks(tensors: dict, i: int) -> list[tuple[str, callable]]:
    """Return (out_key, thunk) pairs for layer i. All thunks are independent."""
    is_attn = i == 3 or ((i - 3) % 4 == 0)

    def gate_up(tg=tensors[f"model.layer.{i}.mlp.gate_exps_proj.weight"],
                tu=tensors[f"model.layer.{i}.mlp.up_exps_proj.weight"]):
        g = convert_to_bf16(tg).reshape(NUM_EXPERT, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        u = convert_to_bf16(tu).reshape(NUM_EXPERT, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        return torch.cat([g, u], dim=1)

    tasks = [
        (f"model.language_model.layers.{i}.input_layernorm.weight",
         lambda t=tensors[f"model.layer.{i}.input_layernorm.weight"]: (convert_to_bf16(t) - 1).contiguous()),
        (f"model.language_model.layers.{i}.post_attention_layernorm.weight",
         lambda t=tensors[f"model.layer.{i}.post_attention_layernorm.weight"]: (convert_to_bf16(t) - 1).contiguous()),
        (f"model.language_model.layers.{i}.mlp.experts.down_proj",
         lambda t=tensors[f"model.layer.{i}.mlp.down_exps_proj.weight"]: convert_to_bf16(t)),
        (f"model.language_model.layers.{i}.mlp.experts.gate_up_proj", gate_up),
        
        (f"model.language_model.layers.{i}.mlp.gate.weight",
         lambda t=tensors[f"model.layer.{i}.moe_router.weight"]: convert_to_bf16(t)),
        (f"model.language_model.layers.{i}.mlp.shared_expert.down_proj.weight",
         lambda t=tensors[f"model.layer.{i}.mlp.share_down_exps_proj.weight"]: convert_to_bf16(t)),
        (f"model.language_model.layers.{i}.mlp.shared_expert.gate_proj.weight",
         lambda t=tensors[f"model.layer.{i}.mlp.share_gate_exps_proj.weight"]: convert_to_bf16(t)),
        (f"model.language_model.layers.{i}.mlp.shared_expert.up_proj.weight",
         lambda t=tensors[f"model.layer.{i}.mlp.share_up_exps_proj.weight"]: convert_to_bf16(t)),
        (f"model.language_model.layers.{i}.mlp.shared_expert_gate.weight",
         lambda t=tensors[f"model.layer.{i}.shared_expert_gate.weight"]: convert_to_bf16(t)),
    ]

    if is_attn:
        tasks += [
            (f"model.language_model.layers.{i}.self_attn.k_norm.weight",
             lambda t=tensors[f"model.layer.{i}.self_attn.k_norm.weight"]: (convert_to_bf16(t) - 1).contiguous()),
            (f"model.language_model.layers.{i}.self_attn.k_proj.weight",
             lambda t=tensors[f"model.layer.{i}.self_attn.k_proj.weight"]: convert_to_bf16(t)),
            (f"model.language_model.layers.{i}.self_attn.o_proj.weight",
             lambda t=tensors[f"model.layer.{i}.self_attn.o_proj.weight"]: convert_to_bf16(t)),
            (f"model.language_model.layers.{i}.self_attn.v_proj.weight",
             lambda t=tensors[f"model.layer.{i}.self_attn.v_proj.weight"]: convert_to_bf16(t)),
            (f"model.language_model.layers.{i}.self_attn.q_norm.weight",
             lambda t=tensors[f"model.layer.{i}.self_attn.q_norm.weight"]: (convert_to_bf16(t) - 1).contiguous()),
            # Q has gate interleaved: converter applied '(g p h) c -> (p g h) c' with p=2, h=256
            (f"model.language_model.layers.{i}.self_attn.q_proj.weight",
             lambda t=tensors[f"model.layer.{i}.self_attn.q_proj.weight"]:
                 rearrange(convert_to_bf16(t), '(pp g h) c -> (g pp h) c', pp=2, h=256).contiguous()),
        ]
    else:
        tasks += [
            (f"model.language_model.layers.{i}.linear_attn.A_log",
             lambda t=tensors[f"model.layer.{i}.linear_attn.ssm_a"]:
                 torch.log(-convert_to_bf16(t)).contiguous()),
            (f"model.language_model.layers.{i}.linear_attn.conv1d.weight",
             lambda t=tensors[f"model.layer.{i}.linear_attn.ssm_conv1d.weight"]:
                 convert_to_bf16(t).T.contiguous()),
            (f"model.language_model.layers.{i}.linear_attn.dt_bias",
             lambda t=tensors[f"model.layer.{i}.linear_attn.ssm_dt.bias"]: convert_to_bf16(t)),
            (f"model.language_model.layers.{i}.linear_attn.in_proj_a.weight",
             lambda t=tensors[f"model.layer.{i}.linear_attn.ssm_alpha_proj.weight"]: convert_to_bf16(t)),
            (f"model.language_model.layers.{i}.linear_attn.in_proj_b.weight",
             lambda t=tensors[f"model.layer.{i}.linear_attn.ssm_beta_proj.weight"]: convert_to_bf16(t)),
            (f"model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight",
             lambda t=tensors[f"model.layer.{i}.linear_attn.qkv_proj.weight"]: convert_to_bf16(t)),
            (f"model.language_model.layers.{i}.linear_attn.in_proj_z.weight",
             lambda t=tensors[f"model.layer.{i}.self_attn.gate_proj.weight"]: convert_to_bf16(t)),
            (f"model.language_model.layers.{i}.linear_attn.norm.weight",
             lambda t=tensors[f"model.layer.{i}.linear_attn.ssm_norm.weight"]: convert_to_bf16(t)),
            (f"model.language_model.layers.{i}.linear_attn.out_proj.weight",
             lambda t=tensors[f"model.layer.{i}.linear_attn.ssm_out_proj.weight"]: convert_to_bf16(t)),
        ]

    return tasks


def convert():
    print(f"Loading Q4NX tensors from {INPUT_Q4NX_FILE_PATH}")
    tensors = load_file(INPUT_Q4NX_FILE_PATH)
    out_tensors = load_file(OUPUT_SAFETENSOR_PATH)

    all_tasks: list[tuple[str, callable]] = [
        ("model.language_model.embed_tokens.weight",
         lambda t=tensors["model.embed_tokens.weight"]: convert_to_bf16(t)),
        ("lm_head.weight",
         lambda t=tensors["lm_head.weight"]: convert_to_bf16(t)),
    ]
    for i in range(40):
        all_tasks.extend(_layer_tasks(tensors, i))

    # torch C++ kernels release the GIL, so threads give real CPU parallelism without pickling
    NUM_WORKERS = 16
    results: dict[str, torch.Tensor] = {}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        future_to_key = {pool.submit(thunk): key for key, thunk in all_tasks}
        done = 0
        for fut in as_completed(future_to_key):
            key = future_to_key[fut]
            results[key] = fut.result()
            done += 1
            print(f"  [{done}/{len(all_tasks)}] {key}")

    # apply results sequentially — no concurrent mutation of out_tensors
    for key, tensor in results.items():
        replace_tensor(out_tensors, key, tensor)

    print(f"Saving to {OUPUT_SAFETENSOR_PATH}")
    save_file(out_tensors, OUPUT_SAFETENSOR_PATH)
    print("Done.")


if __name__ == "__main__":
    convert()
