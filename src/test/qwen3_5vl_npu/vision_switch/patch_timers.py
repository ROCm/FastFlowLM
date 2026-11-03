import sys
path = sys.argv[1]
src = open(path).read()

def sub(old, new):
    global src
    n = src.count(old)
    assert n == 1, (n, old)
    src = src.replace(old, new)

sub('#include "vision/norm.hpp"\n', '#include "vision/norm.hpp"\n#include "vision_timer.hpp"\n')
sub('    auto encoder_start_time = std::chrono::high_resolution_clock::now();\n',
    '    auto encoder_start_time = std::chrono::high_resolution_clock::now();\n'
    '    vision_timer_t vt;\n    vt.start();\n    const bool vt_probe = vision_timer_t::probe_enabled();\n')
sub('    constexpr bool DEBUG_NO_REGRESSION_ERROR = false;\n',
    '    constexpr bool DEBUG_NO_REGRESSION_ERROR = false;\n    vt.mark("pre_layers");\n')

sub('''        auto time_start_qkv_projection = std::chrono::high_resolution_clock::now();
        mm_qkv_input.sync_to_device();
        attn_qkv_weight[layer_idx].sync_to_device();
        qkv_projection_app( mm_qkv_input,  attn_qkv_weight[layer_idx], mm_qkv_output_mha_input   );
        mm_qkv_output_mha_input.sync_from_device();
''', '''        vt.mark("ln1");
        auto time_start_qkv_projection = std::chrono::high_resolution_clock::now();
        mm_qkv_input.sync_to_device();
        vt.mark("qkv_sync_in");
        attn_qkv_weight[layer_idx].sync_to_device();
        vt.mark("qkv_w_sync");
        qkv_projection_app( mm_qkv_input,  attn_qkv_weight[layer_idx], mm_qkv_output_mha_input   );
        vt.mark("qkv_run");
        if (vt_probe) {
            qkv_projection_app( mm_qkv_input,  attn_qkv_weight[layer_idx], mm_qkv_output_mha_input   );
            vt.mark("qkv_run_again");
        }
        mm_qkv_output_mha_input.sync_from_device();
        vt.mark("qkv_sync_out");
''')

sub('''        auto time_start_mha = std::chrono::high_resolution_clock::now();
        mm_qkv_output_mha_input.sync_to_device();
        mha_qkv_output_mm_o_input.sync_to_device();
        flash_attention_app(mha_qkv_output_mm_o_input, mm_qkv_output_mha_input);
        mha_qkv_output_mm_o_input.sync_from_device();
''', '''        vt.mark("bias_rope");
        auto time_start_mha = std::chrono::high_resolution_clock::now();
        mm_qkv_output_mha_input.sync_to_device();
        mha_qkv_output_mm_o_input.sync_to_device();
        vt.mark("attn_sync_in");
        flash_attention_app(mha_qkv_output_mm_o_input, mm_qkv_output_mha_input);
        vt.mark("attn_run");
        if (vt_probe) {
            flash_attention_app(mha_qkv_output_mm_o_input, mm_qkv_output_mha_input);
            vt.mark("attn_run_again");
        }
        mha_qkv_output_mm_o_input.sync_from_device();
        vt.mark("attn_sync_out");
''')

sub('''        auto time_start_o_projection = std::chrono::high_resolution_clock::now();
        mha_qkv_output_mm_o_input.sync_to_device();
        o_proj_weight[layer_idx].sync_to_device();
        o_projection_app( mha_qkv_output_mm_o_input,  o_proj_weight[layer_idx], mm_o_output_fc1_proj_input   );
        mm_o_output_fc1_proj_input.sync_from_device();
''', '''        vt.mark("host_after_attn");
        auto time_start_o_projection = std::chrono::high_resolution_clock::now();
        mha_qkv_output_mm_o_input.sync_to_device();
        vt.mark("o_sync_in");
        o_proj_weight[layer_idx].sync_to_device();
        vt.mark("o_w_sync");
        o_projection_app( mha_qkv_output_mm_o_input,  o_proj_weight[layer_idx], mm_o_output_fc1_proj_input   );
        vt.mark("o_run");
        if (vt_probe) {
            o_projection_app( mha_qkv_output_mm_o_input,  o_proj_weight[layer_idx], mm_o_output_fc1_proj_input   );
            vt.mark("o_run_again");
        }
        mm_o_output_fc1_proj_input.sync_from_device();
        vt.mark("o_sync_out");
''')

sub('''        auto time_start_fc1 = std::chrono::high_resolution_clock::now();
        mm_o_output_fc1_proj_input.sync_to_device();
        mlp_fc1_weight[layer_idx].sync_to_device();
        mlp_fc2_weight[layer_idx].sync_to_device();
        mlp_fc1_app( mm_o_output_fc1_proj_input,  mlp_fc1_weight[layer_idx],   mm_fc1_output_fc2_input );
''', '''        vt.mark("bias_res_ln2");
        auto time_start_fc1 = std::chrono::high_resolution_clock::now();
        mm_o_output_fc1_proj_input.sync_to_device();
        vt.mark("fc1_sync_in");
        mlp_fc1_weight[layer_idx].sync_to_device();
        mlp_fc2_weight[layer_idx].sync_to_device();
        vt.mark("fc_w_sync");
        mlp_fc1_app( mm_o_output_fc1_proj_input,  mlp_fc1_weight[layer_idx],   mm_fc1_output_fc2_input );
        vt.mark("fc1_run");
''')

sub('''        auto time_start_mlp_fc2 = std::chrono::high_resolution_clock::now();
        mlp_fc2_app( mm_fc1_output_fc2_input,  mlp_fc2_weight[layer_idx],   mm_fc2_output );
        mm_fc2_output.sync_from_device();
''', '''        vt.mark("host_after_fc1");
        auto time_start_mlp_fc2 = std::chrono::high_resolution_clock::now();
        mlp_fc2_app( mm_fc1_output_fc2_input,  mlp_fc2_weight[layer_idx],   mm_fc2_output );
        vt.mark("fc2_run");
        mm_fc2_output.sync_from_device();
        vt.mark("fc2_sync_out");
''')

sub('''        auto time_end_mlp_residual_add = std::chrono::high_resolution_clock::now();
''', '''        auto time_end_mlp_residual_add = std::chrono::high_resolution_clock::now();
        vt.mark("residual2");
''')

sub('''    merger_fc1_input.sync_to_device();
    merger_fc1_weight.sync_to_device();
''', '''    vt.mark("merger_ln");
    merger_fc1_input.sync_to_device();
    merger_fc1_weight.sync_to_device();
''')
sub('''    merger_fc2_output.sync_from_device();
''', '''    merger_fc2_output.sync_from_device();
    vt.mark("merger_mm");
''')
sub('''    return final_res;
}''', '''    vt.mark("tail");
    vt.print(seq_len, seq_len_padded);
    return final_res;
}''')
open(path, 'w').write(src)
print("ok")
