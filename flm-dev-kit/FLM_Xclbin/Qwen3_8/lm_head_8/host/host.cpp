// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "typedef.hpp"
#include "buffer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "metrices.hpp"
#include "tensor_2d.hpp"
#include "lm_config.hpp"
#include <map>
#include <cmath>
#include <vector>

// Token rows per dispatch -- the verify phase scores 8 candidates at once. Must
// match L_TOKENS in kernel/qmm.cc and iron/lm_head_aie.py. The L = 1 design is
// ../../lm_head_npu_bin.
static constexpr int L = 8;

// Lanes in the x stream: the L token rows plus one carrying model.norm.weight,
// which the kernel folds into x as it loads it. Must match X_LANES in
// kernel/qmm.cc and iron/lm_head_aie.py.
static constexpr int X_LANES = L + 1;

void generate_seq(npu_sequence* seq, int VOCAB_SIZE, int D);
void reorder_cpy_q80(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks);

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("QWEN3_8_27B"), "Model type, e.g., QWEN3_8_27B");

    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_type = vm["model"].as<std::string>();

    header_print("info", "Testing model: " + model_type);

    // -m picks both the checkpoint and the matching xclbin; MODEL_DIM is baked into
    // the kernel, so a build is not interchangeable across checkpoints.
    static const std::map<std::string, std::string> model_dirs = {
        {"QWEN3_8_27B", "Qwen3.8-27B-NPU2"},
    };
    auto model_dir = model_dirs.find(model_type);
    if (model_dir == model_dirs.end()){
        header_print("error", "Unknown model type: " + model_type);
        return 1;
    }

    std::string model_path     = "/path/to/models/" + model_dir->second;
    header_print("info", "Model path: " + model_path);
    LM_Config lm_config;
    lm_config.from_pretrained(model_path);

    size_t D          = lm_config.hidden_size;
    size_t VOCAB_SIZE = lm_config.vocab_size;
    size_t M_PER_ROUND = 32 * 32;
    size_t VOCAB_SIZE_PADDED = (VOCAB_SIZE + M_PER_ROUND - 1) / M_PER_ROUND * M_PER_ROUND;
    // overide the layer xclbin

    lm_config.lm_head_xclbin_name = "./build/" + model_type + "/xclbins/lm_head.xclbin";
    header_print("info", "Layer xclbin path: " + lm_config.lm_head_xclbin_name);
    header_print("info", "Hidden size (D): " + std::to_string(D));
    header_print("info", "Vocab size: " + std::to_string(VOCAB_SIZE));


    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_path);

    // This checkpoint does NOT tie the word embeddings (config tie_word_embeddings is
    // false, and lm_head.weight / model.embed_tokens.weight really are different
    // bytes), so the Qwen3.5 test -- feed an embedding row in, expect argmax to come
    // back as that same token id -- is meaningless here. Drive the head from the
    // reference dump instead and score against the logits the reference recorded.
    buffer<u8> lm_head_packed;
    q4nx_model->load_weights(lm_head_packed, "lm_head.weight");

    // The kernel applies the final rmsnorm itself, so the input is the *pre*-norm
    // hidden state (layer_63_out), not final_norm_out.
    SafeTensors ref(model_path + "/ref.safetensors");
    buffer<bf16> x;
    ref.load_weights(x, "verify.layer_63_out");
    buffer<float> logits_ref;
    ref.load_weights(logits_ref, "verify.logits");

    const int test_size = (int)(x.size() / D);
    header_print("info", "Reference rows: " + std::to_string(test_size));
    if ((size_t)test_size * VOCAB_SIZE != logits_ref.size()){
        header_print("error", "verify.logits does not match verify.layer_63_out");
        return 1;
    }

    header_print("info", "Testing lm head...");
    header_print("info", "Initializing NPU manager...");

    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* lm_head_app_mgr = xclbin_mgr.register_xclbin(lm_config.lm_head_xclbin_name);

    npu_app lm_head = lm_head_app_mgr->create_app();
    header_print("info", "Creating NPU sequence...");

    size_t lm_head_weight_size = D * VOCAB_SIZE_PADDED + D * VOCAB_SIZE_PADDED / 32 * 2;

    buffer<u8>  lm_head_weights = lm_head.create_bo_buffer<u8>(lm_head_weight_size);
    // L rows of logits per dispatch, and L + 1 lanes of x in -- the L token rows
    // plus model.norm.weight.
    buffer<bf16> lm_head_output = lm_head.create_bo_buffer<bf16>((size_t)L * VOCAB_SIZE_PADDED);
    buffer<bf16> lm_head_input  = lm_head.create_bo_buffer<bf16>((size_t)X_LANES * D);

    lm_head_weights.memset(0);
    lm_head_output.memset((bf16)0.0f);
    lm_head_output.sync_to_device();

    {
        header_print("info", "Vocab got: " + std::to_string(lm_head_packed.size() * 8 / 8.5 / D));
        header_print("info", "Vocab reserved: " + std::to_string(lm_head_weights.size() * 8 / 8.5 / D));
        reorder_cpy_q80(lm_head_weights.data(), lm_head_packed, D, 4);
    }
    buffer<bf16> final_rms_weight;

    q4nx_model->load_weights(final_rms_weight, "model.norm.weight");

    // model.norm runs on the core, the same as in the L = 1 design -- what the
    // host does here is interleave, not normalise. The core cannot hold a whole
    // 5120-wide row (80 KB against 64 KB of core memory), but it does not need
    // to: it *sees* every column of every row across the k loop, so the sum of
    // squares is a reduction it can accumulate on the fly, and the reciprocal it
    // yields is a per-token scalar that commutes through the matmul. Only the
    // per-column half of the norm has to meet x elementwise, and that is why
    // model.norm.weight travels as an extra lane of x. See kernel/qmm.cc.
    //
    // x goes out as [k_block][lane][256] with X_LANES lanes -- the L token rows
    // followed by that column block's slice of the norm weight. The kernel's
    // k-loop wants all lanes of one column block together, and the memtile
    // replays the whole buffer per round in exactly this order.
    //
    // The rows go out RAW. Feeding this bitstream pre-normalised x does not
    // fault; it applies the norm twice and returns fluent, wrong logits.
    //
    // `order[t]` picks which reference row lands in token slot t. Identity for
    // the scoring pass; the repeat pass below permutes it, which is the only
    // way to tell a second dispatch that re-read x from one that replayed the
    // first dispatch's copy.
    auto upload_x = [&](const int* order) {
        const int K_BLOCKS = (int)(D / 256);
        for (int k = 0; k < K_BLOCKS; k++) {
            for (int t = 0; t < L; t++) {
                const bf16* src = x.data() + (size_t)order[t] * D + (size_t)k * 256;
                bf16* dst = lm_head_input.data() + ((size_t)k * X_LANES + t) * 256;
                for (int c = 0; c < 256; c++) dst[c] = src[c];
            }
            const bf16* nw  = final_rms_weight.data() + (size_t)k * 256;
            bf16*       dst = lm_head_input.data() + ((size_t)k * X_LANES + L) * 256;
            for (int c = 0; c < 256; c++) dst[c] = nw[c];
        }
        lm_head_input.sync_to_device();
    };

    // Which reference row currently sits in each token slot. The repeat pass
    // permutes this, and the scoring below follows it, so a permuted rerun is
    // checked against the rows it was actually given.
    int row_of_slot[L];
    for (int t = 0; t < L; t++) row_of_slot[t] = t;
    upload_x(row_of_slot);

    header_print("info", "Weights loaded...");

    generate_seq(lm_head.seq(), VOCAB_SIZE_PADDED, D);

    header_print("info", "Running NPU sequence...");


    lm_head_weights.sync_to_device();
    time_utils::time_with_unit total_ms;
    total_ms.first = 0;
    lm_head_app_mgr->get_ready();
    lm_head_app_mgr->ready();
    
    int argmax_hits = 0;

    // One dispatch for all L rows -- the point of this design. The weights are
    // read once instead of once per token.
    std::vector<std::vector<float>> logits((size_t)L,
                                           std::vector<float>(VOCAB_SIZE_PADDED));
    auto dispatch = [&]() {
        time_utils::time_point start = time_utils::now();
        // Check the state: a timed-out dispatch leaves the output BO holding the
        // previous contents, which a value-only check reads as a pass.
        ert_cmd_state st = lm_head(lm_head_output, lm_head_weights, lm_head_input);
        if (st != ERT_CMD_STATE_COMPLETED)
            header_print("error", "dispatch state = " + std::to_string((int)st));
        time_utils::time_point end = time_utils::now();
        total_ms.first += time_utils::duration_ms(start, end).first;
        lm_head_output.sync_from_device();

        // The cores emit [token][32] per 32-row weight block, so the DDR image
        // is [round][col][core][token][32]. Un-interleave into L contiguous
        // logit rows.
        const size_t M_PER_ROUND = 32 * 32;
        const size_t rounds = VOCAB_SIZE_PADDED / M_PER_ROUND;
        const bf16* src = lm_head_output.data();
        for (size_t r = 0; r < rounds; r++)
            for (size_t blk = 0; blk < 32; blk++)   // 8 cols x 4 cores
                for (int t = 0; t < L; t++)
                    for (size_t e = 0; e < 32; e++)
                        logits[t][r * M_PER_ROUND + blk * 32 + e] =
                            (float)src[((r * 32 + blk) * L + t) * 32 + e];
    };
    dispatch();

    // A one-shot bench cannot see dispatch state. This design fills its memtile
    // x buffer from a PROLOGUE bd -- one that the ring steps over forever after
    // the first pass -- so whether a second dispatch re-reads x is a property
    // no single dispatch can report. Repeat with the rows reversed: the
    // argmaxes must reverse with them.
    // FLM_LM8_REPEAT=same re-dispatches the *identical* x. That separates the two
    // ways a second dispatch can fail: if only the x prologue is spent, the array
    // still holds dispatch 1's x and "same" must score 8/8 while a permuted x
    // comes back stale-but-valid. If "same" also fails, the damage is not x --
    // the weight/y streams themselves are desynchronised.
    // FLM_LM8_REPEAT=N re-dispatches N more times, permuting the token slots each
    // time (FLM_LM8_REPEAT=same keeps the identity permutation instead). The
    // scoring below follows row_of_slot, so a pass means every slot still
    // carries the row that was put into it.
    //
    // Permuting is the point: a design that silently replays the first
    // dispatch's x would keep returning the first dispatch's argmaxes and score
    // 8/8 against an unpermuted expectation. It has to be checked, because the
    // original layout did exactly that -- x was filled by a one-shot PROLOGUE bd
    // that the memtile ring stepped over after the first pass.
    const char* rep = std::getenv("FLM_LM8_REPEAT");
    if (rep && rep[0] != '0') {
        const bool same = (rep[0] == 's');
        const int n_rep = same ? 1 : std::max(1, std::atoi(rep));
        for (int it = 0; it < n_rep; it++) {
            // A rotation rather than a reversal: over several dispatches it puts
            // a different row in each slot, where a reversal just alternates.
            for (int t = 0; t < L; t++)
                row_of_slot[t] = same ? t : (t + it + 1) % L;
            upload_x(row_of_slot);
            dispatch();
        }
        header_print("info", "Re-dispatched " + std::to_string(n_rep)
            + " more time(s); slots " + (same ? "unpermuted" : "rotated")
            + ". Scored against the rows actually fed in.");
    }

    for (int i = 0; i < test_size; i++) {
        // Score slot i against whichever reference row was fed into it. For the
        // plain pass that is row i; after a permuted repeat it is not, and
        // comparing slot i to row i would report a working design as 0/8.
        const int ref_row = row_of_slot[i];
        // get_error_metrics strides by 8 with no tail guard; VOCAB_SIZE is 248320,
        // a multiple of 8, so the valid view can be scored as-is.
        buffer<float> y_float(VOCAB_SIZE);
        buffer<float> y_ref(logits_ref.data() + (size_t)ref_row * VOCAB_SIZE, VOCAB_SIZE);
        for (size_t j = 0; j < VOCAB_SIZE; j++){
            y_float[j] = logits[i][j];
        }

        int max_index = 0, ref_index = 0;
        for (size_t j = 1; j < VOCAB_SIZE; j++){
            if (y_float[j] > y_float[max_index]) max_index = (int)j;
            if (y_ref[j]   > y_ref[ref_index])   ref_index = (int)j;
        }
        // The reference logits are bf16-rounded, so its top-1 is sometimes an exact
        // tie (row 1 of verify has two tokens at 24.75). Which of a tied pair wins is
        // arbitrary, so score against the reference's own value at the token the NPU
        // picked rather than against the index numpy happened to break the tie toward.
        const bool tied = y_ref[max_index] >= y_ref[ref_index];
        if (tied) argmax_hits++;

        error_metrics m = get_error_metrics(y_float, y_ref);
        std::cout << "Row " << i << " (ref row " << ref_row << ")"
                  << ": argmax npu=" << max_index << " ref=" << ref_index
                  << (max_index == ref_index ? " OK" : (tied ? " OK (tie)" : " MISMATCH"))
                  << "; cos=" << m.CosineSimilarity
                  << " nmse=" << m.NormalizedMSE << std::endl;
    }

    header_print("info", "Argmax match: " + std::to_string(argmax_hits) + "/" + std::to_string(test_size));
    header_print("info", "Time spent: " + std::to_string(total_ms.first) + total_ms.second
                        + " for " + std::to_string(L) + " rows ("
                        + std::to_string(total_ms.first / L) + total_ms.second + "/row)");

    return argmax_hits == test_size ? 0 : 1;
}

void generate_seq(npu_sequence* seq, int VOCAB_SIZE, int D){
   static constexpr int y_arg_id = 0;
    static constexpr int w_arg_id = 1;
    static constexpr int x_arg_id = 2;
    static constexpr int M_PER_ROUND = 32 * 32;
    static constexpr int M    = 32;
    static constexpr int M_PER_COL    = M * 4;
    static constexpr int COLS = 8;
    static npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    // Column 0's MM2S 0 is packet-switched -- must match W_PKT_ID in
    // iron/q4k_mm.py. Every other shim MM2S port is circuit-switched.
    static constexpr int W_PKT_ID = 1;
    static constexpr int NO_PKT   = -1;
    // Rounds of shim BD ping-pong; also how far the y waits lag the y issues.
    // Rounds of shim BD ping-pong; also how far the y waits lag the y issues.
    //
    // Two, not three. A round now issues FOUR shim BDs (x, y, wa, wb) where the
    // L = 1 design issued three, so LOOKAHEAD+1 = 4 rounds would need all 16 of
    // the shim's BDs and leave the sequencer no free id to stage the next round
    // into: it hangs, ERT state 8, ~60 s. Three rounds in flight uses 12 and is
    // only ~3% slower than the (unreachable) four.
    //
    // FLM_LM8_LOOKAHEAD overrides it for bisecting stalls.
    static const int LOOKAHEAD = [] {
        const char* e = std::getenv("FLM_LM8_LOOKAHEAD");
        return e ? std::atoi(e) : 2;
    }();
    assert(VOCAB_SIZE % M_PER_ROUND == 0);
    int rounds = VOCAB_SIZE / M_PER_ROUND;
    size_t W_PER_COL = int(M_PER_COL * D * 8.5 / 8) / sizeof(bf16);
    size_t W_PER_ROUND = W_PER_COL * COLS;
    // Each column's weights are split over two shim MM2S channels, one per pair of
    // cores; reorder_cpy_q80 has already laid the two halves out back to back.
    size_t W_PER_STREAM = W_PER_COL / 2;

    seq->clear_cmds();
    // x is no longer one broadcast off IT0, and no longer a one-shot preamble
    // either. Every column takes its own full copy of x into its memtile ONCE PER
    // ROUND, on the same shim channel its group-a weights use, alternating with
    // them: x, wa, x, wa, ...  The memtile's S2MM 1 ring is [x0, wa0, x1, wa1],
    // so that alternation is what keeps the two in phase (see lm_head_memtile).
    //
    // It used to be a single transfer ahead of the weights, matching a PROLOGUE
    // bd in the memtile. That could not be dispatched twice: the ring steps over
    // the prologue after the first pass, so dispatch 2's x landed in the weight
    // BDs and was dequantised as Q8_0 -- NaN logits, and the command still
    // reported COMPLETED. Re-sending x per round costs bandwidth (see below) but
    // removes the per-dispatch state entirely.
    //
    // The cost is real but small: x is X_LANES*D*2 = 90 KB against a round's
    // 340 KB of weights per channel, so this is ~27% more bytes on channel 1 and
    // ~13% on the column. Channel 1 was the lighter of the two before this
    // change. 10 KB of that 90 is model.norm.weight, which is the same bytes
    // every round -- the alternative, a resident copy filled once, is the
    // one-shot BD that broke the second dispatch.
    //
    // Sharing a port between x and weights was safe at L = 1 only because x went
    // once, before any weight was queued. Per-round x on a shared *broadcast*
    // port would deadlock: cores waiting on weights stuck behind x stop draining
    // x, which stalls the port they are waiting on. Landing x in the memtile
    // breaks that cycle, because the memtile sinks all 90 KB and frees the port.

    for (int r = 0; r < rounds; r++) {
        // Four BDs per round now that x is per-round, times LOOKAHEAD+1 = 4 rounds
        // in flight, is exactly the shim's 16 BDs.
        int bd_offset = (r % (LOOKAHEAD + 1)) * 4;
        npu_bd_id bd_x  = npu_bd_id(bd_0 + bd_offset);
        npu_bd_id bd_y  = npu_bd_id(bd_1 + bd_offset);
        npu_bd_id bd_wa = npu_bd_id(bd_2 + bd_offset);
        npu_bd_id bd_wb = npu_bd_id(bd_3 + bd_offset);
        for (size_t col = 0; col < std::size(ITs); col++){
            // x goes out ahead of this round's group-a weights, on the same
            // channel: the memtile ring is [x, wa, x, wa] and this is what holds
            // the two halves of it in phase.
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                x_arg_id,
                MM2S,
                ITs[col],
                bd_x,
                it_channel_1,
                {0, 0, 0, 0},
                {1, 1, 1, (uint32_t)((size_t)X_LANES * D)},
                {0, 0, 0, 1},
                NO_PKT, 0, false, aggressive_cache
            );
            size_t w_col_offset = r * W_PER_ROUND + col * W_PER_COL;
            // Each core now emits L * 32 values per round instead of 32, so a
            // column's slice is L times wider and lands L times further apart.
            // The DDR image is [round][col][core][token][32]; the host
            // un-interleaves it after the dispatch.
            uint32_t y_offset = (uint32_t)((size_t)L * (r * M_PER_ROUND + col * M_PER_COL));
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                y_arg_id,
                S2MM,
                ITs[col],
                bd_y,
                it_channel_0,
                {0, 0, 0, (uint32_t)(y_offset)},
                {1, 1, 1, (uint32_t)((size_t)L * M_PER_COL)},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            // Group a (cores 0/1) on channel 1, group b (cores 2/3) on channel 0.
            // Channel 0 of IT0 is packet-switched, so only that transfer needs an id.
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_wa,
                it_channel_1,
                {0, 0, 0, (uint32_t)(w_col_offset)},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                NO_PKT, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_wb,
                it_channel_0,
                {0, 0, 0, (uint32_t)(w_col_offset + W_PER_STREAM)},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                col == 0 ? W_PKT_ID : NO_PKT, 0, false, aggressive_cache
            );
            // Four sets of shim BDs are in flight, so a round only has to wait for
            // the y of the round three back rather than the one immediately before.
            // That keeps three rounds of weight transfers queued ahead of the
            // sequencer instead of one.
            if (r > LOOKAHEAD - 1){
                seq->npu_dma_wait(
                    ITs[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (size_t col = 0; col < std::size(ITs); col++){
        for (int i = 0; i < LOOKAHEAD; i++){
            seq->npu_dma_wait(
                ITs[col],
                S2MM,
                it_channel_0
            );
        }
    }

    seq->cmds2seq();
}

void reorder_cpy_q80(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
{
     // return;
    const int a_block_size = 32 * 256 + 256 * 2;
    const int blocks_per_row = col / 256;

    const int rows = src.size() / a_block_size / blocks_per_row;

    // A column's weights arrive on two shim channels, one per pair of cores, so each
    // column-round has to be two contiguous halves in DDR rather than one run with all
    // four cores' blocks interleaved: rows r+0/r+1 of the group for the stream that
    // feeds cores 0 and 1, rows r+2/r+3 for the stream that feeds cores 2 and 3. Within
    // a half the order is still k-major, one block per core, because that is exactly one
    // memtile slot. See lm_head_memtile in iron/q4k_mm.py.
    const int rows_per_stream = vertical_blocks / 2;

    u8 *dst_ptr = dst;
    for (int r = 0; r < rows; r += vertical_blocks)
    {
        for (int half = 0; half < 2; half++)
        {
            for (int c = 0; c < blocks_per_row; c++)
            {
                for (int i = 0; i < rows_per_stream; i++)
                {
                    const int row = r + half * rows_per_stream + i;
                    const u8 *src_ptr = (row < rows)
                        ? src.data() + (size_t)(row * blocks_per_row + c) * a_block_size
                        : src.data(); // useless padding
                    memcpy(dst_ptr, src_ptr, a_block_size);
                    dst_ptr += a_block_size;
                }
            }
        }
    }
}
