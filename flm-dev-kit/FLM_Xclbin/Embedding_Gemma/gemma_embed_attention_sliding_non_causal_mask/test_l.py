# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import sys
import numpy as np
import matplotlib.pyplot as plt

BLOCK_SIZE = 16
COLUMNS = 8
CHUNK_SIZE = BLOCK_SIZE * COLUMNS

def masked_softmax(q_pos: int, k_pos: int, k_begin: int, k_end: int, L: int, global_mask: np.ndarray = None):
    idx_vec = np.arange(BLOCK_SIZE - 1, -1, -1)

    cutoff_mask = k_pos + BLOCK_SIZE <= (L + idx_vec)

    l_vec = k_begin + idx_vec
    r_vec = k_end + idx_vec
    
    for row in range(BLOCK_SIZE):
        if row + q_pos >= L:
            break
        current_idx = k_pos + BLOCK_SIZE - 1 - row
        mask_l = current_idx >= l_vec
        mask_r = current_idx <= r_vec

        mask = mask_l & mask_r & cutoff_mask

        # out_str = "\t\t"
        # for i in range(BLOCK_SIZE):
        #     out_str += "1" if mask[i] else "0"
        # print(out_str)
        for i in range(BLOCK_SIZE):
            global_mask[q_pos + row, k_pos + i] = mask[i]



def CT(L: int = 1024, W: int = 257, Column: int = 0, global_mask: np.ndarray = None):
    L_PADDED = (L + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE # pad Q to multiple of CHUNK_SIZE
    CHUNKS = L_PADDED // CHUNK_SIZE
    print(f"CHUNKS: {CHUNKS}, L_PADDED: {L_PADDED}, L: {L}")

    for chunk_idx in range(CHUNKS): # CHUNKS in L of Q
        chunk_q_begin = chunk_idx * CHUNK_SIZE
        chunk_q_end = chunk_q_begin + CHUNK_SIZE - 1
        chunk_k_begin = chunk_q_begin - W + 1
        chunk_k_end   = chunk_q_end + W
        
        chunk_k_begin = chunk_k_begin if chunk_k_begin > 0 else 0 # begin must be non-negative
        chunk_k_end = chunk_k_end if chunk_k_end < L else L       # end must be less than L

        k_begin = chunk_k_begin // CHUNK_SIZE * CHUNK_SIZE
        k_end   = (chunk_k_end + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE

        k_chunks = (k_end - k_begin) // CHUNK_SIZE

        print(f"CHUNK {chunk_idx:4d}: chunk_q_begin: {chunk_q_begin:4d}, chunk_q_end: {chunk_q_end:4d}, chunk_k_begin: {chunk_k_begin:4d}, chunk_k_end: {chunk_k_end:4d}, k_begin: {k_begin:4d}, k_end: {k_end:4d}, k_chunks: {k_chunks:4d}")

        # Now locate the CT at the column
        ct_q_begin = chunk_q_begin + Column * BLOCK_SIZE
        ct_q_end   = ct_q_begin + BLOCK_SIZE - 1
        ct_k_begin = ct_q_begin - W + 1
        ct_k_end   = ct_q_end + W

        ct_k_begin = ct_k_begin if ct_k_begin > -BLOCK_SIZE else -BLOCK_SIZE # begin must be non-negative
        ct_k_end = ct_k_end if ct_k_end < (L_PADDED + W) else (L_PADDED + W)               # end must be less than L + W
        ct_k_end -= BLOCK_SIZE

        print(f"\tCT {Column:4d}: ct_q_begin: {ct_q_begin:4d}, ct_q_end: {ct_q_end:4d}, ct_k_begin: {ct_k_begin:4d}, ct_k_end: {ct_k_end:4d}")
        
        q_pos = chunk_q_begin + Column * BLOCK_SIZE
        for k_chunk_idx in range(k_chunks):
            for k_block_idx in range(COLUMNS):
                k_pos = k_begin + k_chunk_idx * CHUNK_SIZE + k_block_idx * BLOCK_SIZE
                print(f"\t\tQ {q_pos:4d}, k_pos: {k_pos:4d}")
                masked_softmax(q_pos, k_pos, ct_k_begin, ct_k_end, L, global_mask)
        #         input()
        # print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <L> <W>")
        sys.exit(1)
    L = int(sys.argv[1])
    W = int(sys.argv[2])
    L_PADDED = (L + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
    global_mask = np.zeros((L_PADDED, L_PADDED))
    global_mask = global_mask - 1
    for col in range(COLUMNS):
        CT(L, W, col, global_mask)
    
    plt.imshow(global_mask)
    plt.colorbar()
    plt.savefig("global_mask.png")
    plt.close()

    bool_mask = global_mask == 1
    right_part = bool_mask[:, L:]
    print(right_part.sum())
    if right_part.sum() != 0:
        print("Right part is not all zeros")
        sys.exit(1)
    useful_part = bool_mask[0:L, 0:L]

    row_sum = useful_part.sum(axis=1)
    standard_row_sum = np.zeros(L)
    for i in range(L):
        start = max(0, i - W + 1)
        end = min(L, i + W)
        standard_row_sum[i] = end - start

    correct = True
    for i in range(L):
        if row_sum[i] != standard_row_sum[i]:
            correct = False
            break
    if correct:
        print("Row sum is correct")
    else:
        print("Row sum is not correct")
