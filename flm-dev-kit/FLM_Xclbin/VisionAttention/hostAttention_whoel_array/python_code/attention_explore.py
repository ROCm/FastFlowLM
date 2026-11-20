# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import torch
import torch.nn.functional as F
import math

# Use the GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flash_attention_inference_tiled(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    l_q: int = 32,  # Block size for Q (rows)
    l_v: int = 64   # Block size for K, V (columns)
) -> torch.Tensor:
    """
    Conceptual implementation of FlashAttention's tiling logic for inference.

    This function simulates the block-wise computation and running normalization
    trick (m_i and L_i) to avoid storing the full attention matrix S=QK^T.

    Args:
        Q (torch.Tensor): Query tensor (L, D)
        K (torch.Tensor): Key tensor (S, D)
        V (torch.Tensor): Value tensor (S, D)
        B_r (int): Block size for the Query dimension (L).
        B_c (int): Block size for the Key/Value dimension (S).

    Returns:
        torch.Tensor: The output attention tensor (L, D).
    """
    L, D = Q.shape  # L = sequence length of Q, D = hidden dimension
    S = K.shape[0]  # S = sequence length of K/V (source length)
    d_k = D

    # 1. Initialize output tensors and running max/sum for normalization
    O = torch.zeros_like(Q, dtype=Q.dtype, device=DEVICE)

    # m_i (running max score) and L_i (running exponential sum) are (L) dimensional vectors
    # m_i is initialized to -inf, L_i to 0
    m_i = torch.full((L,), -float('inf'), dtype=Q.dtype, device=DEVICE)
    L_i = torch.zeros((L,), dtype=Q.dtype, device=DEVICE)

    scale = 1.0 / math.sqrt(d_k)

    # --- Tiled Processing ---

    # Loop over blocks of Q (rows, i)
    for i in range(0, L, l_q):
        # Q_i: A block of query vectors (B_r, D)
        Q_i = Q[i:i + l_q]
        R_i = Q_i.shape[0] # Actual block size (can be < B_r at the end)

        # Ensure running normalization vectors are correctly sized for the block
        m_i_block = m_i[i:i + l_q].clone()
        L_i_block = L_i[i:i + l_q].clone()
        O_i_block = O[i:i + l_q].clone()

        # Loop over blocks of K and V (columns, j)
        for j in range(0, S, l_v):
            # K_j, V_j: Blocks of key and value vectors (B_c, D)
            K_j = K[j:j + l_v]
            V_j = V[j:j + l_v]

            # 2. Compute Attention Score Block (S_ij)
            # S_ij is (R_i, B_c)
            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

            # 3. Apply the FlashAttention Tiling Trick (Crucial step for memory efficiency)
            # Find the new maximum (m'_ij) across the new block S_ij and the old running max m_i
            m_new = torch.max(m_i_block.unsqueeze(1), torch.max(S_ij, dim=-1).values.unsqueeze(1)).squeeze(1)

            # Compute the correction factors (exponents for scaling O and L)
            # exp(m_i - m_new) is the factor to scale the old L_i and O_i
            exp_old = torch.exp(m_i_block - m_new)

            # exp(S_ij - m_new) is the unnormalized probability for the new block
            P_ij_unnormalized = torch.exp(S_ij - m_new.unsqueeze(1))

            # Update L_i (running exponential sum)
            L_new = exp_old * L_i_block + torch.sum(P_ij_unnormalized, dim=-1)

            # 4. Update Output Block (O_i)
            # O_i_new = diag(exp(m_i - m_new)) * O_i_old + P_ij_unnormalized * V_j
            # Note: We divide by the new L_new at the very end (step 5)
            O_new = exp_old.unsqueeze(1) * O_i_block + torch.matmul(P_ij_unnormalized, V_j)

            # 5. Update running state variables
            L_i_block = L_new
            m_i_block: torch.Tensor = m_new
            O_i_block = O_new

        # 6. Final normalization for the current Q block
        # The final output O_i is O_i_block / L_i_block
        O[i:i + l_q] = O_i_block / L_i_block.unsqueeze(1)

    return O

# --- Example Usage ---
# Hyperparameters
L_SEQ = 512    # Query sequence length
S_SEQ = 512   # Key/Value sequence length (e.g., source context)
D_MODEL = 72  # Hidden dimension
BATCH = 1      # Batch size is typically 1 for inference

# Tiling Parameters (smaller for demo, larger in production)
BLOCK_L_Q = 32
BLOCK_L_V = 64

# Create Mock Tensors (Batch dimension is implicitly handled by the loop over L)
Q_mock = torch.randn(L_SEQ, D_MODEL, dtype=torch.float32, device=DEVICE)
K_mock = torch.randn(S_SEQ, D_MODEL, dtype=torch.float32, device=DEVICE)
V_mock = torch.randn(S_SEQ, D_MODEL, dtype=torch.float32, device=DEVICE)

print(f"--- Running FlashAttention Inference Simulation ---")
print(f"Q shape: {Q_mock.shape}, K/V shape: {K_mock.shape}")
print(f"Block sizes (B_r, B_c): ({BLOCK_L_Q}, {BLOCK_L_V})")

# 1. Run the Tiled (Flash) Attention
output_tiled = flash_attention_inference_tiled(Q_mock, K_mock, V_mock, BLOCK_L_Q, BLOCK_L_V)

print(f"Tiled Output shape: {output_tiled.shape}")

# 2. Run Standard Attention (for verification of result correctness)
def standard_attention(Q, K, V):
    """Standard (Non-tiled) Self-Attention for comparison."""
    d_k = Q.size(-1)
    # 1. Score (L, S)
    S_full = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # 2. Probability (L, S)
    P_full = F.softmax(S_full, dim=-1)
    # 3. Output (L, D)
    O_full = torch.matmul(P_full, V)
    return O_full

# Run Standard Attention
output_standard = standard_attention(Q_mock, K_mock, V_mock)

# 3. Verify numerical equivalence (they should be very close)
diff = torch.norm(output_tiled - output_standard)
print(f"\nNorm of difference between Tiled and Standard Output: {diff.item():.6f}")

if diff.item() < 1e-4:
    print("Verification Successful: Tiled implementation produces numerically equivalent results.")
else:
    print("Verification Failed: Check implementation.")

# Clean up
del Q_mock, K_mock, V_mock, output_tiled, output_standard
torch.cuda.empty_cache()