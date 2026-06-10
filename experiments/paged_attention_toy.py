"""
Toy paged attention prototype for Phase 3.

Goal:
    Verify that storing K/V cache in fixed-size blocks does not change
    the mathematical result of attention.

This file compares:
    1. normal attention over contiguous K/V tensors
    2. paged attention over a block pool + block table

Expected result:
    paged attention output should match normal attention output
    up to small floating point error.

This is a correctness prototype, not a performance implementation.
It does not use Qwen, FastAPI, the real scheduler, CUDA, or Triton.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Standard single-token decode attention.

    q:
        [num_heads, head_dim]

    k:
        [seq_len, num_heads, head_dim]

    v:
        [seq_len, num_heads, head_dim]

    returns:
        [num_heads, head_dim]
    """
    head_dim = q.shape[-1]

    # scores: [num_heads, seq_len]
    scores = torch.einsum("hd,thd->ht", q, k) / math.sqrt(head_dim)

    # weights: [num_heads, seq_len]
    weights = F.softmax(scores, dim=-1)

    # out: [num_heads, head_dim]
    out = torch.einsum("ht,thd->hd", weights, v)

    return out


def write_kv_to_blocks(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Store contiguous K/V tensors into fixed-size blocks.

    k:
        [seq_len, num_heads, head_dim]

    v:
        [seq_len, num_heads, head_dim]

    block_size:
        number of tokens per block

    returns:
        block_pool_k:
            [num_blocks, block_size, num_heads, head_dim]

        block_pool_v:
            [num_blocks, block_size, num_heads, head_dim]

        block_table:
            logical block index -> physical block index
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if k.shape != v.shape:
        raise ValueError(f"k and v must have same shape, got {k.shape} and {v.shape}")

    seq_len, num_heads, head_dim = k.shape
    num_blocks = math.ceil(seq_len / block_size)

    block_pool_k = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        head_dim,
        dtype=k.dtype,
        device=k.device,
    )
    block_pool_v = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        head_dim,
        dtype=v.dtype,
        device=v.device,
    )

    block_table: list[int] = []

    for physical_block_id in range(num_blocks):
        start = physical_block_id * block_size
        end = min(start + block_size, seq_len)
        length = end - start

        block_pool_k[physical_block_id, :length] = k[start:end]
        block_pool_v[physical_block_id, :length] = v[start:end]

        block_table.append(physical_block_id)

    return block_pool_k, block_pool_v, block_table


def read_kv_from_blocks(
    block_pool_k: torch.Tensor,
    block_pool_v: torch.Tensor,
    block_table: list[int],
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct logical contiguous K/V tensors from block pool + block table.

    This function intentionally reconstructs full K/V.
    Later, we will replace this with true block-by-block attention.

    returns:
        k:
            [seq_len, num_heads, head_dim]

        v:
            [seq_len, num_heads, head_dim]
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    k_chunks = []
    v_chunks = []

    remaining = seq_len

    for physical_block_id in block_table:
        if remaining <= 0:
            break

        take = min(block_size, remaining)

        k_block = block_pool_k[physical_block_id, :take]
        v_block = block_pool_v[physical_block_id, :take]

        k_chunks.append(k_block)
        v_chunks.append(v_block)

        remaining -= take

    if remaining != 0:
        raise ValueError(
            f"block_table does not contain enough tokens: {remaining} tokens missing"
        )

    reconstructed_k = torch.cat(k_chunks, dim=0)
    reconstructed_v = torch.cat(v_chunks, dim=0)

    return reconstructed_k, reconstructed_v


def paged_attention(
    q: torch.Tensor,
    block_pool_k: torch.Tensor,
    block_pool_v: torch.Tensor,
    block_table: list[int],
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """
    Toy paged attention.

    Current version:
        1. read logical K/V from blocks
        2. call normal_attention

    This proves correctness of block storage + block table reconstruction.

    Later version:
        compute attention directly from blocks without reconstructing full K/V.
    """
    k, v = read_kv_from_blocks(
        block_pool_k=block_pool_k,
        block_pool_v=block_pool_v,
        block_table=block_table,
        seq_len=seq_len,
        block_size=block_size,
    )

    return normal_attention(q=q, k=k, v=v)


def main() -> None:
    torch.manual_seed(0)

    seq_len = 10
    num_heads = 4
    head_dim = 8
    block_size = 4

    q = torch.randn(num_heads, head_dim)
    k = torch.randn(seq_len, num_heads, head_dim)
    v = torch.randn(seq_len, num_heads, head_dim)

    normal_out = normal_attention(q=q, k=k, v=v)

    block_pool_k, block_pool_v, block_table = write_kv_to_blocks(
        k=k,
        v=v,
        block_size=block_size,
    )

    paged_out = paged_attention(
        q=q,
        block_pool_k=block_pool_k,
        block_pool_v=block_pool_v,
        block_table=block_table,
        seq_len=seq_len,
        block_size=block_size,
    )

    max_diff = (normal_out - paged_out).abs().max().item()

    print("=== Toy Paged Attention Check ===")
    print(f"seq_len: {seq_len}")
    print(f"num_heads: {num_heads}")
    print(f"head_dim: {head_dim}")
    print(f"block_size: {block_size}")
    print(f"block_table: {block_table}")
    print(f"block_pool_k shape: {tuple(block_pool_k.shape)}")
    print(f"block_pool_v shape: {tuple(block_pool_v.shape)}")
    print(f"normal_out shape: {tuple(normal_out.shape)}")
    print(f"paged_out shape: {tuple(paged_out.shape)}")
    print(f"max diff: {max_diff}")

    assert torch.allclose(normal_out, paged_out, atol=1e-6)

    print("Paged attention toy check passed.")


if __name__ == "__main__":
    main()