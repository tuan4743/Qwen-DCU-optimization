# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
[DCU Optimize] Runtime KV-cache helpers for gfx936 / Qwen3.5-27B.

Optimization goal: reduce decode HBM traffic via in-memory KV FP8 (no weight
offline quant). Context: 8k~16k / 16k~32k scoring buckets; DCU HBM ~1.2TB/s.
Bandwidth: FP8 KV halves KV bytes vs bf16 → more room for concurrent decode.
Precision: expect Δ typically <1% with e4m3fnuz; always validate with accuracy
script before relying on score coefficient.

Allowed competition surface: runtime KV dtype selection only (no prune /
distill / persistent quantized weight dump). Block allocate/free stays in
vllm/v1/core (locked); this module only helps dtype / Roofline accounting.
"""

from __future__ import annotations

# Preferred runtime KV dtype on Hygon DCU (gfx936) when platform.supports_fp8().
DCU_RUNTIME_KV_FP8 = "fp8_e4m3fnuz"
# Official DCU HBM peak used for Roofline notes (TB/s).
DCU_HBM_PEAK_TB_S = 1.2


def recommend_kv_cache_dtype(
    *,
    prefer_fp8: bool = True,
    supports_fp8: bool | None = None,
) -> str | None:
    """
    Return a vLLM --kv-cache-dtype string, or None to keep model/default dtype.

    Does not mutate weights or write quantized checkpoints to disk.
    """
    if not prefer_fp8:
        return None
    if supports_fp8 is None:
        try:
            from vllm.platforms import current_platform

            supports_fp8 = bool(current_platform.supports_fp8())
        except Exception:
            supports_fp8 = False
    if not supports_fp8:
        return None
    return DCU_RUNTIME_KV_FP8


def kv_bytes_per_token(
    *,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    layers_full_attn: int | None = None,
) -> int:
    """
    Approximate KV bytes written/read per token for full-attention layers.

    Qwen3.5 is hybrid: only full_attention layers use PagedAttention KV.
    Pass layers_full_attn if known; otherwise returns per-layer bytes.
    """
    per_layer = 2 * num_kv_heads * head_dim * dtype_bytes  # K + V
    if layers_full_attn is None:
        return per_layer
    return per_layer * layers_full_attn


def roofline_kv_bw_util(
    *,
    tokens_per_s: float,
    num_kv_heads: int = 4,
    head_dim: int = 256,
    dtype_bytes: int = 2,
    layers_full_attn: int = 16,
    hbm_peak_tb_s: float = DCU_HBM_PEAK_TB_S,
) -> float:
    """
    Rough KV-only bandwidth utilization vs DCU HBM peak.

    Returns fraction in [0, inf); >1 means estimate exceeds peak (other traffic).
    For notes / A/B only — not used in the hot path.
    """
    if tokens_per_s <= 0 or hbm_peak_tb_s <= 0:
        return 0.0
    bytes_per_tok = kv_bytes_per_token(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        layers_full_attn=layers_full_attn,
    )
    # Decode reads full context roughly once per token per layer (order-of-mag).
    # Caller should scale by seq_len for absolute bytes/s if needed.
    gb_s = tokens_per_s * bytes_per_tok / 1e9
    peak_gb_s = hbm_peak_tb_s * 1000.0
    return gb_s / peak_gb_s


__all__ = [
    "DCU_RUNTIME_KV_FP8",
    "DCU_HBM_PEAK_TB_S",
    "recommend_kv_cache_dtype",
    "kv_bytes_per_token",
    "roofline_kv_bw_util",
]
