# SPDX-License-Identifier: Apache-2.0
# [终稿 5.3 重建骨架] 融合 Chunk 预处理:cumsum(+KKT(+tril(+WU 融核入库,默认关闭。
# 说明文档截图(image 1)只给出 env 门控行(os.environ.setdefault("VLLM_ROCM_GDN_FUSED_PREPROC","0")),
# 未给出 kernel 实现;本文件为接口骨架,实现细节需人工按"融合 chunk 预处理"语义补齐:
#   - cumsum : 序列内 token 位置累加
#   - KKT    : chunk 内 K^T K 上三角构造
#   - tril   : 因果三角掩码
#   - WU     : W·U 递推更新
# 建议参考 fla 生态 fused chunk 预处理 kernel 实现,并把结果接入 fla/ops/chunk.py 的 chunk 前处理段。
"""Fused chunk preprocessing for gated delta rule (skeleton)."""
import os

_ENABLED = os.environ.get("VLLM_ROCM_GDN_FUSED_PREPROC", "0") == "1"


def fused_chunk_preprocess(*args, **kwargs):  # pragma: no cover - 未实现
    """TODO: 按上述四个算子融合语义实现单 kernel 前处理。"""
    raise NotImplementedError(
        "fused_chunk_preprocessing 为 5.3 重建骨架,截图未提供 kernel 实现;默认关闭。"
    )
