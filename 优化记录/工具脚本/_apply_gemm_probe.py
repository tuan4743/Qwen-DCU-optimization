#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GDN 投影 GEMM 探针 patch(一次性,容器内运行)。

把 `_gemm_probe.py` 的 record_function 标注桩注入
`qwen3_next.py:Qwen3NextGatedDeltaNet.forward`(line 634)的三个投影 GEMM 调用点。

桩点(forward 内,line 650/651/689):
    projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)        # line 650
    projected_states_ba, _ = self.in_proj_ba(hidden_states)            # line 651
    ...
    output[:num_tokens], _ = self.out_proj(core_attn_out)              # line 689

注入后(forward_native / cudagraph replay 都会过 forward):
    from gemm_probe import label_in_proj_qkvz
    with label_in_proj_qkvz():
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
    ...

Idempotent: 已注入则跳过(检测 `GEMM_PROBE::` 标记)。

调用方: 容器内 `python _apply_gemm_probe.py`(gemm_probe.py 需在 PYTHONPATH)。
影响 API: qwen3_next.py 的 Qwen3NextGatedDeltaNet.forward。
数据产物: 无独立文件,label 进 torch profiler trace(user_annotation 事件)。
"""
import sys

P = "/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/model_executor/models/qwen3_next.py"
s = open(P, encoding="utf-8").read()

PROBE_MARK = "GEMM_PROBE::"

# ---------- Step 0: 幂等检查 ----------
if PROBE_MARK in s:
    print("gemm_probe already applied, skip.")
    sys.exit(0)

# ---------- Step 1: 三个投影 GEMM 调用点逐个包裹 ----------
# 注意: 用原始行内容做精确匹配,缩进与源码一致(8 空格)。

# 1.1 in_proj_qkvz (line 650)
QKVZ_OLD = "        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
QKVZ_NEW = (
    "        # --- gemm_probe: 包住 in_proj_qkvz 投影 GEMM ---\n"
    "        from gemm_probe import label_in_proj_qkvz\n"
    "        with label_in_proj_qkvz():\n"
    "            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "        # --- end gemm_probe in_proj_qkvz ---\n"
)
if QKVZ_OLD not in s:
    print("ERROR: in_proj_qkvz anchor not found.", file=sys.stderr)
    sys.exit(1)
s = s.replace(QKVZ_OLD, QKVZ_NEW, 1)
print("Wrapped in_proj_qkvz with label_in_proj_qkvz().")

# 1.2 in_proj_ba (line 651)
BA_OLD = "        projected_states_ba, _ = self.in_proj_ba(hidden_states)\n"
BA_NEW = (
    "        # --- gemm_probe: 包住 in_proj_ba 投影 GEMM ---\n"
    "        from gemm_probe import label_in_proj_ba\n"
    "        with label_in_proj_ba():\n"
    "            projected_states_ba, _ = self.in_proj_ba(hidden_states)\n"
    "        # --- end gemm_probe in_proj_ba ---\n"
)
if BA_OLD not in s:
    print("ERROR: in_proj_ba anchor not found.", file=sys.stderr)
    sys.exit(1)
s = s.replace(BA_OLD, BA_NEW, 1)
print("Wrapped in_proj_ba with label_in_proj_ba().")

# 1.3 out_proj (line 689)
OUT_OLD = "        output[:num_tokens], _ = self.out_proj(core_attn_out)\n"
OUT_NEW = (
    "        # --- gemm_probe: 包住 out_proj 投影 GEMM ---\n"
    "        from gemm_probe import label_out_proj\n"
    "        with label_out_proj():\n"
    "            output[:num_tokens], _ = self.out_proj(core_attn_out)\n"
    "        # --- end gemm_probe out_proj ---\n"
)
if OUT_OLD not in s:
    print("ERROR: out_proj anchor not found.", file=sys.stderr)
    sys.exit(1)
s = s.replace(OUT_OLD, OUT_NEW, 1)
print("Wrapped out_proj with label_out_proj().")

open(P, "w", encoding="utf-8").write(s)

# 语法校验
import ast
try:
    ast.parse(open(P, encoding="utf-8").read())
    print("SYNTAX_OK")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {e}", file=sys.stderr)
    sys.exit(2)
print("DONE gemm_probe patch.")
