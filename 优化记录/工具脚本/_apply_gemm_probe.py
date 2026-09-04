#!/usr/bin/env python3
"""One-shot patch: inject GEMM record_function probe stubs into a model file.

Purpose
-------
Wrap the three projection GEMM call sites of
``Qwen3NextGatedDeltaNet.forward`` (e.g. qwen3_next.py) with the
``_gemm_probe.py`` record_function labels, so the offline parser
(_parse_gemm_probe.py) can attribute GPU kernels to each projection.

Probe sites (inside forward):
    projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
    projected_states_ba,   _ = self.in_proj_ba(hidden_states)
    output[:num_tokens],   _ = self.out_proj(core_attn_out)

After injection (both forward_native and cudagraph replay call forward):
    from gemm_probe import label_in_proj_qkvz
    with label_in_proj_qkvz():
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)

Idempotent: skips if the ``GEMM_PROBE::`` marker is already present.

Usage
-----
    python _apply_gemm_probe.py        # inside the container, gemm_probe on PYTHONPATH

Generalization notes
--------------------
- Parameterize: change ``TARGET`` (the file to patch) and the three
  OLD/NEW anchor pairs below for any model file / any operator; the pattern
  (exact-line anchor + marker idempotency + ast.parse check) is generic.
- Anchor matching is exact on source text (8-space indent here); adjust
  whitespace for other files. Verify with ``ast.parse`` before shipping.
"""
import sys

TARGET = "/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/model_executor/models/qwen3_next.py"
P = TARGET
s = open(P, encoding="utf-8").read()

PROBE_MARK = "GEMM_PROBE::"

# ---------- Step 0: idempotency check ----------
if PROBE_MARK in s:
    print("gemm_probe already applied, skip.")
    sys.exit(0)

# ---------- Step 1: wrap the three projection GEMM call sites ----------
# NOTE: match on the exact original line text; indentation must match source (8 spaces).

# 1.1 in_proj_qkvz
QKVZ_OLD = "        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
QKVZ_NEW = (
    "        # --- gemm_probe: wrap in_proj_qkvz projection GEMM ---\n"
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

# 1.2 in_proj_ba
BA_OLD = "        projected_states_ba, _ = self.in_proj_ba(hidden_states)\n"
BA_NEW = (
    "        # --- gemm_probe: wrap in_proj_ba projection GEMM ---\n"
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

# 1.3 out_proj
OUT_OLD = "        output[:num_tokens], _ = self.out_proj(core_attn_out)\n"
OUT_NEW = (
    "        # --- gemm_probe: wrap out_proj projection GEMM ---\n"
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

# syntax check
import ast
try:
    ast.parse(open(P, encoding="utf-8").read())
    print("SYNTAX_OK")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {e}", file=sys.stderr)
    sys.exit(2)
print("DONE gemm_probe patch.")
