#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relocate fill_alloc_probe instrumentation (one-shot patch run inside container):
- Remove begin/end blocks from profile_cudagraph_memory (skipped on ROCm/DCU
  because gpu_worker.py:399 gates it with `not current_platform.is_rocm()`).
- Insert begin block into the REAL capture_model() (def @ ~5702), right before
  its `set_cudagraph_capturing_enabled(True)`.
- The end block already sits inside capture_model after graph_capture; keep it.
Idempotent: safe to run multiple times.

Usage
-----
    python _fix_probe_location.py      # inside the container

Generalization notes
--------------------
- Generic relocate pattern: strip misplaced stub blocks, then re-insert at
  the verified target anchor. Parameterize TARGET (runner file) and the
  anchors for other vLLM versions; verify function ownership of an anchor
  with __probe_dcu.py before patching.
"""
import sys

P = "/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/v1/worker/gpu_model_runner.py"
s = open(P, encoding="utf-8").read()

BEGIN_BLOCK = (
    "        # --- fill_alloc_probe (P0-策略1): 抓 capture 期 256MB int32 分配栈 ---\n"
    "        try:\n"
    "            from fill_alloc_probe import begin_capture_probe\n"
    "            begin_capture_probe()\n"
    "            print('[fill_alloc_probe] capture_model ENTERED', flush=True)\n"
    "        except Exception as _e:\n"
    "            print('[fill_alloc_probe] BEGIN_FAIL', repr(_e), flush=True)\n"
    "        # --- end fill_alloc_probe begin ---\n"
    "        set_cudagraph_capturing_enabled(True)\n"
)

END_MARKER_BEGIN = "        # --- fill_alloc_probe: capture 结束, dump 分配栈 ---"

# --- Step 1: Remove begin block from profile_cudagraph_memory ---
idx = s.find(BEGIN_BLOCK)
if idx == -1:
    print("BEGIN_BLOCK not found - already removed?", file=sys.stderr)
else:
    block_without_settrue = BEGIN_BLOCK[: BEGIN_BLOCK.rfind("set_cudagraph_capturing_enabled(True)\n")]
    if s.startswith(block_without_settrue, idx):
        s = s[:idx] + s[idx + len(block_without_settrue):]
        print("Removed begin block from profile_cudagraph_memory (first occurrence).")
    else:
        print("WARN: BEGIN_BLOCK structure mismatch - left untouched.", file=sys.stderr)

# --- Step 2: Insert begin block into the REAL capture_model() ---
real_anchor = (
    "        # Capture the large shapes first so that the smaller shapes\n"
    "        # can reuse the memory pool allocated for the large shapes.\n"
    "        set_cudagraph_capturing_enabled(True)\n"
    "        with self._freeze_gc(), graph_capture(device=self.device):\n"
)
if real_anchor not in s:
    print("ERROR: real_anchor for capture_model not found!", file=sys.stderr)
    sys.exit(1)

probe_marker_in_capture = (
    "        # --- fill_alloc_probe (P0-策略1): 抓 capture 期 256MB int32 分配栈 ---\n"
    "        try:\n"
    "            from fill_alloc_probe import begin_capture_probe\n"
)
n_probe = s.count(probe_marker_in_capture)
print(f"probe begin markers currently present: {n_probe}")

if n_probe == 0:
    insert_text = (
        "        # --- fill_alloc_probe (P0-策略1): 抓 capture 期 256MB int32 分配栈 ---\n"
        "        try:\n"
        "            from fill_alloc_probe import begin_capture_probe\n"
        "            begin_capture_probe()\n"
        "            print('[fill_alloc_probe] capture_model ENTERED', flush=True)\n"
        "        except Exception as _e:\n"
        "            print('[fill_alloc_probe] BEGIN_FAIL', repr(_e), flush=True)\n"
        "        # --- end fill_alloc_probe begin ---\n"
    )
    s = s.replace(real_anchor, insert_text + real_anchor, 1)
    print("Inserted begin block into real capture_model().")
elif n_probe == 1:
    print("begin block already in capture_model (or one place) - verifying position.")
else:
    print(f"WARN: {n_probe} probe markers - unexpected, leaving as is.", file=sys.stderr)

if END_MARKER_BEGIN in s:
    print("end block present (will be checked for location).")
else:
    print("WARN: end block marker not found!", file=sys.stderr)

open(P, "w", encoding="utf-8").write(s)
print("DONE. File rewritten.")
