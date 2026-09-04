#!/usr/bin/env python3
"""One-shot patch: fill_alloc_probe v2 instrumentation (lifetime recording).

Purpose
-------
v1 bracketed capture_model with begin/end and saw exact_256MB=0 during
capture (the target buffer was NOT allocated during capture). v2 records
from process start with multi-checkpoint snapshots:
  - begin_lifetime_probe()  inserted at the END of load_model() (early in
    EngineCore init, before capture/profile),
  - checkpoint_snapshot(tag) at capture_model entry ('pre_capture') and
    exit ('post_capture').

Idempotent: strips any v1 stubs first, then inserts v2 stubs (skips if
already present).

Usage
-----
    python _apply_probe_v2.py     # inside the container, PYTHONPATH has fill_alloc_probe

Generalization notes
--------------------
- Parameterize ``TARGET`` (the runner file), the load_model tail anchor and
  the capture entry/exit anchors for other vLLM versions; the pattern is
  generic (anchor search + idempotency + ast.parse check).
- The v1 cleanup section is included so re-runs are safe on legacy trees.
"""
import sys
import re as _re

TARGET = "/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/v1/worker/gpu_model_runner.py"
P = TARGET
s = open(P, encoding="utf-8").read()

# ---------- Step 1: strip all v1 stubs ----------
V1_BEGIN = (
    "        # --- fill_alloc_probe (P0-策略1): 抓 capture 期 256MB int32 分配栈 ---\n"
    "        try:\n"
    "            from fill_alloc_probe import begin_capture_probe\n"
    "            begin_capture_probe()\n"
    "            print('[fill_alloc_probe] capture_model ENTERED', flush=True)\n"
    "        except Exception as _e:\n"
    "            print('[fill_alloc_probe] BEGIN_FAIL', repr(_e), flush=True)\n"
    "        # --- end fill_alloc_probe begin ---\n"
)
if V1_BEGIN in s:
    s = s.replace(V1_BEGIN, "", 1)
    print("Removed v1 begin block.")
else:
    print("v1 begin block not found (already clean?).")

V1_END = (
    "        # --- fill_alloc_probe: capture 结束, dump 分配栈 ---\n"
    "        try:\n"
    "            from fill_alloc_probe import end_capture_probe\n"
    "            end_capture_probe(tag='capture_model')\n"
    "        except Exception as _e:\n"
    "            print('[fill_alloc_probe] END_FAIL', repr(_e), flush=True)\n"
    "        # --- end fill_alloc_probe end ---\n"
)
if V1_END in s:
    s = s.replace(V1_END, "", 1)
    print("Removed v1 end block.")
else:
    print("v1 end block not found (already clean?).")

# ---------- Step 2: insert v2 lifetime begin at the tail of load_model() ----------
# load_model() ends with "get_offloader().post_init()" (next def is _get_eagle3_aux_layers_from_config)
LOAD_TAIL_ANCHOR = "        get_offloader().post_init()\n"
V2_MARKER = "begin_lifetime_probe"
if V2_MARKER in s:
    print("v2 begin already present, skip insert into load_model.")
else:
    LIFETIME_BEGIN = (
        "        # --- fill_alloc_probe v2: lifetime recording begin (early EngineCore init) ---\n"
        "        try:\n"
        "            from fill_alloc_probe import begin_lifetime_probe\n"
        "            begin_lifetime_probe()\n"
        "        except Exception as _e:\n"
        "            print('[fill_alloc_probe] LIFETIME_BEGIN_FAIL', repr(_e), flush=True)\n"
        "        # --- end fill_alloc_probe v2 begin ---\n"
    )
    if LOAD_TAIL_ANCHOR in s:
        s = s.replace(LOAD_TAIL_ANCHOR, LOAD_TAIL_ANCHOR + LIFETIME_BEGIN, 1)
        print("Inserted v2 begin after get_offloader().post_init() in load_model tail.")
    else:
        print("ERROR: load_model tail anchor (get_offloader().post_init) not found.", file=sys.stderr)
        sys.exit(1)

# ---------- Step 3: capture_model entry/exit snapshots ----------
entry_marker = "checkpoint_snapshot(tag='pre_capture')"
if entry_marker in s:
    print("pre_capture snapshot already present, skip.")
else:
    a = "        # Trigger CUDA graph capture for specific shapes.\n"
    CAP_ENTRY_SNAP = (
        "        # --- fill_alloc_probe v2: snapshot before capture ---\n"
        "        try:\n"
        "            from fill_alloc_probe import checkpoint_snapshot\n"
        "            checkpoint_snapshot(tag='pre_capture')\n"
        "        except Exception as _e:\n"
        "            print('[fill_alloc_probe] PRE_CAPTURE_FAIL', repr(_e), flush=True)\n"
        "        # --- end fill_alloc_probe v2 pre_capture ---\n"
    )
    if a in s:
        s = s.replace(a, a + CAP_ENTRY_SNAP, 1)
        print("Inserted pre_capture snapshot at capture_model entry.")
    else:
        print("WARN: capture_model entry anchor not found.", file=sys.stderr)

exit_marker = "checkpoint_snapshot(tag='post_capture')"
if exit_marker in s:
    print("post_capture snapshot already present, skip.")
else:
    a = "        set_cudagraph_capturing_enabled(False)\n"
    CAP_EXIT_SNAP = (
        "        # --- fill_alloc_probe v2: snapshot after capture ---\n"
        "        try:\n"
        "            from fill_alloc_probe import checkpoint_snapshot\n"
        "            checkpoint_snapshot(tag='post_capture')\n"
        "        except Exception as _e:\n"
        "            print('[fill_alloc_probe] POST_CAPTURE_FAIL', repr(_e), flush=True)\n"
        "        # --- end fill_alloc_probe v2 post_capture ---\n"
    )
    if a in s:
        s = s.replace(a, CAP_EXIT_SNAP + a, 1)
        print("Inserted post_capture snapshot at capture_model exit.")
    else:
        print("WARN: capture_model exit anchor not found.", file=sys.stderr)

open(P, "w", encoding="utf-8").write(s)

# syntax check
import ast
try:
    ast.parse(open(P, encoding="utf-8").read())
    print("SYNTAX_OK")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {e}", file=sys.stderr)
    sys.exit(2)
print("DONE v2 patch.")
