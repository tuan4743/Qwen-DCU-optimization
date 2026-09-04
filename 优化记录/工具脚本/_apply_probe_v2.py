#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fill_alloc_probe v2 instrumentation patch (one-shot, run inside container).

v1 在 capture_model 内 begin/end → capture 期快照 exact_256MB=0 (256MB buffer 不是
capture 期分配). v2 改为全程记录 + 多检查点快照:
  - begin_lifetime_probe() 插在 load_model() 末尾 (EngineCore 初始化早期, capture/profile 之前)
  - checkpoint_snapshot(tag) 插在 capture_model 入口 (capture前) 与出口 (capture后)

Idempotent: safe to run multiple times (先清旧 v1 桩再插 v2 桩, 已插则跳过).

调用方: 容器内 `python _apply_probe_v2.py` (经 PYTHONPATH 找到 fill_alloc_probe).
影响 API: gpu_model_runner.py 的 load_model() 与 capture_model().
数据产物: /public/home/xdzs2026_c150/zya/logs/fill_alloc_probe_ckpt{N}_{tag}.jsonl
"""
import sys
import re as _re

P = "/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/v1/worker/gpu_model_runner.py"
s = open(P, encoding="utf-8").read()

# ---------- Step 1: 清除所有 v1 桩 ----------
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

# ---------- Step 2: v2 lifetime begin 插在 load_model() 末尾 ----------
# load_model() 以 "get_offloader().post_init()" 结束 (紧邻下一个 def _get_eagle3_aux_layers_from_config)
LOAD_TAIL_ANCHOR = "        get_offloader().post_init()\n"
V2_MARKER = "begin_lifetime_probe"
if V2_MARKER in s:
    print("v2 begin already present, skip insert into load_model.")
else:
    LIFETIME_BEGIN = (
        "        # --- fill_alloc_probe v2: 全程记录 begin (EngineCore init 早期) ---\n"
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

# ---------- Step 3: capture_model 入口/出口插快照 ----------
entry_marker = "checkpoint_snapshot(tag='pre_capture')"
if entry_marker in s:
    print("pre_capture snapshot already present, skip.")
else:
    a = "        # Trigger CUDA graph capture for specific shapes.\n"
    CAP_ENTRY_SNAP = (
        "        # --- fill_alloc_probe v2: capture 前快照 ---\n"
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
        "        # --- fill_alloc_probe v2: capture 后快照 ---\n"
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

# 语法校验
import ast
try:
    ast.parse(open(P, encoding="utf-8").read())
    print("SYNTAX_OK")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {e}", file=sys.stderr)
    sys.exit(2)
print("DONE v2 patch.")
