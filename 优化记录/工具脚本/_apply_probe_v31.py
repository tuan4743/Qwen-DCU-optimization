#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fill_alloc_probe v3.1 cleanup: move the post_first_req stub that landed in
sample_tokens() to before execute_model()'s sync return output.

v3's SYNC_RET_ANCHOR='            return output\n' first occurs in sample_tokens()
(the PP pass-through early return), not in execute_model's main sync return,
so the sync stub went into sample_tokens. v3.1:
  - Deletes the block inside sample_tokens() (located via its preceding line
    'output.kv_connector_output = ...').
  - Re-inserts before execute_model's sync anchor
    ('if not self.use_async_scheduling:\n            return output').

Usage
-----
    python /tmp/_apply_probe_v31.py     # inside the container

Generalization notes
--------------------
- Same general lesson as v3: anchor on a UNIQUE context (preceding line +
  the unless-branch anchor), never a bare string that may occur elsewhere.
- Edits the target file in place; ast.parse gate included. Parameterize
  TARGET for other trees.
"""
import sys
import ast

P = "/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/v1/worker/gpu_model_runner.py"
s = open(P, encoding="utf-8").read()

# ---------- 1. 删除误插在 sample_tokens 的 sync 桩 ----------
# 该块上文是 "            output.kv_connector_output = kv_connector_output\n"
SAMPLE_MISPLACED = (
    "            # --- fill_alloc_probe v3: post_first_req 快照 (sync path, once) ---\n"
    "            try:\n"
    "                if not getattr(self, '_probe_first_req_done', False):\n"
    "                    self._probe_first_req_done = True\n"
    "                    from fill_alloc_probe import checkpoint_snapshot\n"
    "                    checkpoint_snapshot(tag='post_first_req')\n"
    "            except Exception as _e:\n"
    "                print('[fill_alloc_probe] POST_FIRST_REQ_FAIL', repr(_e), flush=True)\n"
    "            # --- end fill_alloc_probe v3 post_first_req ---\n"
    "            return output\n"
)
if SAMPLE_MISPLACED in s:
    s = s.replace(SAMPLE_MISPLACED, "            return output\n", 1)
    print("Removed misplaced post_first_req sync block from sample_tokens.")
else:
    print("Misplaced sync block not found (already clean?).")

# ---------- 2. 在 execute_model 的 sync return output 前插桩 ----------
EXEC_SYNC_ANCHOR = (
    "        if not self.use_async_scheduling:\n"
    "            return output\n"
)
EXEC_SYNC_STUB = (
    "        if not self.use_async_scheduling:\n"
    "            # --- fill_alloc_probe v3: post_first_req 快照 (sync path, once) ---\n"
    "            try:\n"
    "                if not getattr(self, '_probe_first_req_done', False):\n"
    "                    self._probe_first_req_done = True\n"
    "                    from fill_alloc_probe import checkpoint_snapshot\n"
    "                    checkpoint_snapshot(tag='post_first_req')\n"
    "            except Exception as _e:\n"
    "                print('[fill_alloc_probe] POST_FIRST_REQ_FAIL', repr(_e), flush=True)\n"
    "            # --- end fill_alloc_probe v3 post_first_req ---\n"
    "            return output\n"
)
idx = s.find(EXEC_SYNC_ANCHOR)
already = (idx >= 0 and "checkpoint_snapshot(tag='post_first_req')" in s[idx:idx+len(EXEC_SYNC_STUB)])
if "checkpoint_snapshot(tag='post_first_req')" in s and not already:
    # async 桩可能已存在但 sync 未就位; 仅当 EXEC_SYNC_ANCHOR 处无桩时插入
    pass
if EXEC_SYNC_ANCHOR in s and not already:
    s = s.replace(EXEC_SYNC_ANCHOR, EXEC_SYNC_STUB, 1)
    print("Inserted post_first_req sync stub into execute_model (before return output).")
else:
    print("execute_model sync stub already present or anchor not found.")

open(P, "w", encoding="utf-8").write(s)

try:
    ast.parse(open(P, encoding="utf-8").read())
    print("SYNTAX_OK")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {e}", file=sys.stderr)
    sys.exit(2)

lines = open(P, encoding="utf-8").read().splitlines()
def find_enclosing_def(t):
    for i in range(t, 0, -1):
        if lines[i-1].startswith("    def "):
            return lines[i-1].strip()
    return "?"
for i, ln in enumerate(lines, 1):
    if "checkpoint_snapshot(tag='pre_capture')" in ln:
        print(f"  pre_capture   @ line {i} in {find_enclosing_def(i)}")
    if "checkpoint_snapshot(tag='post_capture')" in ln:
        print(f"  post_capture  @ line {i} in {find_enclosing_def(i)}")
    if "checkpoint_snapshot(tag='post_first_req')" in ln:
        print(f"  post_first_req @ line {i} in {find_enclosing_def(i)}")
print("DONE v3.1.")
