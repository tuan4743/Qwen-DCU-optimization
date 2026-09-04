#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fill_alloc_probe v3 instrumentation patch (one-shot, run inside container).

修正 v2 的关键 bug:
  v2 用 set_cudagraph_capturing_enabled(False) 作 anchor 插 post_capture 快照,
  但该字符串在文件里 *第一次出现* 在 profile_cudagraph_memory() (line ~5682),
  而非 capture_model() (line ~5754)。DCU/ROCm 上 profile_cudagraph_memory 被
  gpu_worker.py:398 门控掉 (not is_rocm) → post_capture 桩跑不到, 故只有
  pre_capture (在 capture_model 入口) 出现, post_capture 永不触发。

v3 做三件事 (在 v2 已 patch 的源码上叠加, 幂等):
  1) 把误插在 profile_cudagraph_memory() 里的 post_capture 块删掉, 改插到
     capture_model() 真正的出口 (return cuda_graph_size 之前, 用更唯一的 anchor).
  2) 在 execute_model() 末尾加 post_first_req 检查点 (只取一次), 用于确认
     稳态 serving 期是否仍有 256MB alloc (P0.5).
  3) begin_lifetime_probe (load_model 末尾) 与 pre_capture (capture_model 入口)
     已在 v2 正确就位, 不动.

调用方: 容器内 `python _apply_probe_v3.py` (PYTHONPATH 含 /public/home/.../zya).
影响 API: gpu_model_runner.py 的 capture_model() 与 execute_model().
"""
import sys
import ast

P = "/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/v1/worker/gpu_model_runner.py"
s = open(P, encoding="utf-8").read()

POST_CAPTURE_BLOCK = (
    "        # --- fill_alloc_probe v2: capture 后快照 ---\n"
    "        try:\n"
    "            from fill_alloc_probe import checkpoint_snapshot\n"
    "            checkpoint_snapshot(tag='post_capture')\n"
    "        except Exception as _e:\n"
    "            print('[fill_alloc_probe] POST_CAPTURE_FAIL', repr(_e), flush=True)\n"
    "        # --- end fill_alloc_probe v2 post_capture ---\n"
)

# ---------- Step 1: 删除所有误插的 post_capture 块 (可能 1~2 处) ----------
n_removed = s.count(POST_CAPTURE_BLOCK)
s = s.replace(POST_CAPTURE_BLOCK, "")
print(f"Removed {n_removed} misplaced post_capture block(s).")

# ---------- Step 2: 把 post_capture 重插到 capture_model() 真正出口 ----------
# capture_model() 的最后几行 (DCU 上唯一会跑的 capture 出口):
#     logger.info_once(
#         "Graph capturing finished in %.0f secs, took %.2f GiB",
#         ...
#         scope="local",
#     )
#     return cuda_graph_size
# 用 "        return cuda_graph_size\n" 作 anchor (文件里唯一, profile_cudagraph_memory
# 用的是 return int(total_estimate)).
CAP_EXIT_ANCHOR = "        return cuda_graph_size\n"
POST_CAPTURE_V3 = (
    "        # --- fill_alloc_probe v3: capture_model 出口快照 (DCU 实际跑的 capture) ---\n"
    "        try:\n"
    "            from fill_alloc_probe import checkpoint_snapshot\n"
    "            checkpoint_snapshot(tag='post_capture')\n"
    "        except Exception as _e:\n"
    "            print('[fill_alloc_probe] POST_CAPTURE_FAIL', repr(_e), flush=True)\n"
    "        # --- end fill_alloc_probe v3 post_capture ---\n"
)
if "checkpoint_snapshot(tag='post_capture')" in s:
    print("post_capture already present somewhere, skip re-insert.")
elif CAP_EXIT_ANCHOR in s:
    s = s.replace(CAP_EXIT_ANCHOR, POST_CAPTURE_V3 + CAP_EXIT_ANCHOR, 1)
    print("Inserted post_capture at capture_model exit (return cuda_graph_size).")
else:
    print("ERROR: capture_model exit anchor (return cuda_graph_size) not found.", file=sys.stderr)
    sys.exit(1)

# ---------- Step 3: execute_model() 末尾加 post_first_req (只取一次) ----------
# execute_model 末尾两处 return: ~4079 `return output` (非 async, 在 if not use_async_scheduling 块内),
# ~4102 `return async_output` (函数末尾). 用 instance flag self._probe_first_req_done 保证只快照一次.
# 安全做法: 直接在 "return output" / "return async_output" 行前插独立快照块, 不动原 if 结构.
SYNC_RET_ANCHOR = "            return output\n"
FIRST_REQ_BLOCK_SYNC = (
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
ASYNC_RET_ANCHOR = "        return async_output\n"
FIRST_REQ_BLOCK_ASYNC_TAIL = (
    "        # --- fill_alloc_probe v3: post_first_req 快照 (async path, once) ---\n"
    "        try:\n"
    "            if not getattr(self, '_probe_first_req_done', False):\n"
    "                self._probe_first_req_done = True\n"
    "                from fill_alloc_probe import checkpoint_snapshot\n"
    "                checkpoint_snapshot(tag='post_first_req')\n"
    "        except Exception as _e:\n"
    "            print('[fill_alloc_probe] POST_FIRST_REQ_FAIL', repr(_e), flush=True)\n"
    "        # --- end fill_alloc_probe v3 post_first_req ---\n"
    "        return async_output\n"
)
if "checkpoint_snapshot(tag='post_first_req')" in s:
    print("post_first_req already present, skip.")
else:
    # return output 在文件里可能多处 (含 pooling path 的 return model_runner_output 等);
    # 用 12 空格缩进的 "return output" (execute_model async 分支前的 if 块内) 作唯一锚.
    if SYNC_RET_ANCHOR in s:
        s = s.replace(SYNC_RET_ANCHOR, FIRST_REQ_BLOCK_SYNC, 1)
        print("Inserted post_first_req before sync return output (execute_model).")
    else:
        print("WARN: execute_model sync return anchor (12-space 'return output') not found.")
    if ASYNC_RET_ANCHOR in s:
        s = s.replace(ASYNC_RET_ANCHOR, FIRST_REQ_BLOCK_ASYNC_TAIL, 1)
        print("Inserted post_first_req before async return (execute_model).")
    else:
        print("WARN: execute_model async return anchor not found.")

open(P, "w", encoding="utf-8").write(s)

# 语法校验
try:
    ast.parse(open(P, encoding="utf-8").read())
    print("SYNTAX_OK")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {e}", file=sys.stderr)
    sys.exit(2)

# 校验桩位置: post_capture 应在 return cuda_graph_size 前; 列出每桩所属 def
lines = open(P, encoding="utf-8").read().splitlines()
def find_enclosing_def(target_line_1based):
    for i in range(target_line_1based, 0, -1):
        if lines[i-1].startswith("    def "):
            return lines[i-1].strip()
    return "?"
for i, ln in enumerate(lines, 1):
    if "checkpoint_snapshot(tag='post_capture')" in ln:
        print(f"  post_capture stub @ line {i} in def: {find_enclosing_def(i)}")
    if "checkpoint_snapshot(tag='post_first_req')" in ln:
        print(f"  post_first_req stub @ line {i} in def: {find_enclosing_def(i)}")
print("DONE v3 patch.")
