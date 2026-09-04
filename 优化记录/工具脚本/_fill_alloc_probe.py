"""fill_alloc_probe.py - locate 256MB int32 ([4096,16384]) alloc via _record_memory_history.

策略 v2 (P0-策略1, 2026-06-29 修正):
  capture 期快照 exact_256MB=0 → 256MB buffer 不是 capture 期分配的。
  fill 是 cudagraph capture 期录进图、replay 期回放;buffer 的"分配"发生在更早
  (init / profile_run / warmup / capture 之前)。所以要全程记录、多检查点快照,
  抓到那一次 256MB int32 分配的 Python 栈。

机制: _record_memory_history 开启后, 所有 torch 分配带 Python 栈被记录;
       _snapshot() 取全量 dict. 多检查点各快照一次, 任一抓到 256MB 栈即成功。

API (2026-06-29 selfcheck 钉死):
  - _record_memory_history(enabled="all", context="all", stacks="python") 开启
  - _snapshot() 取数据 (dict{segments, device_traces, ...}; device_traces=list-per-device,
    每个 device=list[event]; event.keys 含 action/addr/size/stream/time_us/frames;
    alloc 类 action = segment_alloc/alloc/resize/realloc)
  - _record_memory_history(enabled=None) 仅停止 (返回 None, 不能拿数据)
"""
import os

try:
    from torch.cuda.memory import _record_memory_history, _snapshot
    _SUPPORTED = True
    _SUPPORTED_ERR = ""
except Exception as _e:
    _SUPPORTED = False
    _SUPPORTED_ERR = repr(_e)

# [4096,16384] int32 = 67,108,864 elems = 268435456 bytes
_TARGET_NUMEL = 4096 * 16384
_TARGET_BYTES = _TARGET_NUMEL * 4   # 256 MB

_LOG_DIR = "/public/home/xdzs2026_c150/zya/logs"
os.makedirs(_LOG_DIR, exist_ok=True)

_state = {"recording": False, "ckpt_count": 0}


def _scan_snapshot(snap, tag, log_path):
    """从 snapshot 中过滤 256MB / 近大 alloc event, 写入 log_path。返回 (total, matches, near)。"""
    traces = snap.get("device_traces", []) if isinstance(snap, dict) else []
    events = []
    for tr in traces:
        if isinstance(tr, list):
            events.extend(tr)

    total = matches = near = 0
    with open(log_path, "w") as f:
        f.write(f"[fill_alloc_probe] ckpt tag={tag} events_total={len(events)}\n")
        for i, entry in enumerate(events):
            try:
                if not isinstance(entry, dict):
                    continue
                action = entry.get("action")
                if action not in ("segment_alloc", "alloc", "resize", "realloc"):
                    continue
                size = entry.get("size", 0)
                total += 1
                if size == _TARGET_BYTES:
                    matches += 1
                    _write_match(f, i, entry, tag, exact=True)
                elif size > 128 * 1024 * 1024 and size % 4 == 0:
                    near += 1
                    _write_match(f, i, entry, tag, exact=False)
            except Exception as e:
                f.write(f"[entry {i}] PARSE_FAIL: {e!r}\n")
        f.write(f"\n[fill_alloc_probe] SUMMARY tag={tag} total_alloc_events={total} exact_256MB={matches} near_big={near}\n")
    return total, matches, near


def _write_match(f, idx, entry, tag, exact):
    frames = entry.get("frames", [])
    stack_lines = []
    user_frames = []
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        fname = fr.get("filename", "?")
        line = fr.get("line", "?")
        name = fr.get("name", "?")
        stack_lines.append(f"    {fname}:{line} in {name}")
        if "vllm" in fname or "/public/home/" in fname or "zya" in fname:
            user_frames.append(f"{fname}:{line} in {name}")

    f.write(f"\n--- MATCH[{idx}] {'EXACT_256MB' if exact else 'NEAR_BIG'} "
            f"size={entry.get('size')} action={entry.get('action')} "
            f"addr={entry.get('addr')} tag={tag} ---\n")
    f.write("USER/VLLM FRAMES (most relevant):\n")
    if user_frames:
        for uf in user_frames[:20]:
            f.write(f"  >>> {uf}\n")
    else:
        f.write("  (no vllm/user frame)\n")
    f.write("FULL STACK:\n")
    for sl in stack_lines[:40]:
        f.write(sl + "\n")


# ---------- v2: 全程记录 + 多检查点快照 ----------

def begin_lifetime_probe():
    """v2: 在 EngineCore 子进程初始化早期调用, 开启全程 recording。
    capture 期 / 运行期不再开关 recording, 由 checkpoint_snapshot 在各检查点取快照。"""
    if not _SUPPORTED:
        print(f"[fill_alloc_probe] NOT_SUPPORTED: {_SUPPORTED_ERR}", flush=True)
        return
    if _state["recording"]:
        print("[fill_alloc_probe] already recording, skip begin", flush=True)
        return
    # 仅记录 alloc 事件(不记录 free), 减开销; 仍带 python 栈
    _record_memory_history(enabled="all", context="alloc", stacks="python")
    _state["recording"] = True
    print(f"[fill_alloc_probe] LIFETIME_RECORDING_STARTED (alloc-only, python stacks)", flush=True)


def checkpoint_snapshot(tag=""):
    """v2: 在检查点(capture 前/后, 首请求前后等)调用, 取 snapshot 落盘。
    注意: _snapshot() 不停止 recording, 可多次调用。"""
    if not _SUPPORTED or not _state["recording"]:
        print(f"[fill_alloc_probe] CKPT_SKIP supported={_SUPPORTED} recording={_state['recording']} tag={tag}", flush=True)
        return
    _state["ckpt_count"] += 1
    log_path = os.path.join(_LOG_DIR, f"fill_alloc_probe_ckpt{_state['ckpt_count']}_{tag}.jsonl") \
        if tag else os.path.join(_LOG_DIR, f"fill_alloc_probe_ckpt{_state['ckpt_count']}.jsonl")
    try:
        snap = _snapshot()
    except Exception as e:
        print(f"[fill_alloc_probe] SNAPSHOT_FAIL tag={tag}: {e!r}", flush=True)
        return
    total, matches, near = _scan_snapshot(snap, tag, log_path)
    print(f"[fill_alloc_probe] CKPT#{_state['ckpt_count']} tag={tag} "
          f"total_alloc={total} exact_256MB={matches} near_big={near} log={log_path}", flush=True)


def stop_lifetime_probe(tag="final"):
    """v2: 结束 recording 并取最后一次快照。"""
    if not _SUPPORTED or not _state["recording"]:
        checkpoint_snapshot(tag)  # 仍尝试快照(若还开着)
        _record_memory_history(enabled=None)
        _state["recording"] = False
        return
    checkpoint_snapshot(tag)
    _record_memory_history(enabled=None)
    _state["recording"] = False
    print(f"[fill_alloc_probe] LIFETIME_RECORDING_STOPPED tag={tag}", flush=True)


# ---------- 兼容旧接口(capture 期一次性, 已废弃但保留不报错) ----------

def begin_capture_probe():
    """[deprecated] 旧 capture 期一次性接口。v2 改用 begin_lifetime_probe。"""
    begin_lifetime_probe()


def end_capture_probe(tag=""):
    """[deprecated] 旧 capture 期一次性接口。v2 改用 checkpoint_snapshot/stop_lifetime_probe。"""
    checkpoint_snapshot(tag)
    _record_memory_history(enabled=None)
    _state["recording"] = False


def selfcheck():
    print(f"[fill_alloc_probe] supported={_SUPPORTED} err={_SUPPORTED_ERR}", flush=True)
    print(f"[fill_alloc_probe] target_numel={_TARGET_NUMEL} target_bytes={_TARGET_BYTES} ({_TARGET_BYTES/1024/1024:.0f}MB)", flush=True)
    print(f"[fill_alloc_probe] log_dir={_LOG_DIR}", flush=True)
    if _SUPPORTED:
        import torch
        begin_lifetime_probe()
        x = torch.zeros(_TARGET_NUMEL, dtype=torch.int32, device="cuda")
        del x
        checkpoint_snapshot(tag="selfcheck")
        stop_lifetime_probe(tag="selfcheck_final")


if __name__ == "__main__":
    selfcheck()
