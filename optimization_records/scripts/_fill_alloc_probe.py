"""Locate a target-size alloc (e.g. the 256MB int32 [4096,16384] buffer) via
torch.cuda.memory._record_memory_history snapshots.

Purpose
-------
Find WHERE a specific-size GPU allocation came from. Classic use case: a
256MB int32 memset (``FillFunctor<int>``) shows up in profiles but no vLLM
buffer matches its shape; this probe records every allocation with Python
stacks across the whole process lifetime and snapshots at checkpoints, so
the exact alloc site (file:line in the Python stack) can be pinned down.

Strategy v2 (P0-fix, 2026-06-29)
--------------------------------
- Capturing a snapshot during cudagraph capture found exact_256MB=0: the
  buffer was NOT allocated during capture — the fill is replayed from the
  graph, and its allocation happened earlier (init / profile_run / warmup /
  before capture). So: record from process start, snapshot at multiple
  checkpoints, catch that one alloc whenever it fires.

Mechanics (verified on torch 2.10 ROCm, 2026-06-29)
---------------------------------------------------
- ``_record_memory_history(enabled="all", context="alloc", stacks="python")``
  records allocs with Python stacks (context="alloc" to skip free events).
- ``_snapshot()`` returns dict{segments, device_traces, ...}; device_traces
  is list-per-device, each a list of events: action/addr/size/stream/
  time_us/frames. Alloc actions: segment_alloc/alloc/resize/realloc.
- ``_record_memory_history(enabled=None)`` only stops (returns None).

API (called by stubs injected into the runner, see _patch_*.py)
----------------------------------------------------------------
    begin_lifetime_probe()      # start recording early in EngineCore init
    checkpoint_snapshot(tag="") # snapshot + write scan log (call anywhere)
    stop_lifetime_probe(tag="") # final snapshot + stop
    selfcheck()                 # run standalone smoke test

Run standalone (self-test)
--------------------------
    python _fill_alloc_probe.py

Generalization notes
--------------------
- Configure via env vars: ``FILL_ALLOC_TARGET_BYTES`` (default 268435456,
  i.e. 256MB) and ``FILL_ALLOC_LOG_DIR`` (default the project log dir).
- The "user frame" filter (vllm / /public/home/ / zya path substrings) is
  project-specific: remove it or extend it for your environment in
  ``_write_match`` — the full stack is always written anyway.
- The deprecated capture-phase API is kept for compatibility.
"""
import os

TARGET_BYTES = int(os.environ.get("FILL_ALLOC_TARGET_BYTES", "268435456"))
LOG_DIR = os.environ.get("FILL_ALLOC_LOG_DIR", "/public/home/xdzs2026_c150/zya/logs")

try:
    from torch.cuda.memory import _record_memory_history, _snapshot
    _SUPPORTED = True
    _SUPPORTED_ERR = ""
except Exception as _e:
    _SUPPORTED = False
    _SUPPORTED_ERR = repr(_e)

# [4096,16384] int32 = 67,108,864 elems = 268435456 bytes
_TARGET_NUMEL = TARGET_BYTES // 4

os.makedirs(LOG_DIR, exist_ok=True)

_state = {"recording": False, "ckpt_count": 0}


def _scan_snapshot(snap, tag, log_path):
    """Filter 256MB / near-big alloc events from a snapshot; returns (total, matches, near)."""
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
                if size == TARGET_BYTES:
                    matches += 1
                    _write_match(f, i, entry, tag, exact=True)
                elif size > 128 * 1024 * 1024 and size % 4 == 0:
                    near += 1
                    _write_match(f, i, entry, tag, exact=False)
            except Exception as e:
                f.write(f"[entry {i}] PARSE_FAIL: {e!r}\n")
        f.write(f"\n[fill_alloc_probe] SUMMARY tag={tag} total_alloc_events={total} "
                f"exact_{TARGET_BYTES//(1024*1024)}MB={matches} near_big={near}\n")
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
        # project-specific heuristic; the FULL stack is always written below
        if "vllm" in fname or "/public/home/" in fname or "zya" in fname:
            user_frames.append(f"{fname}:{line} in {name}")

    f.write(f"\n--- MATCH[{idx}] {'EXACT_TARGET' if exact else 'NEAR_BIG'} "
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


# ---------- v2: lifetime recording + multi-checkpoint snapshots ----------

def begin_lifetime_probe():
    """v2: call early in EngineCore init; starts lifetime recording.
    No further on/off switching: checkpoint_snapshot() grabs snapshots. """
    if not _SUPPORTED:
        print(f"[fill_alloc_probe] NOT_SUPPORTED: {_SUPPORTED_ERR}", flush=True)
        return
    if _state["recording"]:
        print("[fill_alloc_probe] already recording, skip begin", flush=True)
        return
    _record_memory_history(enabled="all", context="alloc", stacks="python")
    _state["recording"] = True
    print("[fill_alloc_probe] LIFETIME_RECORDING_STARTED (alloc-only, python stacks)", flush=True)


def checkpoint_snapshot(tag=""):
    """v2: snapshot + write scan log. _snapshot() does NOT stop recording."""
    if not _SUPPORTED or not _state["recording"]:
        print(f"[fill_alloc_probe] CKPT_SKIP supported={_SUPPORTED} "
              f"recording={_state['recording']} tag={tag}", flush=True)
        return
    _state["ckpt_count"] += 1
    log_path = os.path.join(LOG_DIR, f"fill_alloc_probe_ckpt{_state['ckpt_count']}_{tag}.jsonl") \
        if tag else os.path.join(LOG_DIR, f"fill_alloc_probe_ckpt{_state['ckpt_count']}.jsonl")
    try:
        snap = _snapshot()
    except Exception as e:
        print(f"[fill_alloc_probe] SNAPSHOT_FAIL tag={tag}: {e!r}", flush=True)
        return
    total, matches, near = _scan_snapshot(snap, tag, log_path)
    print(f"[fill_alloc_probe] CKPT#{_state['ckpt_count']} tag={tag} "
          f"total_alloc={total} exact={matches} near_big={near} log={log_path}", flush=True)


def stop_lifetime_probe(tag="final"):
    """v2: final snapshot + stop recording."""
    if not _SUPPORTED or not _state["recording"]:
        checkpoint_snapshot(tag)  # still try to snapshot if it was running
        _record_memory_history(enabled=None)
        _state["recording"] = False
        return
    checkpoint_snapshot(tag)
    _record_memory_history(enabled=None)
    _state["recording"] = False
    print(f"[fill_alloc_probe] LIFETIME_RECORDING_STOPPED tag={tag}", flush=True)


# ---------- legacy capture-phase API (deprecated, kept non-fatal) ----------

def begin_capture_probe():
    """[deprecated] old capture-phase one-shot API; use begin_lifetime_probe."""
    begin_lifetime_probe()


def end_capture_probe(tag=""):
    """[deprecated] old capture-phase one-shot API; use checkpoint/stop."""
    checkpoint_snapshot(tag)
    _record_memory_history(enabled=None)
    _state["recording"] = False


def selfcheck():
    print(f"[fill_alloc_probe] supported={_SUPPORTED} err={_SUPPORTED_ERR}", flush=True)
    print(f"[fill_alloc_probe] target_numel={_TARGET_NUMEL} target_bytes={TARGET_BYTES} "
          f"({TARGET_BYTES/1024/1024:.0f}MB)", flush=True)
    print(f"[fill_alloc_probe] log_dir={LOG_DIR}", flush=True)
    if _SUPPORTED:
        import torch
        begin_lifetime_probe()
        x = torch.zeros(_TARGET_NUMEL, dtype=torch.int32, device="cuda")
        del x
        checkpoint_snapshot(tag="selfcheck")
        stop_lifetime_probe(tag="selfcheck_final")


if __name__ == "__main__":
    selfcheck()
