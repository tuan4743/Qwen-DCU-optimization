#!/usr/bin/env python3
"""Offline parser for GEMM-probe-labeled Chrome traces.

Purpose
-------
Given a trace captured with ``record_function`` labels (user_annotation X
events, e.g. ``GEMM_PROBE::in_proj_qkvz``), attribute the GPU kernels that
executed *inside* each labeled interval to that label. This answers
questions like: which projection GEMM produced the dominant
``Cijk_..._MT64x32x32_..._GSU1`` kernel in the trace?

Matching strategy
-----------------
- CPU labels (cat="user_annotation") and GPU kernel/memcpy/memset events live
  on the same Chrome-trace timestamp axis, but GPU ``ts`` is the GPU-clock
  domain translation done by torch profiler. We use the kernel midpoint
  ``ts + dur/2`` and a two-pointer scan against the sorted label intervals;
  if the midpoint falls inside an interval, the kernel is attributed to it.
- This is intentionally coarse; if ALL labels come back empty, the GPU ts
  domain is misaligned with the CPU annotations and a different strategy
  (e.g. stream order matching) is needed.

Usage
-----
    python _parse_gemm_probe.py <trace.json[.gz]> [--labels in_proj_qkvz,in_proj_ba,out_proj]

    # one-off summary of the Cijk/GSU kernel distribution per label:
    (the script always prints a "Cijk_*_GSU* per-label" section at the end)

Generalization notes
--------------------
- The label prefix ``GEMM_PROBE::`` is matched exactly; change
  ``--labels`` (without prefix) to any record_function names you captured.
- ``user_annotation`` detection follows torch profiler conventions; other
  profilers (nsys, perfetto) need a different parser but the same idea.
- Works for any model: use different label names for different operators.
"""
import json, sys, gzip, argparse
from collections import defaultdict

LABEL_PREFIX = "GEMM_PROBE::"


def load_events(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents") or data.get("events") or []
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--labels", default="in_proj_qkvz,in_proj_ba,out_proj",
                    help="comma-separated probe labels (without the GEMM_PROBE:: prefix)")
    args = ap.parse_args()

    events = load_events(args.trace)
    want = set(f"{LABEL_PREFIX}{x}" for x in args.labels.split(",") if x.strip())

    # collect user_annotation intervals: name -> list of (ts, ts_end)
    labels = defaultdict(list)
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") != "user_annotation":
            continue
        name = e.get("name", "")
        if name in want:
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            labels[name].append((ts, ts + dur))

    # collect GPU events (kernel/gpu_memcpy/gpu_memset): (name, dur, ts)
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel"}
    gpu = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") not in gpu_cats:
            continue
        gpu.append((e.get("name", ""), e.get("dur", 0), e.get("ts", 0)))

    print(f"=== trace: {args.trace}")
    print(f"=== GPU kernel/gpu_memcpy/gpu_memset events: {len(gpu)}")
    print("=== user_annotation labels found:")
    for name in want:
        print(f"    {name}: {len(labels.get(name, []))} intervals")

    if not labels:
        print("\n[!] No GEMM_PROBE::* user_annotation found. Verify the trace was "
              "captured with the probe-patched wheel (_apply_gemm_probe.py applied) "
              "and torch profiler was active (record_function emits annotation "
              "events only while the profiler runs).")
        sys.exit(1)

    # per-label: aggregate kernels whose midpoint falls inside the intervals
    print("\n=== kernel attribution per GEMM_PROBE label ===")
    for name in sorted(want):
        intervals = labels.get(name, [])
        if not intervals:
            print(f"\n[{name}] 0 intervals, skip.")
            continue
        intervals.sort()
        gpu_sorted = sorted(gpu, key=lambda x: x[2])
        agg = defaultdict(lambda: [0.0, 0])  # kernel_name -> [total_dur, count]
        idx = 0
        for kname, kdur, kts in gpu_sorted:
            kmid = kts + (kdur / 2 if kdur else 0)
            # find the first interval with end >= kmid, then check start <= kmid
            while idx < len(intervals) and intervals[idx][1] < kmid:
                idx += 1
            if idx < len(intervals) and intervals[idx][0] <= kmid <= intervals[idx][1]:
                agg[kname][0] += kdur
                agg[kname][1] += 1

        total = sum(v[0] for v in agg.values())
        print(f"\n[{name}] intervals={len(intervals)}, "
              f"kernels inside={sum(v[1] for v in agg.values())}, "
              f"total_dur={total/1e6:.4f}s")
        if not agg:
            print("    (no kernel fell inside — GPU ts may be misaligned with CPU "
                  "annotations; see header. If every label is empty, switch strategy.)")
            continue
        print(f"    {'dur(s)':>9}{'count':>8}{'per_us':>10}{'share':>8}  kernel")
        for kname, (d, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:15]:
            per = d / c if c else 0
            pct = d / total * 100 if total else 0
            print(f"    {d/1e6:>9.4f}{c:>8}{per:>10.1f}{pct:>7.2f}%  {kname[:80]}")

    # summary: Cijk_*_GSU* distribution across the labels
    print("\n=== Cijk_*_GSU* kernel dur distribution per label ===")
    for name in sorted(want):
        intervals = labels.get(name, [])
        if not intervals:
            continue
        intervals.sort()
        gsu = defaultdict(lambda: [0.0, 0])
        idx = 0
        for kname, kdur, kts in sorted(gpu, key=lambda x: x[2]):
            kmid = kts + (kdur / 2 if kdur else 0)
            while idx < len(intervals) and intervals[idx][1] < kmid:
                idx += 1
            if idx < len(intervals) and intervals[idx][0] <= kmid <= intervals[idx][1]:
                if "GSU" in kname or "Cijk" in kname:
                    gsu[kname][0] += kdur
                    gsu[kname][1] += 1
        tot = sum(v[0] for v in gsu.values())
        print(f"\n[{name}] Cijk/GSU kernel total={tot/1e6:.4f}s")
        for kname, (d, c) in sorted(gsu.items(), key=lambda x: -x[1][0])[:8]:
            print(f"    {d/1e6:>9.4f}{c:>8}  {kname[:80]}")


if __name__ == "__main__":
    main()
