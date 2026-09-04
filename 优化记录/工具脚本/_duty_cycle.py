#!/usr/bin/env python3
"""P2 占空比分析: 用批3 chrome trace 的 kernel ts 序列, 量化 GPU kernel 占空比 (duty cycle)
   与 step 间间隙, 坐实 "step 内 1ms 但 tpot 69.8ms" 的非单步开销来源。

思路:
- GPU kernel 事件 cat="kernel", ph="X", 有 ts(us)/dur(us)。
- 占空比 = Σ kernel dur / 窗口 span (最后一个 kernel 结束 - 第一个 kernel 开始)。
  · 若 ≈ 100% -> kernel 占满窗口, step 间无间隙 (矛盾 step=1ms/tpot=69.8ms)。
  · 若 << 100% -> step 间有大段 GPU 空闲, 坐实来源 H (streaming 往返)。
- 进一步: 按 GPU stream timeline 把 kernel 按时间排序, 计算相邻 kernel 间隙分布,
  分离 "step 内多 kernel 紧凑 (<5ms)" 与 "step 间大间隙 (>5ms)" 两段。

用法:
  python _duty_cycle.py <trace.json.gz> [--gap-threshold-us 5000]
"""
import json, sys, gzip, argparse
from collections import defaultdict


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        events = data.get("traceEvents") or data.get("events") or []
    else:
        events = data
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap-threshold-us", type=int, default=5000,
                    help="gap>=此值视为 step 间大间隙 (default 5ms=5000us)")
    args = ap.parse_args()

    events = load_events(args.trace)

    kernels = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat in ("kernel", "gpu", "gpu_memcpy", "gpu_memset") or cat.startswith("kernel"):
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            name = e.get("name", "")
            pid = e.get("pid")
            tid = e.get("tid")
            kernels.append((ts, dur, name, pid, tid, cat))

    by_pid = defaultdict(list)
    for k in kernels:
        by_pid[(k[3], k[4])].append(k)

    print("=== trace: %s" % args.trace)
    print("=== total kernel/memcpy/memset events: %d" % len(kernels))
    print("=== GPU timeline (pid,tid) count: %d\n" % len(by_pid))

    all_ts = [k[0] for k in kernels]
    all_end = [k[0] + k[1] for k in kernels]
    span_start, span_end = min(all_ts), max(all_end)
    span_us = span_end - span_start
    total_dur_us = sum(k[1] for k in kernels)

    print("=== global duty cycle (all GPU activity merged) ===")
    print("  window span      : %.3f s (%.1f ms)" % (span_us/1e6, span_us/1e3))
    print("  GPU activity dur : %.3f s (%.1f ms)" % (total_dur_us/1e6, total_dur_us/1e3))
    print("  duty cycle       : %.2f %%" % (total_dur_us/span_us*100))
    print("  GPU idle         : %.3f s (%.2f %%)\n" % ((span_us-total_dur_us)/1e6, (span_us-total_dur_us)/span_us*100))

    busiest = max(by_pid.items(), key=lambda kv: sum(k[1] for k in kv[1]))
    (bpid, btid), bk = busiest
    bk_sorted = sorted(bk, key=lambda x: x[0])
    busy_total = sum(k[1] for k in bk_sorted)
    bspan = (bk_sorted[-1][0] + bk_sorted[-1][1]) - bk_sorted[0][0]

    print("=== busiest GPU timeline pid=%s tid=%s ===" % (bpid, btid))
    print("  events       : %d" % len(bk_sorted))
    print("  span         : %.3f s" % (bspan/1e6))
    print("  kernel dur   : %.3f s" % (busy_total/1e6))
    print("  duty cycle   : %.2f %%\n" % (busy_total/bspan*100))

    gaps = []
    for i in range(len(bk_sorted) - 1):
        cur_end = bk_sorted[i][0] + bk_sorted[i][1]
        nxt_start = bk_sorted[i + 1][0]
        g = nxt_start - cur_end
        if g > 0:
            gaps.append(g)
    gaps.sort()
    n = len(gaps)
    if n:
        sum_gap = sum(gaps)
        med = gaps[n // 2]
        p90 = gaps[int(n * 0.9)]
        p99 = gaps[int(n * 0.99)]
        maxg = gaps[-1]
        print("=== adjacent-kernel gap distribution (n=%d) ===" % n)
        print("  gap total   : %.3f s (%.1f %% of span)" % (sum_gap/1e6, sum_gap/bspan*100))
        print("  median      : %.3f ms" % (med/1e3))
        print("  p90         : %.3f ms" % (p90/1e3))
        print("  p99         : %.3f ms" % (p99/1e3))
        print("  max         : %.3f ms\n" % (maxg/1e3))

        thr = args.gap_threshold_us
        small = [g for g in gaps if g < thr]
        big = [g for g in gaps if g >= thr]
        print("=== gap bucket (threshold %dms) ===" % (thr//1000))
        print("  <%dms (intra-step tight): %d, total %.3f s" % (thr//1000, len(small), sum(small)/1e6))
        print("  >=%dms (inter-step big  ): %d, total %.3f s" % (thr//1000, len(big), sum(big)/1e6))
        if big:
            big.sort()
            print("     inter-step big gap median: %.3fms, p90: %.3fms, max: %.3fms" % (big[len(big)//2]/1e3, big[int(len(big)*0.9)]/1e3, big[-1]/1e3))
            print("  -> %d inter-step big gaps, total %.3f s, avg %.2f ms/gap" % (len(big), sum(big)/1e6, sum(big)/len(big)/1e3))


if __name__ == "__main__":
    main()
