#!/usr/bin/env python3
"""Duty-cycle analysis v2: idle *position* distribution + periodicity check.

Purpose
-------
The v1 duty-cycle number alone can be biased by how the trace window was
chosen (e.g. "buffer 2s -> grab 8s" prefers the steady-state middle and
hides boundary idle). v2 answers *where* the idle sits:
  1. Extract idle segments on the busiest timeline (keeps their position).
  2. Bucket idle along the timeline -> is idle uniformly spread or clustered?
  3. Idle-size histogram (<1ms / 1-5ms / 5-20ms / 20-50ms / >50ms).
  4. Periodicity check: are big idle segments (>=10ms) evenly spaced?
  5. head/mid/tail 10/80/10% window idle comparison (cuts off bias).

Usage
-----
    python _duty_cycle_v2.py <trace.json[.gz]> [--gap-threshold-us 5000] [--bins 20]

Generalization notes
--------------------
- Fully generic over any Chrome trace (same event filter as _duty_cycle.py).
- ``--bins`` controls the number of timeline buckets; increase for long
  traces. Big-idle threshold (10ms) is a constant near the top of main();
  scale it to your layer/step time.
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
    ap.add_argument("--gap-threshold-us", type=int, default=5000)
    ap.add_argument("--bins", type=int, default=20, help="number of timeline buckets for idle position distribution")
    args = ap.parse_args()

    events = load_events(args.trace)

    kernels = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        if cat in ("kernel", "gpu", "gpu_memcpy", "gpu_memset") or cat.startswith("kernel"):
            kernels.append((e.get("ts", 0), e.get("dur", 0), e.get("name", ""),
                            e.get("pid"), e.get("tid"), cat))

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
    bstart = bk_sorted[0][0]

    print("=== busiest GPU timeline pid=%s tid=%s ===" % (bpid, btid))
    print("  events       : %d" % len(bk_sorted))
    print("  span         : %.3f s" % (bspan/1e6))
    print("  kernel dur   : %.3f s" % (busy_total/1e6))
    print("  duty cycle   : %.2f %%\n" % (busy_total/bspan*100))

    # --- idle segments: keep (position in window, gap duration) ---
    idle_segments = []  # (rel_us, gap_us)  rel = idle start relative to window start
    for i in range(len(bk_sorted) - 1):
        cur_end = bk_sorted[i][0] + bk_sorted[i][1]
        nxt_start = bk_sorted[i + 1][0]
        g = nxt_start - cur_end
        if g > 0:
            idle_segments.append((cur_end - bstart, g))

    gaps = [g for _, g in idle_segments]
    n = len(gaps)
    if n:
        sum_gap = sum(gaps)
        gaps_sorted = sorted(gaps)
        med = gaps_sorted[n // 2]
        p90 = gaps_sorted[int(n * 0.9)]
        p99 = gaps_sorted[int(n * 0.99)]
        maxg = gaps_sorted[-1]
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
            big_sorted = sorted(big)
            print("     inter-step big gap median: %.3fms, p90: %.3fms, max: %.3fms"
                  % (big_sorted[len(big_sorted)//2]/1e3, big_sorted[int(len(big_sorted)*0.9)]/1e3, big_sorted[-1]/1e3))
            print("  -> %d inter-step big gaps, total %.3f s, avg %.2f ms/gap\n"
                  % (len(big), sum(big)/1e6, sum(big)/len(big)/1e3))

        # === 1: idle gap size histogram ===
        print("=== idle gap size histogram ===")
        buckets = [(0, 1000, "<1ms"), (1000, 5000, "1-5ms"), (5000, 20000, "5-20ms"),
                   (20000, 50000, "20-50ms"), (50000, 100000, "50-100ms"), (100000, float('inf'), ">100ms")]
        for lo, hi, label in buckets:
            cnt = [g for _, g in idle_segments if lo <= g < hi]
            print("  %-10s : %5d gaps, total %.3f s" % (label, len(cnt), sum(cnt)/1e6))
        print()

        # === 2: idle position distribution along timeline (even buckets) ===
        nbins = args.bins
        bin_us = bspan / nbins
        bin_idle = [0.0] * nbins
        for rel, g in idle_segments:
            idx = min(int(rel / bin_us), nbins - 1)
            bin_idle[idx] += g
        print("=== idle position distribution along timeline (%d bins, each %.1f ms) ===" % (nbins, bin_us/1e3))
        print("  (bin_idx : idle%%  ->  check for clustering)")
        for i in range(nbins):
            pct = bin_idle[i] / bin_us * 100
            bar = "#" * int(pct / 2)
            print("  bin%2d [%5.1f-%5.1f ms] : %5.1f%%  %s" % (i, i*bin_us/1e3, (i+1)*bin_us/1e3, pct, bar))
        print()

        # === 3: big idle (>=10ms) periodicity check ===
        BIG_THR = 10000  # 10ms
        big_rel = sorted([(rel, g) for rel, g in idle_segments if g >= BIG_THR])
        print("=== big idle (>=10ms) periodicity check (n=%d) ===" % len(big_rel))
        if len(big_rel) >= 2:
            print("  big idle timestamps (relative to window start, ms):")
            for rel, g in big_rel[:30]:
                print("    +%.1f ms  (idle %.2f ms)" % (rel/1e3, g/1e3))
            intervals = [big_rel[i+1][0] - big_rel[i][0] for i in range(len(big_rel)-1)]
            intervals_sorted = sorted(intervals)
            imed = intervals_sorted[len(intervals_sorted)//2]
            print("  big-idle intervals: median %.1f ms, min %.1f ms, max %.1f ms (n=%d)"
                  % (imed/1e3, intervals_sorted[0]/1e3, intervals_sorted[-1]/1e3, len(intervals)))
            # equidistance check: low variance => periodic; clustered at edges => boundary idle
            if len(big_rel) >= 4:
                import statistics
                stdev = statistics.pstdev(intervals)
                mean = statistics.mean(intervals)
                cv = stdev / mean if mean else 0
                print("  interval CV=%.3f (CV<0.2 => strongly periodic; CV>0.5 => no period / edge-clustered)" % cv)
        elif len(big_rel) == 1:
            print("  only 1 big idle (>=10ms) at +%.1f ms, no periodicity to speak of" % (big_rel[0][0]/1e3))
        else:
            print("  no >=10ms idle -> steady state fully busy; v1 duty number is representative")
        print()

        # === 4: head/mid/tail 10/80/10% idle comparison (cut-off bias) ===
        head_end = bspan * 0.10
        tail_start = bspan * 0.90
        head_idle = sum(g for rel, g in idle_segments if rel < head_end)
        mid_idle = sum(g for rel, g in idle_segments if head_end <= rel < tail_start)
        tail_idle = sum(g for rel, g in idle_segments if rel >= tail_start)
        head_span = head_end
        mid_span = tail_start - head_end
        tail_span = bspan - tail_start
        print("=== head/mid/tail idle comparison (cut-off bias check) ===")
        print("  head 10%% (%.1f ms): idle %.3f s (%.1f%%)" % (head_span/1e3, head_idle/1e6, head_idle/head_span*100 if head_span else 0))
        print("  mid  80%% (%.1f ms): idle %.3f s (%.1f%%)" % (mid_span/1e3, mid_idle/1e6, mid_idle/mid_span*100 if mid_span else 0))
        print("  tail 10%% (%.1f ms): idle %.3f s (%.1f%%)" % (tail_span/1e3, tail_idle/1e6, tail_idle/tail_span*100 if tail_span else 0))
        print("  -> if mid 80%% idle is still <3%%, steady state is saturated; cut-off bias rejected")


if __name__ == "__main__":
    main()
