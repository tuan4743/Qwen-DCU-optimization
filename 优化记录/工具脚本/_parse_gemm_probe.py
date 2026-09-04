#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 GDN GEMM 钉死: 离线解析带 gemm_probe 标注的 chrome trace。

对每个 `GEMM_PROBE::in_proj_qkvz` / `GEMM_PROBE::in_proj_ba` /
`GEMM_PROBE::out_proj` user_annotation 区间(ts..ts+dur), 找出落在区间内的
GPU kernel(kernel/gpu_memcpy/gpu_memset 事件), 按 kernel 名聚合 dur + count。

这样能 1:1 回答 `10_gdn_gemm_design.md` §0.5.4 / §5 第1条:
  trace 主力 `Cijk_..._MT64x32x32_..._GSU1`(5.074s) 到底属于哪个投影 GEMM?

判据:
  - 若 MT64x32x32_GSU1 几乎全落在 in_proj_qkvz 区间 → 坐实 §0.5.4 候选A
    (in_proj_qkvz 是绝对大头, 占 GDN 68%).
  - 若某 label 区间为空 → 该投影 GEMM 走的不是 Cijk_GSU1 kernel(可能是别的).

用法:
  python _parse_gemm_probe.py <trace.json[.gz]>
"""
import json, sys, gzip, argparse
from collections import defaultdict


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
                    help="逗号分隔的 probe label(不含 GEMM_PROBE:: 前缀)")
    args = ap.parse_args()

    events = load_events(args.trace)
    want = set(f"GEMM_PROBE::{x}" for x in args.labels.split(",") if x.strip())

    # 收集 user_annotation 区间: name -> list of (ts, ts_end)
    # torch profiler record_function 事件: cat="user_annotation", ph="X", 有 dur.
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

    # 收集 GPU 事件(kernel/gpu_memcpy/gpu_memset): (name, dur, ts)
    # 注意: GPU 事件 ts 是 GPU 时钟域, torch profiler 会把 user_annotation(CPU)
    #       与 kernel 事件放在同一时间轴(chrome trace 统一时间戳), 但 GPU kernel
    #       的 ts 是 dispatch 后的 GPU 时间. 为稳妥, 用 "kernel ts 落在 label 区间内"
    #       做粗匹配, 并同时报告每个 label 内 kernel 的聚合统计.
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
    print(f"=== user_annotation labels found:")
    for name in want:
        print(f"    {name}: {len(labels.get(name, []))} intervals")

    if not labels:
        print("\n[!] 没找到 GEMM_PROBE::* user_annotation. "
              "确认 trace 是带 gemm_probe 标注版(vllm 已装 patched wheel + "
              "_apply_gemm_probe.py 已跑), 且 torch profiler 开启(record_function "
              "仅在 profiler 运行时才产生 annotation 事件).")
        sys.exit(1)

    # 对每个 label, 统计落入区间的 kernel
    print("\n=== 每个 GEMM_PROBE label 区间内的 kernel 聚合 ===")
    for name in sorted(want):
        intervals = labels.get(name, [])
        if not intervals:
            print(f"\n[{name}] 0 intervals, skip.")
            continue
        # 区间排序便于二分; 这里简单线性匹配(trace 不大, ~28万事件)
        intervals.sort()
        # 按 ts 排序 gpu
        gpu_sorted = sorted(gpu, key=lambda x: x[2])
        # 双指针: 对每个 kernel 找它落在哪个区间(kernel 中点 ts)
        agg = defaultdict(lambda: [0.0, 0])  # kernel_name -> [total_dur, count]
        idx = 0
        for kname, kdur, kts in gpu_sorted:
            kmid = kts + (kdur / 2 if kdur else 0)
            # 找第一个 end >= kmid 的区间, 检查 start <= kmid
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
            print("    (无 kernel 落入 —— 可能 GPU ts 域与 CPU annotation 错位, "
                  "见脚本头部说明; 若全 label 都空则需换匹配策略)")
            continue
        print(f"    {'dur(s)':>9}{'count':>8}{'per_us':>10}{'占比':>8}  kernel")
        for kname, (d, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:15]:
            per = d / c if c else 0
            pct = d / total * 100 if total else 0
            print(f"    {d/1e6:>9.4f}{c:>8}{per:>10.1f}{pct:>7.2f}%  {kname[:80]}")

    # 汇总: Cijk_GSU1 系列在三个 label 的分布
    print("\n=== Cijk_*_GSU* 系列在三个 label 的 dur 分布 ===")
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
