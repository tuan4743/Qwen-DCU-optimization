#!/usr/bin/env python3
"""Offline classifier for torch profiler Chrome traces (GPU kernel attribution).

Purpose
-------
Parse a torch.profiler Chrome trace (JSON / JSON.GZ) and split per-step GPU
kernel time into semantic categories (FFN GEMM, GDN/FLA recurrences, full
attention, KV/cache, LayerNorm, sampling, memcpy/memset, elementwise, other),
then print category totals + top-N kernels with per-call latency. Also infers
the decode step periodicity from the most frequent kernel's ts interval.

Usage
-----
    python _parse_profile_trace.py <trace.json[.gz]> [--top 30]
                                   [--ts-min US] [--ts-max US]

    # decode-only slice from a full trace:
    python _parse_profile_trace.py trace.json.gz --ts-min <us> --ts-max <us>

Output
------
- category table (dur / share / kernel-kind count),
- top-N kernel table (aggregated dur / count / per-call us / category),
- step-periodity estimate.

Classification notes (ORDER-SENSITIVE; classify() returns on first hit)
----------------------------------------------------------------------
- FFN_GEMM must come BEFORE GDN/FLA: ROCm GEMM tiles named ``Cijk_*_GSU1``
  would otherwise be swallowed by the GDN regex fragment "GSU" (GSU1/4/8 is
  a rocBLAS GridSplitU block parameter, completely unrelated to GDN's
  GSU fusion). Only the GDN-specific ``PostGSU`` prefix stays in GDN/FLA.
- ``Cijk_…`` is the ROCm/MIOpen GEMM naming convention (main GEMMs), plus
  generic gemm/matmul/linear patterns.
- LayerNorm: includes Triton fused rmsnorm (mean+rsqrt+mul fusion, names
  like ``triton_red_fused_*_rsqrt``).

Generalization notes
--------------------
- The category table is easily edited for new kernels/backends: extend CATS
  (order matters) or add a custom ``--classify`` module if you need
  model-specific buckets (e.g. hydra/Jamba-style recurrent kernels).
- Works on any GPU vendor trace; kernel naming differs (CUDA vs ROCm), so
  tune the regexes for your platform.
"""
import json, sys, re, argparse, gzip
from collections import defaultdict

CATS = [
    # ORDER-SENSITIVE: first-hit wins. FFN_GEMM MUST precede GDN/FLA (see docstring).
    ("FFN_GEMM",     re.compile(r"Cijk_|gemm|matmul|addmm|hipblas|splitk|_linear|linear_|mm\b|bmm|GEMM", re.I)),
    # GDN/FLA kernels (fla, gated_delta, chunk_*, PostGSU)
    ("GDN/FLA",      re.compile(r"chunk_fwd|chunk_scaled_dot_kkt|solve_tril|wy_fast|cumsum|chunk_delta_h|recompute_w_u|merge_\d|gated_delta|fla|gdn|PostGSU", re.I)),
    ("FullAttn",     re.compile(r"triton_attn|attention|sdpa|flash|scaled_dot_product|_attn_", re.I)),
    ("KV_Cache",     re.compile(r"paged|reshape_and_cache|cache|_kv_|kv_|write|gather_cache|copy_kv", re.I)),
    ("LayerNorm",    re.compile(r"rmsnorm|layernorm|norm|rsqrt|_fused_.*mean", re.I)),
    ("Sampling",     re.compile(r"sample|topk|top_p|argmax|softmax|mixture|multinomial", re.I)),
    ("Memset/Copy",  re.compile(r"fill|zero|copy|memcpy|memset|cache\.zero", re.I)),
    ("Elementwise",  re.compile(r"^add$|^mul$|silu|gelu|elementwise|pointwise|relu|sqr|clamp|where|act_and_mul", re.I)),
]

def classify(name):
    for cat, pat in CATS:
        if pat.search(name):
            return cat
    return "Other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--ts-min", type=float, default=None, help="keep only kernels with ts >= this (us), for decode slicing")
    ap.add_argument("--ts-max", type=float, default=None)
    args = ap.parse_args()

    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt", encoding="utf-8") as f:
        data = json.load(f)

    # torch profiler chrome trace: top-level key is "traceEvents" (list)
    if isinstance(data, dict):
        events = data.get("traceEvents") or data.get("events") or []
    else:
        events = data
    kernels = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") not in ("kernel", "Kernel", "gpu"):
            # torch profiler GPU kernels: cat="kernel"; some versions "gpu"
            if not (e.get("cat", "").lower().startswith("kernel") or "dur" not in e):
                continue
        name = e.get("name", "")
        dur = e.get("dur", 0)
        ts = e.get("ts", 0)
        if args.ts_min is not None and ts < args.ts_min:
            continue
        if args.ts_max is not None and ts > args.ts_max:
            continue
        kernels.append((name, dur, ts))

    if not kernels:
        # fallback: any X event with a dur field
        for e in events:
            if e.get("ph") == "X" and "dur" in e:
                kernels.append((e.get("name", ""), e["dur"], e.get("ts", 0)))

    total = sum(d for _, d, _ in kernels)
    by_cat = defaultdict(lambda: [0, defaultdict(int)])  # cat -> [total_dur, {name:dur}]
    for name, dur, ts in kernels:
        cat = classify(name)
        by_cat[cat][0] += dur
        by_cat[cat][1][name] += dur

    print(f"=== trace: {args.trace}")
    print(f"=== total GPU kernel events: {len(kernels)}, total dur: {total/1e6:.3f}s\n")

    print("=== category breakdown (desc) ===")
    print(f"{'category':<14}{'dur(s)':>10}{'share':>10}{'kernel-kinds':>12}")
    for cat, (d, named) in sorted(by_cat.items(), key=lambda x: -x[1][0]):
        print(f"{cat:<14}{d/1e6:>10.3f}{d/total*100:>9.2f}%{len(named):>12}")

    print(f"\n=== TOP {args.top} kernels (aggregated dur desc, with count) ===")
    print(f"{'dur(s)':>9}{'count':>8}{'per_call_us':>13}  {'cat':<12}  kernel")
    agg = defaultdict(lambda: [0.0, 0])  # name -> [total_dur, count]
    for name, dur, ts in kernels:
        agg[name][0] += dur
        agg[name][1] += 1
    for name, (d, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:args.top]:
        per = d / c if c else 0
        print(f"{d/1e6:>9.4f}{c:>8}{per:>13.1f}  {classify(name):<12}  {name[:90]}")

    # decode step periodicity: use the most frequent kernel's ts interval
    print("\n=== decode step periodicity inference ===")
    name_ts = defaultdict(list)
    for name, dur, ts in kernels:
        name_ts[name].append(ts)
    for name, tss in sorted(name_ts.items(), key=lambda x: -len(x[1]))[:5]:
        tss.sort()
        if len(tss) > 3:
            gaps = [tss[i+1]-tss[i] for i in range(len(tss)-1)]
            med = sorted(gaps)[len(gaps)//2]
            print(f"  {name[:50]:<50} count={len(tss):<5} median_gap={med/1e3:.2f}ms  (est step={med/1e3:.1f}ms)")

if __name__ == "__main__":
    main()
