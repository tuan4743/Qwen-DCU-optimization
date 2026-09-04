#!/usr/bin/env python3
"""P2 decode-only profile: 离线解析 torch.profiler 生成的 chrome trace json,
   拆出单步 decode 内各算子 GPU 耗时分布。

torch profiler trace (chrome://tracing 格式) 结构:
- events: list, 每个有 ph (phase) / name / cat / ts (us) / dur (us) / pid / tid
- GPU kernel: cat="kernel", ph="X", dur=GPU 时间, name=kernel 名
- CPU op: cat="cpu_op", ph="X"
- 我们只看 cat=kernel (或 name 含 __kernel), 按 name 聚合 dur, 排序。

拆分维度(按 kernel 名前缀/正则归类):
  GDN/FLA      : chunk_fwd / chunk_scaled_dot_kkt / solve_tril / wy_fast / cumsum / chunk_delta_h / recompute_w_u / merge_*
  FullAttn     : triton_attn / attention / sdpa / flash
  FFN GEMM     : gemm / matmul / addmm / hipblas / splitk / Linear
  KV/Cache     : paged / cache / kv / reshape_cache / write / gather
  LayerNorm    : rmsnorm / layernorm / norm / mul+rsqrt
  Sampling     : sample / topk / softmax / gather / argmax
  Elementwise  : add / mul / silu / gelu / elementwise / pointwise
  Memset/Copy  : fill / zero_ / copy / memcpy / memset
  Other        : 其余

用法:
  python _parse_profile_trace.py <trace.json> [--top N] [--decode-only]
  --decode-only: 若 trace 含完整 prefill+decode, 用 ts 区间只取 decode 段(需手定 ts 范围, 这里默认全取, 配合 decode-only 抓取)

输出: 各类别 total_dur (us) + 占比 + top-N kernel。
"""
import json, sys, re, argparse, gzip
from collections import defaultdict

CATS = [
    # ⚠️ 顺序敏感: classify() 第一个命中即返回。
    # FFN_GEMM 必须在 GDN/FLA 之前 —— 否则 Cijk_*_GSU1 这类 rocBLAS/hipBLASLt GEMM tile
    # 会被 GDN/FLA 正则里的 GSU 片段误吞 (GSU1/4/8 是 rocBLAS GridSplitU 分块参数,
    # 与 GDN 的 GSU 毫无关系, 纯同名缩写撞车)。
    # FFN_GEMM: 含 ROCm/MIOpen GEMM 命名 Cijk_Alik_Bljk_*(主 GEMM) + 通用 gemm/matmul/mm
    ("FFN_GEMM",     re.compile(r"Cijk_|gemm|matmul|addmm|hipblas|splitk|_linear|linear_|mm\b|bmm|GEMM", re.I)),
    # GDN/FLA: GDN prefill/decode kernel(fla, gated_delta, chunk_*, PostGSU 即 GDN 的 GSU 融合)
    # ⚠️ 去掉裸 GSU 片段 (撞 rocBLAS tile 的 GSU1/4/8), 只保留 GDN 专属的 PostGSU 前缀。
    ("GDN/FLA",      re.compile(r"chunk_fwd|chunk_scaled_dot_kkt|solve_tril|wy_fast|cumsum|chunk_delta_h|recompute_w_u|merge_\d|gated_delta|fla|gdn|PostGSU", re.I)),
    ("FullAttn",     re.compile(r"triton_attn|attention|sdpa|flash|scaled_dot_product|_attn_", re.I)),
    ("KV_Cache",     re.compile(r"paged|reshape_and_cache|cache|_kv_|kv_|write|gather_cache|copy_kv", re.I)),
    # LayerNorm: 含 triton fused rmsnorm(rsqrt + mean + mul 融合, 名如 triton_red_fused_*_rsqrt)
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
    ap.add_argument("--ts-min", type=float, default=None, help="只取 ts>=此值(us) 的 kernel, 用于裁 decode 段")
    ap.add_argument("--ts-max", type=float, default=None)
    args = ap.parse_args()

    if args.trace.endswith(".gz"):
        opener = gzip.open
        with opener(args.trace, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(args.trace) as f:
            data = json.load(f)

    # torch profiler chrome trace: 顶层 key 是 "traceEvents"(list),不是 "events"
    if isinstance(data, dict):
        events = data.get("traceEvents") or data.get("events") or []
    else:
        events = data
    kernels = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") not in ("kernel", "Kernel", "gpu"):
            # torch profiler GPU kernel 通常 cat="kernel"; 部分版本 cat="gpu"
            if not (e.get("cat","").lower().startswith("kernel") or "dur" not in e):
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
        # fallback: 任何有 dur 的 X 事件
        for e in events:
            if e.get("ph") == "X" and "dur" in e:
                kernels.append((e.get("name",""), e["dur"], e.get("ts",0)))

    total = sum(d for _,d,_ in kernels)
    by_cat = defaultdict(lambda: [0, defaultdict(int)])  # cat -> [total_dur, {name:dur}]
    for name, dur, ts in kernels:
        cat = classify(name)
        by_cat[cat][0] += dur
        by_cat[cat][1][name] += dur

    print(f"=== trace: {args.trace}")
    print(f"=== total GPU kernel events: {len(kernels)}, total dur: {total/1e6:.3f}s\n")

    print("=== 按类别 (降序) ===")
    print(f"{'类别':<14}{'dur(s)':>10}{'占比':>10}{'kernel种类':>12}")
    for cat, (d, named) in sorted(by_cat.items(), key=lambda x:-x[1][0]):
        print(f"{cat:<14}{d/1e6:>10.3f}{d/total*100:>9.2f}%{len(named):>12}")

    print(f"\n=== TOP {args.top} kernel (按聚合 dur 降序, 含 count) ===")
    print(f"{'dur(s)':>9}{'count':>8}{'per_call_us':>13}  {'类别':<12}  kernel")
    agg = defaultdict(lambda: [0.0, 0])  # name -> [total_dur, count]
    for name, dur, ts in kernels:
        agg[name][0] += dur
        agg[name][1] += 1
    for name, (d, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:args.top]:
        per = d / c if c else 0
        print(f"{d/1e6:>9.4f}{c:>8}{per:>13.1f}  {classify(name):<12}  {name[:90]}")

    # decode step 周期性推断: 找出现频次最高的 kernel, 用它的 ts 间隔估 step
    print("\n=== decode step 周期推断 ===")
    name_ts = defaultdict(list)
    for name, dur, ts in kernels:
        name_ts[name].append(ts)
    for name, tss in sorted(name_ts.items(), key=lambda x:-len(x[1]))[:5]:
        tss.sort()
        if len(tss) > 3:
            gaps = [tss[i+1]-tss[i] for i in range(len(tss)-1)]
            med = sorted(gaps)[len(gaps)//2]
            print(f"  {name[:50]:<50} count={len(tss):<5} median_gap={med/1e3:.2f}ms  (est step={med/1e3:.1f}ms)")

if __name__ == "__main__":
    main()
