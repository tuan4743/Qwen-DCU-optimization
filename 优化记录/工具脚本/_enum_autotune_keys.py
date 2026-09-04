#!/usr/bin/env python3
"""枚举 triton_autotune_cache 中所有 .autotune.json 的 key，用于判断:
- 候选1 是否真的产生了持久化缓存文件
- key 覆盖了哪些形状(init warmup 的固定形状 vs 真实 serving batch 形状)
- 是否存在 re-autotune(同 kernel 多 key = 不同 batch 形状各自调优)
"""
import json, os, glob, sys, time
from collections import defaultdict, Counter

CACHE_DIR = "/public/home/xdzs2026_c150/zya/triton_autotune_cache"

by_kernel = defaultdict(list)
for f in glob.glob(os.path.join(CACHE_DIR, "**", "*.autotune.json"), recursive=True):
    try:
        d = json.load(open(f))
    except Exception as e:
        continue
    kname = os.path.basename(f).replace(".autotune.json", "")
    mtime = os.path.getmtime(f)
    key = d.get("key")
    ct = d.get("configs_timings", [])
    by_kernel[kname].append((mtime, key, len(ct), f))

total = sum(len(v) for v in by_kernel.values())
print(f"=== 共 {total} 个 .autotune.json，{len(by_kernel)} 类 kernel ===")
for kname, entries in sorted(by_kernel.items()):
    entries.sort()
    print(f"\n## {kname}  ({len(entries)} 个 key)")
    for mt, key, ncfg, f in entries:
        ts = time.strftime("%m-%d %H:%M", time.localtime(mt))
        shape_part = [x for x in key if isinstance(x, int)]
        dtype_part = [x for x in key if isinstance(x, str)]
        print(f"  {ts}  shape_int={shape_part}  dtypes={len(dtype_part)}  n_configs={ncfg}")

print("\n=== chunk_fwd_kernel_o 各 key 唯一性(re-autotune 迹象) ===")
keys = [tuple(e[1]) for e in by_kernel.get("chunk_fwd_kernel_o", [])]
c = Counter(keys)
for k, n in c.items():
    flag = "DUP" if n > 1 else "unique"
    print(f"  n={n}  {flag}  key={list(k)}")
