#!/usr/bin/env python3
"""Enumerate Triton autotune cache keys (.autotune.json) for diagnosis.

Purpose
-------
Inspect whether the persistent autotune cache was actually populated, which
shapes the cached keys cover (warmup fixed shapes vs real serving batch
shapes), and whether re-autotune is happening (one kernel with several keys).
Builds the basis for judging: is the startup autotune tax hitting disk, and
is steady-state re-tuning occurring per new shape?

Usage
-----
    python _enum_autotune_keys.py [--cache-dir DIR] [--kernel-filter SUBSTR]

Examples
--------
    # default: inspect the project cache dir, focus on chunk_fwd_kernel_o
    python _enum_autotune_keys.py --kernel-filter chunk_fwd_kernel_o

Generalization notes
--------------------
- Works for any Triton autotune cache: point ``--cache-dir`` at it (any
  ``TRITON_CACHE_DIR`` value) and filter by kernel substring.
- The per-kernel uniqueness block is demonstrated on ``chunk_fwd_kernel_o``
  but applies to every kernel found: pass ``--kernel-filter`` to change it.
"""
import json, os, glob, sys, time, argparse
from collections import defaultdict, Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/public/home/xdzs2026_c150/zya/triton_autotune_cache",
                    help="path to the Triton autotune cache (TRITON_CACHE_DIR)")
    ap.add_argument("--kernel-filter", default="chunk_fwd_kernel_o",
                    help="kernel name substring for the uniqueness (re-autotune) check")
    args = ap.parse_args()

    by_kernel = defaultdict(list)
    for f in glob.glob(os.path.join(args.cache_dir, "**", "*.autotune.json"), recursive=True):
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
    print(f"=== total {total} .autotune.json entries across {len(by_kernel)} kernel names ===")
    for kname, entries in sorted(by_kernel.items()):
        entries.sort()
        print(f"\n## {kname}  ({len(entries)} keys)")
        for mt, key, ncfg, f in entries:
            ts = time.strftime("%m-%d %H:%M", time.localtime(mt))
            shape_part = [x for x in key if isinstance(x, int)]
            dtype_part = [x for x in key if isinstance(x, str)]
            print(f"  {ts}  shape_int={shape_part}  dtypes={len(dtype_part)}  n_configs={ncfg}")

    print(f"\n=== {args.kernel_filter}: per-key uniqueness (re-autotune signal) ===")
    keys = [tuple(e[1]) for e in by_kernel.get(args.kernel_filter, [])]
    c = Counter(keys)
    for k, n in c.items():
        flag = "DUP" if n > 1 else "unique"
        print(f"  n={n}  {flag}  key={list(k)}")


if __name__ == "__main__":
    main()
