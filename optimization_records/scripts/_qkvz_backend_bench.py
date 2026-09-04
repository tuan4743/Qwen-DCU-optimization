#!/usr/bin/env python3
"""Three-way backend benchmark for a skinny projection GEMM (vLLM dispatch vs
rocBLAS vs hipBLASLt vs matmul).

Purpose
-------
Decide whether vLLM's ROCm custom GEMM path (LLMM1/wvSplitK `_ROCM_OP`) beats
the generic backends for a given (M, K, N) skinny matvec. v1 was confounded by
the profiler and by `preferred_blas_library` not switching kernels; v2 fixes:
  1. Kernel-name verification (profiler, few calls) and timing (cuda Events,
     200 iters) are SEPARATE phases, so profiler overhead cannot pollute the
     timing.
  2. B'/C' variants call `torch.matmul` and an explicit hipBLASLt path.

How to confirm a real backend switch:
  - default   -> rocBLAS
  - cublaslt  -> hipBLASLt
  then compare profiler kernel names: expect different kernel families.

Usage
-----
    python _qkvz_backend_bench.py            # runs inside the worker container
    # edit M, K, N, DTYPE at the top per problem shape.

Generalization notes
--------------------
- Change ``M/K/N`` and ``DTYPE`` for any matvec shape; the probe is generic
  and self-contained (uses only torch + a running CUDA/HIP device).
- ``torch.backends.cuda.preferred_blas_library`` only affects ATen paths
  (F.linear/matmul); the custom `_ROCM_OP` path is unaffected — keep it as
  the reference "A" row.
- Useful sanity check for any vLLM ROCm build: if rows B/C collapse to the
  same kernel, the library switch did not happen for that shape.
"""
import inspect
import statistics
import torch

torch.manual_seed(0)
dev = torch.device("cuda", 0)

M, K, N = 16384, 5120, 1
DTYPE = torch.bfloat16
weight = torch.randn(M, K, dtype=DTYPE, device=dev)
x = torch.randn(N, K, dtype=DTYPE, device=dev)
print(f"shape: weight={tuple(weight.shape)} x={tuple(x.shape)} dtype={DTYPE}")

from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
_ROCM_OP = torch.ops.vllm.rocm_unquantized_gemm
print(f"dispatch returns: {dispatch_unquantized_gemm().__name__}")

# ---------- Phase 1: kernel-name verification (profiler; NOT used for timing) ----------
def prof_kernels(fn, label, n=5):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    ev = prof.key_averages()
    real = [k for k in ev if k.self_device_time_total > 0
            and "hip" not in k.key.lower() and "Launch" not in k.key and "Sync" not in k.key]
    real.sort(key=lambda k: -k.self_device_time_total)
    print(f"\n[{label}] CUDA kernels (top3, {n} calls):")
    for k in real[:3]:
        print(f"  {k.key[:95]:95s} dev_total={k.self_device_time_total:7.1f}us count={k.count}")
    return real[0].key[:60] if real else "?"

print("=== Phase 1: kernel-name verification ===")
# A: vLLM live dispatch (LLMM1)
prof_kernels(lambda: _ROCM_OP(x, weight, None), "A. vLLM dispatch (LLMM1)")

# B: F.linear default (cublas=rocBLAS)
torch.backends.cuda.preferred_blas_library("default")
prof_kernels(lambda: torch.nn.functional.linear(x, weight, None), "B. F.linear default(rocBLAS?)")

# B2: F.linear cublaslt (hipBLASLt)
torch.backends.cuda.preferred_blas_library("cublaslt")
prof_kernels(lambda: torch.nn.functional.linear(x, weight, None), "B2. F.linear cublaslt(hipBLASLt)")

# C: F.linear cublas (rocBLAS explicit)
torch.backends.cuda.preferred_blas_library("cublas")
prof_kernels(lambda: torch.nn.functional.linear(x, weight, None), "C. F.linear cublas(rocBLAS)")

# D: matmul(x, wt[k,n])
wt = weight.t().contiguous()
torch.backends.cuda.preferred_blas_library("default")
prof_kernels(lambda: torch.matmul(x, wt), "D. matmul(wt[k,n]) default")

# ---------- Phase 2: timing (cuda Event, no profiler, 200 iters) ----------
def bench(fn, warmup=80, iters=200):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return {
        "median": statistics.median(times),
        "min": times[0],
        "p10": times[len(times) // 10],
        "p90": times[len(times) * 9 // 10],
        "max": times[-1],
    }

print("\n=== Phase 2: timing (cuda Event, warmup=80 iters=200, no profiler) ===")
results = {}
results["A. vLLM dispatch (LLMM1)"] = bench(lambda: _ROCM_OP(x, weight, None))

torch.backends.cuda.preferred_blas_library("default")
results["B. F.linear default"] = bench(lambda: torch.nn.functional.linear(x, weight, None))

torch.backends.cuda.preferred_blas_library("cublaslt")
results["B2. F.linear cublaslt"] = bench(lambda: torch.nn.functional.linear(x, weight, None))

torch.backends.cuda.preferred_blas_library("cublas")
results["C. F.linear cublas"] = bench(lambda: torch.nn.functional.linear(x, weight, None))

torch.backends.cuda.preferred_blas_library("default")
results["D. matmul(wt[k,n])"] = bench(lambda: torch.matmul(x, wt))

print(f"\n{'path':32s} {'median':>9s} {'min':>9s} {'p10':>9s} {'p90':>9s} {'max':>9s}  speedup")
base = results["A. vLLM dispatch (LLMM1)"]["median"]
for name, r in results.items():
    sp = base / r["median"] if r["median"] > 0 else 0
    print(f"{name:32s} {r['median']:9.2f} {r['min']:9.2f} {r['p10']:9.2f} {r['p90']:9.2f} {r['max']:9.2f}  {sp:.2f}x vs A")

print("\nDONE")
