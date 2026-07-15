# 先导杯 DCU 优化 — 环境变量清单（技术方案第13条）

提交/评测时请在启动脚本或评测环境中设置以下变量。
自 2026-07-14 起，`vllm/env_override.py` 在 **`import torch` 之前**会强制/默认写入关键项（即使评测机未 source 脚本也能生效）。

## 必开（Decode 主路径）

| 变量 | 推荐值 | 作用 |
|------|--------|------|
| `VLLM_ROCM_USE_AITER` | `0` | 禁用 AITER（gfx936 易走错路径）；`env_override` setdefault |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `1` | 启用 `_rocm_C` 瘦 GEMV（Decode Linear）；setdefault |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | **强制** 0（防 shell 残留=1）；Decode 用 rocBLAS |
| `HIP_FORCE_DEV_KERNARG` | `1` | 降低 kernel 启动开销 |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | 缓解显存碎片 |

逃逸（仅 Prefill A/B）：`VLLM_ROCM_FORCE_HIPBLASLT=1` → 允许 hipBLASLt（已知会伤 Decode）。

## 可选（需先过精度）

| 变量 | 推荐值 | 作用 |
|------|--------|------|
| `VLLM_ROCM_KV_FP8` | `1` | `cache_dtype=auto` 时切运行时 KV `fp8_e4m3fnuz`（减 KV HBM 读写） |

开启前必须本地跑 `./run_accuracy.sh`，确认精度跌幅 **Δ&lt;1%** 后再用于打分。

也可显式：`--kv-cache-dtype fp8_e4m3fnuz`（与 env 二选一即可）。

## 本地源码生效

```bash
export PYTHONPATH=/public/home/xdzs2026_c150/tang/vllm_cscc:$PYTHONPATH
```

确保评测安装的 wheel / editable 含 `_rocm_C`（`vllm/_rocm_C*.so`）。

## 合规边界

- 仅运行时 KV 量化；禁止离线权重量化落盘、剪枝、蒸馏、投机解码
- 不改 BatchScheduler / `max-num-seqs` 等官方锁死参数
- 不改 Qwen3.5-27B 权重结构 / tokenizer / 对话模板
