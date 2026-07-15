# 精度中性优化（2026-07-11）

## 原则
- 不改 bf16 计算图语义；不开 KV FP8（默认，避免 Δ）
- 不违规：无离线量化/投机解码/改调度器

## 本轮落地
| 项 | 依据 | 精度 |
|----|------|------|
| `TORCH_BLAS_PREFER_HIPBLASLT=0` | down_proj M=1：**0.148ms vs 0.438ms** | 同 bf16 BLAS |
| Skinny LLMM1，`K≤8192` | gate_up/q 1.8–2.4×，rel≤0.0004 | ✅ |
| 本地 `_rocm_C` 21MB（精度修复版） | 替换系统 87MB 包中用于 skinny 的路径 | ✅ |
| `cudagraph_capture_sizes=[1,2,4,8,16]` | 单卡并发≈1，少占显存 | 无 |
| `NUM_PAR_SOFTMAX_SEGMENTS=32` | Flash-Decoding 分段并行（B=1 长上下文） | 数学等价 |
| Prefill BLOCK=32（此前） | 修 LDS 误用 128 | 无 |
| `evict_last`（此前） | 访存提示 | 无 |
| **关闭** `VLLM_ROCM_KV_FP8` | 保精度优先 | — |

## 明确不做（精度或违规）
- K>8192 的 LLMM1（rel>1，错误结果伪装成“加速”）
- KV FP8（先 accuracy 再开）
- TILE/stages 激进调参、AITER、投机解码

## 重启
```bash
cd /public/home/xdzs2026_c150/tang/testdata && ./start_vllm.sh
./run_throughput.sh 8-16K 20
./run_accuracy.sh
```
