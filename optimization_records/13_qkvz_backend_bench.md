# 13 · qkvz 投影 GEMM 后端三方 bench(基准文档)

> **目的**:钉死 `in_proj_qkvz` 在不同后端下的真实耗时,回答"dist-packages 的 LLMM1 是不是负优化""切 rocBLAS/hipBLASLt 有没有收益"。
> **环境**:worker-0 容器(`/usr/local/lib/python3.10/dist-packages/vllm` v0.18.1+das.dtk2604,DCU gfx936 BW3000,DTK 26.04),2026-07-12。
> **脚本**:`tools/_qkvz_backend_bench.py`。
> **形状**:qkvz 真实形状 `weight=(16384,5120)` `x=(1,5120)` bf16 no-bias(decode batch=1)。

---

## 0. 背景:两份代码的分歧

容器里有两份 vllm:

| | `dist-packages`(线上 `vllm serve` 实跑) | `vllm_cscc` 源码(`/public/home/xdzs2026_c150/zya/vllm_cscc`) |
|---|---|---|
| `use_skinny` 条件 | `(on_gfx9() or on_gfx936())` | `and on_gfx9()`(不含 gfx936) |
| utils.py 行数 | 333 | 308 |
| gfx936 命中 | 是 → 走 LLMM1 分支 | 否 → `use_skinny=False` → `F.linear`(hipBLASLt)回退 |
| git | 别人装的 `0.18.1+das.dtk2604` | 你的仓库 `fa71803` |
| 来源 | 镜像/别人 pip 装(非源码编辑安装) | 源码,无 `_C.so` build 产物 |

**线上进程加载的是 dist-packages**(`/proc/2928` + `import vllm.__file__` 均指向 dist-packages,无 `.egg-link`/`.pth` editable 安装)。所以 qkvz 实跑走 **dist-packages 的 LLMM1**,不是源码的 hipBLASLt 回退。

本 bench 回答:dist-packages 的 LLMM1 到底比 hipBLASLt/rocBLAS 快还是慢。

---

## 1. 核名核实(profiler,确认每条路径真走哪个后端)

5 calls,取 self_device_time_total 最高 kernel:

| 路径 | 命中 kernel | 后端 |
|---|---|---|
| **A. vLLM dispatch**(`torch.ops.vllm.rocm_unquantized_gemm`) | `LLGemm1_kernel<c10::BFloat16,4>` | **LLMM1** |
| B. `F.linear` default(`preferred_blas_library("default")`) | `Custom_Cijk_..._MT256x256x16_..._WGM1` | hipBLASLt |
| B2. `F.linear` cublaslt | `Custom_Cijk_..._MT256x256x16_..._WGM1` | hipBLASLt(与 B 同核) |
| C. `F.linear` cublas | `Cijk_Alik_Bljk_BBH_MT64x32x32_SE_AMAS3_..._GSU1_ISA9` | **rocBLAS** |
| D. `matmul(x, wt[k,n].contiguous())` | `Cijk_Ailk_Bljk_BBS_BH_Bias_..._MT256x128x32_...` | rocBLAS |

要点:
- **A 确认命中 LLMM1**(不是 hipBLASLt),证实 dist-packages 的 `or on_gfx936()` 让 qkvz 走了 LLMM1。
- B 与 B2 同核 → default 与 cublaslt 都走 hipBLASLt。
- **C 命中的 `MT64x32x32_GSU1_ISA9` 与线上 trace 真瓶颈 `MT64x32x32_GSU1`(490us/call,18704 次)同名** —— 这是 rocBLAS 在 F.linear 形状下选的 tile。但 qkvz 实跑走 A(LLMM1),不走 C。所以 trace 里那个 MT64x32x32_GSU1 **不是 qkvz**,是别的 GEMM 在走 rocBLAS 的 F.linear,待 op 归因。

## 2. 计时(cuda Event,warmup=80 iters=200,无 profiler 污染)

| 路径 | median(us) | min | p10 | p90 | max | vs A |
|---|---|---|---|---|---|---|
| **A. vLLM dispatch(LLMM1)** | **188.3** | 185.1 | 186.4 | 190.7 | 204.8 | **1.00x(基线,最快)** |
| B. F.linear default(hipBLASLt) | 261.6 | 259.0 | 260.0 | 264.0 | 280.8 | 0.72x |
| B2. F.linear cublaslt(hipBLASLt) | 260.3 | 257.9 | 259.0 | 269.0 | 474.4 | 0.72x |
| C. F.linear cublas(rocBLAS) | 267.2 | 263.5 | 265.1 | 272.6 | 280.3 | 0.70x |
| D. matmul(wt[k,n])(rocBLAS) | 253.1 | 250.6 | 251.4 | 255.5 | 267.4 | 0.74x |

## 3. 结论

1. **dist-packages 的 LLMM1 不是负优化,反而是该形状最优路径**。A(LLMM1)188us 比 hipBLASLt(261us)快 **1.39x**,比 rocBLAS(267us)快 **1.42x**。别人装的这个 `[DCU Optimize]` 版本里 `or on_gfx936()` + LLMM1 对 qkvz 是正向优化。
2. **qkvz 已在最优路径(LLMM1)上**,切 rocBLAS/hipBLASLt 均退化(188→253~267us)。**任务 #13"切 rocBLAS 降 qkvz"前提确证推翻,该任务作废。**
3. **真瓶颈不是 qkvz**(qkvz 188us,已优)。真瓶颈是 trace 里的 `MT64x32x32_GSU1`(490us/call,18704 次)——本次 bench 证实该核是 **rocBLAS 在 F.linear 形状下的 tile**,但 qkvz 不走它。**需 op 归因**:线上哪个 GEMM 在走 rocBLAS 的 F.linear 并命中 MT64x32x32_GSU1。
4. **数值订正**:旧文档 §8.11 写 LLMM1 127us / hipBLASLt 229us,本次实测 LLMM1 188us / hipBLASLt 261us。**排序与结论一致(LLMM1 最快),绝对值以本次为准**(旧值可能来自不同 warmup/测量方式)。

---

## 4. 数据可信度说明

- 计时与核名核实分两个 phase,避免 profiler overhead 污染计时(Phase1 用 profiler 只看核名,Phase2 用纯 cuda Event 200 次计时)。
- warmup 80 次 iters 200 次,p10/p90 稳定(A 的 p10=186.4 p90=190.7,抖动 <3%),数据可信。
- max 偶有尖刺(B2 max 474us),属偶发调度抖动,median 稳定。
- 形状严格取自 qwen3_next.py qkvz(decode batch=1,no bias),与线上 decode 一致。

## 5. 下一步

新主线:**op 归因 `MT64x32x32_GSU1`**。本次 bench 证明该核是 rocBLAS F.linear 的 tile。线上 trace 里它出现 18704 次占 GDN 65%,但 qkvz/ba/out_proj 都走 LLMM1 不走它。需在 vLLM 里对 lm_head/FFN/attention qkv 等候选 Linear 包 `record_function` 抓带 op 标签的 trace,1:1 锁定归属。归属清楚后再定优化手段(可能:该 GEMM 改走 LLMM1/skinny、或 hipBLASLt heuristic 调参、或融合)。

> 本 bench 为后续所有 qkvz 后端讨论的基准。凡旧文档(10_gdn_gemm_design.md §8.11 等)与本文数值冲突处,以本文为准。

---

## 6. 现场 TPOT 三段对照(serve bench,并发=1)

> 来源:容器 `logs/` + `test/*/result.json`,2026-07-12 捞自 e03r1n05 / 173.0.90.2(共享存储与 e03r1n07 一致)。
> 用户原话:"dist-backache 4-8K 有提升,8-32K 提升不大,远没 1.3x"。此表为**基线 vLLM**(无 dist-packages 之外的改动)三段 TPOT,供对照。

| 段 | total_in_tok | total_out_tok | mean TPOT(ms) | median | p95 | mean TTFT(ms) | out_tput(tok/s) | 数据来源 |
|---|---|---|---|---|---|---|---|---|
| **4-8K** | 62196 | 2575 | **68.98** | 69.01 | 69.37 | 3284.95 | 12.20 | `test/4-8K_throughput/result.json` |
| **8-16K** | 134349 | 1717 | **70.09** | 70.06 | 70.59 | 11766.66 | 7.23 | `test/8-16K_throughput/result.json` |
| **16-32K** | 319761 | 2136 | **49.90** | 49.93 | 50.40 | 12251.49 | 7.34 | `test/16-32K_throughput/result.json`(2026-07-12 重跑,e03r2n01/173.1.51.3,15条全成) |

注意:
- 4-8K/8-16K 的 mean TPOT 几乎一致(68.98 vs 70.09 ms),**输入长度 4K→16K 对 decode TPOT 影响极小(<2%)**。这与 decode 是单 token 逐步、KV cache 命中后 GEMM 形状不变的物理事实吻合——TPOT 瓶颈在 decode 步内的 GEMM/递归核,不在 prefill 长度。
- `logs/run_4-8K_10.log` 的 bench 段 `Successful=0`(server 没起),但 `result.json` 是另一次成功跑的结果(68.98ms),以 result.json 为准。
- **16-32K 重跑成功**(2026-07-12,e03r2n01/173.1.51.3):mean TPOT=**49.90ms**,15/15 全成,input_len 20.5K~22.4K。**注意:此值显著低于 4-8K(68.98)/8-16K(70.09),且与"输入越长 TPOT 应不变或略升"的物理预期相反**——需警惕(见下)。
- baseline mean TPOT:4-8K=68.98,8-16K=70.09,16-32K=49.90。16-32K 偏低 ~20ms,可能因素:(a)新容器 dist-packages 装的是 tang 333 行 gfx936 版,与旧容器 e03r1n05 那份 `[DCU Optimize]` 是否逐字一致未核;(b)profiler-config 默认开(record_shapes),4-8K/8-16K 也是同配,但容器/profiler 状态可能不同;(c)cudagraph capture 在长输入下行为差异。**16-32K 此值暂作参考,需与 4-8K/8-16K 在同容器同 wheel 下重跑复核一致性后才可定基线。**

## 7. profiler op 归因(30s decode trace,基线)

> 脚本:`_parse_profile_trace.py`;trace:`profile_traces/rank0.1783777442602026742.pt.trace.json.gz`(30s decode,167 步)。

### 7.1 按类别(总 GPU kernel dur 11.198s)

| 类别 | dur(s) | 占比 | kernel 种类 |
|---|---|---|---|
| **GDN/FLA** | 10.649 | **95.10%** | 7 |
| Other | 0.286 | 2.56% | 11 |
| FFN_GEMM | 0.170 | 1.51% | 7 |
| LayerNorm | 0.052 | 0.46% | 2 |
| FullAttn | 0.028 | 0.25% | 1 |
| Elementwise | 0.005 | 0.05% | 4 |
| Memset/Copy | 0.005 | 0.04% | 4 |
| Sampling | 0.003 | 0.03% | 1 |

> 注:分类器把所有 `Cijk_Alik_Bljk_*`(rocBLAS GEMM)归入 GDN/FLA(因正则含 `GSU` 命中 `PostGSU|GSU`)。实际这些是 **GEMM kernel**,应并入 FFN_GEMM。修正后 GEMM 占比远高于 1.51%——见 §7.2。

### 7.2 TOP kernel(聚合 dur 降序)

| dur(s) | count | per_call(us) | kernel | 真类别 |
|---|---|---|---|---|
| **7.307** | 18704 | **390.7** | `Cijk_..._MT64x32x32_..._GSU1` | **GEMM(rocBLAS)← 真瓶颈** |
| 2.439 | 21543 | 113.2 | `Cijk_..._MT32x16x4_..._GSU2` | GEMM(rocBLAS) |
| 0.604 | 2672 | 225.9 | `Cijk_..._MT128x32x32_..._GSU` | GEMM(rocBLAS) |
| 0.126 | 8016 | 15.7 | `Cijk_..._MT32x32x32_..._GSU8` | GEMM(rocBLAS) |
| 0.105 | 8016 | 13.1 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | GDN 递归核 |
| 0.093 | 10688 | 8.7 | `triton_poi_fused_0` | fused |
| 0.055 | 10688 | 5.2 | `Cijk_B_PostGSU` | GDN GSU |
| 0.041 | 8016 | 5.2 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_4` | FFN fused |
| 0.041 | 8016 | 5.2 | `triton_red_fused_..._rocm_unquantized_gemm_rsqrt_3` | FFN fused(LayerNorm+GEMM) |
| 0.041 | 8016 | 5.1 | `_causal_conv1d_update_kernel` | GDN conv |

### 7.3 MT64x32x32_GSU1 双峰 dur 分布(关键)

`MT64x32x32_GSU1` 共 18704 次,per_step = 112 次/步(18704/167),dur 呈**双峰**:

| 组 | count | per_step | mean(us) | total(s) | 占该核 |
|---|---|---|---|---|---|
| **big(≥350us)** | 10688 | **64** | **505.9** | **5.407** | **74%** |
| small(<350us) | 8016 | 48 | 237.0 | 1.900 | 26% |

- big 组 ts gap 集中在 ~1031us(≈1ms,即每 decode step 1ms 周期内出 64 次 big 调用),与 decode step 周期 1.0ms 吻合。
- **真瓶颈是 big 组**:5.4s/11.2s = **48% 的 GPU 时间**花在 64 次/步 × 506us 的 MT64x32x32_GSU1 上。
- 64 次/步 × 48 层 GDN = **每 GDN 层 ~1.3 次 big MT64x32x32_GSU1** → 强烈指向 **GDN 层内某个 m=1 的 F.linear GEMM**(qkvz/ba/out_proj 之一,或 GDN 内部 conv/norm 后的投影)在走 rocBLAS 而非 LLMM1。
- 与 §3 结论一致:qkvz bench 走 LLMM1(188us)≠ MT64x32x32_GSU1(506us)。**MT64x32x32_GSU1 是另一个 GEMM op**,待 `record_function` 1:1 归因。

### 7.4 decode step 周期

`Cijk_B_PostGSU` count=10688, median_gap=**1.02ms** → decode step ≈ 1.0ms。
但实测 mean TPOT = 69.8ms —— 差 70x,说明 trace 的 167 步是 **cudagraph capture 后的内部 kernel 步**,非 token 级 decode 步。token 级 decode 一步含多个内部 kernel 波次。归因时以 kernel 级 ts 对齐,不直接用 step 数除。

## 8. 待办(更新)

1. **op 归因 MT64x32x32_GSU1(big 组,64 次/步)**:用 `record_function` 包 qwen3_5.py 的 `in_proj_qkvz`/`in_proj_ba`/`out_proj` + FFN `gate_up_proj`/`down_proj` + `lm_head`,抓带标签 trace,1:1 锁定 506us 的 GEMM 是哪个投影。
2. **重跑 16-32K TPOT**(现 result.json 全 0 无效),补齐"8-32K 提升不大"的基线证据。
3. 钉死实跑进程加载哪份 utils.py(含不含 `[DCU Optimize]`)—— §0 已有 `/proc`+`import vllm.__file__` 铁证指向 dist-packages,但新容器 173.0.90.2 上需复核 PID。
