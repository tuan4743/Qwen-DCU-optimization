# 10 · P2-decode GDN GEMM 设计优化调研清单(最终态)

> **阶段:设计/调研 + 实测**(2026-07-12 重写,旧稿存档于 `10_gdn_gemm_design.archive_20260712.md`)。
> **关联**:`03_profile_findings.md`(瓶颈特征)、`09_cpu_sched_overhead_design.md`(§0.5 duty cycle 把 tpot 瓶颈归到本轨道)、`01_constraints_env.md`(锁定约束)、`02_model_arch.md`(架构)、`06_pitfalls.md`(操作要点)。
> **基准**:`13_qkvz_backend_bench.md`(qkvz 后端三方 bench,2026-07-12,所有数值以此为准)。

---

## 0. 问题陈述

> 本节结论由 `09` §0.5 duty cycle 钉死传递而来。

- **duty cycle 钉死**(批3 cudagraph ON trace,`tools/_duty_cycle.py`):GPU duty = **97.3%**(busy 7800ms / span 8018ms),idle 仅 2.5%。GPU 全程满载,**没有 step 间空闲**。
- 窗口 118 decode token = **67.95 ms/token ≈ baseline mean_tpot 69.8ms**,吻合。
- **端到端 tpot 瓶颈 = step 内部 64 层 GPU kernel 串行**,不在 step 之间。
- **64 层里 FFN_GEMM 占 95.55%**(2026-07-12 去污染订正;旧"GDN/FLA 95.17%"是 `GSU` 同名撞车误吞 GEMM tile 的伪值,见 `12` A3),绝对主体是 `Cijk_..._GSU1` 系列 ROCm GEMM。**这是唯一能降 tpot 的地方。**
- baseline = cudagraph ON,12.20 tok/s,mean_tpot=69.8ms。目标:把 64 层 GEMM 串行总耗时压下来 → 直接降 tpot。

---

## 1. GDN 层结构(`qwen3_next.py:Qwen3NextGatedDeltaNet`)

每层 GDN(`layer_type="linear_attention"`)forward 三部分(`forward()`@634-690):

1. **Input Projection(投影 GEMM)**:
   - `in_proj_qkvz` = `MergedColumnParallelLinear(hidden=5120 → key_dim*2+value_dim*2)`,output_size=16384,k=5120。
   - `in_proj_ba` = `MergedColumnParallelLinear(hidden=5120 → num_v_heads*2)`,output_size=96,k=5120。
   - `out_proj` = `RowParallelLinear(value_dim → 5120)`,m=5120,k=6144。
   - 经 `UnquantizedLinearMethod.apply` → `dispatch_unquantized_gemm()` → DCU 走 `rocm_unquantized_gemm` → `torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)`。
   - 三投影 **`bias=False`**(qwen3_next.py:510/537/559)→ `bias is None` 成立,这是决定 LLMM1 命中的关键开关。
2. **Core Attention**(`gdn_attention_core` custom op):decode 走 `_forward_core_decode_non_spec` = `causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`。
   - 递归核核心循环全是 elementwise/outer/reduce,**无 `tl.dot` GEMM**;`causal_conv1d_update` 是 conv1d 短核(kernel_dim=4),也无 GEMM。
   - → **trace 里 95.55% 的 FFN_GEMM 占比、`Cijk_..._GSU1` 主力,绝不来自递归核/conv(递归核仅 0.9%),必来自投影 GEMM 或其它 Linear。**(2026-07-12 去污染订正)
3. **Output Projection**:`norm(core_attn_out, z)`(`RMSNormGated`)+ `out_proj(core_attn_out)`。

---

## 2. 投影 GEMM 实际走哪个后端(2026-07-12 源码 + profiler + bench 钉死)

### 2.1 容器里有两份 vllm(关键:实跑的是 dist-packages)

| | `dist-packages`(线上 `vllm serve` 实跑) | `vllm_cscc` 源码(`/public/home/xdzs2026_c150/zya/vllm_cscc`) |
|---|---|---|
| `use_skinny` 条件 | `(on_gfx9() or on_gfx936())` | `and on_gfx9()`(不含 gfx936) |
| utils.py 行数 | 333 | 308 |
| gfx936 命中 | **是** → 走 LLMM1 分支 | 否 → `use_skinny=False` → `F.linear`(hipBLASLt)回退 |
| git/版本 | 别人装的 `0.18.1+das.dtk2604` | 你的仓库 `fa71803` |
| build 产物 | 有 `_C.abi3.so`(192MB) | 无 `_C.so` |

**线上进程加载的是 dist-packages**(`/proc/2928` + `import vllm.__file__` 均指向 dist-packages,无 `.egg-link`/`.pth` editable 安装)。`vllm_cscc` 源码未被 import,仅供改源码重编参考。

### 2.2 dist-packages 的 `rocm_unquantized_gemm_impl` 分发逻辑(实跑)

```python
# utils.py:179-208 (dist-packages 实跑版本)
use_skinny = (
    envs.VLLM_ROCM_USE_SKINNY_GEMM            # 默认 True
    and (on_gfx9() or on_gfx936())            # ← gfx936 命中
    and rocm_skinny_ops_available()           # True
    and x.dtype in [torch.float16, torch.bfloat16]
    and k % 8 == 0
)
if use_skinny is not True:
    return torch.nn.functional.linear(x, weight, bias)
x_view = x.reshape(-1, x.size(-1))
try:
    if on_gfx936():
        if n == 1 and m % 4 == 0 and k <= 8192 and bias is None:
            out = ops.LLMM1(weight, x_view, 4)          # ← qkvz/ba/out_proj 命中
            return out.reshape(*x.shape[:-1], weight.shape[0])
        return torch.nn.functional.linear(x, weight, bias)
    ...
```

实测确认(2026-07-12 worker-0):`on_gfx9()=False`,`on_gfx936()=True`,`rocm_skinny_ops_available()=True`,device=`BW3000`。

### 2.3 三个 GDN 投影逐个核对命中条件(全部 `bias=False` → `bias is None`)

| 投影 | n | m | m%4 | k | k≤8192 | bias is None | 命中 |
|---|---|---|---|---|---|---|---|
| `in_proj_qkvz` | 1 | 16384 | ==0 | 5120 | ✅ | ✅ | **LLMM1** |
| `in_proj_ba` | 1 | 96 | ==0 | 5120 | ✅ | ✅ | **LLMM1** |
| `out_proj` | 1 | 5120 | ==0 | 6144 | ✅ | ✅ | **LLMM1** |

### 2.4 qkvz 后端三方 bench(基准,详见 `13_qkvz_backend_bench.md`)

形状 `(weight=16384×5120, x=1×5120, bf16, no-bias)`,warmup=80 iters=200,cuda Event 计时 + profiler 核名核实分离:

| 路径 | 后端 | median(us) | 命中 kernel |
|---|---|---|---|
| **A. vLLM dispatch**(`rocm_unquantized_gemm`) | **LLMM1** | **188.3** | `LLGemm1_kernel<c10::BFloat16,4>` |
| B. `F.linear` default | hipBLASLt | 261.6 | `Custom_Cijk_..._MT256x256x16_GSU1_WGM1` |
| B2. `F.linear` cublaslt | hipBLASLt | 260.3 | 同上 |
| C. `F.linear` cublas | rocBLAS | 267.2 | `Cijk_..._MT64x32x32_SE_AMAS3_GSU1_ISA9` ← **与线上真瓶颈核同名** |
| D. `matmul(x, wt[k,n])` | rocBLAS | 253.1 | `Cijk_..._MT256x128x32` |

### 2.5 结论

1. **三个 GDN 投影 GEMM(qkvz/ba/out_proj)实跑全部走 LLMM1 自定义核**,不进 hipBLASLt/rocBLAS。
2. **LLMM1 是该形状最优路径**:188us 比 hipBLASLt(261us)快 **1.39x**,比 rocBLAS(267us)快 **1.42x**。dist-packages 的 `or on_gfx936()` + LLMM1 对 qkvz 是正向优化,**非负优化**。
3. **切 rocBLAS / 切 hipBLASLt 对 qkvz 无意义且退化**(188→253~267us)。原"切 rocBLAS 降 qkvz"路线作废。
4. **qkvz 已不是瓶颈**(188us,已优,不在 trace top-25)。

---

## 3. 真瓶颈重定位 —— `MT64x32x32_GSU1`(390.7us/call 聚合均值)

> ⚠️ 2026-07-12 去污染订正:`tools/_parse_profile_trace.py` 修正正则归类误判后(原 `GDN/FLA` 正则含裸 `GSU` 把所有 `Cijk_*_GSU1/4/8` GEMM tile 误吞),批3 trace 复跑真实占比 = **FFN_GEMM 95.55%**,GDN/FLA 仅 0.9%(递归核)。下表 per_call 改为聚合均值(dur/count),旧"median 489.9us"是步内 median 口径,保留作历史对照。

### 3.1 trace 顶级 kernel(批3 cudagraph ON,去污染后重算)

| kernel | 总耗时 | 次数 | per_call(us,均值) | 类别(去污染) |
|---|---|---|---|---|
| `Cijk_..._MT64x32x32_..._GSU1` | 7.307s | x18704 | **390.7** | FFN_GEMM |
| `Cijk_..._MT32x16x4_..._GSU1` | 2.439s | x21543 | 113.2 | FFN_GEMM |
| `Cijk_..._MT128x32x32_..._GSU4` | 0.604s | x2672 | 225.9 | FFN_GEMM |
| `Cijk_..._MT32x32x32_..._GSU8` | 0.126s | x8016 | 15.7 | FFN_GEMM |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 0.105s | x8016 | 13.1 | GDN/FLA(真 GDN 递归核) |
| `LLGemm1_kernel`(qkvz/ba/out_proj) | — | — | ~188us(不在 top-25) | FFN_GEMM(LLMM1 分支) |

### 3.2 关键发现:MT64x32x32_GSU1 是 rocBLAS 的 F.linear tile

§2.4 bench 证明:rocBLAS 在 `F.linear` 形状下选的 tile 正是 `Cijk_..._MT64x32x32_..._GSU1_ISA9`,与线上真瓶颈核 `MT64x32x32_GSU1` **同名**。

但 qkvz/ba/out_proj 走 LLMM1 不走 F.linear,所以**这个真瓶颈核不是 GDN 三个投影**,是别的 Linear 层在走 rocBLAS 的 `F.linear`/`nn.Linear`。

### 3.3 候选来源(待 shape→tile 正向匹配确认)

线上 `vllm serve` 用 dist-packages 分发,投影 GEMM 进了 LLMM1 分支。但 trace 里大量 MT64x32x32_GSU1(rocBLAS F.linear 核)说明有别处用了 F.linear/nn.Linear。可能来源:

- **(a) lm_head**:`vocab=248320`,trace 里 248320 维度核归属 lm_head(§旧稿闭合,`VocabParallelEmbedding`/`ParallelLMHead`)。lm_head 是否走 `rocm_unquantized_gemm` 取决于它是否经 `UnquantizedLinearMethod`。
- **(b) FFN gate/up/down**:MLP 的 `gate_proj`/`up_proj`/`down_proj` 也是 Linear,若经 `rocm_unquantized_gemm` 但形状不满足 `n==1` 或 `k>8192` → 走末尾 `return F.linear` 分支 → rocBLAS。
- **(c) attention qkv**:linear_attention 层的 q/k/v 投影(若有)。

归属用 **shape→tile 正向匹配 + 频次/邻接反推**(见 §4),op 标签 trace 已降级死路。

---

## 4. 下一步:shape→tile 正向匹配 + 频次/邻接反推(op 标签 trace 已降级死路)

> ⚠️ op 标签 trace 归因在 cudagraph ON 下不可行:`torch.profiler.record_function` 是 CPU 侧标签,cudagraph 捕获静态图时不进图,重放只跑已捕获 kernel,抓出的 trace 无 op 标签无法 1:1 归因;关 eager 抓标签会改变 kernel 选择(A2 LLMM1 结论就是关 eager 抓的),不代表实跑。详见 `12` B11。

**目标**:锁定 `MT64x32x32_GSU1`(390.7us/call,x18704)和 `MT32x16x4_GSU1`(113.2us/call,x21543)到底归属哪个 Linear 层。

**方法**:
1. **shape→tile 正向匹配**:用各候选 Linear 的 `(m,n,k)`(lm_head / FFN gate-up-down / attention qkv)喂 `rocblas-bench`/`hipblaslt-bench`,看 heuristic 选的 tile 名是否 = `MT64x32x32_GSU1`,命中即归属。不依赖 op 标签。
2. **频次反推**:`MT64x32x32_GSU1` x18704 / `MT32x16x4_GSU1` x21543,除以 64 层与每层次数,对齐 trace 的 generation 数(~236)。
3. **trace 内邻接关系**:在 trace 时间轴上看 `MT64x32x32_GSU1` 紧邻哪个已知核 —— 去污染复跑发现 `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_4`(FFN silu 融合,8016 次,5.2us)与 `MT64x32x32_GSU1` 频次量级接近 → 强提示归属 FFN gate/up。

**前置**:确认候选层是否经 `UnquantizedLinearMethod` → `rocm_unquantized_gemm`(若是,且形状满足 n==1 LLMM1 条件则走 LLMM1;否则走末尾 F.linear → rocBLAS,即 MT64x32x32_GSU1 来源)。

**归因清楚后**再定优化手段,候选:
- 若该 GEMM 形状满足 LLMM1 条件却没走 → 检查为何(可能 bias≠None、或 n≠1、或 k>8192),针对性修复。
- 若形状不满足 LLMM1 → 评估 hipBLASLt heuristic 调参 / 融合 / 改 problem 形状。

---

## 5. 锁定约束(来自 `01`/`09`)

- **dtype 锁定 bf16**(FP8 在本架构 segfault,DeepGEMM 失效;稠密无 MoE)。
- **batch=1 decode**(n=1,cudagraph capture sizes `[1,2,4,8,16]`)。
- **不能改 vllm 源码行为级架构**,只改后端分发/环境/算子级优化。
- **duty cycle 97.3%**,GPU 已饱和,唯一手段是减少单 kernel 绝对时间。

---

## 6. 已废路线(不再追)

| 路线 | 废弃原因 |
|---|---|
| 切 rocBLAS/hipBLASLt 降 qkvz | qkvz 已在 LLMM1(188us,最优),切了反退化。详见 §2.4。 |
| hipBLASLt `TUNING_OVERRIDE_FILE` | m=1 matvec 在 hipBLASLt heuristic 下只有 1 个 algo(index 4362),override 无的放矢。 |
| 投影+递归核融合 | 单 step GEMM 占 87%,融合碰不到 GEMM 本身,收益上限封死。 |
| 关 `VLLM_ROCM_USE_SKINNY_GEMM=0` | qkvz 从 LLMM1 188us 退回 hipBLASLt 261us,负收益,仅可作诊断旁证。 |
| §8.4–§8.10 旧"qkvz=MT64x32x32_GSU1=hipBLASLt 220us/638.9us"链 | 前提"qkvz 走 hipBLASLt"被 profiler 推翻(qkvz 走 LLMM1)。详见存档。 |

---

## 7. 铁律

**所有 GEMM 后端归属判断必须用 torch profiler 实测核名校验**,不能只靠源码静态推断(本会话翻车点:旧稿只查 `on_gfx9()` 漏看 `or on_gfx936()`),也不能只靠 bench heuristic 或 trace kernel 名臆测归属。bench 数值必须用 trace 校验实跑 kernel 名。

---

> 本文档为最终态。旧稿(含候选 A/B/C/D 推演、§8.4–§8.10 override/rocBLAS 翻盘/trace 反转全过程)存档于 `10_gdn_gemm_design.archive_20260712.md`,仅供追溯,勿据其做决策。
