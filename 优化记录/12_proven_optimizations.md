# 12 · 当前已证实的优化汇总(2026-07-12)

> 本文档集中记录 P0→P2-decode 阶段所有**已证实**的结论,分两类:
> **A. 已证实的正向优化/事实**(应保留、应依赖的判断基础);
> **B. 已证实的死路/证伪**(已实验否定、不再追的方向)。
>
> 用途:作为后续优化的"已锁定事实底座" —— 任何新方向必须先与本文件核对,
> 不重复已证伪路线,不推翻已证实的正向结论(除非有新的实测反证)。
>
> 关联:`04_cudagraph_experiment.md`、`05_task_tracker.md`、`07_p0_conclusion.md`、
> `09_cpu_sched_overhead_design.md`、`10_gdn_gemm_design.md`、`11_investigation_findings.md`。
> 创建:2026-07-12。

---

## A. 已证实的正向优化 / 事实

### A1. cudagraph ON 必须保留(净正收益 1.77×)

- **来源**:`04_cudagraph_experiment.md` §4.1/§4.3。
- **证据**:
  | 指标 | cudagraph ON(baseline) | cudagraph OFF(eager) |
  |---|---|---|
  | 输出吞吐 | **12.20 tok/s** | 7.39 tok/s |
  | TPOT P99 | 69.0 ms | 122.2 ms |
- **结论**:关 cudagraph 更慢 1.77×,**cudagraph 是净正收益,必须保留**。baseline = cudagraph ON(后续所有 tpot 优化以此为目标基线)。
- **附带事实(§4.3.2)**:256MB fill 不是 cudagraph 凭空引入,而是模型/算子代码本就有的 buffer 初始化,cudagraph 只把它捕获进静态图。优化它对 ON/OFF 两种模式都有效。

### A2. dist-packages 的 `or on_gfx936()` + LLMM1 对三投影是正向优化(非负优化)

- **来源**:`10_gdn_gemm_design.md` §2.2/§2.4/§2.5、`11_qkvz_backend_bench.md`。
- **事实**:线上 `vllm serve` 实跑 dist-packages 版本,其 `use_skinny = (on_gfx9() or on_gfx936())`,gfx936 命中。三个 GDN 投影(`in_proj_qkvz`/`in_proj_ba`/`out_proj`,均 `bias=False`)满足 `n==1 and m%4==0 and k<=8192 and bias is None` → 全部走 `ops.LLMM1`(`LLGemm1_kernel<c10::BFloat16,4>`)。
- **三方 bench 基准**(qkvz 形状,2026-07-12,详见 `11`):
  | 路径 | 后端 | median(us) | 命中 kernel |
  |---|---|---|---|
  | **A. vLLM dispatch(实跑)** | **LLMM1** | **188.3(1.00×)** | `LLGemm1_kernel<c10::BFloat16,4>` |
  | B. F.linear default | hipBLASLt | 261.6(0.72×) | `Custom_Cijk_..._MT256x256x16_GSU1_WGM1` |
  | C. F.linear cublas | rocBLAS | 267.2(0.70×) | `MT64x32x32_..._GSU1_ISA9`(← 同名真瓶颈核) |
  | D. matmul(x, wt[k,n]) | rocBLAS | 253.1(0.74×) | `MT256x128x32` |
- **结论**:LLMM1 188us 是该形状最优路径,比 hipBLASLt 快 1.39×、比 rocBLAS 快 1.42×。dist-packages 的 `or on_gfx936()` + LLMM1 对 qkvz 是**正向优化,非负优化**。qkvz 已不是瓶颈(188us,不在 trace top-25)。

### A3. ⚠️ 去污染订正:FFN_GEMM 占 decode GPU 时间 95.55%(旧"GDN/FLA 95.17%"是正则误判)

- **来源**:`05_task_tracker.md` §P2/§P2.3、`tools/_parse_profile_trace.py`(2026-07-12 修正版,DCU 已同步)、批3 trace `rank0.1783777442602026742.pt.trace.json.gz` 复跑。
- **订正根因(为什么旧值是错的)**:旧版 `_parse_profile_trace.py` 的 `CATS` 列表里 `GDN/FLA` 正则排在 `FFN_GEMM` 之前,且含裸 `GSU` 片段。`classify()` 第一个命中即返回 → 所有 `Cijk_Alik_Bljk_BBH_*_GSU1/4/8`(rocBLAS/hipBLASLt GEMM tile,`GSU`=GridSplitU 分块参数,与 GDN 的 GSU 毫无关系,纯同名缩写撞车)被误吞进 GDN/FLA 桶,得出伪值"GDN/FLA 95.17%"。修正:把 `FFN_GEMM` 提到 `GDN/FLA` 之前 + 从 GDN/FLA 正则删掉裸 `GSU` 只留 `PostGSU` 前缀。
- **去污染后真实占比(批3 cudagraph ON trace,163326 kernel events,total 11.198s)**:
  | 类别 | dur(s) | 占比 | kernel种类 |
  |---|---|---|---|
  | **FFN_GEMM** | **10.700** | **95.55%** | 12 |
  | Other | 0.286 | 2.56% | 11 |
  | **GDN/FLA** | **0.119** | **1.06%** | 2 |
  | LayerNorm | 0.052 | 0.46% | 2 |
  | FullAttn | 0.028 | 0.25% | 1 |
  | 其余 | <0.02 | <0.2% | — |
- **含义订正**:decode GPU 时间绝对主体是 **GEMM(tile 名 `Cijk_Alik_Bljk_BBH_*`)**,不是 GDN 递归核。GDN 自身递归核 `fused_recurrent_gated_delta_rule_packed_decode_kernel` 仅 0.105s(占 0.9%,13.1us/call × 8016)。优化 GEMM 直接降 tpot —— 但 GEMM 归属哪个 Linear 层仍未定(见 A6/§D)。
- **影响连锁**:本订正不推翻 A4(duty 97.3%)、A1(cudagraph)、A2(qkvz 走 LLMM1)。但 A4 里"GDN/FLA 占 95.17%"的措辞需随本订正改为"FFN_GEMM 占 95.55%"。README「当前下一步」/10 文档同步改。

### A4. duty cycle 97.3% 钉死:GPU 全程满载,瓶颈在 step 内部

- **来源**:`09_cpu_sched_overhead_design.md` §0.5、`11_investigation_findings.md` §2.1/§3.3、记忆 `duty_cycle_kills_step_gap_theory.md`。
- **证据(8s + 30s 长窗口双闭合)**:
  - 8s 窗口:GPU duty = 97.3%(busy 7800.1ms / span 8018ms),idle 2.5%。
  - 30s 长窗口复核:duty = **97.37%**(与 8s 一致);166912 个 idle gap **全部 <1ms**(max 0.644ms),无 >0.644ms 空闲;idle 在时间轴 20 桶均匀分布(2.3%–3.1%),head 10%=2.5%/mid 80%=2.6%/tail 10%=2.8%。
- **结论**:
  1. 占空比 97.3% 代表全稳态(非 8s 截取偏置),idle 均匀分散无周期性聚集。
  2. **无 step 间空闲**。`median_gap=1.00ms` 是同一 token 内相邻层间 kernel 间隔(64 层 × ~1ms ≈ 64ms ≈ tpot),非跨 token 间隔。
  3. **端到端 tpot 瓶颈 = step 内部 64 层 GPU kernel 串行**,不在 step 之间。
- **这是把优化重心从 CPU/调度轨道归到 GDN GEMM 轨道的根本依据。**

### A5. Triton autotune cache 持久化机制本身工作正常

- **来源**:`07_p0_conclusion.md`。
- **事实**:`TRITON_CACHE_AUTOTUNING=1` + 持久 `TRITON_CACHE_DIR` 机制能正确把 autotune 结果落盘(profile_run warmup → FLA/GDN 子核 autotune → `do_bench` → cache 落盘)。
- **结论**:机制本身可用(冷启动后 cache 可复用),**但对单请求稳态吞吐无收益**(见 B1)。

### A6. 真瓶颈已重定位为 `MT64x32x32_GSU1`(390.7us/call),归属待 op 归因

- **来源**:`10_gdn_gemm_design.md` §3、批3 trace 复跑(2026-07-12 去污染后)。
- **事实**(批3 cudagraph ON trace 顶级 kernel,**去污染后重算**,per_call = dur/count,与旧"median 490us"口径不同 —— 490us 是步内 median,390.7us 是全程聚合均值):
  | kernel | 总耗时 | 次数 | per_call(us) | 类别(去污染) |
  |---|---|---|---|---|
  | `Cijk_..._MT64x32x32_..._GSU1` | 7.307s | x18704 | **390.7** | FFN_GEMM |
  | `Cijk_..._MT32x16x4_..._GSU1` | 2.439s | x21543 | 113.2 | FFN_GEMM |
  | `Cijk_..._MT128x32x32_..._GSU4` | 0.604s | x2672 | 225.9 | FFN_GEMM |
  | `Cijk_..._MT32x32x32_..._GSU8` | 0.126s | x8016 | 15.7 | FFN_GEMM |
  | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 0.105s | x8016 | 13.1 | GDN/FLA(真 GDN 递归核) |
- **关键发现**:
  1. bench(§A2 路径 C)证明 `MT64x32x32_..._GSU1_ISA9` 是 **rocBLAS 的 F.linear tile**,与线上真瓶颈核同名。但 qkvz/ba/out_proj 走 LLMM1 不走 F.linear → 该真瓶颈核**不是 GDN 三个投影**,是别处走 rocBLAS F.linear/nn.Linear 的层(候选:lm_head / FFN gate-up-down / attention qkv)。
  2. **去污染后确认所有 `Cijk_*` tile 都属 FFN_GEMM 桶**(95.55%),无一条进 GDN/FLA。GDN/FLA 桶只剩 `fused_recurrent_*`(0.9%)+ `reshape_and_cache`(0.01%)。
  3. **频次反推归属线索**:`MT64x32x32_GSU1` x18704。单 token 内若每层 1 次 ×64 层 ≈ 64 次/token,18704/64 ≈ 292 tokens(与 trace ~236 generations 量级不符偏多);若含 lm_head 每步 1 次 + FFN gate/up/down 每层 3 次,64 层 ×3 = 192 + 1 = 193 次/token,18704/193 ≈ 97 tokens(偏少)。频次需配合 shape 正向匹配才能定论(见 §D)。
- **结论**:已排除"qkvz=MT64x32x32_GSU1"的旧误判,真瓶颈核归属需 op 标签 trace 1:1 归因(下一步主线)。**但 record_function 不进 cudagraph 静态图,op 标签 trace 在 cudagraph ON 下抓不到 —— 见 §D 死路说明。**

---

## B. 已证实的死路 / 证伪(不再追)

### B1. 消除 256MB fill 对单请求吞吐无收益

- **来源**:`05_task_tracker.md` §5.3.5、`07_p0_conclusion.md` §7.4.1/§7.5。
- **证据**:
  - 256MB int32 memset 是 Triton autotune 的 L2-cache-flush buffer(`triton/backends/amd/driver.py:718-721` `get_empty_cache_for_benchmark()`),分配在 profile_run warmup → autotune → `do_bench` → `cache.zero_()`。
  - 候选1(消除/绕过 fill)实测 throughput **7.26 < baseline 8.8**,首跑=二跑(无 autotune 复跑开销差异)。
- **结论**:fill 非单请求吞吐主因,消除它无收益。稳态占比复核:pre_capture=161 / post_capture=161 / post_first_req=252(warmup 161 + serving 91)。

### B2. 缩小 cache 至 64MB 倒退(已回滚)

- **来源**:`05_task_tracker.md` §5.1/§5.3.5。
- **证据**:候选2(缩小 Triton cache 至 64MB)实测 throughput **6.58 < 8.8**,倒退,已回滚。
- **结论**:cache 大小改动无效且有害。

### B3. CPU/调度 overhead 不降 tpot(duty 97.3% 满载)

- **来源**:`09_cpu_sched_overhead_design.md`、`11_investigation_findings.md` §3.3。
- **证据**:duty cycle 97.3%(见 A4),GPU 已满载,无重叠/压缩空间。
- **结论**:`09` 优化点 1(async_scheduling)/1'(stream_interval)/3(`.cpu()`)不会降 tpot。CPU/调度轨道彻底关闭。

### B4. hipBLASLt override 路线死刑(m=1 只 1 algo,splitK 退化)

- **来源**:`10_gdn_gemm_design.md` §6、记忆 `hipblaslt_matvec_single_solution_no_splitk_gain.md`。
- **证据**(worker-0 容器 `/opt/dtk-26.04-DCC2602-0317`,hipBLASLt 0.10.0 git `a6254b89-dirty`):
  - `hipblaslt-bench` 实测 m=1 matvec:`in_proj_qkvz`(n=16384,k=5120)和 `lm_head`(n=248320,k=5120)heuristic **都只返回 1 个 algo**(index 4362,`MT16x16x16_..._GSU1_..._WGM1`,638.9us / 8796us)。
  - splitK 手动调优(mix api,splitK=0/2/4/8/16)对 m=1 **无加速**(GSU 被接受但 matvec M=1 无 K-split 空间,参数退化为无效)。
  - 唯一可调 env = `HIPBLASLT_TUNING_OVERRIDE_FILE`;`HIPBLASLT_HEURISTIC` 不存在(环境变量名误记)。
- **结论**:`HIPBLASLT_TUNING_OVERRIDE_FILE` problem-keyed override 对 m=1 形状无第二个 index 可换,override 无的放矢。调 hipBLASLt algo/参数路线死刑。
- **订正(2026-07-12)**:"只 1 algo"事实成立,但原把它当"qkvz 实跑 638.9us"的前提失效 —— qkvz 实跑走 LLMM1 188us(见 A2),压根没喂给 hipBLASLt。override 路线对 qkvz 仍无意义,只是理由从"跑在唯一慢 algo 上"变成"压根不跑 hipBLASLt"。

### B5. skinny 优化路径 gfx936 不可达(三重闭合)

- **来源**:`11_investigation_findings.md` §3.1、`10_gdn_gemm_design.md`。
- **证据**(三重):
  1. **Python 层短路**:`on_gfx9()` 返回 False —— `_ON_GFX9` 硬编码 `["gfx90a","gfx942","gfx950"]`,排除 gfx936。
  2. **C++ 编译宏空壳**:`__HIP__GFX9__` 宏在 gfx936 编译产物里为空(skinny C++ 路径不编译进)。
  3. **CMakeCache build 目标**:`AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938` —— 含 gfx936,**不含 gfx942/gfx950**(skinny 路径依赖的 gfx9 系高端卡)。
- **结论**:gfx936(海光 BW3000,80 CUs)既不在 `_ON_GFX9` 白名单,也不是 skinny C++ 编译目标,skinny 分支对它结构性关闭。
- **注意(订正)**:此结论针对 `vllm_cscc` 源码版本(`use_skinny=and on_gfx9()`)。dist-packages 版本是 `or on_gfx936()` 已命中并走 LLMM1(见 A2),即 dist-packages 的 skinny 之子集 LLMM1 路径对 gfx936 是可达且正向的。B5 废的是"源码版 skinny 全链路",非 dist-packages 的 LLMM1。

### B6. bucket padding 撑大 m 不成立(capture 后 m=1 + 并发=1)

- **来源**:`11_investigation_findings.md` §2.3/§6.3。
- **证据**:`10` §8.4 trace 闭合 —— capture 后实际 **m=1**,未被 bucket padding 撑大(`aten::linear` 输入 `[1,5120]`)。结合并发=1 不可改(见 §C)。
- **结论**:batch=1 decode 下 m=1 是真实情况。强制 padding 撑大 m 走 mid-size GEMM 多 algo(候选 A)对单 token 做 N 倍冗余计算,在并发=1 下是纯浪费,tpot 只增不减。此路不通。

### B7. FP8 低精度不可用(segfault)

- **来源**:`01`/`10`、`11_investigation_findings.md` §4。
- **证据**:FP8 在本架构(gfx936 / DTK 26.04)segfault,DeepGEMM 失效。
- **结论**:FP8 路线不可用。dtype 锁定 bf16。

### B8. 投机解码禁止(锁定约束)

- **来源**:`01_constraints_env.md`。
- **结论**:锁定约束明令禁止投机解码,不在可探范围。

### B9. 切 rocBLAS / 切 hipBLASLt 降 qkvz 反退化(任务 #13 作废)

- **来源**:`10_gdn_gemm_design.md` §2.5/§6、记忆 `rocblas_beats_hipblaslt_3_8x_on_m1_matvec.md` / `trace_reveals_vllm_already_on_fast_hipblaslt.md`。
- **证据**:见 A2 三方 bench —— qkvz 实跑走 LLMM1 188us(最快),切 rocBLAS 267us / 切 hipBLASLt 261us 均退化。
- **结论**:原"切 rocBLAS 降 qkvz 3.8×"路线前提(qkvz 在 hipBLASLt 638.9us)被 profiler 推翻(qkvz 在 LLMM1)。任务 #13 标记 completed/废止。关 `VLLM_ROCM_USE_SKINNY_GEMM=0` 会让 qkvz 从 LLMM1 退回 hipBLASLt 261us,负收益,仅可作诊断旁证。

### B10. 投影+递归核融合收益上限封死

- **来源**:`10_gdn_gemm_design.md` §6。
- **证据**:单 step GEMM 占 87%,递归核核心循环全是 elementwise/outer/reduce 无 `tl.dot` GEMM,`causal_conv1d_update` 是短 conv 无 GEMM。融合 GEMM 碰不到 GEMM 本身。
- **结论**:投影+递归核融合收益上限封死,且受编译环境限制(docker 严重简化、依赖几乎全无、roc 版本过低)。

### B11. op 标签 trace 归因在 cudagraph ON 下不可行(record_function 不进静态图)

- **来源**:本会话 2026-07-12 复盘;`04_cudagraph_experiment.md`。
- **证据**:`torch.profiler.record_function` 是 CPU 侧 op 标签注入,cudagraph 捕获的是 GPU 静态图,重放时只跑已捕获的 kernel 序列,record_function 不会进图 → 抓出的 trace 只有 kernel 名没有 op 标签,无法 1:1 归因。关 eager 能抓标签,但 eager 下 kernel 选择与 cudagraph ON 不同(A2 的 LLMM1 结论就是关 eager 抓的),不代表实跑。
- **结论**:op 标签 trace 归因降级为死路。归属改用 shape→tile 正向匹配 + 频次/邻接反推(见 §D)。

---

## C. 锁定约束(不可改,所有结论的前提)

> 来源:`01_constraints_env.md` / `09` / `11` §1。

- **dtype 锁定 bf16**(FP8 segfault,见 B7;稠密无 MoE)。
- **batch=1 decode**(n=1,cudagraph capture sizes `[1,2,4,8,16]`);并发=1 是官方评测设定,不再质疑。
- **不能改 vllm 源码行为级架构**,只改后端分发/环境/算子级优化。
- **不能改**:模型权重/结构、tokenizer、scheduler、sampling、对外接口、评测作弊。
- **编译/图相关配置**(`enforce-eager`、`compilation_config`、`cudagraph_mode`、`custom_ops`、`pass_config`)**不在锁定清单内**(但 cudagraph ON 必须保留,见 A1)。
- **改源码生效路径**:工作区副本 `vllm_optimize_data/` 改了不生效;须改 `/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/...` → `cd vllm_cscc && python setup.py bdist_wheel` → `pip install --force-reinstall --no-deps dist/vllm-*.whl` → kill 旧 vllm → 重启 `start_vllm.sh`。**dist-packages 是公用的,任何人安装 vllm 都会覆盖,必须审查源码而非 dist-packages。**

---

## D. 当前主线:去污染归类已完,真瓶颈 GEMM 归属靠 shape/tile 正向匹配

**op 标签 trace 归因已降级为死路**(record_function 不进 cudagraph 静态图,cudagraph ON 下抓不到 op 标签;关 eager 抓标签会改变 kernel 选择,A2 的 LLMM1 结论就是关 eager 抓的,不能代表 cudagraph ON 实跑)。

**新主线:shape → tile 正向匹配 + 频次/邻接反推归属**。
- **目标**:锁定 `MT64x32x32_GSU1`(390.7us/call,x18704,占 GEMM 68%)+ `MT32x16x4_GSU1`(113.2us,x21543,占 23%)到底归属哪个 Linear 层(候选:lm_head / FFN gate-up-down / attention qkv)。
- **方法**:
  1. **shape→tile 正向匹配**:用各候选 Linear 的 `(m,n,k)` 喂 `rocblas-bench`/`hipblaslt-bench`,看 heuristic 选的 tile 名是否 = `MT64x32x32_GSU1`(命中即归属)。LMSCA-like shape-only 推断,不依赖 op 标签。
  2. **频次反推**:见 A6 频次线索 —— 用每步该核出现次数 × 64 层 + lm_head(1/step)对齐总次数 18704/21543。
  3. **trace 内邻接关系**:`MT64x32x32_GSU1` 在 trace 时间轴上紧邻哪个已知核(如 `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_4` = FFN silu 融合 → 强提示归属 FFN gate-up)。
- **前置**:确认候选层是否经 `UnquantizedLinearMethod` → `rocm_unquantized_gemm`(若满足 n==1 LLMM1 条件则走 LLMM1,否则走末尾 `F.linear` → rocBLAS,即 `MT64x32x32_GSU1` 来源)。
- **归因清楚后再定优化手段**:若形状满足 LLMM1 却没走 → 针对性修复;若不满足 → 评估 hipBLASLt 调参 / 融合 / 改 problem 形状。

> 除本主线外,B5/B10/算子基线(hipBLAS BF16 403T > DeepGEMM 280T > Triton 175T > CK 144.7T)约束下,backend 更换/融合空间已极度狭窄(详见 `11` §4/§6.5"范围狭窄成因")。

---

## E. 铁律

**所有 GEMM 后端归属判断必须用 torch profiler 实测核名校验**,不能只靠源码静态推断(本会话翻车点:旧稿只查 `on_gfx9()` 漏看 `or on_gfx936()`),也不能只靠 bench heuristic 或 trace kernel 名臆测归属。bench 数值必须用 trace 校验实跑 kernel 名。

---

## F. 未排除的候选方向(2026-07-12 新增,尚未验证也未证伪)

> 背景:真瓶颈 = FFN gate_up_proj(m=1,n=34816,k=5120,bf16),走 rocBLAS F.linear → `MT64x32x32_GSU1`,506us/call,占 decode 48%。
> 已做正收益优化只有一条:**打开 LLMM1/wvSplitK 链**(`or on_gfx936()`,覆盖 n==1/n<=4 小 n 投影,如 qkvz 走 LLMM1 188us)。gate_up n=34816 不在该链覆盖范围。
> 换后端全封死(B4 hipBLASLt 只1algo、B5 skinny不可达、B6 padding有害、B7 FP8 segfault、B9 切rocBLAS/hipBLASLt退化)、融合碰不到GEMM(B10)。
> **rocBLAS algo override 已实测过(容器里有 trace),属错误方向,不追。**
> 以下 N1–N6 是排除上述死路后剩余的、尚未验证也未证伪的方向。

### N1. rocBLAS heuristic override —— ❌ 已作废(曾测过,错误方向)
- rocBLAS 侧 algo override 容器里已有 trace 记录,属错误方向导致丢失的部分。**不追。**
- (B4 只封了 hipBLASLt override;rocBLAS 侧是单独实测否定,记录在此避免重蹈。)

### N2. aiter `gemm_a16w16` 分支(`utils.py:101` `use_aiter_triton_gemm`)—— ❌ 已排除(2026-07-12 核查)
- **事实**:`rocm_unquantized_gemm` 在 skinny 分支**之前**有独立分支:
  ```python
  if use_aiter_triton_gemm(n, m, k, x.dtype):
      from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
      return gemm_a16w16(x, weight, bias)
  ```
- **核查(读 `vllm/model_executor/layers/utils.py:101-125` 完整实现)**:
  ```python
  def use_aiter_triton_gemm(n, m, k, dtype):
      if (not rocm_aiter_ops.is_triton_gemm_enabled()
          or current_platform.is_fp8_fnuz()
          or dtype not in [torch.float16, torch.bfloat16]):
          return False
      # use hipblaslt for the larger GEMMs
      if n > 2048 and m > 512:
          return False
      return (
          (m == 5120 and k == 2880)
          or (m == 2880 and k == 4096)
          or (m == 128 and k == 2880)
          or (m == 640 and k == 2880)
          or (m == 2880 and k == 512)
      )
  ```
  - **白名单只收 5 个 `(m,k)` 精确组合**(全 MI300 系列小 k 投影:`2880/4096/5120` × `2880/4096/512`)。
  - gate_up 代入签名 `use_aiter_triton_gemm(n=1, m=34816, k=5120, bf16)`:白名单无 `(34816, 5120)` → **返回 False**。
  - aiter 实测可 import(`aiter OK /usr/local/lib/python3.10/dist-packages/aiter/__init__.py`、`gemm_a16w16 import OK`),排除"未装"可能。
- **结论**:gate_up 结构性不命中 aiter 分支,走末尾 `F.linear` → rocBLAS `MT64x32x32_GSU1`,与 trace 实测一致。除非改白名单强行塞 `(m=34816,k=5120)` 让 gate_up 走 triton gemm —— 但 §D 已记录 triton 基线 175T < rocBLAS,且函数注释明言"大 GEMM 交给 hipblaslt",改了反而退化。**排除,不追。**

### N3. weight 布局(is_contiguous / stride)—— ❌ 已排除(2026-07-12 核查,真因是 skinny 链未启用,与布局无关)
- **原假设**:rocBLAS F.linear 的 tile 选择对 weight row-major/col-major 敏感,gate_up 的 `MT64x32x32` 可能是次优布局下的 tile。
- **核查**:
  1. `UnquantizedLinearMethod.create_weights`(`linear.py`):weight = `ModelWeightParameter(torch.empty(sum(out), in), input_dim=1, output_dim=0)`,**无 transpose**。
  2. `process_weights_after_loading`(`linear.py:214`):**仅 CPU 分支**(`if current_platform.is_cpu()`),gfx936 直接 return,不动 weight。
  3. `MergedColumnParallelLinear`(`linear.py:604`):gate_up weight 创建后由 `weight_loader` 按 shard `copy_` 填充,无布局变换 → gate_up weight `(34816,5120)` row-major contiguous。
  4. `rocm_unquantized_gemm_impl` 末尾 `F.linear(x, weight, bias)`:PyTorch F.linear 内部自己处理 `weight.t()`,与外部布局无关。
- **结论**:gate_up weight 布局无次优问题,`MT64x32x32` 不是布局导致。**N3 排除。**

### N3′. ★已闭合并实测正收益:打开 gfx936 LLMM1/wvSplitK 链(C1,见文档 15)
- **核查触发**:静态推算 gate_up `(n=1,m=34816,k=5120,bias=False)` 满足 gfx936 LLMM1 条件 `n==1 and m%4==0 and k<=8192 and bias is None` → 本应走 LLMM1,但 trace 实测走 rocBLAS F.linear `MT64x32x32`。矛盾 → 实测上层 `use_skinny` 闸门。
- **实测闸门(容器内 vllm 环境,2026-07-12)**:`on_gfx936()=True`、`on_gfx9()=False`、`VLLM_ROCM_USE_SKINNY_GEMM=True`,但源码版 `use_skinny` 条件是 `... and on_gfx9() ...`(utils.py:170-175)→ gfx936 上 `use_skinny=False` → 所有 GEMM 走 F.linear/rocBLAS,LLMM1 链从未生效。
- **修复(C1,文档 15 §C1)**:utils.py 层改 `use_skinny` 条件为 `(on_gfx9() or on_gfx936())` + 加 `if on_gfx936()` 的 LLMM1-only 分支(注释明确"Keep gfx936 OUT of _ON_GFX9",故不动 `_ON_GFX9` 列表)。**不是 `_rocm_C` 注册问题**,是 use_skinny 闸门把 gfx936 排除。B5 的"编译侧空壳宏"结论在此被实测推翻——ops.LLMM1 在 gfx936 **可用**。
- **实测结果(文档 15 §结果表,cudagraph ON)**:4-8K 吞吐 12.20→**17.40 tok/s(+42.6%)**,8-16K 14.87,16-32K 5.74。**C1 全段正收益,已落地保留。**
- **与 A2 订正**:A2 称"qkvz 走 LLMM1 188us"原本基于 dist-packages + 关 eager,C1 实测证明 cudagraph ON 下 gfx936 LLMM1 链同样生效(否则不会 +42.6%)。A2 的 LLMM1 结论在 cudagraph ON 下成立,前提未失效。
- **剩余差距**:16-32K 段 5.74 vs 目标 16.32,差距最大。真因已闭合(文档 15 §C5):full_attention 的 triton 3D flash-decoding,`NUM_PAR_SOFTMAX_SEGMENTS=32` 恒定 → `reduce_segments` tax 在长 KV 下急升。当前推进 C5b(32→16,构建中)。


### N4. `VLLM_ROCM_USE_SKINNY_GEMM` 对 gate_up 的影响 —— 诊断旁证(非优化)
- skinny 全链路 gfx936 不可达(B5);gate_up n=34816 不命中 `wvSplitK`(n<=4)也不命中 LLMM1(n==1)。
- 此项仅作诊断旁证:关 `VLLM_ROCM_USE_SKINNY_GEMM=0` 会让 qkvz 从 LLMM1 退回 hipBLASLt 261us(B9),负收益,不作为优化手段。
- 与 N2 的区别:N2 的 aiter 分支独立于 skinny 开关,是另一条路径。

### N5. cudagraph 捕获粒度 / graph 拆分 —— ⚠️ 偏禁区,不建议
- 当前整 step 一个大 graph。拆成多个子 graph 理论上可减少静态 buffer 占用、改善 L2 命中。
- **风险**:改 model_runner 的 capture 逻辑,踩"不能改架构"边界;A1 明确 cudagraph ON 必须保留,拆 graph 风险高。**不建议。**

### N6. HBM 带宽天花板判断(已算,支撑 N1 之外仍有空间)—— 判断依据,非优化手段
- **计算**:gate_up weight = 34816×5120×2B = 356MB/层,每步每层从 HBM 读。506us 读 356MB = 703 GB/s。
- **结论**:gfx936 HBM 带宽约 3.2TB/s,只用了 22% → **非带宽 bound,是 tile/launch 效率问题**。
- **含义**:瓶颈在 tile 层有空间,但 N1(换 algo)已作废后,此条只能指向"launch 效率/调度/kernel 启动开销",不指向换 algo。作为"瓶颈非物理天花板"的理论支撑保留。

### F 小结:§F 方向清盘(2026-07-12),N3′ 已实测落地
- **已排除**:N1(rocBLAS override 已测过错误方向)、N2(aiter triton gemm 白名单不含 gate_up 形状)、N3(布局无关)。
- **仅诊断/禁区/判断依据**:N4(`VLLM_ROCM_USE_SKINNY_GEMM=0` 负收益诊断旁证)、N5(cudagraph 拆分偏禁区不建议)、N6(HBM 带宽 22% 非物理天花板,理论支撑)。
- **N3′ 已闭合并实测正收益(C1,详见文档 15)**:打开 gfx936 LLMM1/wvSplitK 链(utils.py 层 `(on_gfx9() or on_gfx936())` + `if on_gfx936()` LLMM1-only 分支)。实测 4-8K 吞吐 12.20→17.40(+42.6%),全段正收益,已保留。**B5 的"skinny C++ 路径 gfx936 不可达/空壳宏"结论被实测推翻——ops.LLMM1 在 gfx936 可用。**
- **当前主线已转文档 15**:§D 的 shape→tile 正向匹配归因早已闭合(memory `trace_adjoint_attribution_mttiles`),真瓶颈 FFN gate_up = MT64x32x32_GSU1。C1 已把 gate_up 切到 LLMM1。剩余瓶颈 = 16-32K 段 full_attention triton 3D flash-decoding 的 `NUM_PAR_SOFTMAX_SEGMENTS=32` 恒定 → `reduce_segments` tax(文档 15 §C5)。**当前推进 C5b(32→16,构建中)**。后续以文档 15 为权威推进,本文档 §F 清盘。
