# 05 · 任务推进追踪(热数据)

> 每次推进任务时读写。关联冷档案:瓶颈特征见 `03_profile_findings.md`,踩坑备忘见 `06_pitfalls.md`。

---

## 5.1 ✅ 已完成

- [x] 模型架构确认:稠密,64 层(48 GDN + 16 FullAttn),非 MoE(见 `02_model_arch.md`)
- [x] **P0 根因钉死(2026-06-30)**:256MB int32 memset = **Triton autotune 的 L2-cache-flush buffer**(`triton/backends/amd/driver.py:718-721` `get_empty_cache_for_benchmark()` 写死 256MB int32,`do_bench` 每次 `cache.zero_()`),由 `profile_run` warmup 的 FLA/GDN 子核 autotune 触发。非 vLLM 业务 buffer,前两轮静态 0 命中正因如此。运行时快照 161 个 `EXACT_256MB` 块实机确认,详见 `07_p0_conclusion.md`。
- [x] Profile 数据解读:瓶颈 = 256MB fill(62%)+ ViT(29%)(见 `03_profile_findings.md` §3.1)
- [x] 256MB fill 特征量化:219μs/次,峰值带宽,37575 次,9.6TB 写入,prefill+decode 都有(见 `03` §3.2)
- [x] cudagraph 对比实验:cudagraph ON 更快,必须保留(见 `04_cudagraph_experiment.md`)
- [x] DeepGEMM 定位修正:稠密模型无 MoE,DeepGEMM 无直接用武之地(见 `01` §1.3)
- [x] inductor 图 dump(2026-06-24):9 个图全是 layernorm/elementwise 小图,**不含 fill 节点** → fill 是 cudagraph 捕获的 eager `at::native::fill`,非 inductor 节点(见 `03` §3.2)
- [x] MLA indexer `expanded_block_table_buffer` 嫌疑排除:本模型不实例化 indexer(`+sparse_attn_indexer` no-op,走 TRITON_ATTN/GDN backend)→ shape 吻合纯属公式巧合
- [x] 形状反推:`cdiv(max_model_len=262144, block_size=16)` = 16384 → `[4096,16384]×int32=256MB` 由 block_size=16 自然产生
- [x] 静态源码第二轮穷尽复核(2026-06-28):全代码库 GPU-resident+int32+`[4096,16384]` 的唯一候选(indexer)已证不实例化,其余大 buffer 要么 UVA/int64/bf16、要么 dim-0=max_num_reqs(8MB)→ 静态分析到极限,需实机钉死
- [x] **fill 发生阶段钉死(2026-06-29)**:qi 分布实测证明 fill **贯穿运行时每个 forward step**(cudagraph replay 回放),非 capture-only。详见 §5.3.1
- [x] **P1 候选1 落地(2026-06-30 晚)**:`start_vllm.sh` 加 `TRITON_CACHE_AUTOTUNING=1` + 持久 `TRITON_CACHE_DIR=/public/.../triton_autotune_cache`。实测 autotune cache 正常生成(FLA 子核 `.autotune.json`),候选1 机制本身工作正常。
- [x] **P1 候选1 实测证伪(2026-07-02 晚)**:候选1 单独(256MB cache + `TRITON_CACHE_AUTOTUNING=1` + 持久 `TRITON_CACHE_DIR`)8-16K `output_throughput` = **7.26 < baseline 8.8**,且**首跑=二跑**(7.253 vs 7.257,duration 236.72 vs 236.60,TTFT/TPOT/ITL 全一致),证明 7.26 **不是 autotune 冷启动税** —— autotune 命中磁盘缓存对单请求吞吐毫无影响。**结论钉死:fill 消除对单请求在线推理(batch=1)无收益,fill 非单请求吞吐主因。** 物证见 §5.3.5。判定分支落地:转 P2-decode 路径。
- [x] **P1 候选2 实测证伪并回滚(2026-06-30 晚)**:缩 `driver.py:718` cache 至 64MB → autotune 选核精度下降(`do_bench` L2-flush 不充分)→ 选次优 config,被候选1 持久化固化 → 8-16K 吞吐 6.58 < baseline 8.8(倒退)。已回滚 `driver.py` 回 256MB(从 `bak_256m` 还原),污染 autotune cache 已清空。**结论:候选2 不可用,§7.5 候选2 的"代价"实测坐实。**
- [x] **probe 桩清除+重编(2026-06-30 晚)**:`gpu_model_runner.py` 5 处 fill_alloc_probe 桩全清(regex),重编 wheel 装好,`PROBE_CLEAN` 验证。生产压测用干净版。
- [x] **节点盘满诊断(2026-06-30 晚)**:节点宿主盘 `/dev/nvme1n1p3` 437G 满根因 = Docker 镜像层 207G(`docker system df`:4 镜像,你用的 `qwen3.5-dtk26.04-03:0512`=127G;另有 onescience 55G/jupyterlab 27G 等他人镜像)。Triton JIT 写 `/tmp` 编译产物 → `HSACOError: No space left` → EngineCore 崩。已清 `/tmp` 编译垃圾(3476 个 `.s` + torchinductor_root + pip cache + vllm_cscc/build),vllm 进程已关停避免恶化。

---

## 5.2 🔄 进行中 / 待办

### P0 — 定位 256MB fill 的确切 buffer 来源(✅ 已完成 2026-06-30)

**结论**:不是 vLLM buffer,是 **Triton autotune 的 L2-cache-flush buffer**。详见 `07_p0_conclusion.md`。
- 分配:`triton/backends/amd/driver.py:718-721` `get_empty_cache_for_benchmark()` 写死 `256*1024*1024` int32。
- memset:`triton/backends/amd/driver.py:723` `clear_cache(): cache.zero_()`,由 `triton/testing.py:178 do_bench()` 反复调用。
- 触发:`profile_run` warmup → `qwen3_next.py:_warmup_prefill_kernels` → FLA/GDN 子核 autotune。物证:`ckpt1_pre_capture.jsonl` 161 个 EXACT_256MB 块,FULL-STACK leaf 161/161 命中 `driver.py:721`。

**遗留(转入 P1 前需确认)**:当前只有 `pre_capture` 快照(init/profile_run 阶段)。稳态 serving 期是否仍占 62.4%,取决于 FLA kernel 是否在首请求后反复 re-autotune。需补 `post_capture` + 首请求后快照确认稳态占比,再定 P1 优先级。

### P1 — 设计 fill 消除/缩小方案(基于 P0 结论)— ✅ 已闭合(2026-07-02)
- **结论:fill 消除对单请求在线推理(batch=1)吞吐无收益。** 候选1 单独实测 7.26 < baseline 8.8,且首跑=二跑(autotune 命中缓存对吞吐无影响)。fill 是 prefill 期 autotune 一次性开销,被 decode 的 ~120s 稀释到可忽略。详见 §5.1 / §5.3.5。
- 候选1(持久化 autotune cache)机制工作正常但**对单请求吞吐无效**;候选2(缩 cache)已实测倒退回滚。**P1 路径终结,转 P2-decode。**

### P0.5 — 稳态占比复核(P1 前置)— ✅ 已闭合(2026-06-30)
- **结论**:256MB fill **贯穿 init + 稳态首请求**,非纯冷启动开销。
  - 三检查点 `EXACT_256MB` 块:`pre_capture`=161(全 warmup)/ `post_capture`=161(全 warmup)/ `post_first_req`=252(warmup 161 + **serving 91**)。
  - `pre_capture→post_capture` 数量不变(161→161),unique addr 仍 1 → **capture 阶段不产生新 autotune,只录进 cudagraph**。
  - `post_capture→post_first_req` 新增 91 serving-path alloc,全程栈含 `do_bench`+`autotuner.py`(91/91)→ **re-autotune**,非 cache 命中。调用栈:`execute_model→_model_forward→cuda_graph.py:251 replay→qwen3_5.py:765→gdn_attention_core→forward_native→FLA 子核 autotune→do_bench`。
  - 原因:`_warmup_prefill_kernels` 仅对 `T∈{16,32,64},B=1` dummy warmup;FLA 子核 key 含 `BT/H/K/V`(固定)但真实 serving batch 形状不同 → 新 key → re-benchmark → 再 `cache.zero_()`(256MB)。
  - 91 块 USER frame 分布:`chunk_o`(33)/`kkt`(27)/`solve_tril`(12)/`wy_fast`(9)/`cumsum`(5)/`chunk_delta_h`(5)—— 与 init 期一致,全 FLA(GDN)子核。
- **对 P1 的决定性影响**:
  - 候选 3(warmup-only、运行期禁用)**单独不成立**(运行期仍 re-autotune)。
  - `TRITON_CACHE_DIR` 持久化(候选 1)**治本** —— 同 key 二次命中走磁盘 cache 跳过 `do_bench`,init+serving 都减。**前提:`TRITON_CACHE_AUTOTUNING` 必须先开**(默认 OFF,实测 `/root/.triton/cache/` 0 个 `.autotune.json`)。
  - 缩小 cache(候选 2)对 init+serving 都减量,可与候选 1 叠加。
- → **P1 主候选 = 候选 1 + 候选 2 叠加**。详见 `07_p0_conclusion.md` §7.4.1 / §7.5。

### P2 — decode 路径(单请求吞吐瓶颈主体)
- **新主攻方向(2026-07-02 定)**。P1 证伪后,单请求 `output_throughput` 瓶颈主体是 decode:`mean_tpot=69.8ms/step`(64 层,每层约 1.1ms)。`output_throughput = total_output_tokens / duration`,`duration` ≈ Σ decode steps × 69.8ms + Σ TTFT。
- ViT 29%(§3.1)是 profile 聚合统计,纯文本 bench 仅 trace 开头 27 次(== ViT depth),属噪声/超纲,**非 decode 路径关注点**。
- 待与用户定具体打法(见 §5.3 第 5 条)。

### P2 — decode-only profile 结论(2026-07-09 钉死,两批 eager profile 跨批验证)
- **方法**:`tools/_decode_only_profile.py`(streaming 请求先于 start_profile 启动,等 TTFT 后 buffer 2s → 抓 8s)在 `--enforce-eager`(无 cudagraph)下抓两批 decode trace,离线解析 `tools/_parse_profile_trace.py`。两批 trace:
  - 批1 `rank0.1783564816046488746.pt.trace.json.gz`(12.89MB,47873 事件,总 dur 3.329s,TTFT=353ms)
  - 批2 `rank0.1783567816017430570.pt.trace.json.gz`(12.94MB,48622 事件,总 dur 2.145s,TTFT=106.6s ⚠️ 因与另一终端压测争抢同进程排队所致,不影响 decode 段占比)
- **按类别占比(两批对比)**:

  | 类别 | 批1 dur(s) | 批1 占比 | 批2 dur(s) | 批2 占比 |
  |---|---|---|---|---|
  | **GDN/FLA** | 3.143 | **94.40%** | 1.914 | **89.23%** |
  | Elementwise | 0.060 | 1.80% | 0.058 | 2.72% |
  | LayerNorm | 0.056 | 1.67% | 0.064 | 3.00% |
  | Memset/Copy | 0.029 | 0.87% | 0.064 | 2.96% |
  | Other | 0.030 | 0.90% | 0.026 | 1.22% |
  | FullAttn | 0.011 | 0.33% | 0.010 | 0.48% |
  | Sampling | 0.001 | 0.03% | 0.008 | 0.38% |

- **TOP kernel(批2,按聚合 dur 降序)**:
  - `Cijk_..._MT32x16x4_..._GSU1_`:1.4363s / 11086 次 / 129.6us (GDN/FLA) ← GSU matmul 主力
  - `Cijk_..._MT32x32x4_..._GSU1_`:0.4336s / 2944 / 147.3us (GDN/FLA) ← GSU matmul 次主力
  - `fused_recurrent_gated_delta_rule_packed_decode_kernel`:0.0404s / 2208 / 18.3us (GDN/FLA 递归核)
  - `kernel_unified_attention_3d`:0.0102s / 736 / 13.9us (FullAttn,唯一 attention 核)
  - (批1 TOP 见历史,主力是 `Cijk_..._MT64x32x32_..._GSU1` 2.1479s + `MT32x16x4_GSU1` 0.7234s)
- **核心结论(跨批钉死)**:
  1. **decode 单步 GPU kernel 耗时 ~90% 在 GDN/FLA**(批1 94.4% / 批2 89.2%),绝对主体是 ROCm GEMM `Cijk_..._GSU1` 系列(即 GDN 递归层的 GSU gated-sigmoid-unit matmul),两批合计占比 ~86%。**GDN GEMM 主导结论跨批稳定,确定性。**
  2. FullAttn 两批均 <0.5%(只有 1 种 kernel `kernel_unified_attention_3d`),Sampling <0.4%,LayerNorm/Memset/Elementwise 各 ~3%,合计 <10%,可忽略 —— **attention/FFN/sampling 不是 decode GPU 瓶颈。**
  3. decode step 周期 ~2.3–2.5ms(批1 act_and_mul median_gap ~2.3ms / 批2 ~2.5ms),但 baseline `mean_tpot=69.8ms` —— 差距 ~28–30× 来自 CPU 调度/Python overhead/同步等待,非纯 GPU kernel。**GPU 端 decode 瓶颈 = GDN 的 GEMM;端到端 tpot 瓶颈 = CPU/调度 overhead。**
- **✅ 代表性说明(已验证,2026-07-09 钉死)**:三批 profile 覆盖 eager(批1/2)+ cudagraph ON(批3,见 §P2.3)。**结论:GDN GEMM 主导在 baseline 路径(cudagraph ON)下比 eager 更成立,代表性疑虑消除。** 三批占比:批1 eager 94.40% / 批2 eager 89.23% / **批3 cudagraph ON 95.17%**(三批最高)。方向上与 §核心结论3 一致(cudagraph replay 压缩 step 内 CPU/调度 overhead),**程度被批3 钉死**:cudagraph ON 下 decode step 周期从 eager ~2.3–2.5ms 压到 **~1.0ms**(批3 GSU/PostGSU kernel median_gap=1.00ms)—— cudagraph 把单步从 ~2.4ms 压到 ~1.0ms(~2.4×)。**但 baseline `mean_tpot=69.8ms` 仍远高于此 1.0ms 单步**,说明端到端 tpot 仍有 ~70× 的非单步开销(streaming 协议往返 / 跨步调度间隙 / 等待),非 GPU kernel,非 cudagraph 可消除 —— **真正端到端瓶颈在 step 之间,不在 step 内部**。

### P2.3 — 批3 cudagraph ON decode profile(2026-07-09 钉死,代表性验证)
- **方法**:vllm 已删 `--enforce-eager` 重启(cudagraph ON,baseline 路径,PID 21896),`_decode_only_profile.py` 抓 8s,`_parse_profile_trace.py` 离线解析。
- **trace**:`profile_traces/rank0.1783570538023481078.pt.trace.json.gz`(2.84MB,280909 总事件 / **115404 kernel 事件**,总 dur 7.800s,抓取窗口 span 8018ms)

> ⛔ **2026-07-09 晚 duty cycle 钉死,推翻本节"核心结论3"**:用新增 `tools/_duty_cycle.py` 算出 **GPU duty cycle = 97.3%**(busy 7800ms / span 8018ms),GPU 全程满载。下文"核心结论3"把 `Cijk_B_PostGSU median_gap=1.00ms` 解读为"decode step ≈ 1.0ms(跨 token 间隔),step 间有 ~70× 非单步开销"**前提错误**:1.00ms 是同一 token 内**相邻层之间**的 kernel 间隔(64 层各 ~1ms = 64ms ≈ tpot),非跨 token 间隔。窗口内 118 token(attn 1888 ÷ 16 层)= 67.95 ms/token ≈ tpot 69.8ms,完美吻合。**端到端 tpot 瓶颈在 step 内部 64 层 GPU kernel 串行,不在 step 之间。** CPU/调度轨道(09)的 step 间开销框架作废,转交 GDN GEMM 轨道。详见 `09` §0.5。
- **按类别占比(批3 cudagraph ON)**:

  | 类别 | 批3 dur(s) | 批3 占比 | 对比 批1(eager) | 对比 批2(eager) |
  |---|---|---|---|---|
  | **GDN/FLA** | 7.423 | **95.17%** | 94.40% | 89.23% |
  | Other | 0.196 | 2.51% | 0.90% | 1.22% |
  | FFN_GEMM | 0.117 | 1.50% | (未单列) | (未单列) |
  | LayerNorm | 0.035 | 0.45% | 1.67% | 3.00% |
  | FullAttn | 0.020 | 0.26% | 0.33% | 0.48% |
  | Elementwise | 0.003 | 0.04% | 1.80% | 2.72% |
  | Memset/Copy | 0.003 | 0.04% | 0.87% | 2.96% |
  | Sampling | 0.002 | 0.03% | 0.03% | 0.38% |

- **TOP kernel(批3 cudagraph ON,按聚合 dur 降序)**:
  - `Cijk_..._MT64x32x32_..._GSU1`:5.074s / 13216 次 / 383.9us (GDN/FLA) ← **GSU matmul 绝对主力(占 GDN 68%)**
  - `Cijk_..._MT32x16x4_..._GSU1_`:1.717s / 15222 / 112.8us (GDN/FLA) ← GSU matmul 次主力(频次最高)
  - `Cijk_..._MT128x32x32_..._GSU`:0.422s / 1888 / 223.7us (GDN/FLA) ← 大 tile config
  - `Cijk_..._MT32x32x32_..._GSU8`:0.089s / 5664 / 15.6us (GDN/FLA)
  - `fused_recurrent_gated_delta_rule_packed_decode_kernel`:0.075s / 5664 / 13.2us (GDN/FLA 递归核)
  - `Cijk_B_PostGSU`:0.037s / 7552 / 4.9us (GDN/FLA GSU 融合)
  - `kernel_unified_attention_3d`:0.020s / 1888 / 10.6us (FullAttn,唯一 attention 核)
  - FFN GEMM(`triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_4` 等)合计 ~1.5%,LayerNorm ~0.45% —— 全部可忽略
- **批3 tile config**:cudagraph ON 下 Triton autotune 选 `MT64x32x32`(主力)+ `MT32x16x4`(次主力)+ `MT128x32x32`(大 tile),与 eager 批1(`MT64x32x32`+`MT32x16x4`)方向一致。批3 `MT64x32x32` 单 call 383.9us 远高于 eager 批1 的 162.6us —— 反映 cudagraph capture 后实际 batch 增大(每 step 处理更多 seq),非 kernel 变慢。
- **decode step 周期(批3 钉死)**:`Cijk_B_PostGSU` count=7552 median_gap=**1.00ms**,`triton_poi_fused_0` median_gap=1.00ms → **cudagraph ON 下 decode step ≈ 1.0ms**,对比 eager 批1 ~2.3ms / 批2 ~2.5ms → **cudagraph 把单步压到 ~1.0ms(约 2.4×)**,与 cudagraph replay 消除 Python/调度 overhead 的预期一致。
- **批3 核心结论钉死**:
  1. **GDN/FLA 占比 95.17%(三批最高),GDN GEMM 主导结论在 baseline cudagraph ON 路径下比 eager 更强成立** —— 代表性疑虑消除,GDN GEMM 主导是路径无关的确定性结论。
  2. **FFN GEMM 仅 1.50%、FullAttn 0.26%、Sampling 0.03%** —— cudagraph ON 下 attention/FFN/sampling 同样非 GPU 瓶颈,与 eager 一致。
  3. **关键新发现:cudagraph ON 下 decode step ≈ 1.0ms,但 baseline mean_tpot=69.8ms → step 之间有 ~70× 非单步开销**。即 cudagraph 已把"step 内部"开销压到极低(1ms),**端到端 tpot 瓶颈完全在"step 之间"**(streaming 返回往返 / 跨步调度间隙 / 等待),GPU kernel 不再是端到端瓶颈。这把 §核心结论3 的"28–30× gap 来自 CPU overhead"细化钉死:**cudagraph 消除的是 step 内调度(2.4→1.0ms),但 step 间开销(~69ms)独立存在,cudagraph 管不到 —— 优化重心应转向 step 间(见 §P2 CPU/调度 overhead 调研)。**

> ⛔ **【勘误 2026-09】本条(第 3 条)已被本节顶部推翻块(2026-07-09 晚 duty cycle 97.3%)整体推翻**:1.00ms 是**同一 token 内相邻层(kernel)间隔**,不是跨 token 间隔;窗口内 118 token × 67.95ms/token ≈ tpot 69.8ms,端到端瓶颈 = **step 内部 64 层 GPU kernel 串行**,不存在"~70× step 间非单步开销"。第 1/2 条(GDN 主导、FFN/FullAttn 非 GPU 瓶颈)不受影响,仍然成立。

### P2 — decode 优化点定位

### P2 — CPU/调度 overhead 调研(2026-07-09,仅设计/调研阶段)
- **分工**:本窗口走 CPU/调度轨道(另一窗口走 GDN GEMM 轨道)。产出形式 = 仅设计/调研(不改源码、不进容器、不实测)。详见 `09_cpu_sched_overhead_design.md`。
- **⚠️ 重要修正**:旧判断(§5.3 第 4 条)"未设 `--async-scheduling` → 默认关闭 → `max_concurrent_batches`=1 串行"**前提可能有误**。静态代码 `config/vllm.py:706-773`:本配置(TP=1→`uni` executor、`speculative_config=None`、未显式设 async_scheduling → 字段=`None`)会走 `elif ... is None` → `else` → **`async_scheduling=True`**。即 async_scheduling **代码推断已开**,30× gap 主因不能简单归到"async 串行"。**待落地第一步验证**:启动日志 `grep "Asynchronous scheduling is" vllm_start.log` → enabled/disabled 定主因。
- **30× gap 来源候选(A–G)** + **可执行优化点(1–5)** + **落地路线图** 全在 `09`。最高优先 = 优化点1(确认/开启 async_scheduling,若实测关着)。其余:消除 `.cpu()` 同步点(来源 C)、调大 `stream_interval`(来源 F,需确认锁定)、cudagraph bucket 对齐(来源 G)、preprocess 微优化(来源 D)。
- **落地前置**:等另一窗口 cudagraph ON 的 decode profile(`05` §P2 ⚠️ 代表性说明:eager 下 30× gap 不能外推到 cudagraph ON)。
- **2026-07-11 质疑审查结论(`11` §6)**:CPU/调度轨道能否复活,**唯一取决于质疑1(占空比 97.3% 代表性)**。duty cycle 测的是"kernel 间 gap"——若长窗口稳态中段出现 idle(在 kernel 之间),正好复现 step 间空闲 → 轨道复活。复核动作:抓 30s+ 长窗口 trace 重算 duty,**看 idle 时间分布不只看平均数**(8s 中段截取偏置可能掩盖请求边界 idle)。此为当前最高优先方向性复核,优先于 GDN 轨道候选 B/C/D。

### P3 — FFN GEMM 实测复核
- FFN GEMM 总占比 ~9% 内,Triton GEMM 仅 hipBLAS 43%。P0/P1 拿下 fill 后 FFN 才成新瓶颈,届时评估切 hipBLAS 路径或低精度。

---

## 5.3 下一步行动(下次开工先读这里)

1. **P0 已闭合(2026-06-30)**:256MB fill 根因 = Triton autotune 的 L2-cache-flush buffer,详见 `07_p0_conclusion.md`。无需再做 fill 定位工作。
2. **P0.5 已闭合(2026-06-30)**:稳态 serving 首请求仍 re-autotune 91 次 256MB alloc —— fill 贯穿 init+稳态,非纯冷启动。详见 `07_p0_conclusion.md` §7.4.1。
3. **P1 已闭合(2026-07-02 晚)**:候选1 单独实测证伪 —— 8-16K `output_throughput`=7.26 < baseline 8.8,且首跑=二跑(7.253 vs 7.257),证明 autotune 命中磁盘缓存对单请求吞吐毫无影响。**fill 消除对单请求在线推理(batch=1)无收益,fill 非单请求吞吐主因。** 候选2 已早先回滚。详见 §5.1 / §5.3.5。
4. **下一步 = P2-decode 路径(待与用户定打法)**:
   - 单请求 `output_throughput` 瓶颈主体是 decode:`mean_tpot=69.8ms/step` × ~1717 tokens ≈ 120s(duration 主体)。fill 是 prefill 期 autotune 一次性开销,被 decode 稀释到可忽略。
   - decode 真实耗时分布(prefill vs decode 拆分、单步内各算子占比)待 profile 复核 —— 现有 profile(§3.1)是聚合统计,需拆出 decode-only 视图。
   - **可改方向(在锁定约束内)**:KV Cache/显存、Decode 算子、执行调度、算子级非持久化低精度、改 vLLM 框架代码。**不可改**:模型权重/结构、scheduler/sampling 参数、接口、投机解码。
   - ViT 29%(§3.1)是聚合噪声,纯文本 bench 不触发,非 decode 关注点。
5. **若需重测 fill 实际占比**:probe 桩已从源码移除(§5.1),`tools/_apply_probe_v2.py` 仍可重跑插桩;但生产压测务必用无桩干净版。

### 5.3.0 当前代码状态(2026-06-30 晚,容器已关停)
- `start_vllm.sh`:**已含候选1**(`TRITON_CACHE_AUTOTUNING=1` + `TRITON_CACHE_DIR=/public/.../triton_autotune_cache`),`bak` 备份在同目录。
- `driver.py`(容器内 triton dist-packages):**已回滚 256MB**,`bak_256m` 备份在旁。⚠️ 容器重启后 triton 包会重置回镜像默认 256MB,候选1 的 env 仍在 start_vllm.sh 里持久。
- `gpu_model_runner.py`(vllm_cscc 源码):**probe 桩已全清**,重编 wheel 已装(`PROBE_CLEAN`)。⚠️ 容器重启后 dist-packages vllm 会丢,需重装 `dist/vllm-*.whl`(Jun 30 19:10 编译的干净版)。
- `triton_autotune_cache/`:已清空(0 个 `.autotune.json`),下次启动重新生成(用正确 256MB 精度选核)。

### 5.3.1 fill 发生阶段:运行时每步回放(2026-06-29 钉死)
- **物证(本地 CSV)**:`vector_kernel_params.csv` 全量 54619 条采样,qi 1→38690。big int-fill(`FillFunctor<int>`+grid=16,777,216)7327 条,qi 跨 2398→33398(占 80%),每 1000-qi 桶 ~226–250 次连续不聚集。若是 capture-only 应只在 trace 起点成簇 → 实测相反,贯穿整个压测。
- **步数反推**:聚合 `result_23940` 全量 `FillFunctor<int>`=37575 次 ÷ 64 层 ≈ 587 步 ≈ 每 forward 每层 1 次,与一次 benchmark 的 prefill+decode 步数吻合。
- **结论**:capture 期录进图,replay 期每个 forward step 回放。capture 期插桩不足以钉死它(那一次 forward 可能根本没 256MB fill)。

### 5.3.2 DCU 访问链路 → 见 `08_dcu_access_link.md`
- **完整、最新的访问方法已移到 `08_dcu_access_link.md`**(MCP ssh-sessions 嵌套 ssh:login → 计算节点 → `root@173.0.8.2` worker-0 容器)。
- 本节旧内容(本地 `ssh` + `docker exec` 进容器)已废弃:docker exec 被组委会修复不可用、本地直连 173.0.8.2 三把密钥 Permission denied。**别再用旧方法,直接看 `08`**。
- 仍需记住的两点(细节见 `08`):① 节点名随作业变,每次用 `/opt/gridview/slurm/bin/squeue`(全路径)重取;② 容器内命令一律 `bash -lc` 包裹(否则 `libgalaxyhip.so.5` ImportError)。

### 5.3.3 vllm 源码改动与生效流程(钉死)
- 改源码路径:`/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/...`(工作区副本 `vllm_optimize_data/` 改了**不生效**)。
- **不是 editable 安装**:dist-packages 是真实拷贝(md5 与 vllm_cscc 改后一致才生效)。流程:`cd vllm_cscc && python setup.py bdist_wheel` → `pip install --force-reinstall --no-deps dist/vllm-*.whl` → kill 旧 vllm → 重启 `start_vllm.sh`。
- 改完立即 `python -c "import ast; ast.parse(open(<file>).read())"` 校验语法,别等 bdist_wheel 失败才发现。

### 5.3.4 fill_alloc_probe v2(策略1 全程快照,2026-06-30 已部署待验证)
- **背景**:v1(capture 期一次性快照)实测 `exact_256MB=0` → 256MB buffer 不在 capture 期分配。v2 改全程记录 + 多检查点快照。
- **probe 文件**:`/public/home/xdzs2026_c150/zya/fill_alloc_probe.py`(本地 `tools/_fill_alloc_probe.py`,md5 `f730332421788407051ca0cfe17a014e`)。新增接口:`begin_lifetime_probe()`(开 alloc-only recording)、`checkpoint_snapshot(tag)`(检查点快照,不停止 recording)、`stop_lifetime_probe(tag)`。旧 `begin_capture_probe/end_capture_probe` 保留为兼容包装。
- **插桩(2026-06-30 已 patch 进 vllm_cscc 源码,SYNTAX_OK,但尚未 bdist_wheel/装/重启——容器即将关闭,留明早)**:
  - `load_model()` 末尾(`get_offloader().post_init()` 后)插 `begin_lifetime_probe()` —— EngineCore 初始化早期开启全程记录。
  - `capture_model()` 入口(`# Trigger CUDA graph capture` 注释后)插 `checkpoint_snapshot(tag='pre_capture')` —— capture 前快照。
  - `capture_model()` 出口(`set_cudagraph_capturing_enabled(False)` 前)插 `checkpoint_snapshot(tag='post_capture')` —— capture 后快照。
  - patch 脚本:`tools/_apply_probe_v2.py`(一次性,幂等:先清 v1 桩再插 v2 桩)。已在容器内跑过一次,产物在源码里;明早重编前若 vllm_cscc 源码被重置需重跑此 patch。
- **明早验证步骤**:
  1. 重取作业/节点/容器(§5.3.2)。
  2. 确认 `fill_alloc_probe.py` 在 `/public/home/xdzs2026_c150/zya/` 且 gpu_model_runner.py 含 `begin_lifetime_probe`/`checkpoint_snapshot` 桩(若被重置,重跑 `tools/_apply_probe_v2.py`)。
  3. `cd vllm_cscc && python setup.py bdist_wheel` → `pip install --force-reinstall --no-deps dist/vllm-*.whl`。
  4. kill 旧 vllm → `start_vllm.sh` 重启 → 等待 `LIFETIME_RECORDING_STARTED` / `CKPT#1 tag=pre_capture` / `CKPT#2 tag=post_capture` 日志。
  5. 跑一次压测触发 capture+forward(prefill+decode)。
  6. 取 `logs/fill_alloc_probe_ckpt1_pre_capture.jsonl` 和 `ckpt2_post_capture.jsonl`,看 `SUMMARY exact_256MB=` 是否 >0;>0 则读 `MATCH[..] EXACT_256MB` 块的 `USER/VLLM FRAMES` → 直接给分配点文件:行号。
  7. 若两个检查点都 0:说明 256MB buffer 在 `post_capture` 之后(运行时首请求)才分配 → 需在首请求后再加一个 `checkpoint_snapshot(tag='post_first_req')` 检查点(可插在 `execute_model` 首次调用末尾,或直接靠 `stop_lifetime_probe` 在进程退出时抓)。
- **日志位置**:`/public/home/xdzs2026_c150/zya/logs/fill_alloc_probe_ckpt{N}_{tag}.jsonl`。

### 5.3.5 候选1 实测证伪物证(2026-07-02 晚)

候选1 单独(`TRITON_CACHE_AUTOTUNING=1` + 持久 `TRITON_CACHE_DIR` + 256MB cache 未改)8-16K 单请求压测两次,物证来自 `/public/home/xdzs2026_c150/zya/test/8-16K_throughput/result.json`(worker 容器):

| 指标 | run1 | run2 | baseline(6-30 候选1+2 叠加前) |
|---|---|---|---|
| `output_throughput` | **7.2532** | **7.2570** | **8.8** |
| `total_token_throughput` | 574.75 | 575.09 | — |
| `duration` | 236.722 | 236.598 | — |
| `completed` / `failed` | 10 / 0 | 10 / 0 | — |
| `total_output_tokens` | 1717 | 1717 | — |
| `mean_ttft_ms` | 11730.87 | 11724.16 | — |
| `mean_tpot_ms` | 69.78 | 69.79 | — |
| `mean_itl_ms` | 69.16 | 69.12 | — |

**两个决定性事实**:

1. **7.26 < baseline 8.8**(倒退 ~17.5%)—— 候选1 单独不仅无收益,反而低于 baseline。
2. **首跑 = 二跑**(run1 vs run2:`output_throughput` 7.2532 vs 7.2570,差 0.05%;`duration` 236.72 vs 236.60;TTFT/TPOT/ITL 三档全一致到小数点后两位)—— 证明 **7.26 不是 autotune 冷启动税**:若首跑含 autotune 开销而二跑命中磁盘缓存,二跑应明显更快。实测两跑几乎完全相同 → autotune 命中磁盘缓存对单请求吞吐**毫无影响**。

**为什么候选1 对单请求吞吐无效**:`output_throughput = total_output_tokens / duration`。`temperature=0` → token 序列固定(两跑 `total_output_tokens` 都 1717)。`duration` 主体是 **decode**:`mean_tpot=69.8ms × ~1717 tokens ≈ 120s` + 10 请求 TTFT 合计 ~117s。fill(256MB memset)是 **prefill 期 autotune 一次性开销**,在 batch=1 单请求下占比被 decode 的 ~120s 稀释到可忽略。autotune 命中缓存省的是"那一次 prefill autotune 的 `do_bench` 时间",而非每个 decode step 的时间 → 单请求 throughput 不动。

**结论钉死**:**fill 消除对单请求在线推理(batch=1)无收益,fill 非单请求吞吐主因。** decode(`mean_tpot=69.8ms`)才是吞吐瓶颈主体。判定分支(§5.3 第 4 条)落地:转 **P2-decode 路径**。

**候选1 机制本身仍正常**(非误判):autotune cache 15 个 `.autotune.json` 落盘(6 类 kernel)、`TRITON_CACHE_AUTOTUNING=1` + `TRITON_CACHE_DIR` 在 APIServer 进程 environ 里、07-02 新生成 5 个 key(含 int32 dtype 新形状)、`chunk_scaled_dot_kkt_fwd`/`chunk_local_cumsum_scalar` 沿用 06-30 旧文件(命中磁盘缓存)。机制工作正常,只是对单请求吞吐无效。更正了上次会话"0 个 `.autotune.json` = 候选1 未激活"的错误判断(实为 find 路径/通配问题,文件一直存在)。
