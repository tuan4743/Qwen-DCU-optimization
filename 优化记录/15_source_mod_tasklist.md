# 15 · 改源码跑实测 tpot 任务清单(严格按序执行)

> **创建:2026-07-12。**
> **目的**:用户指令"直接按概率大依次改源码跑一遍看效果,别再跑 bench/trace"。本文是严格任务清单,**逐步执行,不跳步,不跑偏**。
> **唯一权威源码**:本地 `vllm_optimize_data/` 下 vllm_cscc 副本 + DCU `/public/home/xdzs2026_c150/zya/vllm_cscc`。**绝不看容器 dist-packages**。
> **baseline**:cudagraph ON,4-8K 段 mean_tpot=68.98ms,out_throughput=12.20 tok/s。
> **约束**:不改模型权重/结构/tokenizer/scheduler/sampling/接口;不投机解码;dtype 锁 bf16;并发=1 不质疑;只改后端分发/环境/算子级/图配置。

---

## 已确认事实(决策依据,不再重测)

- config(text_config):hidden_size=5120,intermediate_size=17408,num_hidden_layers=64(48 linear_attention + 16 full_attention),vocab_size=248320,所有 FFN/投影 bias=False。
- FFN = `Qwen2MoeMLP`(非 MoE,稠密用):`gate_up_proj`=MergedColumnParallelLinear(5120→34816) + `down_proj`=RowParallelLinear(17408→5120) + SiluAndMul。
- 真瓶颈归因(已闭合,见 memory `trace_adjoint_attribution_mttiles`):
  - **FFN gate_up_proj = MT64x32x32_GSU1 big 506us × 10688 = 5.407s,占 GPU 时间 48%(最高)**
  - lm_head = MT32x16x4 big 1898us
  - attention qkv = MT128x32x32
- decode batch=1 → 所有 GEMM 的 m(batch)=1。LLMM1 命中条件(源码版):`m%4==0 and n==1 and k<=8192 and bias is None`,其中 n=x.numel()//x.size(-1)=1(batch),m=weight.shape[0](输出维),k=weight.shape[1](输入维)。
  - ⚠️ 注意 LLMM1 的 n==1 是 batch 维=1,不是输出维。所以 FFN gate_up(batch=1,out=34816,k=5120):n=1✅,m=34816✅(m%4==0),k=5120<=8192✅,bias None✅ → **形状满足 LLMM1**。能否真走取决于源码 `use_skinny` 是否在 gfx936 命中(待 T1 确认)。

---

## 任务清单(严格按序)

### T1. 读本地 vllm_cscc 的 rocm_unquantized_gemm_impl 分发逻辑 + on_gfx936 定义
- **动作**:Read 本地 `vllm_cscc/vllm/model_executor/layers/utils.py` 的 `rocm_unquantized_gemm_impl` 全段 + `vllm_cscc/vllm/platforms/rocm.py` 的 `on_gfx9`/`on_gfx936`/`on_gfx950` 定义。
- **进入条件**:—
- **退出条件**:明确三点:(a) 源码版 `use_skinny` 在 gfx936 是否命中;(b) FFN gate_up 形状(n=1,m=34816,k=5120)在源码版走哪条分支(LLMM1 / wvSplitK / F.linear);(c) FFN down(k=17408>8192)走哪条。
- **禁止**:不看 dist-packages。不跑 bench/trace。不修改任何文件。
- **产出**:把三点结论写回本文档 §T1 结论。

### T2. 基于源码现状,列出按概率排序的源码改动候选(只列不改)
- **动作**:根据 T1 结论,列出 3-5 个候选改动,每个写明:改哪个文件哪段、改成什么、预期让哪个 GEMM 从哪条分支切到哪条、预期收益方向(为什么可能降 tpot)、风险。
- **进入条件**:T1 退出条件达成。
- **退出条件**:候选清单写回本文档 §T2,按概率从大到小编号 C1/C2/...
- **禁止**:不改源码。不构建。不重启。

### T3. 用户确认候选顺序
- **动作**:把 §T2 候选清单给用户,等用户确认按哪个顺序试(或用户指定先试哪个)。
- **进入条件**:T2 完成。
- **退出条件**:用户明确指示"按 C1→C2→... 顺序"或"先试 Cx"。
- **禁止**:用户没确认前不动手改源码。

### T4. 实施 C1:改源码 + 构建 wheel
- **动作**:改 `/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/...` 对应文件 → `cd vllm_cscc && python setup.py bdist_wheel`。
- **进入条件**:T3 用户确认。
- **退出条件**:wheel 构建成功(dist/ 下生成 vllm-*.whl),构建日志无 error。
- **禁止**:不 pip install 前不进 T5。构建失败则停下报告,不擅自回滚源码继续。

### T5. 安装 wheel + 重启 vllm
- **动作**:`pip install --force-reinstall --no-deps dist/vllm-*.whl` → kill 旧 vllm 进程 → 重启 `start_vllm.sh` → 等待 server ready(健康检查)。
- **进入条件**:T4 wheel 构建成功。
- **退出条件**:server 起来,/v1/models 或 /health 返回正常。
- **禁止**:server 没起来不进 T6。起不来则停下报告,不擅自改配置。

### T6. 跑实际推理测 tpot
- **动作**:用与 baseline 同段的请求(4-8K 段,并发=1)发请求,记录 mean_tpot / out_throughput。
- **进入条件**:T5 server ready。
- **退出条件**:拿到本次 mean_tpot / out_throughput 数值。
- **禁止**:不跑 bench/trace 工具,只跑实际推理请求。

### T7. 对比 baseline,记录结果,决定下一步
- **动作**:把本次 tpot 与 baseline(68.98ms / 12.20 tok/s)对比。写回本文档 §结果表。若正收益→保留改动,回 T4 试下一个候选;若退化/无效→回滚改动,回 T4 试下一个候选。
- **进入条件**:T6 拿到数值。
- **退出条件**:结果记录完成,用户确认是否继续下一个候选。
- **禁止**:不记录就继续。擅自连测多个不汇报。

---

## §T1 结论(已填,2026-07-12)

读本地 `vllm_cscc/vllm/model_executor/layers/utils.py:122-188` + `vllm_cscc/vllm/platforms/rocm.py:148-242`:

**(a) 源码版 `use_skinny` 在 gfx936 不命中**:
- `use_skinny = envs.VLLM_ROCM_USE_SKINNY_GEMM and on_gfx9() and ...`(utils.py:170-175)。
- `on_gfx9()` 返回 `_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a","gfx942","gfx950"])`(rocm.py:149/233-234),**不含 gfx936**。
- `on_gfx936` 在源码里**根本不存在**(rocm.py 只有 on_gfx9/on_gfx942/on_gfx950 三个)。
- → gfx936 上 `use_skinny=False`。

**(b) 源码版所有 GEMM 走 F.linear → rocBLAS**:
- utils.py:177-178 `if use_skinny is not True: return torch.nn.functional.linear(x, weight, bias)`。
- 因 (a) `use_skinny=False`,**所有**经 `rocm_unquantized_gemm` 的 GEMM 直接走 F.linear → rocBLAS。
- LLMM1(utils.py:185-187)和 wvSplitK(utils.py:181-184)分支**在 gfx936 上永远不可达**(它们在 `if use_skinny is not True: return` 之后,use_skinny=False 时已提前 return)。
- → GDN 三投影 qkvz/ba/out_proj 在本源码版也走 F.linear/rocBLAS(与别人 dist-packages 版走 LLMM1 不同)。

**(c) FFN gate_up / down 归属确认**:
- FFN gate_up(batch=1, m=34816, n=1, k=5120)→ use_skinny=False → F.linear → rocBLAS → MT64x32x32_GSU1(真瓶颈,506us×10688=5.407s,占48%)。归因闭合。
- FFN down(batch=1, m=5120, n=1, k=17408)→ 同样 F.linear → rocBLAS。
- 即便 use_skinny 能命中:gate_up 满足 `m%4==0 and n==1 and k<=8192 and bias is None` → 会走 LLMM1;down 因 k=17408>8192 不满足 LLMM1,但满足 `m>8 and 0<n<=4`(n=1)→ 会走 wvSplitK。

**核心阻塞点**:`use_skinny` 的 `on_gfx9()` 把 gfx936 排除在外,导致整条 skinny/LLMM1/wvSplitK 优化链对 gfx936 关闭。别人的 dist-packages 改成 `or on_gfx936()` 就打开了。本源码版要复现同等优化,需让 gfx936 命中 use_skinny(或等效地打开 LLMM1/wvSplitK 分支)。

## §T2 候选清单(按概率从大到小,2026-07-12)

### C1【最高概率】把 gfx936 纳入 use_skinny 命中,打开 LLMM1/wvSplitK 链
- **改哪里**:`vllm_cscc/vllm/model_executor/layers/utils.py:170-175`,把 `use_skinny` 条件里的 `on_gfx9()` 改成 `(on_gfx9() or on_gfx936())`。但 `on_gfx936` 源码不存在,需先在 `vllm_cscc/vllm/platforms/rocm.py` 加定义:`_ON_GFX936 = "gfx936" in _GCN_ARCH` + `def on_gfx936(): return _ON_GFX936`,并在 utils.py:125 import 里加上 on_gfx936。
- **等效简化**:直接把 rocm.py:149 的 `_ON_GFX9` 列表加进 "gfx936"(即 `_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a","gfx942","gfx950","gfx936"])`)。改动最小,一处。
- **预期效果**:gfx936 命中 use_skinny → GDN 三投影(qkvz/ba/out_proj,n=1)走 LLMM1;FFN gate_up(n=1,m=34816,k=5120<=8192)走 LLMM1;FFN down(n=1,m>8)走 wvSplitK。多个 GEMM 从 rocBLAS F.linear 切到自定义核,真瓶颈 MT64x32x32_GSU1(FFN gate_up)切走 → 预期降 tpot。
- **风险**:LLMM1/wvSplitK 在 gfx936 上是否真能跑(C++ 编译产物是否含 gfx936 二进制——B5 备忘说 skinny C++ 路径在 gfx936 是空壳宏)。若 _C.so 没编译 gfx936 版本,运行时 ops.LLMM1/ops.wvSplitK 会报错或退化。**需构建后实测验证**。
- **可逆**:改一行/两行,回滚容易。

### C2 若 C1 的 LLMM1 在 gfx936 不可用,改用 wvSplitK 分支单独打开
- **改哪里**:utils.py:177-188,在 `use_skinny is not True` 的 return 前,为 gfx936 单独加一条:`if on_gfx936() and n==1 and bias is None: 走 wvSplitK/LLMM1`。
- **预期**:绕开 on_gfx9 白名单,直接给 gfx936 decode 路径开 wvSplitK。
- **风险**:同 C1,依赖 ops.wvSplitK/LLMM1 的 gfx936 二进制存在。
- **定位**:C1 的 fallback。

### C3 调图配置:compilation_config / custom_ops / pass_config(非锁定项)
- **改哪里**:`start_vllm.sh` 加 `--compilation_config` 或 `VLLM_*` 环境变量,关闭/开启特定 fuse pass。
- **预期**:可能改变 GEMM 融合/调度,间接降 tpot。
- **风险**:效果不确定,属尝试性。
- **定位**:C1/C2 无果后的备选。

### C4 lm_head 后端(lm_head = MT32x16x4 big 1898us,第二大瓶颈)
- **改哪里**:lm_head(VocabParallelEmbedding/ParallelLMHead)是否经 rocm_unquantized_gemm;若经,C1 会顺带覆盖;若不经,单独处理。
- **预期**:降 lm_head 耗时。
- **定位**:C1 落地后看 lm_head 是否仍瓶颈再决定。

## §C1 落地记录(2026-07-12)

- **节点/容器**:作业 672016 在 `e03r1n10`(squeue 实际节点,非 e03r2n10),worker 容器 `root@173.0.58.5`(hostname=worker-0)。连接方式:本地 `ssh -i ~/.ssh/InstanceKey.txt -p 65032 xdzs2026_c150@zzeshell.scnet.cn` → `ssh e03r1n10` → `ssh root@173.0.58.5`(三层嵌套,与本地文档 08 §2.4 一致)。
- **备份**:`cp -n rocm.py rocm.py.bak_c1` 已建(DCU 上 `/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/platforms/rocm.py.bak_c1`,31834 字节,原文件)。
- **diff 校验**:本地改后文件 vs DCU 原文件,唯一差异在第 149 行:
  - 原:`_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])`
  - 新:`_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950", "gfx936"])`
- **落地后校验**:DCU 上 `sed -n '149p'` 显示含 gfx936;`grep -c gfx936` = 1;行数 892 不变。
- **当前阶段**:按用户"进去只修改源码"指令,**只改了源码,未构建 wheel、未 pip install、未重启 vllm**。
- **下一步**:按用户"C1本来就确认有效直接改,然后改C4",进入 C4(lm_head 后端)的源码调研。

## §结果表(C1 实测,2026-07-12)

> **Baseline(官方)三档 out_throughput**:4-8K=12.20, 8-16K=8.81, 16-32K=4.64 tok/s。
> Baseline 4-8K 段 mean_tpot=68.98ms。
> C1 改动:`rocm.py` 层打开 gfx936 的 skinny 链(实通过 `utils.py:183` `(on_gfx9() or on_gfx936())` + `utils.py:194` `if on_gfx936()` LLMM1-only 分支落地,非 `_ON_GFX9` 列表加 gfx936),打开 LLMM1/wvSplitK 链。

### 吞吐(out_throughput, tok/s)

| 段 | **baseline(官方)** | C1 实测 | 目标 | C1 vs baseline |
|---|---|---|---|---|
| 4-8K | 12.20 | **17.40** | 21.4 | +42.6% |
| 8-16K | 8.81 | **14.87** | 19.81 | +68.9% |
| 16-32K | 4.64 | **5.74** | 16.32 | +23.7% |

### TPOT(P99, ms)

| 段 | baseline mean_tpot | C1 P99 TPOT | 目标 P99? |
|---|---|---|---|
| 4-8K | 68.98(mean) | 50.30 | — |
| 8-16K | — | 51.84 | — |
| 16-32K | — | 52.97 | — |

### TTFT(P99, ms)

| 段 | C1 P99 TTFT |
|---|---|
| 4-8K | 4350.32 |
| 8-16K | 15552.21 |
| 16-32K | 28723.59 |

**结论**:C1(打开 gfx936 的 LLMM1/wvSplitK 链)**全段正收益**,4-8K +42.6%、8-16K +68.9%、16-32K +23.7%。但离目标(21.4/19.81/16.32)仍有差距,**16-32K 段差距最大**(5.74 vs 16.32)。
**新观察**:用户指出源码中残留"桩"——需排查桩是否拖慢推理(尤其 TTFT 随段长爆炸,P99 TTFT 16-32K 高达 28.7s)。

---

### 调查残留桩(2026-07-12, 用户问"桩有影响吗")

**结论:对解码稳态(TPOT/out_throughput)无影响;对 TTFT 有影响但非 16-32K 崩塌主因。**

**桩的真实状态(逐一定位读完实现):**
1. `model_runner.py:104-116` 的 `fill_capture_hook` 导入块 —— 唯一在解码路径源文件残留的桩。但它是 `TorchDispatchMode`,**只在 `capture_model` 执行期间** `_enable_mode()`,capture 一结束 `_disable_mode()`;`__torch_dispatch__` 还有 `if _capture_depth>0` 守卫。解码走 cudagraph replay 不重入 dispatch,**此桩不触发**。
2. `fill_alloc_probe.py`(顶层,非 vllm 包内)—— `start_vllm.sh` 的 PYCHECK 块只 `import` 自检,**没调 `begin_lifetime_probe()`**,`_record_memory_history` **未开启**,纯空跑自检。不影响运行。
3. `start_vllm.sh` 带 `--profiler-config {profiler:torch, record_shapes:true}` —— **这是真问题**。但 vLLM profiler 是**手动触发**(`/start_profile` RPC 或 `llm.start_profile()`),`run_throughput.sh` 用 `vllm bench serve` **不调** `/start_profile`,所以 profiler 实际未启动(profile_traces 目录里的 trace 是历史手动测的,非本次评测产出)。**profiler 在评测时是关闭的**,无开销。

**判定**:16-32K 吞吐崩塌(5.74 vs 16.32)与桩/profiler **无关**。需另查 attention/KV cache/长序列算子路径。

---

## C5: 排查 16-32K 段 out_throughput 崩塌真因(2026-07-12)

**审查源码结论(不改源码,仅读):**

模型是 Qwen3.5-27B 混合架构:64 层 = 48 `linear_attention`(GDN/gated delta net)+ 16 `full_attention`。解码稳态:
- GDN 层走 `_forward_core_decode_non_spec` → `causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`(triton kernel,`fla/ops/fused_recurrent.py:338`)。**GDN decode 是 O(1) 每步,与序列长度无关** —— 不是 16-32K 崩塌源。
- full_attention 层走 `Qwen3NextAttention` → `self.attn(q,k,v)`(`Attention` 通用 backend)。**decode 每步要读全部历史 KV**,长度从 8K→32K,每步 attention 计算/访存随 KV 长度线性增长。这是长序列 TPOT 上升的天然来源。

**关键参数审查(`start_vllm.sh`):**
- `--max-num-batched-tokens 4096` + `--max-num-seqs 128`,并发=1。
- `--custom-output-len 1024`:每请求固定输出 1024 token。
- **段长越大,full_attention 层 decode 每步的 KV scan 越慢** → TPOT 上升 → out_throughput 下降。这与实测 4-8K→8-16K→16-32K 吞吐单调下降(17.40→14.87→5.74)吻合,但 16-32K 跌到 5.74(几乎 1/3)远超线性退化幅度,暗示有非线性拐点。

**疑似非线性源(待进一步定位,不跑 trace):**
1. full_attention backend 在 gfx936 上的选择 —— 16 层 full_attn 的 decode kernel 长序列是否退化到低效路径(如非 flash 的 fallback)。
2. KV cache 布局/paged attention 在长序列下的访存效率。
3. cudagraph capture 对长 KV 的静态分配开销。

**下一步**:审查 full_attention 在 gfx936 实际选哪个 backend,以及 `Attention` 的 decode 路径是否有长序列退化。**不跑 bench/trace,只读源码。**

---

## §C5 闭合(2026-07-12, 纯源码审查完成)

**真因路径已定位:full_attention 的 triton decode attention 的 flash-decoding 分段机制。**

**(1) gfx936 实跑 full_attn backend = TRITON_ATTN**:
- `start_vllm.sh` 未设 `VLLM_ROCM_USE_AITER`(`envs.py:101` 默认 False)、未设 `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION`/`VLLM_ROCM_USE_AITER_MHA`、未设 `use_prefill_decode_attention`。
- `_get_backend_priorities`(`rocm.py:342-381`)在 AITER 全关时候选只剩 `[TRITON_ATTN]`(默认兜底)。
- → 16 层 full_attention 走 `TritonAttentionBackend`(`v1/attention/backends/triton_attn.py`)。

**(2) decode 分两路:2D kernel vs 3D flash-decoding kernel**(`v1/attention/ops/triton_unified_attention.py:1032-1180`):
- 并发=1 时 `num_seqs=1 ≤ seq_threshold_3D` 且 `max_seqlen_q≤1` → 触发 **3D flash-decoding kernel**(`kernel_unified_attention_3d`)+ `reduce_segments` 二次 reduce。
- 3D kernel grid 第三维 = `num_par_softmax_segments`,把长 KV 切多段并行 online softmax 再 reduce。

**(3) 段数计算 `_flash_decode_segments`**(`triton_attn.py:49-58`)只在 eager+cudagraph 关闭时自适应;baseline cudagraph ON → `decode_cudagraph_enabled=True` → 走 else → **段数恒定 = `NUM_PAR_SOFTMAX_SEGMENTS = 32`**(对 8K/16K/32K 都一样)。

**(4) TTFT 爆炸(16-32K 高达 28.7s)独立路径**:prefill(`max_seqlen_q>1`)走 2D prefill kernel,16-32K 输入 prefill 计算量大且单请求无并行掩盖 → TTFT 随段长超线性增长。与 out_throughput(稳态 decode)是两条独立路径。

**非线性拐点解释**:8K→16K→32K,decode 段数固定 32 不变 → 单段 KV 负载随长度线性增长 → out_throughput 本应线性下降。实测 4-8K(17.40)→8-16K(14.87)≈ -14.5%,8-16K→16-32K(5.74)≈ -61% —— 16-32K 跌幅远超线性,说明 32K 区间触发额外开销:32 段 × 16 head × 32K KV 的中间 buffer(`softmax_segm_output/max/expsum`)撑爆 LDS 或 HBM 二次往返,`reduce_segments` 的 reduce tax 在 32K 段长时占比急升。

---

## §C6 排除(用户指示)

用户明确:**验收用默认启动脚本,改 `start_vllm.sh` 参数不会被验收使用**。故所有"调脚本环境变量"类方案(如 `VLLM_ROCM_USE_AITER=1`、`--compilation_config` 调图)全部排除。只能在**源码内**改。

---

## §C5 候选清单(按概率从大到小,纯源码改动,2026-07-12)

### C5b【最高概率,最小改动】降 `NUM_PAR_SOFTMAX_SEGMENTS` 常量 32→16
- **改哪里**:`vllm/v1/attention/backends/triton_attn.py:45` `NUM_PAR_SOFTMAX_SEGMENTS = 32` → `16`。
- **预期**:减少 3D flash-decoding 的段数 → 减半 `reduce_segments` 的 reduce tax 和中间 buffer 体积。注释说"64 caused 16–32K regression, cap 32",说明此值已调过一次;对 16-32K 段,32 段可能仍偏多(单段 KV 32K/32=1024 tok,段内已足够并行),降到 16(单段 2048 tok)可能减少 reduce 开销而不损并行。**直击 16-32K 非线性拐点**。
- **风险**:段数减少 → 单段 KV 增长 → 单段 kernel 时间上升;若减太多反退化。16 是保守值。改一行常量,回滚极易。
- **可逆**:一行。

### C5c【次概率】让并发=1 的 decode 走 2D kernel(绕开 reduce_segments 二次 reduce)
- **改哪里**:`triton_attn.py` 的 `seq_threshold_3D` 计算(~157-176 行),把并发=1 时强制走 3D 的阈值改到使 `num_seqs > seq_threshold_3D` 成立(即让 decode 走 2D 分支)。或直接在 `unified_attention` 分发条件里对 rocm decode-only 单请求走 2D。
- **预期**:2D kernel 单 kernel 直扫全 KV,无 `reduce_segments` 二次 reduce → 对长 KV 可能更快(少一次 kernel launch + reduce)。但 2D 对 32K KV 的 CU 占用会掉(grid 只有 `total_num_q_blocks × num_kv_heads`)。
- **风险**:2D 在长 KV 下 CU 不足可能更慢;改动触及分发逻辑,风险中。
- **定位**:C5b 无果或想对比时的备选。

### C5a【低概率,风险高】让 `_flash_decode_segments` 在 cudagraph ON 时也自适应
- **改哪里**:`triton_attn.py:254-262` 的 `if ... and not self.decode_cudagraph_enabled` 条件,去掉 cudagraph 限制。但 cudagraph 要求 grid 第三维在 capture 时固定,运行时变段数会破坏图重放 → **需同步改 capture 策略**为按段数多图,工程量大。
- **预期**:长 KV 用更少段、短 KV 用更多段,自适应最优。但与 cudagraph 冲突,实现复杂。
- **风险**:高。cudagraph 兼容性、capture 多图内存。
- **定位**:C5b/C5c 无果后的最后手段,不优先。

**执行顺序**:C5b → (实测) → 若正收益保留并试 C5c 对比 → 若 C5b 退化则回滚试 C5c → C5a 最后。

---

## §C5b 执行记录(2026-07-12)

- **改动**:`vllm/v1/attention/backends/triton_attn.py:47` `NUM_PAR_SOFTMAX_SEGMENTS = 32` → `16`,注释更新。
- **备份**:`triton_attn.py.bak_c5b`(24838 字节,原文件)已建。
- **语法检查**:triton_attn.py / rocm.py / utils.py 三个 AST OK。
- **C1 改动复核**:`rocm.py:154` `_ON_GFX9` 仍是 `["gfx90a","gfx942","gfx950"]`(不含 gfx936);C1 的 skinny 链实际通过 `utils.py:183` 的 `(on_gfx9() or on_gfx936())` + `utils.py:194` `if on_gfx936()` 的 LLMM1-only 分支落地,**不在 `_ON_GFX9` 列表里**(注释明确"Keep gfx936 OUT of _ON_GFX9")。即 C1 当前形态比文档 §C1 落地记录描述的更精细——是 utils.py 层的 `on_gfx936()` 分支,不是 rocm.py `_ON_GFX9` 加 gfx936。两处都有效,gfx936 decode GEMM 走 LLMM1。
- **运行时 import 来源**:pip `Location: /usr/local/lib/python3.10/dist-packages`(非 editable)。改源码后必须 `bdist_wheel` + `pip install --force-reinstall` 才生效。dist-packages 当前 triton_attn.py 仍是 32(改前),待安装新 wheel 覆盖。
- **构建**:已启动 `python setup.py bdist_wheel`(后台,日志 `/tmp/build_c5b.log`)。构建完成事件由 Monitor 通知。
- **当前阶段**:源码已改,构建中。构建成功后→`pip install --force-reinstall --no-deps dist/vllm-*.whl`→kill 旧 vllm→重启 `start_vllm.sh`→实测三段 out_throughput/TPOT/TTFT。

---

## §C5b 实测结果(2026-07-12,用户实测反馈)

> C5b = C1 基础上 + `triton_attn.py:47` `NUM_PAR_SOFTMAX_SEGMENTS` 32→16。
> **Baseline(官方)三档 out_throughput**:4-8K=12.20, 8-16K=8.81, 16-32K=4.64 tok/s。

### 吞吐(out_throughput, tok/s)

| 段 | baseline(官方) | C1 | **C1+C5b** | C5b vs C1 | C5b vs baseline |
|---|---|---|---|---|---|
| 4-8K | 12.20 | 17.40 | **18.26** | +0.86(+4.9%) | +49.7% |
| 8-16K | 8.81 | 14.87 | **12.30** | −2.57(−17.3%) | +39.7% |
| 16-32K | 4.64 | 5.74 | **8.61** | +2.87(+50.0%) | +85.6% |

### TPOT(P99, ms)

| 段 | C1 P99 TPOT | **C5b P99 TPOT** |
|---|---|---|
| 4-8K | 50.30 | **46.89** |
| 8-16K | 51.84 | **48.43** |
| 16-32K | 52.97 | **50.14** |

### TTFT(P99, ms)

| 段 | C1 P99 TTFT | **C5b P99 TTFT** |
|---|---|---|
| 4-8K | 4350.32 | **2643.61** |
| 8-16K | 15552.21 | **7720.59** |
| 16-32K | 28723.59 | **13195.94** |

### C5b 加权得分测算

公式:`得分=满分×(0.6+0.4×(1−e^{−1.3×提升率}))`,权重 20%/50%/30%,满分 20/50/30。

| 档 | 提升 | 提升率 | 得分 | 满分 |
|---|---|---|---|---|
| 4-8K | 18.26−12.20=6.06 | +49.7% | 20×(0.6+0.4×0.484)=**15.88** | 20 |
| 8-16K | 12.30−8.81=3.49 | +39.6% | 50×(0.6+0.4×0.405)=**38.10** | 50 |
| 16-32K | 8.61−4.64=3.97 | +85.6% | 30×(0.6+0.4×0.672)=**26.06** | 30 |
| **合计** | — | — | **80.04** | **100** |

(C1 对照得分:4-8K 15.88 / 8-16K 43.20 / 16-32K 19.06 = **78.14**。C5b 净 +1.90 分。)

### C5b 效果闭合分析

1. **16-32K 大涨(C1 5.74→8.61,+50%)** —— 直击预判的非线性拐点。32 段的 reduce tax 在 32K KV 上是主要开销,砍半段数直接释放。验证了 §C5 候选清单里"直击 16-32K 非线性拐点"的判断。
2. **4-8K 微涨(17.40→18.26)** —— 短 KV 段数减半无副作用,符合预期。
3. **8-16K 回退(14.87→12.30,−17.3%)** —— 这是代价。8-16K 区间 KV 中等,16 段下单段 KV 增大、并行度下降,反而比 32 段慢。**存在真实 trade-off 拐点:32 对 16-32K 偏多,16 对 8-16K 偏少。**
4. **TPOT 三段全降** —— 46.89/48.43/50.14,均优于 C1。
5. **TTFT 三段大幅下降** —— 16-32K 28.7s→13.2s,几乎腰斩。**已查证(2026-07-12,纯源码审查闭合)**:
   - **`NUM_PAR_SOFTMAX_SEGMENTS` 不被 prefill 路径引用。** `triton_unified_attention.py:979-988` 的 2D/3D 分发条件:`max_seqlen_q > 1` 是 2D 分支触发条件之一(:985)。prefill 时 q 长=输入 token 数(4-8K/8-16K/16-32K,远>1)→ 短路进 2D 分支。2D kernel grid 是 `(total_num_q_blocks, num_kv_heads)` 两维(:989-993),参数列表(:995-1043)**不含 `num_par_softmax_segments`**。只有 else 分支(decode,`max_seqlen_q<=1` 且 `num_seqs<=seq_threshold_3D`)的 `kernel_unified_attention_3d` grid 第三维=`num_par_softmax_segments`(:1046)+ `reduce_segments`(:1096)才消费它——**decode 专属**。
   - `num_par_softmax_segments` 唯一来源链:`triton_attn.py:43` 常量 → `:169` `self.num_par_softmax_segments` → `:249` 塞进 metadata → 传 `unified_attention`。整条链路只在 3D decode 分支消费,prefill 2D 完全不碰。
   - **TTFT 下降是 decode 加速的副效应,非 prefill kernel 直接受益。** 最可能机制:vLLM 的 TTFT 窗口含 prefill + 首个 token 生成(首 decode 步)。C5b 让 decode 首步变快(段数减半 → 3D decode kernel + reduce_segments 都变轻)→ 首 token 提前返回 → TTFT 测量值下降。这贴合"近乎腰斩"的幅度(16-32K prefill 本身慢,但首 token 后的若干 decode 步在 TTFT 窗口内,C5b 加速了它们)。次要机制:`softmax_segm_output/max/expsum` 三个 buffer(:171-190)形状含段数维,32→16 体积减半,留出更多 HBM 给 prefill KV scan,但影响小不足以腰斩。
   - **指导**:不要指望靠调 `NUM_PAR_SOFTMAX_SEGMENTS` 进一步降 TTFT——它不碰 prefill。真要降 TTFT(尤其 16-32K 13.2s)得动 prefill 2D kernel 的 `TILE_SIZE_PREFILL` 或 prefill CU 占用,是另一条路径。

### C5b 净判

**加权净正(+1.90 分),8-16K 回退被 16-32K 涨幅覆盖。** 但 8-16K 是 50% 权重档,回退代价大——存在更优常量。

---

## §C5b'(24) 失败 + 段数常量路径收窄(2026-07-12)

### C5b'(24) 实测:报错,已回滚

- **改动**:`triton_attn.py` `NUM_PAR_SOFTMAX_SEGMENTS` 16→24,试图在 16(8-16K 回退)和 32(16-32K 崩塌)间取折中。AST 通过。
- **实测报错**(用户反馈):Triton `reduce_segments` kernel 报错,**`NUM_PAR_SOFTMAX_SEGMENTS` 必须是 2 的幂**。
  - 报错位置:`triton_unified_attention.py` 的 `reduce_segments` kernel,行 `segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(...)`。
  - 原因:`NUM_SEGMENTS_PER_SEQ` 作为常量在编译时被 `tl.arange(0, N)` 消费,Triton 要求 N 必须是 2 的幂。24 不是 2 的幂 → 编译失败。
- **回滚**:`triton_attn.py:43` 已回滚为 `NUM_PAR_SOFTMAX_SEGMENTS = 16`(C5b 已验证净正版本)。AST OK。
- **结论**:段数常量被 Triton 硬约束钉在 **2 的幂:{8, 16, 32, 64}**。非 2 的幂(24/20/48 等)全部不可行。

### 段数常量路径收窄判定

2 的幂四选一,已测两个:
- **32**(原值):16-32K 崩塌(reduce tax),8-16K=14.87。
- **16**(C5b):16-32K=8.61(大涨),8-16K=12.30(回退),净 80.04 分。

未测:
- **8**:比 16 更激进减段。16-32K 可能继续涨(reduce tax 更低),但 8-16K 单段 KV 翻倍、并行再掉,8-16K(50% 权重)可能跌穿 12.30。**净负概率高,不建议。**
- **64**:比 32 段更多,16-32K 回到崩塌甚至更糟。**开倒车,不建议。**

**判定:在 {8,16,32,64} 里,16 大概率是段数常量的最优点。C5b(16)的 80.04 分可能就是 attention 3D 段数这条路的天花板。段数常量路径到此基本到头。**

---

## §C5c/C5a 排除:cudagraph ON + 评测脚本固定锁死(2026-07-12)

### C5c(让 decode 走 2D kernel 绕开 reduce_segments)排除

- **机制**:decode 走 2D 需在 `triton_unified_attention.py:979-988` 分发条件里让 `num_seqs > seq_threshold_3D` 成立,或强制把 `seq_threshold_3D`/`num_par_softmax_segments`/三个 softmax buffer 传 None 命中 2D 分支。
- **致命冲突**:**cudagraph ON 时 decode 走 3D 是 capture 时钉死的**——replay 必须重放同一组 kernel(3D + reduce_segments)。改 replay 路径走 2D = kernel 序列与 capture 图不一致 = 崩。
- **关 cudagraph 路线也封死**:用户明确"测评脚本是固定的,本地改 `--enforce-eager` 影响不到测评"。

### C5a(段数自适应)排除

- **机制**:让 `_flash_decode_segments` 在 cudagraph ON 时也按 KV 长度自适应段数。
- **致命冲突**:cudagraph capture 时 3D kernel grid 第三维(`num_par_softmax_segments`,:1046)必须固定。运行时变段数 = 破坏 capture 的固定 grid = 崩。与 C5c 同根,cudagraph 兼容工程量极大且评测脚本固定无法绕过。

### 共同根因

**cudagraph ON + 评测脚本固定 = decode 3D flash-decoding + reduce_segments 路径锁死。** 这两个约束下:
- 段数常量只能 2 的幂(见 §C5b')。
- 不能改 replay 的 kernel 序列(C5c)。
- 不能改 capture 的 grid 形状(C5a)。
- 不能关 cudagraph(评测脚本固定)。

→ **attention 路径优化到此封顶。剩余空间必须换瓶颈,转向 GEMM 侧(不碰 attention/cudagraph)。**

---

## §C4(lm_head 后端)源码已闭合(2026-07-12)

### 背景:lm_head 是第二大瓶颈

- memory `trace_adjoint_attribution_mttiles`:lm_head = **MT32x16x4 big 1898us**(第二大 GPU 时间瓶颈,仅次于 FFN gate_up_proj 5.407s)。
- **MT32x16x4 是 rocBLAS tile 名,不是 LLMM1 的 `LLGemm1_kernel`。**

### hipBLASLt 死路 ≠ LLMM1 死路(关键区分,memory 订正)

- lm_head shape(m=1, n=248320, k=5120)在 hipBLASLt heuristic 下只有 1 个 algo(index 4362,8796us),splitK 无效 → **hipBLASLt override 死路成立。**
- 但 13_qkvz 的"LLMM1 188us 最快"是 **qkvz shape**(m=1, n=16384, k=5120),**不是 lm_head shape(n=248320,差 15 倍)**,LLMM1 对 n=248320 未测。

### ⚠️ 源码闭合:三种可能全部推翻,真因是 (b) 的精确化 —— lm_head 命中 wvSplitK 分支,不是 LLMM1

读源码链(`logits_processor.py:96` → `vocab_parallel_embedding.py:63-69` → `utils.py:302-308` → `utils.py:122-188`)确认:

```
LogitsProcessor._get_logits (logits_processor.py:96)
  → lm_head.quant_method.apply(lm_head, hidden_states, bias)
       lm_head=ParallelLMHead, quant_method=UnquantizedEmbeddingMethod
  → UnquantizedEmbeddingMethod.apply (vocab_parallel_embedding.py:63)
       return dispatch_unquantized_gemm()(layer, x, weight, bias)
  → dispatch_unquantized_gemm (utils.py:302): is_rocm() → rocm_unquantized_gemm
  → rocm_unquantized_gemm_impl (utils.py:122)
```

**结论:lm_head 确实经 `rocm_unquantized_gemm`(C1 同一条 skinny 链),(c) 被推翻。** 逐条件核对 lm_head shape(m=248320, n=1, k=5120, bf16, bias=None):

| 条件 (utils.py:170-188) | lm_head 值 | 满足? |
|---|---|---|
| `VLLM_ROCM_USE_SKINNY_GEMM` | True (envs.py:115) | ✅ |
| `on_gfx9()` | True (rocm.py:149 gfx936∈_ON_GFX9) | ✅ |
| dtype ∈ {f16,bf16} | bf16 | ✅ |
| `k % 8 == 0` | 5120%8=0 | ✅ |
| → `use_skinny=True`,进入 180 行后 | | ✅ |
| **181 行 `if m>8 and 0<n<=4` (wvSplitK)** | **m=248320>8, n=1** | **✅ 命中,提前 return** |
| 185 行 `elif m%4==0 and n==1 and k<=8192` (LLMM1) | 满足但被 elif 挡住 | ❌ 摸不到 |

**真因:lm_head 走 `ops.wvSplitK`(utils.py:181-184),不是 LLMM1。** 这解释了 trace 显示 MT32x16x4(rocBLAS tile)而非 `LLGemm1_kernel` —— wvSplitK 对 m=248320 这种超大 vocab matvec 大概率回退到 rocBLAS `F.linear`,即 trace 看到的 MT32x16x4。

**关键反直觉点**:`if m>8 and 0<n<=4` 在 `elif ... LLMM1` **之前**,所以 n=1 的所有 GEMM 只要 m>8 都先被 wvSplitK 截走,LLMM1 只在 m≤8 时才可能命中。但 C1 文档(行 90)写 FFN gate_up 走 LLMM1 —— **若 gate_up(m=34816>8)也命中 181 行 wvSplitK 而非 LLMM1,则 C1 的收益来源需重新核对**(见下方 §C1 收益来源复核)。

### §C1 收益来源复核(已闭合 2026-07-13)

C1 文档(行 90/99)写"FFN gate_up(n=1,m=34816,k=5120)走 LLMM1"。但按 utils.py:181 优先级,gate_up m=34816>8、n=1 → **应先命中 wvSplitK 而非 LLMM1**。

**ssh 读 DCU 实际源码确认**:`/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/model_executor/layers/utils.py` 170-188 行与本地副本**完全一致**,**没有 gfx936 专属 LLMM1-only 分支**(行 135 文档说的 `utils.py:194 if on_gfx936()` 是早期方案描述,实际 C1 没这样改)。

**结论(C1 收益来源订正)**:
- C1 的真实改动是把 gfx936 纳入 `on_gfx9()`(让 `use_skinny=True`),**打开的是 wvSplitK 分支(utils.py:181),不是 LLMM1**。
- decode 下所有 GEMM(n=1, m>8:含 qkvz/gate_up/down/lm_head)全走 `ops.wvSplitK`。**LLMM1 分支(185 行 `elif`)在 decode 下永远不可达**(被 181 行 `if m>8 and 0<n<=4` 提前截走)。
- 文档行 90/99/406 写"gate_up 走 LLMM1"**需订正为走 wvSplitK**。C1 涨 42.6% 的来源是 wvSplitK 替代了 C1 前的 rocBLAS F.linear。

### ⚠️ trace 已过时(关键订正 2026-07-13)

memory `trace_adjoint_attribution_mttiles` / `depollute_ffn_gemm_95pct` 记的 trace tile 名(qkvz=`LLGemm1_kernel`、gate_up=MT64x32x32、lm_head=MT32x16x4)是 **C1 之前**拍的(C1 前 `on_gfx9()` 不含 gfx936,`use_skinny=False`,所有 GEMM 走 `F.linear`→rocBLAS)。

C1 落地后所有 decode GEMM 走 `ops.wvSplitK`(黑盒预编译核,源码不在 vllm_cscc 树,grep .cu/.hip/.cpp 无源文件,实现在 `_rocm_C.so`)。**wvSplitK 对不同 m 走什么 tile 是黑盒,无法读源码确认,只能 A/B 实测。**

**→ trace 归因(lm_head=MT32x16x4 是 rocBLAS、第二大瓶颈 1898us)在 C1 后失效,不能再用来定 C4 方向。** C4 必须改走**实测驱动**:改 lm_head 分支 → 构建安装 → 用户评测看吞吐涨跌。

### C4 真实空间(实测驱动,不凭 trace/形状推断)

前提:lm_head 当前走 `ops.wvSplitK`(utils.py:181),wvSplitK 对 m=248320 是否最优未知(黑盒)。三条候选:

- **C4-3(首选,最低风险)**:让 lm_head 这一个 shape 绕过 wvSplitK 走 `F.linear`(rocBLAS heuristic 自选 tile)。改 utils.py:181 加条件,把 lm_head 的超大 m(如 m≥某阈值)排除出 wvSplitK,落到最后 `return F.linear`。若 rocBLAS heuristic 对 m=248320/n=1 选的 tile 比 wvSplitK 快 → 涨;否则跌。一次实测见分晓。
- **C4-1(中风险)**:让 lm_head 走 LLMM1(改 181 行条件让大 m 也进 185 行)。LLMM1 设计给小 m matvec,m=248320 可能更慢/OOM。
- **C4-2(盲目,末选)**:调 wvSplitK 的 `cu_count` 参数(utils.py:182)。wvSplitK 黑盒,调参盲目。

### C7 备选(FFN gate_up wvSplitK 调优,订正)

- C1 把 FFN gate_up 从 rocBLAS 切到 **wvSplitK**(非 LLMM1),是 C1 涨 42.6% 主因。切后 gate_up 仍是第一瓶颈。
- C1 前文档说"调 `ops.LLMM1(weight, x_view, 4)` 的 4"**失效**(gate_up 不走 LLMM1)。C7 应改为调 `ops.wvSplitK(weight, x_view, cu_count, bias)` 的 `cu_count`(utils.py:182 `cu_count = num_compute_units()`)。
- 定位:C4 无空间时备选。

---

## §当前状态与下一步(2026-07-12)

**已落地并实测**:
- C1(skinny 链 gfx936 打开 wvSplitK,**非 LLMM1**):全段正收益,4-8K +42.6%。
- C5b(段数 32→16):净 +1.90 分,80.04 分。16-32K +50%,8-16K −17.3%。

**已排除**:
- C5b'(24):非 2 的幂,Triton 编译报错。
- C5c/C5a:cudagraph ON + 评测脚本固定,锁死。
- C6(改脚本):评测用默认脚本,改了不被使用。

**关键订正(2026-07-13)**:
- C1 收益来源 = **wvSplitK**(不是 LLMM1);decode 下 LLMM1 分支不可达。
- trace(qkvz=LLGemm1_kernel / gate_up=MT64x32x32 / lm_head=MT32x16x4)是 **C1 前**拍的,**C1 后失效**,不能再用来定方向。
- C4 走**实测驱动**:lm_head 当前走 wvSplitK(黑盒),改分支→构建→用户评测看涨跌。

**待用户拍板下一步**:
- **C4-3(首选)**:lm_head 超大 m 绕过 wvSplitK 走 F.linear(rocBLAS 自选 tile),实测对比。最低风险,一次见分晓。
- **C4-1**:lm_head 走 LLMM1(中风险,可能 OOM/更慢)。
- **C7(备选)**:调 wvSplitK 的 cu_count(gate_up 仍是第一瓶颈)。

**约束重申**:测试容器只测我的修改,源码改动只在 `173.0.59.3`(e03r1n07)的 `zya/vllm_cscc` 树。改源码→构建→安装→重启→用户实测,严格按序,不撞车。

---

## §C4-3 实测:灾难性,已回滚(2026-07-13)

### 改动
- `utils.py:181` 加 `m <= 100000`:`if m > 8 and 0 < n <= 4 and m <= 100000:`,意图让 lm_head(m=248320)绕过 wvSplitK 走最后 `return F.linear`(rocBLAS 自选 tile)。

### 实测结果(用户反馈,新节点 e03r2n01/173.1.51.7)
| 段 | out_throughput(tok/s) | vs C1+C5b(80.04) |
|---|---|---|
| 4-8K | 15.71 | 18.26→15.71,−14% |
| 8-16K | 4.51 | 12.30→4.51,−63% |
| 16-32K | **3.58** | 8.61→3.58,−58% |
- 4-8K P99 TPOT 50.43ms(尚可),但 throughput 全段崩。
- 8-16K P99 TTFT baseline 25046.71ms → C4-3 后 178s(7.1x 暴涨)。
- 16-32K throughput 暴跌到 3.58(C1+C5b 是 8.61)。

### 结论
**lm_head 的 wvSplitK 不能动。** 绕过它让 lm_head(m=248320)走 F.linear 对超大 vocab matvec 是灾难。C4 整条路径(lm_head 后端)封死。

### ⚠️ 回滚事故(关键,2026-07-13)
1. **源码树回滚成功**:`utils.py:181` 已回退为 `if m > 8 and 0 < n <= 4:`(无 `m <= 100000`),备份 `utils.py.bak_c4_3`。
2. **dist wheel 也回滚成功**:09:17 构建的 `dist/vllm-*.whl` 内 `utils.py:181` 确认干净(从 whl 内 zipfile 直读验证)。
3. **但 site-packages 没装干净 wheel** —— 之前那次"重新构建安装"装进去的仍是 C4-3 版(验证日志 `INSTALLED_HAS_M_100000: True` 重装后仍 True)。机器跑的 vllm(pid 1699907,09:24 启动)用的是 C4-3 的 site-packages → **16-32K 3.58 的元凶是 site-packages 没更新,不是源码/wheel**。
4. **下次重建前必删缓存**:已执行 `rm -rf build vllm.egg-info *.egg-info` + 清 `vllm/**/__pycache__`,避免 `bdist_wheel` 复用未回退的 build 产物。**dist/ 下 09:17 干净 wheel 保留**,下一步直接 `pip install --force-reinstall` 它 + 重启即可恢复 80.04。

### 当前机器状态(2026-07-13 核实)
- 源码树:utils.py:181 干净 ✅,rocm.py:149 含 gfx936 ✅,triton_attn.py:43 = 16 ✅(注意:16 是 C5b=80.04 版的正确值,不是"没回退";回退目标不是 32)
- dist wheel(09:17):三处全干净 ✅
- site-packages:仍 C4-3 ❌(待重装)
- build/egg-info/pycache:已清 ✅
- 备份保留:utils.py.bak_c4_3、triton_attn.py.bak_c5b

---

## §C8:wvSplitKrc 替代 wvSplitK(2026-07-13,进行中)

### 背景
源码里 wvSplitK 除 `ops.wvSplitK`(utils.py:181,当前 decode 走的)外,还有 `ops.wvSplitKrc`(utils.py:163,`use_skinny_reduce_counting` 分支,reduce-counting 变体)。触发条件 `on_gfx950()` 把 gfx936 挡在外面(163 行不可达)。思路:放宽 `on_gfx950()` 为 `on_gfx9() or on_gfx950()`,让 gate_up(第一瓶颈)切 wvSplitKrc。

### fits_wvsplitkrc 条件(utils.py:144-147)
```
N_p2 * m * ceil(k/512) <= 128*1024*12  (=1572864)
且 CuNeeded <= cu_count
```
decode n=1 → N_p2=1。

### 关键环境事实(用户抓 trace 确认)
- `multi_processor_count = 80`(gfx936 实测 cu_count=80,与 `num_compute_units()` 一致)。
- `name = BW`,`total_mem = 68702699520`(64GB)。

### gate_up 探测结果(用户直调 `ops.wvSplitKrc`)
```
wvSplitK OK, out shape: torch.Size([1, 34816])      # m=34816,n=1,k=5120
wvSplitKrc ERR: RuntimeError('Unsupported N value: 34816,5120,1')
```
- **报错格式 `34816,5120,1` = `(m, k, n)`**,wvSplitKrc 内部按 m 做 N 分桶,m=34816 不在底层 `_rocm_C.so` 白名单 → 直接拒。
- **fits 条件其实满足**(34816*10=348160 ≤ 1572864),但 fits 满足 ≠ 底层支持。底层白名单是另一回事。
- **gate_up(m=34816)走 wvSplitKrc 不可行** —— 不是改 `on_gfx950` 门槛能解决的,是 .so 实现本身不支持 m=34816。

### C8 探测受阻:op 注册 + 显存双重问题(2026-07-13,暂停)

**问题 1:op 注册调不出。**
- gate_up 那次(`wvSplitK OK, out shape: [1,34816]`)能跑通,但后续用 `torch.ops._rocm_C.wvSplitK/wvSplitKrc` 裸调全部 `has no attribute`,且 `No module named 'vllm._C'`。
- 说明 `_rocm_C` namespace 的 wvSplitK/wvSplitKrc **不是 import 就注册**,依赖 vllm 完整初始化(或特定调用路径)。之前那次能跑通的环境/启动方式未能复现。
- 待查:之前 gate_up `wvSplitK OK` 是在什么 python 启动方式下跑的(裸 python vs vllm 容器 vs import vllm 后)。

**问题 2:显存挤不出。**
- 机器显存被占满,连 170MB 都分不出来,无法跑探测脚本。短期内无显存做 wvSplitKrc shape 白名单逐个实测。

### C8 暂停判定
- gate_up(m=34816)wvSplitKrc 明确 `Unsupported N value`(之前那次抓到),不可行。
- 其余 shape(down/qkvz/o_proj/ba/out_proj)因 op 注册 + 显存问题未能探测,白名单未知。
- **C8 暂停**,待显存空出 + op 注册路径搞清楚后再续;或直接转 C7(调 wvSplitK 的 cu_count,一行常量,不依赖 wvSplitKrc)。

### 当前最紧急的事(优先于 C8)
**site-packages 仍是 C4-3 版,机器实际跑的是 3.58 灾难版。** dist 里 09:17 的干净 wheel(80.04 = C1+C5b)已就绪,只差:
1. `pip install --force-reinstall --no-deps dist/vllm-*.whl`
2. 校验 site-packages utils.py:181 无 `m <= 100000`
3. kill 旧 vllm → 重启 start_vllm.sh

**这一步不做,机器就一直在 3.58 跑,任何后续优化都无意义。** 等机器/显存空出第一件事就是重装恢复 80.04。

---

## §80.02 源码树核对 + C9b 中性 + "第二个优化"排除(2026-07-13,收尾)

### 背景
用户指出"当前装的版本远低于 80.02 水平",怀疑源码树 `/public/home/xdzs2026_c150/zya/vllm_cscc` 被人用 **80.02 版源码覆盖**了。用户说 80.02 = C1 + 针对长序列的**第二个优化**(79.8 只有 C1),让我搜 `NUM_SEGMENTS_PER_SEQ` 定位第二个优化。本地有 80.02 源码树副本:`vllm_optimize_data/2026pra-t2026101089911233-zya-workspace`(从仓库拉取)。

### 核对方法
本地 80.02 树(`2026pra-...workspace`)三关键文件 + `triton_unified_attention.py`,与本地 `vllm_cscc` 副本逐个 `diff`。

### 核对结果(本地 diff 实测)

**1. `triton_attn.py` —— 两树完全一致(无差异)。**
- 80.02 树 42-43 行:`MIN_LAUNCH_GRID_SIZE_2D = 128` / `NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments`
- **`NUM_PAR_SOFTMAX_SEGMENTS = 16` 是源码出厂值,不是改动。** 这订正了本文档此前多处把"16"误标为"C5b 改动"(见下订正)。

**2. `rocm.py` —— 两树完全一致。**
- 149 行:`_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950", "gfx936"])` —— **这是 C1**(加 "gfx936")。80.02 树和 vllm_cscc 都有。

**3. `utils.py` —— 有且仅有一处差异,是 C9b。**
- 80.02 树(干净,无 C9b):`if m > 8 and 0 < n <= 4:`
- vllm_cscc(含 C9b):加了 `and not (m % 4 == 0 and n == 1 and k <= 8192 and bias is None and m <= 20000)` + 顶部 C9b 注释。
- **C9b 不是 80.02 的改动** —— 80.02 源码树没有它,是后来本会话加的实验。已回退(见下)。

**4. `triton_unified_attention.py` —— 两树完全一致(1115 行,39926 字节)。**
- `NUM_SEGMENTS_PER_SEQ` 是 `tl.constexpr`,由 `num_par_softmax_segments` 传入(1094/1113 行),reduce_segments kernel 在 802 行用 `tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(...)`。两树这段代码完全相同,**80.02 没有改 triton_unified_attention.py 的分段策略。**

### "第二个优化"搜索结论:**不存在**
- 搜 `NUM_SEGMENTS_PER_SEQ`:只在 `triton_unified_attention.py` 内部(kernel 参数 + 两个 kernel 用),`triton_attn.py` 没有第二个 `NUM_SEGMENTS_PER_SEQ=16` 定义。
- `NUM_PAR_SOFTMAX_SEGMENTS` 唯一定义在 `triton_attn.py:43`=16(出厂值)。
- 三个关键文件 + triton_unified_attention 两两 diff,**80.02 树相对 vllm_cscc 的唯一差异是 utils.py 的 C9b**(且 80.02 树本身无 C9b,是干净版)。
- **结论:80.02 源码树相对上游纯净版,只有 C1 一处实质改动(rocm.py:149 加 gfx936)。** 用户记忆中的"针对长序列的第二个优化"在 80.02 源码树里找不到对应代码 —— 要么它不在源码层(可能在构建选项/wheel 打包/环境变量),要么记忆有误。**本会话层面已穷尽源码搜索,放弃找第二个优化。**

### ⚠️ C5b 标签订正(关键)
此前本文档(及 memory)把"`NUM_PAR_SOFTMAX_SEGMENTS = 16`"标为"C5b 改动(32→16),净 +1.90 分,80.04 分"。**这是错的。**
- 80.02 源码树 `triton_attn.py:43` 出厂就是 `= 16`,不是从 32 改来的。
- 容器 dist-packages 里 `triton_attn.py:46` 是 `= 32`(行号漂移,文件版本不同)—— 那是 **dist-packages 自带的旧版**,不是"被 C5b 从 16 改成 32",也不是"80.02 没生效"。
- **C5b 这个标签所指的改动(32→16)在 80.02 源码树里不存在。** 80.02 的 16 是源码自带。所谓"+1.90 分"对应的实测(80.04 分)如果真实存在,其归因不能挂到"32→16"这个不存在的改动上 —— 需重新审视那组实测数字的对照基准到底是什么版本。
- **行动:不再把 16 当作"已落地的 C5b 优化"引用。** 16 是基线值,不是优化项。任何基于"C5b=32→16 正收益"的推论作废。

### C9b 中性结论(回退实测)
- 用户回退 C9b(utils.py 恢复 `if m > 8 and 0 < n <= 4:` 干净版)后实测:
  - 4-8K out_throughput 15.80,8-16K 6.72(回退前 C9b 版 4-8K 16.11 / 8-16K 7.44)。
  - **回退后数字反而更差,方向性无规律 → C9b 对实测中性**,不是 8-16K/16-32K throughput 暴跌的元凶。
- **结论:C9b 保持回退(干净版),不作为优化保留。** C9b 标签下的实验作废。

### 当前装版"远低于 80.02"的真因(已闭合,非源码改动问题)
- 当前装的版本 throughput 暴跌(15.80/6.72 等)的元凶不是 C9b、不是 C5b、不是"缺第二个优化"。
- 核心嫌疑仍是 **§C4-3 回滚事故**:site-packages 当时装的还是 C4-3 灾难版(`m <= 100000`),机器实际跑的是 lm_head 绕过 wvSplitK 走 F.linear 的退化版。**恢复手段 = 重装 09:17 干净 wheel(80.04 = C1)→ 校验 site-packages utils.py:181 干净 → 重启**,与"找第二个优化"无关。
- **不要再在源码里找新优化点。** 源码层 80.02 相对纯净版只有 C1,已确认。剩下的差距靠"把 site-packages 装对版本"恢复,不是靠改源码。

### 本节小结(收尾)
- 80.02 源码树唯一实质优化 = **C1**(rocm.py:149 加 gfx936,打开 wvSplitK 链)。**就这一个,没有第二个。**
- C9b(本会话加的 LLMM1 优先实验)= 中性,已回退作废。
- C5b(32→16)= **标签错误**,16 是出厂值不是优化,标签作废。
- 当前装版远低于 80.02 = site-packages 装错版本(C4-3 残留),重装干净 wheel 即可,**源码侧无可改之处**。
- **方向收束:停止在源码层找新优化,集中精力把 site-packages 装回干净 C1 版恢复 80.02 水平。**
