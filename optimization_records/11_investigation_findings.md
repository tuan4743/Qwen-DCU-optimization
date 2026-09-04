# 11 · 近期调查发现汇总(2026-07-09 ~ 07-11)

> 本文档汇总 P2-decode 阶段 GDN GEMM 轨道与 CPU/调度轨道的调查钉死结论,
> 以及用户审查后确认有效的质疑点与不应质疑的官方设定。
> 关联:`05_task_tracker.md` §P2/§P2.3、`09_cpu_sched_overhead_design.md` §0.5、`10_gdn_gemm_design.md`。
> 创建:2026-07-11。

---

## 0. 背景与文档定位

P2-decode 阶段分两条轨道并行:
- **GDN GEMM 轨道**(唯一能降 tpot 的轨道):见 `10_gdn_gemm_design.md`。
- **CPU/调度轨道**(已被 duty cycle 推翻主框架):见 `09_cpu_sched_overhead_design.md`。

用户审查"剩余优化方向"后,认为**范围异常狭窄(范围小的有点反常)**。本文档把造成"狭窄"的各钉死结论、用户认可的有效质疑、以及不应质疑的官方设定,集中记录在一处,便于复核与下一步决策。

---

## 1. ⛔ 不应质疑的官方设定:评估场景并发度 = 1

- **用户明示**:评估场景的并发度官方设定为 **1**。
- **处置**:不再质疑并发=1 的合理性,所有优化方向都基于单请求(batch=1)在线推理前提。`09`/`10` 中若出现对"并发=1 是否合理"的质疑,应停止。
- **影响范围**:这意味着
  - 任何依赖"提高并发/batch 撑大 GEMM 的 m 维"来提速的方向,与官方评测场景不符 —— 评测的是单请求 tpot/throughput。
  - `09` 优化点 4(cudagraph bucket 对齐到 batch=1)的合法性前提成立(batch=1 是真实场景,非人为制造)。
  - 单请求下 `mean_tpot=69.8ms` 是要优化的真实目标值,baseline `output_throughput=8.8` 是对照基准。

---

## 2. ✅ 用户认可的有效质疑(三类,需记录并复核)

### 2.1 占空比 97.3% 的代表性

- **结论来源**:`tools/_duty_cycle.py` 解析批3 cudagraph ON trace,得 GPU duty cycle = 97.3%(busy 7800.1ms / span 8018ms)。
- **✅ 已复核(2026-07-11 晚,长窗口钉死)**:抓 30s(`PROFILE_SECONDS=30`)cudagraph ON decode trace(`rank0.1783777442602026742.pt.trace.json.gz`,167167 kernel 事件,实际录 11.524s 窗口),用 `tools/_duty_cycle_v2.py` 重算并查 idle 分布:
  - duty = **97.37%**(与 8s 窗口 97.3% 一致),idle 2.63%。
  - **idle gap 大小直方图**:166912 个 idle gap **全部 <1ms**;1-5ms/5-20ms/20-50ms/50-100ms/>100ms **每档 0 个**;max gap 仅 0.644ms。→ 任意两 kernel 间从无 >0.644ms 空闲,无 GC/KV 整理/re-autotune 周期性 idle。
  - **idle 时间轴位置分布(20 桶,每桶 576ms)**:idle 占比 2.3%–3.1% 均匀抖动,无桶异常偏高 → idle 均匀分散全程,不聚集首尾、不聚集中段 → 排除"8s 截取偏置掩盖边界 idle"。
  - **head/mid/tail 对比**:head 10%=2.5% / mid 80%=2.6% / tail 10%=2.8% → 稳态中段 idle 不高于边界。
- **✅ 质疑 1 闭合:不成立**。占空比 97.3% 确代表全稳态(非 8s 截取偏置),CPU/调度轨道彻底关闭。duty cycle 测的是"kernel 间 gap",长窗口里也无 idle → step 间空闲假说彻底作废,端到端 tpot 瓶颈确在 step 内部 64 层 GPU kernel 串行。重心回 GDN GEMM 轨道候选 B/C/D。

### 2.2 lm_head vs GDN 投影的区分(trace 直接可见的 ≠ GDN 投影)

- **事实**:2026-07-11 trace 闭合(`10` §8.4)显示 capture 后实际 **m=1** 的 32 个 GEMM,其权重形状 `[248320, 5120]` —— **248320 = vocab_size,是 lm_head**,不是 GDN 投影。
- **有效质疑**:trace 能直接 expose 的 m=1 GEMM 是 lm_head(每个 token 一次 vocab 投影)。而真正占大头的 GDN 投影 GEMM 形状(`in_proj_qkvz`: m=1, n=16384, k=5120)是**靠源码推断**的,未被 trace 直接捕获/标注。
  - 风险:源码推断的 GDN 投影形状可能因 cudagraph capture / bucket padding / 实际调用路径与源码路径有出入而失真。
  - 复核方法:给 GDN 投影 GEMM 打专属 label(类似 §5.0.0 的 `gemm_probe` 桩),在 trace 里直接读出其真实 m/n/k,验证源码推断。
- **结论暂定**:GDN 投影靠源码推形状(可信但未直接验证),lm_head 是 trace 直接可见的 m=1 GEMM。两者不可混为一谈 —— 此前若有"trace 里的 m=1 GEMM 就是 GDN 投影"的表述,属误读,应澄清为 lm_head。

### 2.3 capture 后 m=1 未被撑大(推翻 bucket padding 源码假设)

- **事实**:`10` §8.4 trace 闭合 —— capture 后实际 m=1,**未被 bucket padding 撑大**。
- **有效质疑(核心)**:vLLM v1 cudagraph 源码假设是"按 `max_num_seqs=128` bucket padding 撑大 batch",若成立则 m 应远大于 1(可命中更优 GEMM algo)。**但实测 m=1,假设被推翻。**
  - 影响:这直接否决了"靠 cudagraph bucket padding 自然撑大 m"这条优化路径(`10` 候选 A 失效的前提之一)。既然 batch=1 真实场景下 m 确实是 1,则 m=1 时的 GEMM algo 优化(override 路线)才是真正的着力点 —— 而 override 路线已实测死刑(见 §3.2)。
  - 复核方法:确认 capture 的 bucket sizes 列表里是否真的不含"能把单请求 m 撑到 >1"的档位;或确认 vLLM v1 在单请求 decode 时是否根本走"m=1 直传"而非 padding。
- **结论暂定**:batch=1 decode 下 m=1 是真实情况(非 bucket 撑大),源码 bucket padding 假设对本场景不成立。这与 §1 并发=1 一致 —— 单请求场景 m 本就该是 1。

---

## 3. 钉死结论(近期调查的关键闭合)

### 3.1 skinny 分支 gfx936 不可达(三重闭合,`10` §0.5.3/§0.5.4)

GDN 投影 GEMM 走 hipBLASLt 回退,**skinny 优化路径(`wvSplitK`/`LLMM1`)全链路不可达**,与 n/env/capture batch 全无关。三重证据:

1. **Python 层短路**:`on_gfx9()` 返回 False —— `_ON_GFX9` 硬编码 `["gfx90a","gfx942","gfx950"]`,**排除 gfx936**(`utils.py:122-188`)。
2. **C++ 编译宏空壳**:`__HIP__GFX9__` 宏在 gfx936 编译产物里为空(skinny C++ 路径不编译进)。
3. **CMakeCache build 目标**:`AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938` —— 含 gfx936,**不含 gfx942/gfx950**( skinny 路径依赖的 gfx9 系高端卡)。

→ gfx936(海光 BW3000,80 CUs)既不在 `_ON_GFX9` 白名单,也不是 skinny C++ 的编译目标,skinny 分支对它**结构性关闭**。

### 3.2 hipBLASLt override 路线实测死刑(`10` §8.4.5)

- **m=1 problem 只有 1 个 algo**:hipblaslt-bench 实测,m=1 时 `in_proj_qkvz`(n=16384,k=5120)问题只有 1 个算法(index 4362,`Cijk_Ailk_Bljk_BBS_BH_Bias_AS_SAV_MT16x16x16_..._GSU1_GSUAMB_..._WGM1`,638.9us)。无可选余地。
- **splitK 退化无效**:m=1 时手动改 splitK / GSU 参数对结果无加速(参数退化为无效)。
- **唯一可调 env = `HIPBLASLT_TUNING_OVERRIDE_FILE`**;`HIPBLASLT_HEURISTIC` **不存在**(环境变量名误记)。
- **版本信息**:hipBLASLt 1000/0.10.0,git `a6254b89-dirty`,容器 `/opt/dtk-26.04-DCC2602-0317`。
- → 通过"调 hipBLASLt algo/参数"优化 GDN 投影 GEMM 的路线**死刑**。

### 3.3 duty cycle 97.3% 推翻 step 间空闲假说(`09` §0.5,记忆已存)

- GPU duty 97.3%(busy 7800.1ms / span 8018ms),**无 step 间空闲**。
- `median_gap=1.00ms` 是**同一 token 内相邻层间**的 kernel 间隔(64 层 × ~1ms ≈ 64ms ≈ tpot),**非跨 token 间隔**。
- → CPU/调度轨道的"step 间 ~70× 开销"框架作废:优化点 1(async_scheduling)/1'(stream_interval)/3(.cpu())**不会降 tpot**(GPU 已满载,无重叠/压缩空间)。
- → 端到端 tpot 瓶颈在 **step 内部 64 层 GPU kernel 串行**,归 GDN GEMM 轨道。
- 记忆文件:`duty_cycle_kills_step_gap_theory.md`(type=project)。

### 3.4 重心转移(`10` §8.5)

override 死刑后,GDN GEMM 轨道的重心从"调 GEMM algo 参数"转向:
- **方向 A(已封死)**:强制 batch padding 撑大 m —— 但 §2.3 证明 batch=1 真实 m=1,且 §1 并发=1 不可改 → 此路不通。
- **方向 B**:换 backend(hipBLASLt 回退 → 其他)—— 受 §3.1 skinny 不可达 + 算子基线(hipBLAS BF16 403T > DeepGEMM 280T > Triton 175T > CK 144.7T;FP8 segfault)约束,候选有限。
- **方向 C(融合)**:减 kernel 数量/访存 —— 融合 GDN 投影 GEMM,受编译环境限制。
- **方向 D(接受现状)**:若 A/B/C 均无路,则 GDN 投影 GEMM 在当前硬件/软件栈下已达可及上限,需重新评估"是否还有可改方向"(回应"范围狭窄"的疑问)。

---

## 4. "范围狭窄"的成因分析(回应用户疑问)

用户认为剩余优化方向范围异常狭窄,成因是**多个方向被钉死性证据同时关闭**:

| 方向 | 状态 | 关闭证据 |
|---|---|---|
| 消除 256MB fill 降 tpot | ❌ 已证伪 | `05` §5.3.5:候选1 实测 7.26 < 8.8,fill 非单请求吞吐主因 |
| CPU/调度 overhead 降 tpot | ❌ 推翻 | `09` §0.5:duty cycle 97.3%,无 step 间空闲 |
| 调 hipBLASLt algo/参数 | ❌ 死刑 | `10` §8.4.5:m=1 只 1 algo,splitK 退化 |
| skinny 优化路径 | ❌ 不可达 | `10` §0.5.3/§0.5.4:gfx936 三重闭合 |
| bucket padding 撑大 m | ❌ 不成立 | §2.3:capture 后 m=1 未撑大 + 并发=1 |
| FP8 低精度 | ❌ segfault | `01`/`10` §2.4:FP8 不可用 |
| 投机解码 | ❌ 禁止 | `01`:锁定约束禁止 |

剩余可探方向收敛到 **方向 B(换 backend)/C(融合)/D(接受现状)** 三条,且 B/C 受编译环境/算子基线强约束 —— 这就是"范围狭窄"的客观成因,非遗漏。

**下一步决策点(待与用户定)**:
1. 方向 B:在锁定约束 + 算子基线下,是否还有未试的 backend(如手写 Triton GEMM 是否能超过当前 hipBLASLt 回退?需对照 `10` §2.4 基线评估)。
2. 方向 C:融合 GDN 投影的编译环境可行性复核。
3. 方向 D:若 A/B/C 确实无路,正式确认"GDN 投影 GEMM 已达当前栈上限",优化重心是否转向其他可改维度(KV Cache/显存布局、decode 算子级低精度非 FP8 路线等)。
4. 复核 §2 的三类有效质疑(占空比代表性、lm_head vs GDN 投影、m=1 未撑大),其中任一被推翻都可能重新打开某条已关闭方向。

---

## 5. 待办(本文档派生)

- [x] **审查三类质疑方向(2026-07-11,见 §6)**:逐项判定证据强度 + 是否能重开已关闭方向。结论:**仅质疑 1(占空比代表性)是方向性赌注**(若推翻能复活 CPU/调度轨道),质疑 2/3 即使坐实也不改变方向格局。
- [x] **复核占空比代表性(2026-07-11 晚,钉死)**:抓 30s+ 长窗口 trace 重算 duty,**且看 idle 分布不只看平均数**。结果:duty 97.37%(与 8s 一致),idle gap **全部 <1ms**(max 0.644ms),idle 在时间轴上均匀分布(head/mid/tail 均 ~2.5-2.8%)。**质疑 1 不成立,占空比代表全稳态,CPU/调度轨道彻底关闭。** 详见 §2.1。
- [ ] 复核 GDN 投影形状(§2.2 → §6 判为精度性复核):给 GDN 投影 GEMM 打 label,trace 直接读 m/n/k 验证源码推断。**不影响方向,优先级降低。**
- [ ] 复核 m=1 未撑大(§2.3 → §6 判为根因澄清):确认 cudagraph capture bucket sizes 与单请求 decode 实际 m 路径。**重开不了候选 A(并发=1 下撑大 m 是负收益),仅澄清 vLLM v1 capture 逻辑。**
- [ ] 方向 B/C 可行性复核(§4):对照算子基线评估手写 Triton GEMM / 融合编译环境。
- [ ] 与用户定方向 B/C/D 取舍。

---

## 6. 三类质疑方向审查结论(2026-07-11)

> 本节是对 §2 三类有效质疑的逐项审查:对每个质疑判断 (a) 证据强度、(b) 若坐实能否重开某条已关闭方向、(c) 性质与复核优先级。审查依据:`10` §8.4(trace 闭合)/§8.5(候选 A–D)、`09` §0.5(duty cycle 钉死)、`01`(锁定约束 + 算子基线)、`02`(架构)。

### 6.1 质疑 1 —— 占空比 97.3% 的代表性

| 维度 | 判定 |
|---|---|
| 证据强度 | **中(自洽非外证)**。批3 trace 仅 8s/118 token,duty 97.3% 靠"67.95ms/token ≈ tpot 69.8ms 吻合"支撑。这只能说明"这 8s 内 GPU 满载",不能排除"稳态里 8s 窗口外存在周期性 idle"。 |
| 能否重开方向 | **✅ 能(唯一的方向性赌注)**。duty cycle 测的是"GPU kernel 之间的 gap",它推翻的是"step 间空闲"。**若长窗口里出现 idle,那 idle 必然落在 kernel 之间(否则 duty 算法统计不到),正好复现 step 间空闲** → 复活整个 CPU/调度轨道(`09` §0.5 的推翻失效)。 |
| 性质 | **方向性**(三类质疑里唯一一条若成立能直接重开已关闭轨道的) |
| 复核优先级 | **最高**。复核成本极低(抓一次长窗口 trace),收益不对称(若推翻则整条 CPU/调度轨道复活)。 |

**复核要点(超出原 §2.1)**:8s 窗口是"buffer 2s → 抓 8s"截取的,截取逻辑偏好"稳态中段"。若长窗口的 idle 集中在请求边界(首尾),8s 中段天然避开。所以复核**不只看长窗口的 duty 平均数,要看 idle 的时间分布** —— 即便长窗口 duty 仍 >95%,也只能说"稳态中段满载",不能说"全程满载"。要复核的是 **idle 是否在稳态中段也周期性出现**,而非单一平均数。

### 6.2 质疑 2 —— lm_head vs GDN 投影的区分

| 维度 | 判定 |
|---|---|
| 证据强度 | **高(已实质闭合)**。`10` §8.4.2 已做完整归属核对:248320=vocab=lm_head(trace 直接可见的 32 个 m=1 GEMM),GDN 投影形状靠源码 + `§5.0.0` 邻接推断定位,且有频次反推独立佐证(`MT64x32x32` 3584 个/32 step=112/step,与"48 GDN 层 × 多投影"吻合,lm_head 仅 1/step 不吻合)。 |
| 能否重开方向 | **❌ 不影响任何已关闭方向**。源码推断的 GDN 投影形状非孤证(§5.0.0 邻接独立佐证)。给 GDN 投影打 label 直接读形状是有价值的复核,但即使证实 GDN 投影形状与推断不同,override 死刑(对 `(m=1,n=16384,k=5120)` 和 `(m=1,n=248320,k=5120)` 都只有 1 个 algo,§8.4.5)+ skinny 不可达(§0.5.3 三重闭合)**仍成立**。 |
| 性质 | **精度性复核**(消除"源码推断可能失真"的不确定性,非方向性) |
| 复核优先级 | **中**(低于质疑 1)。值得做 label 复核,但不改变方向格局。 |

### 6.3 质疑 3 —— capture 后 m=1 未被撑大

| 维度 | 判定 |
|---|---|
| 证据强度 | **强(已作结论钉死,非悬而未决)**。trace 直接可见 `aten::linear` 输入 `[1,5120]`(§8.4.1),推翻 §8.2 源码假设。这条质疑本身已被 `10` §8.4 当结论用。 |
| 能否重开方向 | **❌ 候选 A 重开但与并发=1 相悖**。`10` §8.5 候选 A="强制 batch padding 撑大 m 走 mid-size GEMM(多 algo 可选)",技术上可探。**但结合 §1 并发=1 不可改**,候选 A 的前提是"把单请求 m 从 1 撑到 128"——对 1 个 token 的投影做 128 倍冗余计算,在并发=1 下是纯浪费,只让 tpot 变慢不会变快。**候选 A 即便可行也与降 tpot 目标相悖,重开无意义。** |
| 性质 | **根因澄清**(澄清 vLLM v1 capture 为何 m=1,而非方向性) |
| 复核优先级 | **低**。值得补"为何 m=1"的根因解释(澄清 vLLM v1 capture 逻辑:是命中 m=1 bucket / 还是 pad 到 128 但图内跑 m=1 / 还是 m 维本就=1),但**重开不了候选 A**,反而强化"m=1 是 batch=1 真实形状"前提,让 override 死刑 + skinny 不可达更稳。 |

> §8.4.1 留的潜在弱点:"m=1 适用于全部 decode GEMM"是基于"lm_head 与 GDN 投影共享同一 hidden_states 的 batch=1 输入"的推断。这个推断在并发=1 前提下是稳固的(单请求 decode 的 hidden_states batch 维必然=1,无论走哪个投影),所以质疑 3 即使复核出"capture 走的是 m=1 bucket"也只澄清机制、不改变 m=1 这一形状事实。

### 6.4 审查总览

| 质疑 | 证据强度 | 若成立能否重开方向 | 性质 | 优先级 |
|---|---|---|---|---|
| 1. 占空比代表性 | **✅ 已复核不成立**(长窗口 97.37%,idle 全 <1ms 且均匀) | ❌ CPU/调度轨道彻底关闭 | ~~方向性~~ 已闭合 | ~~最高~~ 已完成 |
| 2. lm_head vs GDN 投影 | 高(已实质闭合) | ❌ 不影响任何已关闭方向 | 精度性复核 | 中 |
| 3. m=1 未撑大 | 强(已作结论) | ❌ 候选 A 重开但与并发=1 相悖 | 根因澄清 | 低 |

### 6.5 对"范围狭窄"疑问的最终回应

用户感觉"剩余优化方向范围狭窄得反常"。审查后确认:

- 三类质疑里**只有质疑 1 是真正的方向性赌注**,质疑 2/3 即使坐实也不改变方向格局。
- `10` §8.5 候选 A/B/C/D 的真实约束:A 被并发=1 否决(撑大 m 是负收益)、D 是接受现状、C 受编译环境限制(docker 严重简化、依赖几乎全无、roc 版本过低)、B 受 skinny 不可达(§3.1)+ 算子基线(hipBLAS BF16 403T > DeepGEMM 280T > Triton 175T > CK 144.7T;FP8 segfault)双重约束。
- **范围狭窄是约束叠加的客观结果,非遗漏。** 但在最终确认"已达上限"(方向 D)之前,应先排除质疑 1 这个方向性赌注 —— 因为它是唯一能以极低复核成本重开一整条轨道的。**下一步动作 = 质疑 1 长窗口复核优先于 GDN 轨道的候选 B/C/D 推进。**

### 6.6 派生的下一步动作(覆盖 §5 待办优先级)

1. **【最高优先,方向性】质疑 1 长窗口复核**:抓 30s+ 稳态 decode trace → `tools/_duty_cycle.py` 重算 → **看 idle 时间分布**(不只平均数),判断稳态中段是否有周期性 idle。
   - 若 idle 仍 <3%(duty >97%):质疑 1 不成立,duty cycle 97.3% 确代表稳态,CPU/调度轨道彻底关闭,重心回 GDN 轨道候选 B/C/D。
   - 若稳态中段出现显著 idle(如 GC / KV cache 整理 / 偶发 re-autotune):质疑 1 成立,**CPU/调度轨道复活**,重新评估 `09` 优化点 1/1'/3。
2. **【中优先,精度性】质疑 2 label 复核**:给 GDN 投影打专属 label(`10` §5.0.0 gemm_probe 桩思路),trace 直接读 m/n/k。不影响方向,消除源码推断不确定性。
3. **【低优先,根因澄清】质疑 3 capture 逻辑澄清**:确认 vLLM v1 cudagraph capture bucket sizes 列表 + 单请求 decode 命中哪个 bucket。重开不了候选 A,仅留档。
4. **【GDN 轨道,待质疑 1 结果】候选 B/C/D**:仅在质疑 1 复核确认 CPU/调度轨道确无空间后,推进 backend 更换 / 融合 / 接受现状的取舍。
