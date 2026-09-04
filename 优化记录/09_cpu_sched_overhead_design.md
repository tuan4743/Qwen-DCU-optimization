# 09 · P2-decode CPU/调度 overhead 调研设计清单

> **阶段:仅设计/调研**(不改源码、不进容器、不实测)。
> 分工:本窗口走 **CPU/调度** 轨道;另一窗口走 GDN GEMM 轨道。
> 等待:另一窗口在 **cudagraph ON** 下重抓 decode profile(见 `05` §P2"⚠️ 代表性说明":eager 下 30× gap 不能直接外推到 cudagraph ON)。本文结论待该 profile 出来后定向落地。
> 关联:`03_profile_findings.md`(瓶颈特征)、`05_task_tracker.md`(热追踪)、`06_pitfalls.md`(操作要点)、`08_dcu_access_link.md`(容器访问)。

---

## 0. 问题陈述(2026-07-09 三批 profile 钉死后重写)

> ⛔ **2026-07-09 晚重大修正(duty cycle 钉死,推翻本节 §0/§3 的"step 间开销"框架)**:用 `tools/_duty_cycle.py` 解析批3 cudagraph trace,得 **GPU duty cycle = 97.3%**(busy 7800ms / span 8018ms),GPU 全程满载,**根本没有 step 间空闲**。下文 §0 旧框架("cudagraph 把单步压到 1ms,step 间有 ~70× 非单步开销")**前提错误**:`median_gap=1.00ms` 是**同一 token 内相邻层之间的 kernel 间隔**(64 层各 ~1ms = 64ms ≈ tpot),不是跨 token 间隔。详见 §0.5(本次修正钉死)。§3 的来源 H/B/E "step 间 streaming 往返 / IPC" 框架失效,优化点 1(async_scheduling)/优化点 1'(stream_interval 调大削减 IPC)失去前提。**端到端 tpot 瓶颈仍在 GPU kernel 串行(64 层 × 多 kernel),归 GDN GEMM 轨道,本 CPU/调度轨道无 step 间 gap 可压缩。** 本文件保留旧框架作误读档案,落地据此修正。

- **三批 decode profile**(eager 批1/批2 + cudagraph ON 批3,见 `05` §P2/§P2.3):
  - GPU kernel 占比:GDN/FLA 批1 94.4% / 批2 89.2% / **批3 cudagraph ON 95.17%**(三批最高)→ **GDN GEMM 主导结论路径无关,确定性**,GPU 侧瓶颈归 GDN 轨道。
  - decode step 周期:eager ~2.3–2.5ms(批1/批2)→ **批3 `Cijk_B_PostGSU` median_gap=1.00ms**。~~cudagraph 把"step 内部"开销压到 ~1ms~~ **(已证伪:1.00ms 是层间 kernel 间隔,非 token 间隔,见 §0.5)**。
  - baseline `mean_tpot=69.8ms/step`(`05` §5.3.5,两跑 69.78 vs 69.79 一致)。
- ~~gap 钉死细化~~(批3 决定性转折)**已被 §0.5 推翻**:
  - 旧"step ≈ 1.0ms,但 tpot 仍 69.8ms → step 之间有 ~70× 非单步开销" → **误读**。1.0ms 是层间间隔;真实 token 间隔 = 64 层 × ~1ms ≈ 64ms,与 tpot 吻合,GPU 全程连续满载。
  - 即 ~~cudagraph 已把"step 内部"压到极低(1ms),端到端 tpot 瓶颈完全在"step 之间"~~ **(已证伪)**。
- **本轨道聚焦(duty cycle 修正后)**:CPU/调度 overhead 在 decode 稳态下**不是 tpot 主因**(GPU duty 97.3% 证明 GPU 全程忙,无 step 间间隙)。本轨道的优化点(async_scheduling/stream_interval/.cpu() 同步点)即便实施也**不会降 tpot**——GPU 已满载,无重叠/压缩空间。端到端 tpot 瓶颈 = 64 层 GDN GEMM 串行,归 GDN 轨道。本文件保留作误读档案 + async_scheduling 开关的纯记录性验证。
- **批2 旁证**:批2 TTFT=106.6s(与另一终端争抢同进程排队)→ 坐实来源 B/E(单进程 EngineCore 串行处理 + busy-loop/sleep 让步)对**排队/TTFT** 有效,但不构成**稳态 decode tpot** 主因(GPU duty 97.3% 已证稳态 GPU 满载)。

## 0.5 ⛔ duty cycle 钉死(2026-07-09 晚,推翻 §0/§3 旧框架)

- **工具**:`tools/_duty_cycle.py`(本次新增,对 chrome trace 算 GPU kernel/memcpy/memset 的全局占空比 + 相邻 kernel gap 分布)。
- **trace**:批3 `rank0.1783570538023481078.pt.trace.json.gz`(cudagraph ON,baseline 路径)。
- **结果**:
  - **GPU duty cycle = 97.3%**(busy 7800.1ms / span 8018ms),idle 仅 0.203s(2.5%)。**GPU 全程满载,没有 step 间空闲可供 IPC/调度填充**。
  - 窗口内 118 个 decode token(attn kernel `kernel_unified_attention_3d` count=1888 ÷ 16 FullAttn 层 = 118 token)→ **67.95 ms/token ≈ tpot 69.8ms,完美吻合**。
  - 各高频 kernel 的 per-token count 都是层倍数,证明它们是**每层每 token 触发**,而非每 token 一次:
    | kernel | count | ÷118 token | 含义 |
    |---|---|---|---|
    | `Cijk_B_PostGSU` | 7552 | **64.0** | 每层1个(64层) |
    | `fused_recurrent_gated_delta_rule` | 5664 | **48.0** | 48 GDN 层各1个 |
    | `MT64x32x32_GSU1` | 13216 | 112.0 | GDN 层多个 GEMM |
    | `MT32x16x4_GSU1` | 15222 | 129.0 | GDN 层多个 GEMM |
  - **`Cijk_B_PostGSU` median_gap=1.00ms 的真义**:这是**同一 token 内,64 层之间相邻 kernel 的间隔**(每层贡献 ~1ms),不是跨 token 间隔。跨 token 间隔 = 64 × ~1ms ≈ 64ms ≈ tpot。
- **对旧框架的推翻**:
  - `05` §P2.3 核心结论3 + 本文件 §0/§3 把 "median_gap=1.00ms" 解读为 "cudagraph 把 decode step 压到 1ms" → 推出 "step 间有 ~70× 非单步开销(streaming/IPC/调度)" → **整个推理链前提错误**。1ms 是层间间隔,真实 token 间隔 ~68ms 由 64 层 kernel 串行填满,GPU 全程连续。
  - §3 来源 H(step 间 streaming 往返)、来源 B/E(step 间线程占用)、来源 F(stream_interval 放大 H)**失去前提**——没有 step 间空闲,IPC/调度往返不在 GPU 空闲里,即便削减也不降 tpot(GPU 本就满载)。
  - §4 优化点 1(async_scheduling)、优化点 1'(stream_interval 调大)、优化点 3(.cpu() 同步点)**不会降 tpot**:GPU duty 97.3%,无重叠/压缩空间。async_scheduling 开关(§1)验证价值从"决定主因方向"降为"纯记录"。
- **对两轨道分工的决定性影响**:
  - **GDN 轨道(另一窗口)**:端到端 tpot 瓶颈 = 64 层 GDN GEMM 串行(~68ms),正是它要优化的对象。GDN GEMM 主导(95.17%)+ duty 满载 → 这是唯一能降 tpot 的轨道。
  - **CPU/调度轨道(本窗口)**:**稳态 decode 无 step 间 gap 可优化**,本轨道在稳态 tpot 上无收益空间。批2 TTFT=106.6s 的排队问题属 TTFT(并发争抢),非稳态 tpot,若优化排队可单列但与 tpot 无关。
- **结论**:端到端 tpot 瓶颈**不在 step 之间,在 step 内部的 64 层 GPU kernel 串行**。本 CPU/调度轨道的"step 间开销"调研方向作废,转交 GDN 轨道。

---

## 1. ⚠️ 关键前置:async_scheduling 到底开没开?(待验证点 #0,最高优先级)

### 1.1 静态代码结论(可能推翻档案旧判断)

`vllm/config/vllm.py:706-773` 的解析逻辑:

```python
executor_backend = self.parallel_config.distributed_executor_backend  # 本配置 = "uni"
executor_supports_async_sched = executor_backend in ("mp", "uni", "external_launcher")  # True

if self.scheduler_config.async_scheduling:           # 显式开 → 检查兼容性
    ...
elif self.scheduler_config.async_scheduling is None: # 未显式设(本配置就是 None)
    if (speculative_config is not None and ...):     # 本配置 speculative_config=None → 跳过
        ...; self.scheduler_config.async_scheduling = False
    elif (speculative_config is not None and ...):   # 跳过
        ...
    elif not executor_supports_async_sched:          # False(uni 支持)→ 跳过
        ...
    else:
        self.scheduler_config.async_scheduling = True   # ← 命中这里!
```

`start_vllm.sh` 既不设 `--async-scheduling` 也不设 `--no-async-scheduling` → `async_scheduling` 字段保持默认 `None`(`config/scheduler.py:138`)。而:
- `--tensor-parallel-size 1` → `world_size==1` → `distributed_executor_backend=None` → `parallel.py:802-803` 兜底为 `"uni"`。
- `speculative_config` 默认 `None`(`vllm/config/vllm.py:276`),`start_vllm.sh` 无任何 spec-decode 参数。

**两条 if 都不命中 → 走 else → `async_scheduling = True`**。

### 1.2 与档案旧判断的矛盾

`05` §5.3 第 4 条旧判断("未设 `--async-scheduling` → 默认关闭 → `max_concurrent_batches`=1,调度与执行串行")**前提错误**:它假设默认 = False,但代码里默认 = `None` → `None` 时会自动 resolve 为 `True`(只要无 spec-decode 不兼容 + executor 支持)。

### 1.3 为什么这最关键

- 若 async_scheduling **实际已开**:`max_concurrent_batches=2`(`uniproc_executor.py:63`),调度与执行**已重叠**,CPU/调度 overhead 的主要来源就不是"串行不重叠",而要重新归因到别处(§3 的其他来源)。`05` §P2 中"async_scheduling 关闭导致串行"作为 30× 主因的结论**站不住**。
- 若实测确实关着:可能是 `start_vllm.sh` 实际启动时被某处覆盖,或本 fork 改过默认。需实测确认。

### 1.4 验证方法(给落地阶段用,**本轮不执行**)

进 worker 容器看启动日志:
```bash
bash -lc 'grep -iE "Asynchronous scheduling is" /public/home/xdzs2026_c150/zya/logs/vllm_start.log'
```
`config/vllm.py:775-778` 会 `logger.info_once("Asynchronous scheduling is %s.", "enabled"|"disabled")`。
- 输出 `enabled` → async_scheduling 开着 → §3 主因重排,§4 优化点 1(async_scheduling)从"开它"变"已开,验证收益"。
- 输出 `disabled` → 关着 → §3 旧归因成立,§4 优化点 1 是首选项。

**这是落地阶段第一步,决定后续所有 CPU/调度优化的方向。**

---

## 2. 锁定约束复核(来自 `01_constraints_env.md`)

| 类别 | 是否锁定 | 对本轨道的影响 |
|---|---|---|
| `--async-scheduling` / `--no-async-scheduling` | **未锁定**(`01` 明确:`--enforce-eager`/`compilation_config`/`cudagraph_mode`/`custom_ops`/`pass_config` 不在锁定清单;async_scheduling 属 scheduler **运行模式**而非 scheduler **参数**) | ✅ 可改 |
| scheduler 参数:`max-num-seqs`/`max-num-batched-tokens`/`max-model-len` | **锁定** | ❌ 不能动 batch 配置 |
| `--tensor-parallel-size 1` | 锁定(单卡 DCU) | executor 必然 `uni` |
| 执行调度 / 修改 vLLM 框架代码 | **允许** | ✅ 可改 `gpu_model_runner`/`engine/core.py`/executor |
| `stream_interval`(每 N token 推一次输出) | **未锁定**(scheduler 行为参数,非 max-num-seqs 类硬锁定) | ⚠️ 需确认;若可调是低成本优化点 |
| 投机解码 | **禁止** | ❌ 不能用 spec-decode 降 tpot |

**注意**:`async_scheduling` 一旦显式开启会强制走 `AsyncScheduler`(`config/scheduler.py:160-165`),需确认与本 fork 的 GDN/FLA 路径无回归。本 fork 已在多处 `if self.use_async_scheduling:` 分支(`gpu_model_runner.py` 13 处)做了适配,**说明代码预期 async_scheduling 可用**。

---

## 3. step 间 ~70× overhead 来源拆解(候选,按假设贡献降序)

> **批3 钉死后的归因转向**:cudagraph ON 下 step ≈ 1.0ms,但 tpot 69.8ms → overhead 主体在 **step 之间**(两次 kernel 之间的 CPU/调度/IO 间隙),不在 step 内部。下列候选按"是否制造 step 间间隙"重新排序。原 §3 的"step 内串行"归因(旧来源 A/D)被批3 弱化——cudagraph 已把 step 内 Python 调度压掉,即便 async_scheduling 关着、preprocess 慢,只要在 1ms 以外不阻塞下一次 replay,就不构成主因。
>
> **判定标准**:一个候选是不是主因,看它是否让 GPU 在两次 1ms kernel 之间空等。批3 trace 有 280909 事件 / 8018ms 窗口,但 kernel 总 dur 仅 7.8s 中的一部分 → 若 kernel 占满窗口则无 step 间间隙;若 kernel 稀疏则 step 间有大段空闲。**落地第一步应用 trace 的 ts 序列算 GPU kernel 占空比(duty cycle)**,直接量化 step 间间隙大小。

### 来源 H(新增,最高优先):step 间 streaming 输出往返 —— 每个 token 走一遍 EngineCore↔API server↔client

- **批3 暴露的矛盾**:cudagraph 把单步压到 1ms,理论上 1000 tok/s,但实测 ~14 tok/s(69.8ms/step)。差的 ~68ms/step 几乎全在两次 kernel 之间。
- **机制**(`stream_interval=1`,来源 F 的强化版):每个 decode token:
  1. EngineCore `step()` 末尾把 output 塞 `output_queue`(进程内)。
  2. EngineCore 通过 ZMQ/socket 把 output 推给 API server(**跨进程 IPC,单请求 decode 每步一次**)。
  3. API server 反序列化、组装 SSE chunk、推回 client(**跨网络往返**)。
  4. EngineCore 进入下一轮 `run_busy_loop`,可能 `time.sleep(0.001)` 让步(来源 E)。
- **关键**:这条链的 IPC + 序列化 + 网络 RTT 在 DCU/单进程环境下是**每步固定税**。1ms kernel + ~68ms IPC/调度 = 69.8ms step。**这是 step 间间隙最可能的主体**。
- **为什么 cudagraph 管不到**:cudagraph 只优化 step 内 model forward,不碰 IPC/输出路径。
- **可量化**:trace 里两次 kernel 之间的 CPU 事件(`cat=cpu_op` + 无 kernel 的 ts 空白段)时长 = step 间间隙。

### 来源 B/E(强化):UniProc 单进程 busy-loop 串行 + sleep 让步

- 保留(见 §3 来源 B 既有分析)。批2 TTFT=106.6s 已坐实。
- **批3 新增**:单请求稳态下,`run_busy_loop`(`engine/core.py:1127`)每轮 `step()` 后若 output_queue 非空会处理输出,处理完才能进下一轮。**输出处理与下一轮 schedule 在同一线程** → 即便 async_scheduling 开着把 schedule 重叠到 GPU 执行,输出处理仍串行占用 EngineCore 线程 → 制造 step 间间隙。这与来源 H 是同一链的两面(H 看 IPC,B/E 看线程占用)。

### 来源 F:stream_interval=1 放大来源 H

- `stream_interval=1` 让来源 H 的 IPC 每个 token 触发一次。调大(4/8)直接按比例减少 IPC 次数 → 是来源 H 最直接的缓解。详见 §4 优化点 3。

### 来源 C(降级):`.cpu()` 同步点 —— 在 cudagraph ON 下可能不在热路径

- 保留既有表格(`gpu_model_runner.py:1360/1960/4854` 等)。**但批3 转向后需复核**:cudagraph replay 路径下,这些 `.cpu()` 是否仍每步触发?sampling 的 `_bookkeeping_sync` 在 async_scheduling 下走 `copy()` 而非 `_to_list`(`gpu_model_runner.py:3156`),可能已经避开了 `.cpu()`。
- **降级理由**:若 step 内已被 cudagraph 压到 1ms,说明 step 内同步点没拖慢;真正拖慢的是 step 间。来源 C 从"主因候选"降为"待 profile 确认是否残留"。

### 来源 A(基本证伪):async_scheduling 关闭 → step 内串行

- **批3 证伪**:即便 async_scheduling 关着(step 内串行),cudagraph ON 下 step 仍只有 1ms → step 内串行的开销 < 1ms,远小于 step 间的 ~68ms。**async_scheduling 开关不是 70× 主因**。
- 但 §1 验证仍有价值:async_scheduling=True 能把 schedule 重叠到 GPU 执行,腾出 EngineCore 线程处理输出 → 间接缓解来源 B/H。**优先级从"最高"降为"辅助"**。

### 来源 D(基本证伪):step 内 preprocess 纯 Python 开销

- **批3 证伪**:preprocess 在 cudagraph 捕获范围外,但 step 仍 1ms → preprocess 耗时 < 1ms(64 层 metadata 构造在 1ms 内完成)。不是 70× 主因。
- 保留优化点 5 作低优先备选,仅在 profile 显示 preprocess > step 间间隙的显著比例时才动。

### 来源 G(基本证伪):cudagraph bucket padding

- **批3 证伪**:step ≈ 1.0ms 说明 replay 路径本身高效,bucket padding 没制造大间隙。保留优化点 4 作低优先。

### 来源 B:UniProcExecutor 单进程同步执行(与 async_scheduling 开关无关的固有特性)

- `uniproc_executor.py` 是 TP=1 单卡 executor,**driver_worker 与 EngineCore 同进程**。`collective_rpc` 非 `non_block` 时直接 `run_method` 同步调用(`uniproc_executor.py:77-79`)。
- 即便 `non_block=True`,也只是把已完成的 result 包 Future(`uniproc_executor.py:81-94`),**没有把 model 执行挪到独立线程/进程**。
- 对比 `multiproc_executor.py`(TP>1):worker 在独立进程,EngineCore 提交后可并行做调度。UniProc 无此能力。
- 后果:TP=1 下 EngineCore 的 schedule/preprocess 与 model forward **天然无法跨进程重叠**(只能靠 async_scheduling 在同进程内用 batch queue 交错)。
- 这条**无法通过开关消除**(单卡必然 UniProc),只能靠 async_scheduling 的 batch queue 缓解。
- **批2 旁证(2026-07-09)**:批2 TTFT=106.6s,系"与另一终端压测争抢同进程排队"所致(`05` §P2 批2 备注)。两个终端的请求打进**同一个 EngineCore 进程**,在 `run_busy_loop`(`engine/core.py:1127`)单线程里串行消化 → 队头请求阻塞队尾 → TTFT 被拉到 106s。这**直接坐实了来源 B/E**:单进程 busy-loop 串行处理 + `time.sleep(0.001)` 让步(`engine/core.py:1183-1184`),在高并发/争抢下放大成巨大排队延迟。即单个请求稳态 decode 的 28–30× gap,本质也是同一串行链的稳态表现。

### 来源 C:GPU→CPU 同步点(`.cpu().numpy()` / `synchronize_input_prep`)

静态扫到的强制流同步点(`gpu_model_runner.py`):

| 行号 | 代码 | 作用 | decode 频率 |
|---|---|---|---|
| `1360` | `.cpu().numpy()` | (上下文待核)取张量回 CPU | 每 step? |
| `1960` | `.cpu().numpy()` | (上下文待核) | 每 step? |
| `4854` | `.cpu().numpy()` | (上下文待核) | 每 step? |
| `3244-3256` | `synchronize_input_prep`:`prepare_inputs_event.synchronize()` | 等上一步 CPU→GPU 传输完成才开始本步 preprocess | 每 step(async_scheduling 下才生效;`prepare_inputs_event is None` 时 no-op) |
| `2926/2927` | `.copy()` | `req_ids`/`req_id_to_index` 拷贝(防 async 下被改) | async_scheduling 下每 step |
| `3148/3149` | `.copy()` | 同上,output 路径 | async_scheduling 下每 step |
| `4160` | `.copy()` | (上下文待核) | 每 step? |

- **每个 `.cpu()` 都强制 GPU stream 同步**(等所有排队 kernel 跑完才能拷)。cudagraph ON 下这些点是否在 step 间热路径,待 trace 的 cpu_op 事件 + kernel 占空比确认(见 §3 判定标准)。`synchronize_input_prep` 注释(`gpu_model_runner.py:3249-3251`:"Ensure prior step has finished with reused CPU tensors. This is required in the async scheduling case")说明它专为 async_scheduling 设计,async 关时 no-op(`gpu_model_runner.py:3245-3247`)。

### 来源 E:`step()` 串行链 + `time.sleep(0.001)` 让步

- `_process_engine_step`@1168:无 model execution 但有 unfinished requests 时 `time.sleep(0.001)`(`engine/core.py:1183-1184`)。decode 稳态一般每步都有 execution,不应触发。
- `step()` 内 `schedule()`→`execute_model`→输出处理→`update_from_output` 串行,**输出处理占用 EngineCore 线程** → 与来源 H/B 同链。

---

## 4. 可执行优化点(锁定约束内,按批3 钉死后的预期收益/风险重排)

> ⛔ **2026-07-09 晚 duty cycle 修正后**:GPU duty 97.3%,稳态 decode 无 step 间空闲。下列优化点 1/1'/3 即便实施也**不会降 tpot**(GPU 已满载,无重叠/压缩空间)。仅保留作"误读档案 + 纯记录性验证"。**真正能降 tpot 的是 GDN GEMM 轨道(64 层 kernel 串行),不在本节。**
>
> ⏳ 全部**待落地验证**(批3 profile 已就绪)。本节为设计清单,不实施。**批3 转向后,优化重心从"开 async"转向"削减 step 间输出往返"**(~~此判断已被 §0.5 推翻~~)。

### 优化点 1(最高优先,来源 H/F):调大 `stream_interval`

- **动作**:`start_vllm.sh` 加 `--stream-interval 4`(或 8,从默认 1 调大)。
- **机制**:每 N 个 token 才推一次输出 → output_queue IPC / SSE 序列化次数按比例减少 → 直接削减 step 间输出往返(来源 H)。这是对 step 间 ~68ms 间隙最直接的缓解。
- **风险**:① 流式体验变粗(bench 是非交互压测,影响小);② **需先确认 `stream_interval` 是否在锁定清单内**(`01` 锁定点名 max-num-seqs/max-num-batched-tokens/max-model-len 等,stream_interval 未点名,倾向可调,但落地前需用户确认)。
- **约束合规**:⚠️ 待确认(§2)。
- **验证**:对比 `mean_tpot` baseline 69.8ms。**注意**:若 tpot 是按 token 间延迟测的,stream_interval 调大可能让"每 token 的 itl"看起来不变(因为输出攒批),但 `output_throughput`(总 token/总时长)应提升 —— bench 看 `output_throughput` 更准。

### 优化点 2(高优先,来源 H/B):确认 async_scheduling 实际开关 + 输出路径重叠

- **第一步(诊断)**:§1.4 grep 启动日志 `Asynchronous scheduling is enabled/disabled`。
  - 若 `disabled`:开 async_scheduling(`--async-scheduling`)让 schedule 重叠到 GPU 执行,腾出 EngineCore 线程处理输出 → 间接缓解来源 H/B。
  - 若 `enabled`(代码推断):async 已开,schedule 已重叠,主因不在这 → 来源 H 的 IPC 本身就是瓶颈,转优化点 1。
- **机制**:`max_concurrent_batches` 1→2,`AsyncScheduler` 把 schedule/preprocess 与当前步 GPU 执行重叠。
- **风险**:① 本 fork GDN/FLA 路径在 async 分支是否有回归(代码已有 13 处 `use_async_scheduling` 适配,需实测不崩);② `async_scheduling=True` 强制 `AsyncScheduler`(`config/scheduler.py:160-165`),需确认与 GDN 自定义 attn metadata 兼容。
- **约束合规**:✅ async_scheduling 不在锁定清单(§2)。
- **批3 后预期调整**:async_scheduling 缓解的是 step 内调度重叠,而 step 间输出往返是另一条链 —— 即便 async 开着,输出 IPC 仍按 token 触发。**优化点 2 单独收益可能有限,需与优化点 1 叠加**。

### 优化点 3(中优先,来源 C):消除/减少 decode 路径的 `.cpu()` 同步点

- **动作**:逐个核 `gpu_model_runner.py:1360/1960/4854` 的 `.cpu().numpy()`,确认是否每步必要。若为 logprobs/统计可延迟或条件化(如 `not request_logprobs` 跳过)。
- **机制**:每去掉一个 `.cpu()` 就少一次强制 stream 同步。
- **批3 后降级**:step 内已被 cudagraph 压到 1ms,`.cpu()` 若在 step 内则非主因;**仅当 trace 显示 `.cpu()` 落在两次 kernel 之间的 step 间间隙时才是主因**。必须先 profile 定位。
- **风险**:改 framework 代码,需 `bdist_wheel` + 重装(见 `06`)。
- **约束合规**:✅ 改 vLLM 框架代码允许。

### 优化点 4(低优先,来源 G):cudagraph bucket 对齐

- **动作**:确认 batch=1 是否命中已捕获的 cudagraph bucket(`capture_model`@5694)。若 decode batch 频繁 padding,调整 `cudagraph_capture_sizes`(`compilation_config`,未锁定)使 batch=1 直接命中。
- **批3 后降级**:step ≈ 1.0ms 说明 replay 高效,bucket padding 没制造大间隙。低优先。
- **约束合规**:✅ `compilation_config`/`cudagraph_mode` 未锁定(§2)。

### 优化点 5(低优先,来源 D):preprocess 路径微优化

- **动作**:若 profile 显示 `_build_attention_metadata`(64 层每步构造)占 step 间间隙显著比例,考虑对 GDN 层做 metadata 复用。
- **批3 后降级**:step 1ms 说明 preprocess < 1ms,非主因。仅作备选。
- **约束合规**:✅ 改框架代码允许。

---

## 5. 落地路线图(等 cudagraph ON profile + §1 验证后)

```
[第一步] §1.4 验证 async_scheduling 实际开关(看启动日志)
   ├─ enabled  → 跳过优化点1,主因重排到 C/D/F/G,直接进 [第二步]
   └─ disabled → [优化点1] start_vllm.sh 加 --async-scheduling → 跑 bench 对比 mean_tpot
                 若 tpot 大降 → 证实来源 A/B 是主因,继续看残余 gap
                 若几乎不变   → 说明 UniProc 单进程同步是硬限制,转 [第二步]

[第二步] 拿 cudagraph ON 的 decode profile(另一窗口产出)
   ├─ 看 cat=cpu_op 总时长 + gpu_model_runner: preprocess/sample 区间
   ├─ 看是否有 .cpu() 事件每步触发 → [优化点2]
   ├─ 看 cudagraph replay 前的 CPU 间隙 → [优化点4]
   └─ 拆出 CPU 端各段(schedule/preprocess/forward-replay/sample/bookkeeping)占比

[第三步] 按 profile 指向的来源,逐个试优化点 2/3/4/5,每次只动一个,bench 对比
```

---

## 6. 待验证清单(汇总,本轮不执行)

1. **[最高] §1**:启动日志里 "Asynchronous scheduling is enabled/disabled" 到底是哪个 → 决定主因归因。
2. **§3 来源 C**:`.cpu()` 三处(`1360/1960/4854`)在 decode 热路径的实际上下文(本轮只 grep 到行号,未读上下文),需 cudagraph ON profile 的 cpu_op 事件 + 读上下文确认。
3. **§2/优化点3**:`stream_interval` 是否在锁定清单内(需用户确认)。
4. **§3 来源 D**:`_build_attention_metadata` 64 层每步构造的 CPU 耗时占比(待 profile)。
5. **优化点1 风险**:async_scheduling=True 走 AsyncScheduler 是否与本 fork GDN 路径兼容(代码有适配但未实测)。

---

## 7. 与另一窗口(GDN GEMM 轨道)的边界

- **GDN 轨道**:GPU 侧 94.4% 的 GDN GEMM kernel 优化(更优 GEMM config / fused / 算子级低精度)。
- **本轨道**:CPU/调度 overhead(GPU 之外的那 ~30× gap)。
- **不重叠**:GPU kernel 内部耗时归 GDN;GPU idle/CPU 忙的时间归本轨道。
- **协同点**:cudagraph ON profile 同时服务两条轨道(GDN 看 kernel 占比是否仍主导;本轨道看 CPU 区间占比)。若 cudagraph ON 后 30× gap 大幅收窄(符合 `05` §P2"⚠️ 代表性说明"预期),则本轨道主因可能从"async 串行"转为"残余 CPU overhead",优化点 1 的优先级相应调整。
