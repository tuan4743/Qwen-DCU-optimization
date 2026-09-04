# 07 · P0 结论:256MB int32 memset 的确切源码定位

> 2026-06-30 钉死。运行时内存快照(`_record_memory_history` + `_snapshot`)抓到 256MB 分配的 Python 栈,根因确认。
> 关联冷档案:瓶颈特征见 `03_profile_findings.md`,踩坑备忘见 `06_pitfalls.md`,任务追踪见 `05_task_tracker.md`。

---

## 7.1 一句话结论

**那个吃掉 62.4% GPU 时间的 256MB int32 memset(`at::native::FillFunctor<int>`,shape `[4096,16384]`=256MB/4B),不是 vLLM 的业务 buffer,而是 Triton autotune 的 L2-cache-flush buffer。**

来源:`triton/backends/amd/driver.py` 的 `get_empty_cache_for_benchmark()` 写死分配 256MB int32,`triton/testing.py` 的 `do_bench()` 在每次 benchmark 前调 `cache.zero_()` 清 L2 cache —— 这就是被 rocprof 抓到的 `at::native::FillFunctor<int>`。

---

## 7.2 确切源码位置

### 分配点(256MB int32 buffer 的诞生)
```
triton/backends/amd/driver.py:718-721  get_empty_cache_for_benchmark():
        cache_size = 256 * 1024 * 1024                              # ← 写死 256MB
        return torch.empty(int(cache_size // 4), dtype=torch.int,   # ← int32, 67,108,864 元素
                           device='cuda')
```
- `256*1024*1024 // 4 = 67,108,864` 元素 = `[4096,16384]` reshape → **与 rocprof 抓到的 shape 完全吻合**。

### 触发点(memset 本体 = 那个 62.4% 内核)
```
triton/backends/amd/driver.py:723  clear_cache(cache): cache.zero_()
triton/testing.py:178              runtime.driver.active.clear_cache(cache)   # do_bench 主循环每轮调
```
- `cache.zero_()` → eager `at::native::fill` → `at::native::FillFunctor<int>`(int32,256MB)→ 即 profile TOP1 的那个 kernel。
- `clear_cache` 在 `do_bench` 的 estimate 阶段(每 5 次)、warmup 阶段(每轮)、benchmark 阶段(每轮 `n_repeat` 次)都被调用 → 反复 `zero_()` → kernel 统计上极高频。

---

## 7.3 触发链路(来自 161 个 EXACT_256MB 块的聚合栈)

| 项 | 值 |
|---|---|
| 快照文件 | `logs/fill_alloc_probe_ckpt1_pre_capture.jsonl`(4.2MB) |
| SUMMARY | `total_alloc_events=6552  exact_256MB=161  near_big=634` |
| 256MB alloc 次数 | **161 次**,全部 `addr=140434380685312`(同一地址反复 alloc = 原址重分配) |
| **FULL-STACK leaf(161/161)** | `triton/testing.py` `do_bench` → `driver.py:721 get_empty_cache_for_benchmark` |
| **USER frame 主导** | 105/161 → `fla/ops/chunk_o.py:166 chunk_fwd_o`(FLA chunk attention O 投影) |
| 其余 USER frame | `chunk_scaled_dot_kkt.py:141`(27)、`solve_tril.py:545`(12)、`wy_fast.py:139`(9)、`cumsum.py:183`(4)、`chunk_delta_h.py:325`(4)——全是 FLA(GDN)子核 |
| 调用方(自底向上) | `qwen3_next.py:_warmup_prefill_kernels` → `_forward_core` → `gdn_attention_core` → **`profile_run`** → `gpu_worker.py:388 determine_available_memory` |

### 因果链
1. Qwen3.5-27B 的 **GDN(FLA chunk attention)层在 `profile_run` warmup 时触发 Triton autotune**。
2. Autotune 的 `do_bench` 用 AMD driver 写死的 **256MB int32 cache**(`get_empty_cache_for_benchmark`),在每次 benchmark 前 `cache.zero_()` 清 L2。
3. 这个 `zero_()` 就是回放/运行期反复出现、吃 62.4% GPU 时间的 `at::native::FillFunctor<int>` memset。

---

## 7.4 关键判定

- ❌ **不是** vLLM 业务 buffer(已彻底排除 MLA indexer、block_table、KV cache 等所有静态候选,见 `03` §3.2)。前两轮静态源码排查 0 命中,正是因为它根本不是 vLLM 的 buffer,是 **Triton 工具自身的 benchmark cache**。
- ✅ **是** Triton autotune 的 L2-cache-flush buffer,属于 benchmark 工具自身开销。
- ⚠️ **阶段**:161 次 alloc 全在 `pre_capture` 之前/之中(profile_run/init 阶段,recording 窗口内),地址恒定 → autotune 在 init 期一次性完成;`zero_()` 在 `do_bench` 循环里被反复调用,故 kernel 统计上极高频。

## 7.4.1 P0.5 稳态占比复核(2026-06-30 已闭合)

补了 `post_capture` 与首请求后 `post_first_req` 两个检查点,确认稳态 serving 期是否仍触发 256MB alloc(re-autotune)。

**三检查点汇总(`EXACT_256MB` 块,addr 来自 `driver.py:721`)**:

| 快照 | total | warmup 路径 | serving 路径 | unique addr |
|---|---|---|---|---|
| `ckpt1_pre_capture` | 161 | 161 | 0 | 1 |
| `ckpt2_post_capture` | 161 | 161 | 0 | 1 |
| `ckpt3_post_first_req` | 252 | 161 | **91** | 3 |

**关键结论:稳态 serving 首请求会再触发 91 次 256MB alloc(serving 路径,非 warmup)。**

- `pre_capture` → `post_capture` 之间 256MB alloc 数不变(161→161),且 unique addr 仍为 1 → **capture 阶段不产生新的 autotune,仅录进 cudagraph**。
- `post_capture` → `post_first_req` 新增 91 次 serving-path alloc(unique addr 从 1→3),**全程栈含 `do_bench` + `autotuner.py`(91/91)** → 是 **re-autotune**,不是 cache 命中。
- 91 个 serving 块的调用栈自底向上:`execute_model` → `_model_forward`(line 3282)→ `cuda_graph.py:251`(cudagraph replay)→ `qwen3_5.py:765 forward` → `gdn_attention_core` → `_forward_core` → `forward_native` → FLA 子核 autotune → `do_bench`。**即稳态 forward 路径上 FLA kernel 对真实 batch 的 key 触发了 autotune。**
- 91 块的 top USER frame 分布:`chunk_o.py:166 chunk_fwd_o`(33)、`chunk_scaled_dot_kkt.py:141`(27)、`solve_tril.py:545`(12)、`wy_fast.py:139`(9)、`cumsum.py:183`(5)、`chunk_delta_h.py:325`(5)——与 init 期 161 块的分布一致,均为 FLA(GDN)子核。

**为什么 serving 期会 re-autotune**:`_warmup_prefill_kernels` 只对 `T∈{16,32,64}`、`B=1` 跑 dummy warmup;FLA 子核 autotune key 含 `BT`(=`chunk_size`=64,固定)与 `H/K/V`(head 维度,固定)——但真实 serving batch 的 token 数 / head 排布与 dummy 不同 → 新 key → `if key not in self.cache:` 命中 → re-benchmark → 再调 `do_bench` → 再 `cache.zero_()`(256MB)。

**对 P1 选型的决定性影响**:
- 256MB fill **贯穿 init + 首请求**,不是纯冷启动开销 → 候选 3(warmup-only、运行期禁用)**不能**单独成立(运行期仍 re-autotune)。
- `TRITON_CACHE_DIR` 持久化(候选 1)**是治本**:同一 key 二次命中走磁盘 cache,跳过 `do_bench` → 既减启动期又减运行期。**但前提是 `TRITON_CACHE_AUTOTUNING` 必须先开**(默认 OFF,见 §7.5 注)。
- 缩小 cache(候选 2)对 init + serving **都有效**(每次 `zero_()` 量变小),与候选 1 可叠加。

→ **P1 主候选 = 候选 1(开 `TRITON_CACHE_AUTOTUNING=1` + 持久 `TRITON_CACHE_DIR`)+ 候选 2(缩 `driver.py:718` cache)叠加**,详见 §7.5。

---

## 7.5 优化方向(P1 候选,依赖本结论)

> 方向仍是"减量/减次/缩小尺寸"(memset 带宽已打满 1.17TB/s,无法靠"加速")。Triton autotune 在锁定约束外(编译/图配置不锁定),可改。
> **P0.5 已确认 256MB fill 贯穿 init + 稳态首请求(见 §7.4.1)**,故候选 3 单独不成立,主候选 = 候选 1 + 候选 2 叠加。

1. **持久化 autotune 结果(治本,减次)**:开 `TRITON_CACHE_AUTOTUNING=1`(默认 OFF!实测 `/root/.triton/cache/` 有 168 个编译产物但 0 个 `.autotune.json`)+ 设 `TRITON_CACHE_DIR` 指向持久目录,避免每次启动重跑 `do_bench` → 启动期 memset 次数大幅下降。FLA kernel autotune 一次后 cache 命中,**稳态 re-autotune 也命中磁盘 cache 跳过 `do_bench`**。
   - 关键代码路径(已核对):`autotuner.py:37` `self.cache_results = cache_results or (knobs.autotuning.cache and not knobs.runtime.interpret)`;`knobs.py:369` `cache = env_bool("TRITON_CACHE_AUTOTUNING")`(默认 False);`autotuner.py:258` `if self.cache_results:` 才走 `check_disk_cache` → 写/读 `{fn}.autotune.json`,否则 `benchmark()` 直接跑(不落盘)。
2. **缩小 cache(治标,减量)**:改 `driver.py:718` 的 `256*1024*1024` 为更小值(如 64MB/32MB)。直接砍掉单次 `zero_()` 开销;代价是 benchmark 时序精度下降(对生产推理无影响,只影响 autotune 选核精度)。对 init + serving 两次 autotune 都减量。
3. **warmup-only autotune + 运行期禁用**:**P0.5 证伪** —— serving 首请求仍 re-autotune 91 次,运行期非"不重跑"。仅可作为候选 1/2 之外的补充(如扩大 `_warmup_prefill_kernels` 覆盖更多真实 batch 形状,减少 serving 期新 key)。
4. ~~稳态占比复核~~:**已闭合(§7.4.1)**。

---

## 7.6 复现路径(物证留档)

- 探针:`fill_alloc_probe.py` v2(`begin_lifetime_probe` / `checkpoint_snapshot` / `stop_lifetime_probe`),插桩在 `gpu_model_runner.py`:`load_model()` 末尾开 recording、`capture_model()` 入口/出口各快照。
- 产物:`/public/home/xdzs2026_c150/zya/logs/fill_alloc_probe_ckpt1_pre_capture.jsonl`。
- 解析:`MATCH[..] EXACT_256MB` 块的 `USER/VLLM FRAMES`(leaf)与 `FULL STACK`(leaf=`driver.py:721`)。本地解析脚本 `tools/_analyze_probe.py`。
- DCU 访问链路见 `08_dcu_access_link.md`;编译生效流程见 `05_task_tracker.md` §5.3.3。
