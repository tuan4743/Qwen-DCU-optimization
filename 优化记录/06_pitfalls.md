# 06 · 关键判断备忘(避免重复踩坑)

> 冷数据档案,设计方案前快速过一遍。每条 = 一条已钉死的判断,带物证。

## 256MB fill 本身
- ✅ baseline(cudagraph ON,12.20 tok/s)是优化基线与回归对照。关 cudagraph 更慢(`04_cudagraph_experiment.md`)。
- ✅ fill 是带宽打满的 256MB int32 memset(219μs/次,1.17 TB/s 峰值),优化路径只能是"减量/减次/缩小",不是"加速"。
- ✅ fill 是 cudagraph 捕获的 eager `at::native::FillFunctor<int>`,非 inductor 节点(inductor 图 dump 已证)。
- ✅ **【2026-06-30 根因钉死】256MB fill 的真凶 = Triton autotune 的 L2-cache-flush buffer**,非 vLLM 业务 buffer。`triton/backends/amd/driver.py:718-721` `get_empty_cache_for_benchmark()` 写死 `256*1024*1024` int32,`driver.py:723` `clear_cache(): cache.zero_()` 由 `triton/testing.py:178 do_bench()` 反复调用 → 即 `FillFunctor<int>`。触发:`profile_run` warmup 的 FLA/GDN 子核 autotune。物证:`ckpt1_pre_capture.jsonl` 161 个 EXACT_256MB 块,FULL-STACK leaf 161/161 命中 `driver.py:721`。详见 `07_p0_conclusion.md`。
- ✅ **fill 贯穿运行时每个 forward step(cudagraph replay 回放),非 capture-only**。qi 分布实测:big-fill 跨 qi 2398→33398(占全 qi 80%),每 1000-qi 桶 ~226–250 次连续不聚集;37575 次 ÷ 64 层 ≈ 每步每层 1 次。
- ⚠️ **"qi 贯穿运行时"与"autotune init 期一次性"的关系**:qi 贯穿运行时是 kernel 统计层面(replay 回放/`do_bench` 循环反复 `zero_()`);而 256MB *alloc* 在 init 期一次性完成(pre_capture 快照 161 次同地址)。两者不矛盾:alloc 一次,`zero_()` 在 capture 期录进图 + `do_bench` 循环里反复执行。**稳态 serving 期 `zero_()` 是否持续高频仍待 P0.5 复核**。
- ❌ 不要把 256MB fill 归因到 MLA indexer `expanded_block_table_buffer`(`mla/indexer.py:265`)。它是源码层面唯一 shape 完全吻合 `[4096,16384]×int32` 的 GPU int32 buffer,但本模型走 TRITON_ATTN/GDN backend,`+sparse_attn_indexer` no-op,**不实例化 indexer**。吻合纯属 `max_num_batched_tokens × cdiv(max_model_len,16)` 公式巧合。第二轮穷尽复核后彻底排除。**最终真相是 Triton autotune cache,前两轮静态 0 命中正因它不在 vLLM 源码里。**
- ✅ 静态源码已到极限:GPU-resident+int32+`[4096,16384]` 的唯一候选(indexer)已证不实例化;其余 GPU 大 buffer 要么 UVA 非 GPU、要么 int64/bf16、要么 dim-0=max_num_reqs(8MB)。**确切根因已靠实机 `_record_memory_history` 快照钉死**(非 vLLM buffer,是 Triton autotune cache)。
- ❌ 不要为了去 fill 关 cudagraph —— 实测更慢。
- ❌ 不要把 DeepGEMM grouped GEMM 套到稠密 FFN —— 没有"分组"。
- ❌ 不要再 dump inductor 图找 fill —— fill 不在 inductor 图。

## 已废的定位路线(不要再走)
- ❌ **capture 期 Python `fill_`/`zero_` hook(`fill_capture_hook`)**:replay 不走 Python runtime,hook 抓不到运行时 fill;且 capture 期那一次 forward 本就无 256MB fill。三重死结:(1) 早期 hook 被 import 进 APIServer 父进程而非 EngineCore 子进程(capture 在子进程,方法对象不同);(2) spawn 子进程不继承父 env,`VLLM_TRACE_FILL`/`PYTHONPATH` 丢失 → env 门控恒 False;(3) `/proc/<pid>/maps` 实证 hook 模块从未被任何 vllm 进程加载。路线废弃,改抓分配点。
- ❌ **capture 期 `_record_memory_history` 抓分配点(`fill_alloc_probe` v1)**:**2026-06-29 实测证伪**。probe 正确进了 `capture_model`、抓到 69403 个 alloc event,但 `exact_256MB=0`。根因:256MB buffer **不是 capture 期分配**的(更可能在 init/profile_run 阶段分配,capture 期只是录进图),v1 的 recording 只开在 capture 窗口 → 抓不到。
  - **v2 修正(2026-06-30 选定)**:改全程记录 + 多检查点快照(`begin_lifetime_probe` 在 `load_model` 末尾开 alloc-only recording,`checkpoint_snapshot(tag)` 在 capture 前/后各快照一次),筛 `size==268435456` 的 alloc event 读 Python 栈。`_snapshot()` 不停止 recording 可多次调;`_record_memory_history(enabled=None)` 仅停止返回 None。详见 `05_task_tracker.md` §5.3.4。
  - 留备复用的 API 细节:`_record_memory_history(enabled="all", context="alloc", stacks="python")` 开启(alloc-only 减开销);`_snapshot()` 返回 dict{`segments`,`device_traces`,...},`device_traces` 是 list-per-device,每 device list[event],event.keys 含 `action/addr/size/stream/time_us/frames`,alloc 类 action=`segment_alloc`/`alloc`/`resize`/`realloc`。DCU/torch2.10.0/HIP6.3.26093 实测支持。
- ❌ **rocprof 运行时对齐**:海光预装 rocprof 版本低,大概率无法开 `--call-stack`,只能间接对齐 region,**给不了变量名**(用户 2026-06-30 明确否决)。定位确切 Python 源码改用 `_record_memory_history` 快照。

## 实机操作要点
- ⚠️ **改 `vllm_cscc` 源码后必须 `bdist_wheel` + `pip install --force-reinstall --no-deps`** 才生效(非 editable 安装,dist-packages 是拷贝)。改工作区副本 `vllm_optimize_data/` 是空忙。
- ⚠️ **改完立即 `ast.parse` 校验语法**。2026-06-29 `_fix_probe_location.py` 用切片移除带缩进代码块时误吃 `set_cudagraph_capturing_enabled(True)` 行的 8 空格缩进 → `IndentationError`。教训:切片移除带缩进块务必校验边界行缩进。
- ✅ **`profile_cudagraph_memory`(def @ 5593)在 DCU/ROCm 被跳过**:`gpu_worker.py:399` 门控 `not current_platform.is_rocm()`,DCU 是 ROCm → 整方法跳过。插桩必须插在 `capture_model`(def @ 5694,`gpu_worker.py:608` 仅门控 `not enforce_eager`,DCU 正常执行)。
- ⚠️ **spawn 子进程不继承父 env**(`VLLM_TRACE_FILL`/`PYTHONPATH` 等丢失,保留 `PATH`/`LD_LIBRARY_PATH`)。凡靠 env 触发又要在 EngineCore 子进程生效的插桩,不能假定 env 继承 —— 在子进程入口点 import 或显式传 env,别用父进程 env 门控。`start_vllm.sh` 开头 `export PYTHONPATH=...` 是为解决此。
- ⚠️ **selfcheck 用独立临时进程,其输出不代表真正 vllm 进程**。验证插桩真在 vllm 进程跑,看 `vllm_start.log` 里由 EngineCore 子进程打的日志(带 `(EngineCore pid=...)` 前缀),或查 `/proc/<pid>/maps`。
- ⚠️ **DCU 访问链路**:见 `08_dcu_access_link.md`(MCP ssh-sessions 嵌套 ssh:login → 计算节点 → `root@173.0.8.2` worker-0)。**旧路径已废**:本地直连 173.0.8.2 三把密钥 Permission denied;`docker exec` 进容器被组委会修复不可用。**不要用 `UserKnownHostsFile=none`/`GlobalKnownHostsFile=none`**(Windows 下触发 `Host key verification failed`)。
- ⚠️ 非交互 ssh 进容器不加载 `~/.bashrc`(`[ -z "$PS1" ] && return`)→ `import vllm` 需显式 `export PYTHONPATH=/usr/local/`。
- 🐛 **OOM 与 hook 改动无关**:启动报 `Free memory ... less than desired GPU memory utilization` 多是旧 vllm 进程未杀干净占显存。重启前先 `pkill`/查显存。别误判成 hook bug。

## HTTP/容器网络(2026-07-09 钉死)
- 🐛 **NO_PROXY 根因**:`/start_profile`、`/v1/chat/completions` 等请求在容器里返回 squid 错误页 / `Connection failed` / `urllib.error.HTTPError: 404`,**根因不是路由没挂载,而是请求走了容器 squid 代理**。容器内设了 `http_proxy`/`https_proxy` 指向 squid,`urllib` 默认走代理 → 代理把 `127.0.0.1:8001` 也代理出去,vllm 收不到。**治本:发请求前 `os.environ["no_proxy"]="127.0.0.1,localhost"` + `NO_PROXY` 同设**(urllib 看 `no_proxy`/`NO_PROXY` 两变量)。已固化进 `_decode_only_profile.py:18-19`。诊断法:对比设不设 NO_PROXY 的返回体,设了才返回 vllm JSON 而非 squid HTML。
  - **反例(误判)**:`/start_profile 返回 404` 第一反应是"profiler-config 没设/路由没挂载",在已设 `--profiler-config` 的前提下,**先怀疑代理再怀疑路由**。404 的 body 是 squid 的还是 vllm 的,看一眼就分得清。
- 🐛 **空 trace 根因**:`/stop_profile` 返回 200 但 `profile_traces/` 下空 / 无 `.pt.trace.json` 文件,**根因不是 profiler 没启动,而是 profile 窗口内没有 GPU 活动**。`/start_profile`→`/stop_profile` 之间若没有正在 decode 的请求(torch profiler 只录这段时间内的 kernel),trace 几乎为空。**治本:profile 窗口必须覆盖一段**正在 streaming 的 decode(先发 streaming 请求,等 TTFT 确认进入 decode,再 `start_profile` 抓 ~8s 覆盖 ~110 个 step)。`_decode_only_profile.py` 的策略即按此设计(streaming 请求先于 start_profile 启动)。
  - 次因:TTFT 放宽到 180s —— 首请求含 cudagraph capture + Triton autotune warmup,实测 TTFT 可 >60s,按 30s 超时会误判"卡死"。
