# Profile 与插桩工具集(工具脚本)

> 全部工具均为**通用化**设计:凡原本写死路径/形状的地方,均已参数化为 CLI/env 参数(默认值=原值),因此每个脚本默认行为与已验证版本完全一致、可直接复现;换模型、换卡、换目录只需传参。
> 每个脚本开头有英文 docstring(Purpose / Usage / Generalization Notes),同一结构便于维护。这里是中文索引。

## 分组

### A. 离线计算流分析(无需 GPU)

| 脚本 | 干什么 | 关键参数 |
|---|---|---|
| `_duty_cycle.py` | 从 Chrome trace 算 GPU kernel 占空比:最忙流 duty、相邻 kernel 间隙分布、步内紧凑 vs 步间大间隙分桶 | `trace`, `--gap-threshold-us`(默认 5000) |
| `_duty_cycle_v2.py` | v1 + idle **位置**分析:时间轴分桶、idle 尺寸直方图、大 idle 周期性(CV)、首/中/尾偏置检查 | `trace`, `--gap-threshold-us`, `--bins`(默认 20) |
| `_parse_profile_trace.py` | 离线 kernel 分类器:类别表(FFN GEMM / GDN-FLA / FullAttn / KV / LayerNorm / Sampling / Memcpy / Elementwise / Other)、top-N kernel、decode 步周期估计 | `trace`, `--top`, `--ts-min`, `--ts-max` |
| `_parse_gemm_probe.py` | 把 GPU kernel 归属到 `GEMM_PROBE::*` 标注区间(钉死某主力 kernel 属于哪个投影 GEMM) | `trace`, `--labels`(默认 in_proj_qkvz,in_proj_ba,out_proj) |
| `_enum_autotune_keys.py` | 盘点 Triton autotune cache:key、覆盖形状、每 kernel 是否 re-autotune | `--cache-dir`, `--kernel-filter` |

> **通用化**:五个脚本都只依赖 Chrome trace(任意框架/profiler 均可);唯一平台相关的是 `_parse_profile_trace.py` 的分类正则(CUDA 与 ROCm 核名风格不同,按需要调整)。

### B. 实时探针(在 worker 容器内运行)

| 脚本 | 干什么 | 关键参数 |
|---|---|---|
| `_stream_probe.py` | HTTP streaming 探针:连接时延、首 token 时延(TTFT)、chunk 数 | `--host`, `--port`, `--model`, `--prompt`, `--max-tokens` |
| `_decode_only_profile.py` | 经 `/start_profile`+`/stop_profile` 抓**纯 decode 段** trace(先等 TTFT) | `--host/--port/--model`, `--max-tokens`, `--profile-seconds` |
| `__probe_dcu.py` | 环境自检:vllm 位置/版本、哪个 runner 模块有 `GPUModelRunner.capture_model`(文件+行号) | 无(编辑 `_RUNNER_CANDIDATES`) |
| `_qkvz_backend_bench.py` | 瘦 GEMM 三方 bench:vLLM 自定义算子 vs rocBLAS vs hipBLASLt vs matmul(核名核验 + cuda-Event 计时两个独立 phase) | 顶部 `M/K/N`、`DTYPE` |

### C. 内存分配溯源

| 脚本 | 干什么 | 关键参数 |
|---|---|---|
| `_fill_alloc_probe.py` | 全程记录带 Python 栈的 alloc;多检查点快照,定位特定尺寸(默认 256MB)缓冲的确切分配点 | env `FILL_ALLOC_TARGET_BYTES`、`FILL_ALLOC_LOG_DIR`;API `begin_lifetime_probe/checkpoint_snapshot/stop_lifetime_probe` |
| `fill_capture_hook.py` | `__torch_dispatch__` capture 包夹 hook:包裹 `GPUModelRunner.capture_model`,在 cudagraph capture 期记录 fill 类算子 | 自门控 `_ENABLED`(env) |
| `_patch_v1_runner.py` | 一次性:向 V1 `GPUModelRunner.__init__` 注入 hook import | 编辑 `TARGET` |
| `_patch_hook_candidates.py` | 一次性:重排 `_RUNNER_CANDIDATES`(V1 优先)并记录实际被包裹的模块 | 编辑 `TARGET` |

> **通用化**:`FILL_ALLOC_TARGET_BYTES` 使探针与目标尺寸无关;`_write_match` 里的"user frame"子串过滤是项目特定的(完整栈始终写出,可安全删除)。

### D. 一次性 runner 插桩 patch(历史存档)

`_apply_gemm_probe.py`、`_apply_probe_v2.py`、`_apply_probe_v3.py`、`_apply_probe_v31.py`、`_fix_probe_location.py` —— 一次性向特定 vLLM 源码树注入探针的补丁。保留作**证据 + 可复用模式**:每个文件都记录了它修掉的坑(v2:锚点串首次出现在别的函数;v3.1:锚点撞车)以及通用化方法(参数化 `TARGET`+锚点、用 `__probe_dcu.py` 核实锚点归属、`ast.parse` 门禁、幂等标记、绝不拿裸字符串当锚点)。

### E. 启动器

| 脚本 | 干什么 | 关键参数 |
|---|---|---|
| `_start_vllm_profiler.sh` | 启动带 torch profiler 路由的 vLLM server、持久化 Triton autotune cache、日志 tee | env `MODEL_DIR`, `PORT`, `TRITON_CACHE_*`, `ZYA_HOME` |

## 原调查流程串联(工具如何配合)

```
__probe_dcu.py            -> 定位 runner 模块 + capture_model 位置
_patch_v1_runner.py       -> 注入 fill_capture_hook import
fill_capture_hook.py      -> capture 期抓 fill 类算子(256MB 溯源)
_fill_alloc_probe.py      -> 全程 alloc 记录(Python 栈)
_apply_probe_v2/v3/v31    -> 在核实过的锚点放 lifetime+快照桩
_gemm_probe.py            -> 对投影 GEMM 打 record_function 标签
_apply_gemm_probe.py      -> 把标签注入模型文件
_decode_only_profile.py   -> 经 HTTP 抓 decode-only chrome trace
_parse_profile_trace.py   -> 分类/top-N/步周期
_parse_gemm_probe.py      -> kernel 归属 GEMM 标签
_duty_cycle[_v2].py       -> 占空比 + idle 位置/周期性
_qkvz_backend_bench.py    -> 目标形状的后端对比
_enum_autotune_keys.py    -> autotune cache 健康检查(启动税诊断)
```
