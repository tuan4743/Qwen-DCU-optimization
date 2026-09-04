# vLLM (Qwen3.5-27B) DCU 性能优化 — 总索引

> 本文件是工作主索引,**只放导航 + 极高频结论 + 当前下一步行动指针**。
> 详细事实按主题拆分到 `01`–`06`,按需读取,避免单文件膨胀吃掉上下文窗口。
> 数据来源:`importance/提供的数据.txt`、`profile/`、`log/`、`model_config/`。
> 最后更新:2026-07-12(**去污染订正 + 主线转向**:`tools/_parse_profile_trace.py` 修正正则归类误判后,批3 trace 复跑得出真实占比 = **FFN_GEMM 95.55%**(旧"GDN/FLA 95.17%"是 `GSU` 同名撞车误吞 GEMM tile 的伪值)。真瓶颈 `MT64x32x32_GSU1`(390.7us/call,x18704)+ `MT32x16x4_GSU1`(113.2us,x21543)全是 rocBLAS F.linear tile。op 标签 trace 归因降级为死路 B11(record_function 不进 cudagraph 静态图)。新主线 = shape→tile 正向匹配 + 频次/邻接反推归属。详见 `12_proven_optimizations.md` A3/A6/§D、`10_gdn_gemm_design.md`)
> **⚠️ 2026-09 收尾勘误**:本目录 01-18 记录的是 07-13 前的分析/实验窗口;比赛终稿(五轮优化)与此不同且更完整,唯一权威 = `基于国产加速卡的千问大模型推理服务优化说明文档.md`(含 `图片和附件/` 40 张截图)。本地代码与终稿的逐项校验、终稿代码规格、19 号文档订正,见工作区顶层 `qwen3_dcu_optimize/`。

---

## 一句话现状

**P0/P0.5/P1 已闭合(2026-07-02)。256MB int32 memset 根因 = Triton autotune 的 L2-cache-flush buffer(非 vLLM buffer),但 P1 实测证明:消除该 fill 对单请求在线推理(batch=1)吞吐无收益 —— 候选1 单独 8-16K = 7.26 < baseline 8.8,且首跑=二跑,autotune 命中缓存对吞吐毫无影响。** 单请求 `output_throughput` 瓶颈主体是 decode(`mean_tpot=69.8ms/step` × ~1717 tokens ≈ 120s),fill 是 prefill 期 autotune 一次性开销,被 decode 稀释到可忽略。**转 P2-decode 路径。** 关 cudagraph 反而更慢,cudagraph 必须保留。

## 硬约束一句话摘要

可改:KV Cache/显存、Decode 算子、执行调度、算子级非持久化低精度、改 vLLM 框架代码。
不可改:模型权重/结构、锁定参数(model/tokenizer/scheduler/sampling/接口)、投机解码、评测作弊。
**编译/图相关配置(enforce-eager、compilation_config、cudagraph_mode、custom_ops、pass_config)不在锁定清单内。** 全文见 `01_constraints_env.md`。

## 当前下一步

**P0/P0.5/P1 已闭合;P2-decode 调查钉死(2026-07-12 去污染订正)**:duty cycle 97.3% 推翻 step 间空闲假说 → 端到端 tpot 瓶颈在 step 内部 64 层 GPU kernel 串行。**去污染后真实占比 = FFN_GEMM 95.55%**(旧"GDN/FLA 95.17%"是 `GSU` 同名撞车误吞 GEMM tile 的伪值,见 `12` A3)。GDN 三投影(qkvz/ba/out_proj)实走 LLMM1(188us,最优,非瓶颈)。真瓶颈 = `MT64x32x32_GSU1`(390.7us/call,x18704)+ `MT32x16x4_GSU1`(113.2us,x21543),全是 rocBLAS F.linear tile,非 GDN 三投影,非 GDN 递归核(递归核仅占 0.9%)。详见 `12_proven_optimizations.md`、`10_gdn_gemm_design.md`、`11_investigation_findings.md`。

**下一步 = shape→tile 正向匹配 + 频次/邻接反推,锁定 `MT64x32x32_GSU1`/`MT32x16x4_GSU1` 归属哪个 GEMM**(候选:lm_head / FFN gate-up-down / attention qkv):
- 用各候选 Linear 的 `(m,n,k)` 喂 `rocblas-bench`/`hipblaslt-bench`,看 heuristic 选的 tile 名是否命中 `MT64x32x32_GSU1`(shape→tile 正向匹配,不依赖 op 标签)。
- 辅以频次反推(18704/21543 ÷ 64 层 ÷ 每层次数对齐 token 数)+ trace 内邻接关系(`MT64x32x32_GSU1` 紧邻 `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice` → 强提示 FFN gate-up)。
- 归因清楚后定优化手段:满足 LLMM1 条件却没走 → 针对性修复;不满足 → 评估 hipBLASLt 调参/融合/改 problem 形状。
- ⚠️ op 标签 trace 归因已降级死路 B11(record_function 不进 cudagraph 静态图)。已证伪死路不追(见 `12` §B):消除 fill / 缩小 cache / CPU 调度 / hipBLASLt override / skinny 源码版 / bucket padding / FP8 / 投机解码 / 切 rocBLAS 降 qkvz / 投影+递归核融合 / op 标签 trace。

---

## 索引(按主题,冷热分离)

| 文件 | 内容 | 冷/热 |
|---|---|---|
| `01_constraints_env.md` | 硬约束全文 + 环境 + 算子性能基线 + DeepGEMM 定位 | 冷 |
| `02_model_arch.md` | 模型架构表 + 稠密结论 | 冷 |
| `03_profile_findings.md` | profile 数据 + 256MB fill 特征(含 06-24 修正)+ ViT 特征 | 冷 |
| `04_cudagraph_experiment.md` | cudagraph ON/OFF 对比实验 + 配置差异 + 结论 | 冷 |
| `05_task_tracker.md` | 已完成 / 进行中 / P0–P3 任务清单 + 下一步行动 | **热** |
| `06_pitfalls.md` | 关键判断备忘(踩坑记录) | 冷 |
| `07_p0_conclusion.md` | **P0 结论:256MB fill 根因 = Triton autotune cache(已钉死)** | 冷 |
| `08_dcu_access_link.md` | **DCU 容器访问链路:MCP ssh-sessions 嵌套 ssh 进 worker-0** | 冷 |
| `09_cpu_sched_overhead_design.md` | **P2-decode CPU/调度 overhead 调研设计清单**:30× gap 来源拆解 + 可执行优化点 + 落地路线图。⛔ §0.5 duty cycle 97.3% 推翻 step 间空闲框架,本轨道稳态 tpot 无收益空间,归 GDN 轨道 | 热 |
| `10_gdn_gemm_design.md` | **P2-decode GDN GEMM 设计优化调研清单(最终态,2026-07-12 重写)**:duty cycle 97.3% → GDN/FLA 占 95.17% → 真瓶颈 `MT64x32x32_GSU1`(490us)= rocBLAS F.linear tile。qkvz 实走 LLMM1 188us(三方 bench 钉死,非 hipBLASLt,非瓶颈)。§4 下一步 = op 标签 trace 归因。§6 已废路线表 | 热 |
| `11_investigation_findings.md` | **近期调查发现汇总(2026-07-09~07-11)**:并发=1 官方设定(不质疑)+ 三类有效质疑 + 钉死结论(skinny 不可达/override 死刑/duty cycle 推翻 step gap)+ "范围狭窄"成因分析 + §6 三类质疑审查结论 | 热 |
| `12_proven_optimizations.md` | **当前已证实优化汇总(2026-07-12,事实底座)**:§A 正向优化/事实(A1 cudagraph ON / A2 LLMM1 正向 / A3 去污染订正 FFN_GEMM 95.55%(旧 GDN/FLA 95.17% 是 GSU 撞车误判)/ A4 duty 97.3% / A5 Triton cache 机制可用 / A6 真瓶颈重定位 MT64x32x32_GSU1 390.7us)+ §B 已证伪死路(B1–B11,含 B11 op 标签 trace 归因死路)+ §C 锁定约束 + §D 主线(shape→tile 正向匹配 + 频次/邻接反推归属)+ §E 铁律。新方向必先与此核对 | 热 |
| `13_qkvz_backend_bench.md` | **qkvz 投影 GEMM 后端三方 bench 基准**:qkvz 形状下 LLMM1 188us(最快)/hipBLASLt 261us/rocBLAS 267us,钉死 qkvz 已在最优路径非瓶颈,切后端反退化 | 冷 |
| `15_source_mod_tasklist.md` | **改源码落地任务清单**:C1(rocm.py gfx936,+42.6%)/C5b/C5b′/C5a/C4-3 灾难回滚/C8/C9b/80.02 核对,含全部 C 系列订正与实测数字 | 热 |
| `16_splitk_gemm_tasklist.md` | **split-K GEMM 算子任务:v6→v17 演化链与三关键优化**(v15c 406.7us beat rocBLAS) | 冷 |
| `17_dual_path_splitk.md` | **双路并行作战**:标量路 B(hipcc/clang18)+ mmac 路 A(DCC clang17) | 冷 |
| `18_v12_integration_buildguide.md` | **v12 集成编译指南**(4-5 对接点+wheel 编译,集成失败卡点记录) | 冷 |
| `19_final_summary.md` | **最终总结**(已加 2026-09 勘误标注:融合≠死路,见 `qwen3_dcu_optimize/`) | 热 |
| `基于国产加速卡的千问大模型推理服务优化说明文档.md` | **★ 比赛终稿**:五轮优化(瘦GEMV/瓦片校准/长上下文档/FLA定参/大瓦片+融合)+汇总效果表;截图在 `图片和附件/` | **热/唯一权威** |

---

## 维护规则

- **新尝试 / 新发现**:追加到对应的主题文件(`01`–`06`),并在该文件内更新结论。不要把详情塞进本 README。
- **进度推进**:更新 `05_task_tracker.md`(热数据);若产生新的"下一步",同步改本 README 的「当前下一步」。
- **索引同步**:新增主题文件后,在本表追加一行;修正某文件核心结论时,可更新本 README 对应摘要。
- 每进入下一步前更新本文件。
