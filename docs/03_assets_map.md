# 03 · 工作区资产地图与恢复路径

> 项目原始工作区 = `vllm_optimize_data/`(09-04 从远端/评测环境组装拷贝,mtime 保留原时间)。
> 本文件是"哪个目录有什么、哪些要保留、哪些是噪音"的总地图。

```
deep-workspace/
├── memory/MEMORY.md                       持久记忆(本会话已更新)
├── qwen3_dcu_optimize/                    ★ 本项目交付(本目录)
└── vllm_optimize_data/
    ├── vllm_cscc/                         vLLM v0.18.1+das.dtk2604 fork 源码树(权威基线)
    │   ├── vllm/                          Python 包(本地=07-13 快照,仅 3 处真实改动)
    │   ├── csrc/                          C++ 扩展(06-20 基线;无 gfx936 宏)
    │   ├── dist/vllm-0.18.1+das.dtk2604-….whl   06-20 基线 wheel(非终稿)
    │   ├── build/lib.linux-x86_64-cpython-310/  06-20 构建镜像(=wheel 同源)
    │   └── build/…CMakeCache              GPU_TARGETS=gfx906;926;928;936;938 证据
    ├── tasks/
    │   ├── README.md                      内部文档总索引(07-12 停更,本会话已补 15-19/终稿条目)
    │   ├── 01~19_*.md                     内部任务文档(分析/实验/算子开发/集成失败记录)
    │   ├── 基于国产加速卡的千问大模型推理服务优化说明文档.md   ★ 比赛终稿(唯一最终版权威)
    │   └── 图片和附件/                     ★ 40 张截图(终稿代码对比+效果图+指标图)
    ├── tools/                             profile/插桩/归因脚本(fill_capture_hook、_apply_*, _parse_*, 等 19 个)
    ├── profile/                           baseline profile(dump、pmc、vector_kernel_params.csv 等)
    ├── profile_traces/                    rank0.*.pt.trace.json.gz(批1-批3 trace)
    ├── log/                               启动日志/ERROR/历史 dump(06-24~28)
    ├── model_config/                      config.json + 模型官方 README(92KB)
    ├── importance/提供的数据.txt           官方给定环境/限制/基线数据(16-32K=5.38 口径)
    ├── dcu_profile_tool/                  PMC 监控工具(device_manager/monitor/monitor_performance)
    ├── deepgemm-main/                     光合社区 DeepGEMM(适配 BF16 的 DCU 版,算子基线用)
    └── _baseline_decode/                  api_server/model_runner/qwen3_next 早期基线副本(对照用)
```

## 关键取舍

| 资产 | 价值 | 保留建议 |
|---|---|---|
| `tasks/01-19` | 方法论+踩坑(19 号文档含全部算子开发经验) | 保留;19 已加勘误标注 |
| 说明文档+截图 | **唯一终稿代码证据** | 保留;建议连同本项目一起备份(截图已补回) |
| `vllm_cscc/vllm`+`csrc` | 恢复基线与 3 处真实改动 | 保留;**恢复时按 `docs/01` 规格改,勿用当前树当终稿** |
| `vllm_cscc/dist`、`build/` | 基线产物/证据 | 占空间大,可留作证据(CMakeCache)或删除后按需重建 |
| `tools/`(19 个脚本) | 插桩/归因方法论 | 保留,已收录进 README 附件说明 |
| `profile*`、`log/`、`model_config/`、`importance/` | 原始数据/官方输入 | 保留 |
| `deepgemm-main/`、`dcu_profile_tool/` | 基线/工具链 | 按需保留(与终稿无直接关系) |
| `_baseline_decode/` | 早期副本 | 可删(与 06-20 树同源) |

## 恢复路径(有机器时)

1. **基线**:`vllm_cscc` + `dist` wheel 装入 DTK 26.04 容器(python3.10 / torch2.10 / vllm 0.18.1+das.dtk2604),模型 `Qwen3.5-27B`(锁定,不可改);评测口径:并发=1、RPS=1.0、每段 50 成功请求、长度段 4-8K/8-16K/16-32K。
2. **顺序**(每步打包 A/B,先对比段序再叠加):
   - 0) 纯基线 wheel 复现 12.20/8.81/4.64(注意 16-32K 用 4.64 还是官方 5.38 口径的确认);
   - 1) 1.1b csrc 宏 → rebuild wheel(必须先,否则 wvSplitK 空壳/assert)→ 1.1a 路由;
   - 2) 1.2→1.3→1.4(与 2.1 一起评估);
   - 3) 2.1+2.2 → 3.1+3.2 → 4.1+4.2+4.3;
   - 4) 5.1+5.2(5.3 默认关,最后);
   - 5) 终验:与说明文档五轮数字逐轮对表,记录差异,优先核对 R1 16-32K(3.64<4.64 异常)。
3. **正确性门**:端到端输出一致性 + `ast.parse` 语法校验 + 每轮 profile(TTFT/TPOT/吞吐三指标)后再累加。
4. **复现工具**:`tools/` 中 `_decode_only_profile.py`(NO_PROXY 已固化)、`_parse_profile_trace.py`(去污染版)、`fill_capture_hook.py` 已在 `06/08/15` 文档中说明使用方法。

## 结论

- 本地唯一能重建"终稿"的路径 = 说明文档截图 + 本交付 `docs/01` 规格表;仓库树仅支持"基线+C1"状态。
- 不可逆事实:机器/容器已收回,`/public/home/xdzs2026_c150/zya` 与评测容器不可达;任何需要上机验证的决策(如 1.1a 是否让 m=248320 走 LLMM1)只能暂记为"待验证"。
