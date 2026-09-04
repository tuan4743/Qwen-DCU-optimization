# 国产加速卡大模型推理优化实录:Qwen3.5-27B × 海光 DCU gfx936 × vLLM

> **将生产级大语言模型推理引入本土加速器** —— 在 Hygon DCU gfx936（非 CUDA、ROCm 系列架构）上的完整适配及内核级优化记录，实测 **输入长度分段的吞吐量提升 60% ~ 163%**。
> 终稿说明文档见 `优化记录/基于国产加速卡的千问大模型推理服务优化说明文档.md`
> 本项目为个人独立完成（除PyTorch/HIP基础库外），**未使用任何第三方手写高性能算子库**，所有适配逻辑均为针对gfx936微架构的原创调优。
> **Agent 说明**:Agent 用于以下方面:代码采样、profile、插桩、代码校验与文档整理。

## 01. TL;DR — 最终成果

| 指标 | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| 吞吐 4-8K | 12.20 tok/s | **19.56 tok/s** | **+60.3%** |
| 吞吐 8-16K | 8.81 tok/s | **14.92 tok/s** | **+69.4%** |
| 吞吐 16-32K | 4.64 tok/s | **12.22 tok/s** | **+163.4%** |
| P99 TPOT | 69.00 ms | **45.14 ms** | **−34.6%** |
| P99 TTFT | 4789.70 ms | **1964.41 ms** | **−59.0%** |

在Hygon DCU gfx936上实测 Qwen3.5-27B : 长上下文段最高 2.6 倍吞吐、首字时延腰斩,可复现与泛化。

## 02. 项目背景

本项目以 **Qwen3.5-27B**(MoE:64 层 = 48 层 GDN 门控增量网络 + 16 层全注意力,hidden 5120,vocab 248320)为对象,在**非 CUDA 主流栈(ROCm 家族)的海光 DCU gfx936** 上完成 vLLM 全链路适配与算子级重写适配。

profile分析栈:
- decode 单步内 **64 层 GPU kernel 串行**(cudagraph 下每层 ~1ms ≈ TPOT),GPU duty cycle 97.3%——**瓶颈在 kernel 内部,不在调度**;
- GDN 层的瘦 GEMM(M=1 matvec)占单步 **~87%**,而 m=1 在通用 BLAS 上无算法可选(单 algo 下限);
- 长上下文 Prefill 受共享内存(LDS)预算约束,tile 尺寸必须按 DCU 特性重排;
- 因此优化最终组合为: **适配层(适配 gfx936)+ 算子层(瘦 GEMM 路由 / Flash-Attention 瓦片 / 跨阶段 in_proj 融合 / 融合 Chunk 预处理)**。   

## 03. 适配文档

### 硬件

| 项 | 值 |
|---|---|
| 加速卡 | 海光 DCU **BW3000**(amdsmi: `market_name=BW3000`,vendor 成都 C-3000) |
| 架构 | **gfx936**,80 CU,64GB HBM,实测带宽 ~3.2 TB/s |
| 矩阵单元 | **注意**:AMD 标准 MFMA(`v_mfma_*`)在 gfx936 **编译可过、运行时非法指令/VMFault**——硬件未实现;实际可用的是海光自有 **`v_mmac_*`**(bf16 16×16×8/16)与标量 FMA 流水 |
| 特殊点 | 此卡有768个VGPR,相较于其他海光或AMD卡更多 |
| 算力定位 | 理论 BF16 峰值 490TFlops,但 Triton 实测环境平均值为 175.4TFlops, 远<1/3 峰值;分析结论:**非带宽 bound,是 tile/指令效率 bound** |

### 软件栈

| 层 | 版本 |
|---|---|
| OS / Python | Linux · Python 3.10 |
| 编译器栈 | DTK 26.04(`/opt/dtk-26.04-DCC2602-0317`),DCC clang 17,HIP 6.2/6.3 |
| 深度学习 | PyTorch 2.10(ROCm/HIP 构建) |
| 推理框架 | vLLM fork(`OpenDAS/vllm_cscc`,v0.18.1+das.dtk2604,HEAD `fa71803`) |
| 数学库 | 海光适配 rocBLAS / hipBLASLt(0.10.0)/ aiter;Triton(光合社区可找,暂未适配,慎用) |

### 适配(踩坑)记录 —— 仅简要展示

1. **架构白名单**:原始版本vLLM出于通用性考虑, `_ON_GFX9` 仅列 `["gfx90a","gfx942","gfx950"]`,`gfx936` 被排除 → `use_skinny=False`,全部 skinny GEMM 自定义核(LLMM1/wvSplitK)对 DCU 关闭,decode 投影全部落入通用 `F.linear`分支。理论分析,GEMM内核绝大部分情况下一定比常规分支更快.**修复:Python 侧 `(on_gfx9() or on_gfx936())` + LLMM1-only 分支**。
2. **矩阵指令缺失**:`v_mfma_*` 运行时崩溃(HSA ILLEGAL_INSTRUCTION + KERNEL VMFault,微基准隔离可复现);`v_mmac_*` 可用。判断路径用反汇编 + 微基准,不能使用编译器默认的 mattr。
3. **Triton 实测崩盘**:gfx936 上 Triton 实测 <1/3 峰值,有记录文档;所有 Triton 路线按低效预期避开;性能敏感算子走 HIP C++ 或原生库,并一律用 torch profiler 实测核名归因防止翻车(之前因为静态推断失误导致浪费了大量时间)。
4. **capability 错误路径**:gfx936 上报 capability 居然为9.x,触发 vLLM 按 BLOCK=128 选 tile;对 Qwen head_dim=256 的 Prefill 会爆 LDS/严重 thrash → 按共享内存预算出**强制 BLOCK=32** 同时收窄 autotune(BKV=[64], BT=64)为最优划分。
5. **工具链踩坑,gfx936未适配**:`hipcc` 找不到 clang,通过设置 `HIP_CLANG_PATH=/opt/dtk/lib/llvm/bin` 解决;gfx936 LLVM 后端缺 global→LDS 绕寄存器指令(Cannot select);`-real-size 32`、TF32、FP8(segfault)等 CUDA 盘常规手段在此均不可用。
6. **运行级保障**:`TORCH_BLAS_PREFER_HIPBLASLT=0`(M=1 不进慢 BLAS 后端)、k>8192 硬封顶回退(避免数值漂移)、cudagraph + duty-cycle 校准(97.3% 满载,端到端瓶颈在 GPU kernel 内部而非调度)。

## 04. 优化汇总

> 详版见 `最终优化报告.md`;逐文件改动与置信度见 `vllm_final/README_变更清单.md`。

| 轮 | 主题 | 关键改动 | 阶段效果(4-8K 吞吐) |
|---|---|---|---|
| 一 | 瘦 GEMV + Prefill 瓦片校准 | 打开 gfx936 的 LLMM1 路由(仅 n==1 & k≤8192 & 无 bias);BLOCK=32 防 LDS 爆;Flash-Decoding 段数 16→32;KV 访存 evict_last | **12.20→16.20(+32.8%)**,破局 |
| 二 | 长上下文 | 自适应 Flash-Decoding;非对称 Prefill 瓦片 32×16 + stages 3 | 稳定化,16-32K 回升 |
| 三 | GDN/FLA 定参 | BLOCK_M 规范为 2 次幂;钉 BKV=[64]、BT=64,收窄 autotune 空间 | **16.20→17.25;16-32K 3.64→6.99** |
| 四 | 大瓦片 + 初步融合 | Prefill FA M=128/N=32;Decode GDN BV=128;单阶段 in_proj 融合 | 17.25→19.12;TTFT 首破 2s |
| 五 | **算子融合** | TILE_N 32→64;**跨阶段 in_proj 权重重排融合**(`_fused_in_proj_weight`);融合 Chunk 预处理(默认关,宏开关) | **19.56 / 14.92 / 12.22 tok/s,TPOT 45.14ms** |

- **归因以 profiler 实测核名为准**vllm在运行过程中函数名会动态改变,静态分析无法查证,并且kernel名称也会撞车,只有将profile桩插入正确内层位置才能得到标准kernel名称;

## 05. 目录结构

```
qwen3_dcu_optimize/
├── README.md                       ← 本文件
├── 最终优化报告.md                  ★ 全貌报告(背景/基线/五轮/经验/风险)
├── vllm_final/                      ★ 最终修改版代码(12 文件修改)
│   ├── README_变更清单.md           每文件改动/置信度/未改动项
│   ├── csrc/rocm/skinny_gemms.cu
│   └── vllm/…(utils.py、rocm.py、triton_*.py、fused_recurrent.py、chunk_o.py、
│              env_override.py、qwen3_next.py、qwen3_5.py、fla/ops/fused_chunk_preprocessing.py)
├── vllm_origin/                     ★ 修改前(fork 基线)同 12 文件,路径镜像
├── 优化记录/                        ★ 任务文档 01-19 + 终稿说明文档 + 19 张截图
│   ├── 图片和附件/                   代码对比/效果/指标截图(终稿唯一代码证据)
│   ├── 工具脚本/                    profile 插桩/归因脚本(19 个)
│   └── log/                        启动/错误日志
├── profile/                        ★ profile 物证:批3 trace(gz)、pmc 结果、hipkernel 结果、kernel 参数表
└── docs/                           ← 面向读者的说明
    ├── 00_audit_local_vs_final.md  成果总览(30 秒看完)
    ├── 01_final_changes_spec.md    15 项改动详解(给要复现的人)
    ├── 02_summary_corrections.md   为什么"算子融合"是最有价值项
    ├── 03_assets_map.md            仓库导览
    └── 04_ocr_crosscheck.txt       截图核对底稿(可跳过)
```

## 06. 引用与延伸

- 环境与知识沉淀(工具链坑、矩阵指令事实、duty-cycle 方法)详见 `优化记录/` 01-19 任务文档。
- 若在新一批国产卡(如 gfx938 或更新 DTK)复现,按 `docs/01_final_changes_spec.md`(复现指南)+ `vllm_final/README_变更清单.md` 逐项应用,并优先 A/B 验证 LLMM1 路由对 lm_head 的边界。

## 07. 赛题背景(尾注)

本项目源自 2026 年国产算力推理优化赛题——Qwen3.5-27B 单请求在线推理(评测窗口 4-8K/8-16K/16-32K)。同台队伍追得很紧,**最终卡线止步:与前方差距不到 1 分钟**。赛题期间长上下文配置存在偶发抖动(瓦片/LDS 冲突致性能抖动 >5%);**赛后已通过 BV=128 定参与瓦片校准(autotune 收窄)解决,当前版本已稳定**。本文数据为赛题收官值,仓库为赛后整理稿。
