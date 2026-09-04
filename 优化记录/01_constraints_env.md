# 01 · 硬约束、环境与算子基线

> 数据来源:`importance/提供的数据.txt`(硬边界,不可违反)。
> 本文件为冷数据档案,做方案前按需读取。

---

## 1.1 硬约束

### 允许修改
- KV Cache 与显存管理优化
- Decode 阶段算子优化
- 执行路径与调度优化
- 推理过程中的**非持久化、算子级低精度计算**:激活值动态量化、KV Cache 运行时量化、kernel 内临时类型转换、低精度矩阵乘、Attention 内核优化
- 镜像定制:装 Python 包、编译 custom kernel、**修改 vLLM 框架代码**

### 禁止修改
- 模型权重与结构、剪枝、量化操作边界、后训练/微调
- 推理框架核心参数(锁定):`model`/`tokenizer`/`tokenizer-mode`/chat template;`--max-model-len`/`--max-num-seqs`/`--max-num-batched-tokens` 及所有 batch scheduler 参数;采样 `temperature=0`/`max_tokens`;`--served-model-name`/OpenAI API 路径/host/port
- 投机解码 / 辅助模型
- 评测作弊

> ⚠️ 注意:`--enforce-eager`、`compilation_config`、`cudagraph_mode`、`custom_ops`、`pass_config` 这类**编译/图相关配置不在锁定清单内**(锁定的明确是 model/tokenizer/scheduler/sampling/接口)。已用 enforce-eager 做过对比实验(见 `04_cudagraph_experiment.md`)。

### 环境
- GPU:海光系列(保密),参数与 AMD Instinct 高度相似;VGPR 数量 768;Max Clock 1500MHz;8 节点;Max Queue 128
- PyTorch 2.10.0 / Python 3.10.12 / vLLM 0.18.1+das.3266200.dtk2604 / Transformers 5.5.0
- **docker 环境被严重简化,依赖几乎全无,roc 版本过低,无法手动编译复杂工具;只能用预装 rocprof / hipprof**。任何依赖额外编译的 profiling 工具(omniperf 等)都不可用。

---

## 1.2 算子性能基线(gfx936/海光,实测)

| 项 | 实测 |
|---|---|
| HBM copy 带宽 | 1247 GB/s(峰值) |
| hipBLAS BF16 GEMM | 403 TFlops |
| Triton BF16 GEMM | 175 TFlops(仅 hipBLAS 的 43.5%) |
| CK GEMM nt_cshuffle 4096³ | 144.7 TFlops |
| **DeepGEMM**(光合社区改版,支持 DCU BF16) | BF16 contig 8192³ = **280 TFlops**;BF16 asym K=12032 = **294 TFlops**;FP8 = segfault 不可用 |

## 1.3 DeepGEMM 定位(重要修正)
- DAS DeepGEMM 的核心接口是 **`m_grouped_bf16_gemm_nt_contiguous`(grouped GEMM,为 MoE 多专家设计)**。
- **Qwen3.5-27B 是稠密模型,FFN 是普通 GEMM,不是 grouped** → DeepGEMM 不能直接喂给 FFN。
- 稠密 GEMM 上:hipBLAS BF16 = 403T > DeepGEMM BF16 = 280T。**DeepGEMM 在稠密 GEMM 上无优势**。
- DeepGEMM 在本任务里的可能用途仅剩:把 dense GEMM 当"1 group"跑,或用其 `tf32_hc_pernorm_gemm`——但都不比 hipBLAS 快。
- **结论:DeepGEMM 这张牌对本稠密模型基本失效,不应作为主线。** 除非未来证实 FFN GEMM 实际走的不是 hipBLAS(需 profile 确认 FFN GEMM 实际 kernel 与 TFlops)。
