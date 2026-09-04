# Qwen3.5-27B · 海光 DCU(gfx936)· vLLM 推理服务优化 —— 项目定稿

> 终稿说明文档见 `优化记录/基于国产加速卡的千问大模型推理服务优化说明文档.md`
> 全貌阅读顺序:**本文 → `最终优化报告.md` → `最终修改版/README_变更清单.md` → `docs/00`**

## 0. 最终成绩

**最终输出吞吐 4-8K 12.20→19.56 tok/s(1.60×)、8-16K 8.81→14.92(1.69×)、16-32K 4.64→12.22(2.63×);P99 TPOT 69.00→45.14ms(−34.6%);P99 TTFT 4789.70→1964.41ms(−59%)。**

## 1. 目录结构

```
qwen3_dcu_optimize/
├── README.md                       ← 本文件(索引)
├── 最终优化报告.md                  ★ 全貌报告(背景/基线/五轮/经验/风险)
├── 最终修改版/                       ★ 最终修改版代码(12 文件修改)
│   ├── README_变更清单.md           每文件改动/置信度/未改动项
│   ├── csrc/rocm/skinny_gemms.cu
│   └── vllm/…(utils.py、rocm.py、triton_*.py、fused_recurrent.py、chunk_o.py、
│              env_override.py、qwen3_next.py、qwen3_5.py、fla/ops/fused_chunk_preprocessing.py)
├── 修改前原版/                       ★ 修改前(fork 基线)同 12 文件,路径镜像
├── 优化记录/                        ★ 任务文档 01-19 + 终稿说明文档 + 40 张截图
│   ├── 图片和附件/                   代码对比/效果/指标截图(终稿唯一代码证据)
│   ├── 工具脚本/                    profile 插桩/归因脚本(19 个)
│   └── log/                        启动/错误日志
├── profile/                        ★ profile 物证:批3 trace(gz)、pmc 结果、hipkernel 结果、kernel 参数表
└── docs/                           ← ai 校验工作文档
    ├── 00_audit_local_vs_final.md  本地代码 ↔ 终稿逐项校验报告(15 项)
    ├── 01_final_changes_spec.md    终稿改动规格表(行级锚点+代码,双源转录)
    ├── 02_summary_corrections.md   19 号文档结论勘误(融合=最有价值论证)
    ├── 03_assets_map.md            原始工作区资产地图与恢复路径
    └── 04_ocr_crosscheck.txt       截图 OCR 交叉校验底稿
```

**AI辅助说明**:AI用于采样profile与插桩,校验与代码审查
