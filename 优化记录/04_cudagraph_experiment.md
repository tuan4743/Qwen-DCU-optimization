# 04 · cudagraph 对比实验

> 已完成,数据位于 `log/CUDA_graph/`。
> 本文件为冷数据档案,讨论开关 cudagraph 时按需读取。

## 4.1 结果对比

| 指标 | baseline(cudagraph ON) | eager(cudagraph OFF) |
|---|---|---|
| 输出吞吐 | **12.20 tok/s** | 7.39 tok/s |
| TPOT P99 | 69.0 ms | 122.2 ms |
| TTFT P99 | 4789.7 ms | 4678.0 ms |

## 4.2 配置差异(baseline vs eager)
- baseline:`compilation_config.mode=VLLM_COMPILE`,`cudagraph_mode=FULL_AND_PIECEWISE`,`capture_sizes=[1..256]`,`custom_ops=['+sparse_attn_indexer','none']`,`fuse_norm_quant=False`,`fuse_act_quant=False`
- eager:`enforce_eager=True`,`mode=NONE`,`cudagraph_mode=NONE`,`custom_ops=['+sparse_attn_indexer','all']`,`fuse_norm_quant=True`,`fuse_act_quant=True`

## 4.3 结论
1. 关 cudagraph 更慢(1.77×)。cudagraph 是净正收益,**必须保留**。
2. fill 在 cudagraph ON 下占 62%;关掉反而更慢,说明 **fill 不是 cudagraph 凭空引入的,而是模型/算子代码里本就有的 buffer 初始化,cudagraph 只是把它捕获进静态图**。优化它对两种模式都有效,baseline(cudagraph ON)是优化目标基线。
