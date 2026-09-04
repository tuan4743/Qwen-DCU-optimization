# 最终修改版 · 变更清单与还原说明

> 本目录 = 说明文档五轮优化后的**最终修改版**(按截图+行级证据重建);`../修改前原版/` = 同文件的 fork 基线(修改前)。
> ⚠️ **重建品,非机器原样**:机器已收回,最终代码只能由说明文档截图(视觉转录+OCR 双源)配本地基线重建;未上机验证。每项标注置信度:**【高】**=截图给出了完整代码;**【中】**=截图给出主体、接线/实现细节由推断补全;**【低/骨架】**=截图只有门控或概念描述。
> 本文档与 `../docs/01_final_changes_spec.md`(逐项规格)配套;存疑点以 `图片和附件/` 原始截图为准。

## 文件清单(全部 12 文件,同路径镜像 vllm 仓库布局)

| 文件 | 轮次/子项 | 置信度 | 改动摘要 |
|---|---|---|---|
| `csrc/rocm/skinny_gemms.cu` | 1.1b | 【中】 | `__HIP__GFX9__` 宏名单加入 `__gfx936__`/`__gfx938__`(LLMM1/wvSplitK 可编译);新增 `__HIP_DCU_GFX93X__` 宏(改 shfl 归约、禁 mov_dpp)——宏定义已还原,宏在 kernel 内的使用点截图未给,需人工按 19 号文档 §1.4/1.5 补 |
| `vllm/platforms/rocm.py` | 1.1a | 【高】 | 恢复基线 `_ON_GFX9`(90a/942/950),新增 `_ON_GFX936` + `on_gfx936()` 谓词(终稿以独立谓词代替 C1 的列表加法,效果等价) |
| `vllm/model_executor/layers/utils.py` | 1.1a | 【高】 | `use_skinny` 加 `(on_gfx9() or on_gfx936()) and rocm_skinny_ops_available()`;新增 LLMM1-only 路由分支(`on_gfx936() and n==1 and m%4==0 and k<=8192 and bias is None → ops.LLMM1(weight, x_view, 4)`);`rocm_skinny_ops_available()` 实现为 `import vllm._rocm_C` 探测【中,截图只出现调用】;本地 C9b(m≤20000 carve-out)已删除 |
| `vllm/v1/attention/ops/triton_prefill_attention.py` | 1.2+2.2 | 【中】 | `get_block_size` 增 `head_size` 参数并加 ROCm∧hd≥256→32 分支(1.2);新增 `get_prefill_tiles`(返回 Bm,Bn,warps,stages:长输入 32,16,8,3;否则 32,32,8,2)+ forward 分流(2.2)——分流接线为推断【中】 |
| `vllm/v1/attention/backends/triton_attn.py` | 1.3+2.1 | 【中】 | `NUM_PAR_SOFTMAX_SEGMENTS` 16→32;新增 `_flash_decode_segments(max_seq_len, max_segments)`;`build()` 中 eager 按 `max_seq_len` 自适应、cudagraph 恒 32(接线为推断) |
| `vllm/v1/attention/ops/triton_unified_attention.py` | 1.4+3.1+4.1+5.1 | 【中-高】 | K/V 的 4 处 `tl.load` 加 `eviction_policy="evict_last"`(1.4,高);BLOCK_M 区:3.1 的 ROCm∧hd≥256 二次幂规范 + 4.1 的 `use_large_qwen_prefill_tile(head_size==256∧GQA==6∧q>1∧k≥4096)`→BLOCK_M≥128(高);TILE:5.1 的 `VLLM_ROCM_FA_PREFILL_TILE`(默认 64)∧`num_warps,num_stages=8,1`,仅大瓦片路径生效(中,launch 透传为推断);2.2 的"Unified TILE=16/stages=3"与 4.1/5.1 在大瓦片路径互斥,以 5.1 终态为准(恢复期可再加回非大瓦片分支) |
| `vllm/model_executor/layers/fla/ops/fused_recurrent.py` | 4.2 | 【高】 | 全 T 循环与 packed decode 两处:`is_qwen35_decode = HV==48 and V==128 and K==128` → `BV=128`、`num_warps=4`(否则维持原值) |
| `vllm/model_executor/layers/fla/ops/chunk_o.py` | 3.2 | 【高】 | ROCm:`BKV_LIST=[64]`(禁 128)、`NUM_WARPS=[4,8]`、`_NUM_STAGES=[2,3]`(去 stages=4),autotune 遍历改用 `_NUM_STAGES`;`BT=64` 由既有 `FLA_GDN_FIX_BT` env 生效 |
| `vllm/env_override.py` | 5.3 | 【高】 | `os.environ.setdefault("VLLM_ROCM_GDN_FUSED_PREPROC", "0")`(默认关闭) |
| `vllm/model_executor/models/qwen3_next.py` | 4.3+5.2 | 【中】 | `self._fused_in_proj_weight=None`(init);decode forward 改单次 `rocm_unquantized_gemm`(权重 cat 输出维),按 `in_proj_qkvz.weight.shape[0]` 切 qkvz/ba,`num_tokens==1` 时 `contiguous()`;权重拼接时机=首次 forward(截图未给 load_weights 细节,接线为推断) |
| `vllm/model_executor/models/qwen3_5.py` | 4.3+5.2 | 【中】 | 同构(qwen3_5 自身 forward 的混合拆分保持原逻辑) |
| `vllm/model_executor/layers/fla/ops/fused_chunk_preprocessing.py` | 5.3 | **【骨架】** | **新文件**:环境门控与接口骨架;融合 kernel(cumsum⊕KKT⊕tril⊕WU)截图未给出实现,标注 NotImplementedError;默认 0 不参与推理,可在恢复期跳过完整实现 |

## 未重建 / 未收录(如实说明)

| 文件/项 | 原因 | 建议 |
|---|---|---|
| `csrc/rocm/attention.cu` | 文档 1.1 路径列出,但全文件与瘦 GEMM/LLMM1 无关(grep 零命中),无可见改动 | 无需改动;不收录 |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | 文档 2.2 路径列出,无具体改动描述/截图 | 不收录;若恢复期需要再对截图复核 |
| `vllm/v1/attention/backends/utils.py` | 文档 3.2 路径列出,无具体改动描述 | 同上 |
| `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` | 3.2 只展现了 chunk_o 定参;chunk_delta_h 的 BV/warps 收窄为同类推断 | 恢复期按 chunk_o 风格同步收窄 |
| `fla/mamba/ops/causal_conv1d.py`(vendored `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`) | 同上,文档路径(可能指 site-packages fla)与 vendored 对应关系需确认 | 恢复期确认 |
| `vllm/version.py` | 非优化项,是环境级构建 hack(`0.18.1+das.dtk2604`,含自引用 import 瑕疵) | 原版见 `修改前原版/`,建议恢复 upstream 写法 |
| `vllm/model_executor/layers/fla/ops/utils.py` | 3.2 的 `FLA_GDN_FIX_BT` env 本地 fork **已有** | 无需改动 |
| 2.2 非大瓦片分支的 Unified TILE=16/stages=3 | 与 4.1/5.1 互斥未合入 | 恢复期按需补 |

## 与本地工作树的差异提示

- `vllm_cscc` 工作树 = 07-13 快照:含 C1(`rocm.py` 列表加 gfx936,**终稿已改为独立谓词**)与 C9b(utils.py m≤20000 分支,**终稿已删除**)与 version hack——均已被本"最终修改版"取代;恢复时**不要**基于工作树逐文件叠加,直接以本目录的"原版(基线)+最终修改版"diff 为准。
- diff 方法:`git diff --no-index 修改前原版/<f> 最终修改版/<f>` 即可得到每文件完整改动。
