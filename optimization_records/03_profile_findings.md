# 03 · Profile 数据与瓶颈特征

> 全部来自 baseline 即 cudagraph ON 的一次运行。
> 本文件为冷数据档案,复核瓶颈时按需读取。

## 3.0 数据文件
- `profile/result_23940.hipkernel.csv` — 全量 kernel 聚合耗时(85589 calls,总 13.2s)
- `profile/vector_kernel_params.csv` — kernel 级采样(54619 行,约全量 64%),含 grid/wgr/queue-index
- `profile/pmc_results_25232.txt` — 仅 flash_fwd 单 kernel 的硬件计数器

## 3.1 全量耗时 TOP(占比)

| kernel | calls | 总耗时 | 占比 |
|---|---|---|---|
| `vectorized_elementwise_kernel` (FillFunctor<int>, **256MB memset**) | 37575 | 8.25s | **62.4%** |
| `flash_fwd_kernel_16x64_prefetch` (ViT 注意力) | 27 | 3.86s | **29.2%** |
| `chunk_fwd_kernel_o` (GDN prefill) | 18769 | 0.26s | 2.0% |
| 其余(GEMM/attention 等) | — | ~1.1s | ~8.4% |

---

## 3.2 256MB fill 的特征(已确认)

- 聚合:`result_23940` 把所有 `FillFunctor<int>`(不分 grid)合并为一条 —— **37575 calls、8.25s、62.4%**。avg=219μs ⇒ 时间被大尺寸 fill 主导(小 fill 如 grid=256 仅 ~1μs,不可能拉出 219μs 均值)。
- 采样(`vector_kernel_params`)里 `FillFunctor<int>` 的 grid 分布:**grid=16,777,216(=256MB int32)占 7327 次**,grid=256 仅 23 次。即绝大多数 fill 是 256MB 这一种。
- 单次 **219μs**,写 256MB ⇒ **1.17 TB/s = 峰值 HBM 带宽**(实测 copy 1247 GB/s)。带宽打满,**无法靠"更快清零"优化,只能减少次数/大小**。
- 256MB fill:grid=16,777,216;`FillFunctor<int>`(int32);元素数 = 67,108,864 = 2^26 = **4096 × 16384**(`max_num_batched_tokens × max_num_blocks`)。
- 计数口径:采样 7327 次大 fill,聚成 ~19 个簇(≈请求数),每簇 ~410 次;全量聚合 37575 次(采样对该 kernel 覆盖约 20%,故真实大 fill 次数远多于采样值)。总写入量按"全为 256MB"计为上界 **9.6 TB / 8.25s**。
- **发生阶段:运行时每个 forward step 都在重放(cudagraph replay),prefill(chunked)和 decode 都有**(2026-06-29 qi 分布实测钉死)。
- **【2026-06-29 qi 分布实测,推翻"capture-only"假说】**:`vector_kernel_params.csv` 中 big int-fill(grid=16,777,216)样本 7327 条,queue-index 跨 **qi 2398 → 33398**,全量 kernel qi 范围 **1 → 38690**(big-fill 覆盖 80%)。按 1000-qi 分桶每桶稳定 ~226–250 次、连续不聚集 —— 若 capture-only 应只在 trace 起点成簇。聚合 37575 次 ÷ 64 层 ≈ 587 步 ≈ 每次 forward 每层 1 次。**结论:capture 期录进图,replay 期每步回放;fill 的"来源"是 capture 期某处 `torch.zeros/zero_/fill_`,但"耗时间"贯穿整个运行时。** 完整物证与步数反推见 `05_task_tracker.md` §5.3.1。
- **Python `Tensor.fill_`/`zero_` hook 抓不到它** → 被 cudagraph 捕获进静态图,replay 时不走 Python runtime,故 hook 不命中;capture 期 `_record_memory_history` 探针也抓不到(replay 期不记录)。已废定位路线见 `06_pitfalls.md`。
- ⚠️ **修正(P0 dump 复核,2026-06-24)**:原判断"它是 inductor/torch.compile 编译图节点"**不成立**。依据:
  1. `result_23940.hipkernel.csv` 里真凶 kernel 全名是 `at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, ...>` —— 这是 **PyTorch eager 的 `at::native` fill kernel**,**不是** inductor 生成的 `triton_poi_fused_*`/`triton_red_fused_*` 融合 kernel。
  2. dump 的 9 个 inductor 图(`profile/run_*_pid_1983/torchinductor/model__*_inference_*`)**全部不含任何 `aten.fill`/`aten.zero`/`new_zeros` 节点**;9 个图都是 layernorm/elementwise 小图(piecewise 编译,RMSNorm 的 f32 计算,如 `model__4` 真实尺寸 `[4096,4,256]`=FullAttn q/k_norm),唯一命中 "16384" 的是 `model__4/output_code.py:75` 的 `size_hints={'x':16384}`(reduction 的 size hint,**非 buffer 形状**)。
  3. dump 的图编号不连续(0,1,3,4,5,6,8,9,11,缺 2/7/10),均为小图(22-25 行)——只覆盖 profile/warmup 触发的 piecewise 图,**主解码 forward 大图未被捕获**(可能在 dump 前退出/目录被截断)。
  - **新结论**:256MB fill 是 **cudagraph 在 capture 期录进去的一次 eager `at::native::fill`(来自某处 `torch.zeros/zero_/fill_`)**;capture 后由 cudagraph replay 回放,所以每次 forward 都重放、但 Python runtime 不再触发 → hook 抓不到、也不在 inductor 图里。优化它仍对 baseline(cudagraph ON)有效,且方向仍是"减量/减次/缩小尺寸"。
- **静态源码排查已排除(2026-06-28 第二轮穷尽复核)**:对 vLLM V1 全部 GPU-resident int32 大 buffer 逐一比对 shape/dtype/backing,均不等于 `[4096,16384]×int32=256MB`,排除项如下:
  - **MLA sparse indexer `expanded_block_table_buffer`**(`v1/attention/backends/mla/indexer.py:265`,**唯一在源码层面 shape 完全吻合** `[max_num_batched_tokens=4096, cdiv(max_model_len,block_size)=16384]×int32=256MB`,GPU 常驻):**已确认本模型不实例化该 indexer**。证据 `log/baseline/vllm_start_log.txt` 第 27 行 `Using TRITON_ATTN attention backend`、第 28 行 `WARNING ... Op 'sparse_attn_indexer' not present in model, enabling with '+sparse_attn_indexer' has no effect`。即 Qwen3.5 FullAttn→`FullAttentionSpec`→`TRITON_ATTN`,GDN→`MambaSpec`→`GDN_ATTN`,**走 GDN backend 不创建 MLA indexer**;`+sparse_attn_indexer` 是 no-op。→ **彻底排除 indexer 假说**,该 shape 吻合纯属 `max_num_batched_tokens × cdiv(max_model_len,block_size=16)` 公式巧合。
  - **NEW 活跃 block_table 路径**(`v1/worker/gpu/block_table.py`,`model_runner.py:334` 构造,本模型实际走这条):
    - `block_tables` = `StagedWriteTensor((max_num_reqs=128, max_num_blocks=cdiv(262144,16)=16384), int32)` = **[128,16384]×int32=8MB/组**(dim-0 是 max_num_reqs,**不是** max_num_batched_tokens)→ **不是 256MB**。
    - `slot_mappings` = `torch.zeros(num_kv_cache_groups, max_num_batched_tokens=4096, int64)` = **int64**(非 int32)。
    - `get_dummy_slot_mappings` 的 `.fill_(PAD_SLOT_ID)` 命中 int64,非 256MB int32。
  - **OLD block_table 路径**(`v1/worker/block_table.py`,`CpuGpuBuffer` 版,本模型未走):同样 [128,16384]×int32=8MB,`slot_mapping=[4096]×int64`。
  - `gpu/states.py` `all_token_ids` = `StagedWriteTensor((128,262144), int32, uva_instead_of_gpu=True)` = **UVA(CPU pin_memory,非 GPU 常驻)→ 不会表现为 GPU fill**;且 shape 不符。
  - `gpu/rope.py` `prefill_positions` = `StagedWriteTensor((max_num_reqs*num_dims, max_model_len), int32, uva_instead_of_gpu=True)` = **UVA,非 GPU**;`positions` = `torch.zeros((num_dims, max_num_tokens+1), int64)` 小。
  - `attention/backends/mamba_attn.py` `state_indices_tensor_d`:**仅当 `mamba_cache_mode=="all"` 时为大** `[decode_cudagraph_max_bs=128, cdiv(max_model_len,block_size)=16384]×int32=8MB`;Qwen3.5 `mamba_cache_mode="none"`(且 `Qwen3_5ForCausalLMBase` 显式禁止 `"all"`,raise NotImplementedError)→ 为小 `[128, 1+num_spec]`。排除。
  - `attention/backends/triton_attn.py` `softmax_segm_output/max/expsum` = **float32 小尺寸**;`build_for_cudagraph_capture` 只 `seq_lens.fill_(1)`。排除。
  - `attention/backends/gdn_attn.py` 各 buffer = **[decode_cudagraph_max_bs=128, ...] 小 int32**,`.fill_(PAD_SLOT_ID)` 只清小切片。排除。
  - `attention/backends/utils.py:200` `make_local_attention_virtual_batches` 的 `block_table_local` 由索引产生,**非 256MB zero**;`mamba_get_block_table_tensor` 对 "none" 原样返回。排除。
  - `model_executor/models/qwen3_next.py:665` `core_attn_out` = `torch.zeros((num_tokens, num_v_heads//tp, head_v_dim), bf16)` = **bf16、token 形状**。排除。
  - `model_executor/layers/fla/ops/kda.py:759` `A`/`Aqk` = `torch.zeros(B,T,H,BT)`(BT=64,**token 形状小**)。排除。
  - `model_executor/models/qwen3_vl.py:1426` `deepstack_input_embeds` = `[torch.zeros(max_num_batched_tokens=4096, hidden_size, bf16)]` = **bf16**(非 int32);`_clear_deepstack_input_embeds` 只 `[:num_tokens].zero_()`。排除。
  - `gpu/input_batch.py` `InputBuffers`:`input_ids[4096] int32`、`positions[4096] int64`、`query_start_loc[129] int32`、`seq_lens[128] int32`、`dcp_local_seq_lens[128] int32`,dummy 仅 `[:num_tokens].zero_()` → **远小于 256MB**。排除。
- **【2026-06-30 P0 钉死,根因已实机确认】** 该 256MB int32 fill 的确切来源 = **Triton autotune 的 L2-cache-flush buffer,非 vLLM 业务 buffer**。
  - 分配点:`triton/backends/amd/driver.py:718-721` `get_empty_cache_for_benchmark()` 写死 `cache_size=256*1024*1024`,`torch.empty(int(cache_size//4), dtype=torch.int, device='cuda')` → int32,67,108,864 元素 = `[4096,16384]` reshape,**与 shape 完全吻合**。
  - memset 本体:`triton/backends/amd/driver.py:723` `clear_cache(cache): cache.zero_()`,由 `triton/testing.py:178 do_bench()` 在每次 benchmark 前调用 → 即 `at::native::FillFunctor<int>`。
  - 触发链路:`profile_run` warmup → `qwen3_next.py:_warmup_prefill_kernels` → `gdn_attention_core` → FLA chunk attention 子核(`chunk_o`/`chunk_scaled_dot_kkt`/`solve_tril` 等)触发 Triton autotune → `do_bench` 反复 `cache.zero_()`。
  - 物证:`fill_alloc_probe_ckpt1_pre_capture.jsonl` 中 161 个 `EXACT_256MB` 块,FULL-STACK leaf 161/161 命中 `driver.py:721 get_empty_cache_for_benchmark`;USER frame 105/161 = `fla/ops/chunk_o.py:166 chunk_fwd_o`。详见 `07_p0_conclusion.md`。
- **静态源码已到极限(已闭合)**:全代码库中 GPU-resident、int32、且 shape 恰为 `[4096,16384]` 的只有 indexer 那一处(已证不实例化)。其余 GPU 大 buffer 要么 UVA 非 GPU、要么 int64/bf16、要么 dim-0 是 max_num_reqs(8MB)。**静态 0 命中正是因为它不是 vLLM 的 buffer,是 Triton 工具自身的 benchmark cache** —— 必须靠实机运行时手段,现已钉死。原"静态无法 100% 钉死,需实机"判断已闭环,结论见 `07_p0_conclusion.md`。

---

## 3.3 flash_fwd(ViT)的特征
- 27 次 == ViT depth=27,**就是视觉编码器的逐层注意力**(dim96 = ViT head_dim 72 pad 到 96)。
- 仅出现在 trace 开头(qi 1747–2349),**一次性**(共 27 次 == ViT depth=27),与 256MB fill(qi 2398 起、贯穿至 qi 33398)起点几乎相接但不重叠。
- ⇒ benchmark 首请求带了图像,或 warmup 触发了 ViT。**纯文本任务里这是噪声/超纲成本**,优化文本推理时不应把它算进目标(但若 benchmark 强制带图,则需单独评估)。
