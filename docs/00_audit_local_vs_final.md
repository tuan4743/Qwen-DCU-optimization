# 00 · 校验报告:本地源码 ↔ 说明文档最终版

> 目的:回答"本地 `vllm_cscc` 树是不是最终版,差在哪"。
> 结论先行:**本地树是 07-13 快照(止于 C1+C9b),不是终稿;终稿唯一权威载体 = 说明文档 + 40 张截图(09-04 组装)。五轮 15 个改动子项中,本地只含 1.1 的"旧式路由"(存在但写法不同),其余 14 项全部缺失(fork 基线原样)。**
> 本报告只陈述证据;截图级最终代码见 `01_final_changes_spec.md`。

---

## 1. 审计基础

### 1.1 本地树证据状态(三个独立证据,互相印证)

| 证据 | 结果 |
|---|---|
| `git -C vllm_cscc status/diff`(vs HEAD `fa71803`,上游 v0.18.1+OpenDAS fork) | 真实改动仅 3 文件:`vllm/platforms/rocm.py`(+1−1,加 `gfx936`)、`vllm/model_executor/layers/utils.py`(+6−1,C9b 注释+分支)、`vllm/version.py`(版本 hack);其余 66 个 Modified 均为 LF↔CRLF 噪音 |
| 文件 mtime | `vllm/**` 全 06-20(基线提取时刻);`rocm.py` 07-12 13:59;`utils.py` 07-13 11:17 → 树止于 07-13 |
| wheel/构建产物 | `dist/vllm-0.18.1+das.dtk2604-….whl`(06-20 20:12)与 `build/lib.linux…`(06-20 20:13)哈希一致 → **基线构建**;`build/` CMakeCache 显示 `GPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938` |

### 1.2 本地"最终态"能确认的改动(全量,共 3 处 + 1 处环境 hack)

1. **`rocm.py:149`(C1,唯一实证收益改动)**:`_ON_GFX9 = any(arch in ["gfx90a","gfx942","gfx950","gfx936"])` —— 给 `on_gfx9()` 打开 gfx936,使 `use_skinny` 链可用。内部实测(07-12):4-8K 12.20→17.40(+42.6%)、8-16K 14.87、16-32K 5.74。
2. **`utils.py:181-191`(C9b)**:`rocm_unquantized_gemm_impl` 的 wvSplitK 分支增加 carve-out——`m%4==0 and n==1 and k<=8192 and bias is None and m<=20000` 的"中型 matvec"(qkvz m=16384/k=5120)改走 `ops.LLMM1(weight, x_view, 4)`;`m>20000`(lm_head 248320、gate_up 34816)与 `n∈[2,4]` 仍走 `wvSplitK`。内部 07-13 判定"方向性无规律→中性",并在 DCU 上回退;**本地树残留未回退版本**。
3. **`version.py`(构建 hack)**:`__version__="0.18.1"`+`__hcu_version__='0.18.1+das.dtk2604'`;代码含自引用 `from vllm.version import …` 与 `f"..." + str(e)` 拼错的 warn(前者运行时会自洽通过,后者为瑕疵)。"das" 疑为队名(登崇队)构建标签,待用户确认;不参与性能。
4. **`rocm.py` fork 自带差异**(非本队改动):模块级 `import vllm._rocm_C` 被注释、改 `import_kernels()` 方法延迟导入;capability 解析新增 major<9/major>12 raise。与性能无关。

### 1.3 ⚠️ 空壳 kernel 矛盾(本地态最大疑点)

- `csrc/rocm/skinny_gemms.cu:24-27`:`__HIP__GFX9__` 仅 `gfx90a/942/950`;**gfx936 在全部 csrc 零出现**;`wvSplitK_hf_sml_/hf_/hf_big_` 真体在 `#if __HIP__GFX9__` 内,`#else` 为 `UNREACHABLE_CODE`(= `assert(false)`,L520-531/749-759/1093-1104);`wvSplitKrc_` 仅 `#if defined(__gfx950__)`。
- 而 DCU 构建目标 = `gfx906;926;928;936;938`(`build/` CMakeCache + `CMakeLists.txt:40 HIP_SUPPORTED_ARCHS` 新增了 gfx928/936/938)→ **按本地态 csrc 构建的 DCU wheel 里,wvSplitK 系是 assert 空壳,LLMM1 是无宏保护的真码**。
- 由此:①本地态(07-13)"C1 打开 wvSplitK 链 +42.6%"在本地 csrc 前提下无法解释;②终稿 1.1 的 **HIP 侧宏修正(image 5:把 gfx936 纳入 `__HIP__GFX9__` + 新增 `__HIP_DCU_GFX93X__` 用 shfl 归约、禁 mov_dpp)正是补齐真 kernel 的修复**,且发生在本地同步之后(07-14/15,DCU 上) —— 这反过来证明"终稿≠本地树"且"终稿改动真实存在"。
- 结论:空壳矛盾**仅存在于本地快照**;终稿版本已通过 1.1 的 HIP 改动解决。恢复时必须把该改动放第一位。

---

## 2. 逐项校验(五轮 × 15 子项)

判定:🟢 已在本地(证据) | 🟡 部分(本地有旧式写法,与终稿不同) | 🔴 缺失(本地=fork 基线) | ⏳ 无法判定(依赖 site-package fla 等本地无件)

| # | 子项 | 文档声称(终稿) | 本地树现状(文件:行) | 判定 | 截图 |
|---|---|---|---|---|---|
| 1.1a | Decode 瘦 GEMV 路由 | `use_skinny = … and (on_gfx9() or on_gfx936()) and rocm_skinny_ops_available()`;新增 `if on_gfx936() and n==1 and m%4==0 and k<=8192 and bias is None: out=ops.LLMM1(weight, x_view, 4)` | `utils.py:170-191`:仍按 `use_skinny`=`env∧on_gfx9()∧…`;LLMM1 分支条件=本地 C9b 版(`m<=20000` 且无 `on_gfx936()`、无 `rocm_skinny_ops_available()`) | 🟡 旧式在、终稿式缺 | image 8 |
| 1.1b | HIP 侧打开 gfx936 编译 | `skinny_gemms.cu` 宏:`__HIP__GFX9__` 条件加入 `__gfx936__`(OCR 显示还含 `__gfx938__` 类),新增 `__HIP_DCU_GFX93X__`(shfl 归约、禁 mov_dpp) | `csrc/rocm/skinny_gemms.cu:24-31`:无 gfx936;全 csrc grep gfx936=0 命中;`attention.cu` 同 | 🔴 | image 5 |
| 1.2 | Prefill 瓦片上限 BLOCK=32 | `if is_rocm() and head_size is not None and head_size>=256: return 32`(替代 `elif is_cuda_alike() and has_device_capability(80): return 128`) | `v1/attention/ops/triton_prefill_attention.py:180-188`:仍 `fp32→32 / cc80→128 / 其余→64`,无 ROCm/hd 分支 | 🔴 | image 4 |
| 1.3 | Flash-Decoding 分段 16→32 | `NUM_PAR_SOFTMAX_SEGMENTS = 32` (# 提升长上下文 Decode 并行度) | `v1/attention/backends/triton_attn.py:43` = 16;缓冲按 16 分配(L171-190) | 🔴 | image 19 |
| 1.4 | KV 访存 evict_last | `tl.load(K_ptrs, …, eviction_policy="evict_last")`、同 V | `triton_unified_attention.py` 全文件 eviction_policy/cache_modifier 0 命中;唯一 evict_last 在 `chunked_prefill_paged_decode.py:167,180`(rocm_attn 路径,非终稿所指) | 🔴 | image 9 |
| 2.1 | 自适应 Flash-Decoding | `NUM_PAR_SOFTMAX_SEGMENTS=32` + `_flash_decode_segments(max_seq_len, max_segments)`(≤2048→min(16,ms);else `segs=1<<max((ms+255)//256-1,0).bit_length()`;`max(16,min(ms,segs))`) | `triton_attn.py:169` 恒定 16,无自适应函数(内部 15 文档曾把同类方案 C5a 判死路) | 🔴 | image 7 |
| 2.2 | 非对称 Prefill 瓦片 + 软件流水 | `get_prefill_tiles(dtype, head_size, max_input_len)`:is_rocm∧hd≥256 → `max_input_len≥8192` 返回 `(Bm,Bn,warps,stages)=(32,16,8,3)`,否则 `(32,32,8,2)`;Unified Prefill TILE=16,长 KV warps=8/stages=3 | `triton_prefill_attention.py` / `triton_unified_attention.py`:无此函数;`_get_tile_size` prefill 恒 32(L879-880),decode 16/32(L881) | 🔴 | image 20 |
| 3.1 | BLOCK_M 二次幂对齐 | `if is_rocm() and head_size>=256 and max_seqlen_q>1 and num_queries_per_kv>1: BLOCK_M=triton.next_power_of_2(max(BLOCK_M,32))`;`BLOCK_Q=BLOCK_M//num_queries_per_kv`(GQA=6→Bm=32/BQ=5) | `triton_unified_attention.py:942-945`:`BLOCK_M=16 if qpk<=16 else next_pow2(qpk)`,无 ROCm/长序列特化 | 🔴 | image 16 |
| 3.2 | FLA Chunk 定参 | `if is_rocm(): BKV_LIST=[64]; NUM_WARPS=[4,8]; _NUM_STAGES=[2,3]`(# 禁 BKV=128、去 stages=4);`FLA_GDN_FIX_BT → BT=64` | vendored `vllm/model_executor/layers/fla/ops/chunk_o.py:21` 仍 `[64,128] if check_shared_mem() else [32,64]`、stages `[2,3,4]`(L33-37);`chunk_delta_h.py` 同基线;`fla/ops/utils.py:27` 已有 `FLA_GDN_FIX_BT` env | 🔴 | image 10 |
| 4.1 | Prefill 大瓦片 FA(M=128,N=32) | `use_large_qwen_prefill_tile = is_rocm() and head_size==256 and GQA==6 and max_seqlen_q>1 and max_seqlen_k>=4096`;`BLOCK_M=next_power_of_2(max(BLOCK_M,128 if … else 32))`;`TILE_SIZE_PREFILL=32; num_warps,num_stages=8,1` | `triton_unified_attention.py` 无该门控;唯一形状特化=Gemma3(sw=1024∧hd∈{128,256}→decode TILE 32,L852-876) | 🔴 | image 18 |
| 4.2 | GDN Decode 值维瓦片 BV=128 | `is_qwen35_decode = HV==48 and V==128 and K==128`;`BV=128 if is_qwen35_decode else min(next_pow2(V),32)`;`num_warps=4 if is_qwen35_decode else 1` | `vllm/model_executor/layers/fla/ops/fused_recurrent.py:436,438`(decode 入口 L338-477)仍 `BV=min(next_pow2(V),32)`、`num_warps=1`;全文件无该特化 | 🔴 | image 21 |
| 4.3 | Decode in_proj 融合 + hipBLASLt 旁路 | `if self.fused_in_proj_weight is not None and num_tokens==1: fused=rocm_unquantized_gemm(h, self.fused_in_proj_weight, None); qkvz,ba=split(fused)`;`os.environ["TORCH_BLAS_PREFER_HIPBLASLT"]="0"` | `qwen3_next.py:650-651`(及 qwen3_5.py:183-189)仍两次独立 Linear;全树无 `fused_in_proj_weight`(`_fused_in_proj` grep=0);无 TORCH_BLAS_PREFER_HIPBLASLT 代码 | 🔴 | image 12 |
| 5.1 | FA TILE_N 32→64 | `tile_n=int(os.getenv("VLLM_ROCM_FA_PREFILL_TILE","64"))`;`TILE_SIZE_PREFILL=tile_n if tile_n in (16,32,64) else 64`;`num_warps,num_stages=8,1` | `triton_unified_attention.py` 无该 env;TILE_SIZE_PREFILL 由 `_get_tile_size` 恒 32;`envs.py` 无 VLLM_ROCM_FA_PREFILL_TILE | 🔴 | image 30 |
| 5.2 | 跨阶段 in_proj 融合 | `load_weights` 后拼接 `_fused_in_proj_weight`(qkvz+ba 输出维);forward:`if self._fused_in_proj_weight is not None: fused=rocm_unquantized_gemm(h,…); qkvz,ba=fused[...,:split],fused[...,split:]; if num_tokens==1: qkvz,ba=qkvz.contiguous(),ba.contiguous()`(# Prefill 用 view 避免 T×out 二次 HBM 写) | `qwen3_next.py`/`qwen3_5.py` 无任何权重拼接/`_fused_in_proj_weight`(grep=0);load_weights 为 stacked 参数映射(upstream 标准 packed) | 🔴 | image.png(及 image 12) |
| 5.3 | 融合 Chunk 预处理 | **新增** cumsum⊕KKT⊕tril⊕WU 融核;`os.environ.setdefault("VLLM_ROCM_GDN_FUSED_PREPROC","0")`(默认 0,按需门控) | vendored `fla/ops/` 无 `fused_chunk_preprocessing.py`(目录清单 16 文件,无此项);`env_override.py`(484 行)=上游原版,无任何 VLLM_ROCM_* 逻辑 | 🔴 | image 1 |

**汇总:🟢 0 项 / 🟡 1 项(1.1a)/ 🔴 14 项 / ⏳ 0 项。** 即:本地树不是终稿,终稿 93% 的改动本地无代码痕迹;唯一相符的是 1.1 的**意图**(瘦 GEMV 路由),但写法是旧版。

---

## 3. 内部文档 ↔ 终稿的一致性结论(据 05/12/15 全量盘点)

- 内部文档(止于 07-13)有**实测落地记录**的只有:**C1(rocm.py +gfx936,+42.6%)**;其余 C5b(标签作废:16 是出厂值)、C4-3(灾难回滚)、C9b(中性回退)、C5a(死路,即终稿 2.1 的逆命题)、B10(投影+递归核融合死刑)、手写 split-K 算子(集成失败)。
- 终稿 13 子节中 **11 个(1.2/1.4/2.2/3.1/3.2/4.1/4.2/4.3/5.1/5.2/5.3)内部零记录**;1.3 与内部结论**方向相反**(内部:32 段致 16-32K 崩塌、16 为出厂值;终稿:16→32);2.1 即内部排除的 C5a;1.1 与内部"C1=wvSplitK 非 LLMM1"的订正**冲突**(终稿:on_gfx936()+LLMM1 路由)。
- **最可能的真相**(据此重建叙事):07-14 评测基线截图(17:06)后至 07-15 上午,在"评测/演示窗口"完成五轮并截图(09:04-09:13);19_final_summary(07-15 12:14)由独立执行窗口撰写,只掌握 C1+手写算子失败,未反映五轮状态;说明文档 09-04 组装定稿,成为唯一权威。**用户已确认"融合已改出来、最有价值",接受此叙事。**

## 4. 指标对照与异常

| 数据源 | 4-8K | 8-16K | 16-32K | 备注 |
|---|---|---|---|---|
| 官方给定(importance/提供的数据.txt) | 12.20 | 8.81 | **5.38** | 官方 baseline 16-32K = 5.38 |
| 内部基线(04/12/15,06-28 实测) | 12.20 | 8.81 | **4.64** | 内部统一用 4.64 |
| 终稿 baseline(汇总表) | 12.20 | 8.81 | **4.64** | TTFT 28744.39 与内部 28.7s 同量级 |
| 终稿五轮 | 16.20→…→**19.56** | 9.44→…→**14.92** | 3.64→…→**12.22** | 逐轮递增见 README §2 |
| 内部 C1 | 17.40 | 14.87 | 5.74 | 07-12 |
| 内部 C1+C5b | **18.26** | 12.30 | 8.61 | 内部纪录上限 |
| 官方目标(15 文档) | 21.4 | 19.81 | 16.32 | 未达标 |

**异常(A)**:终稿 R1 16-32K=3.64、R2=4.39 均 **低于基线 4.64**;R2 4-8K=16.05 低于 R1 的 16.20(不单调);终稿 R1 8-16K TTFT 15772.61≈内部 C1 的 15552.21,R1 16-32K TTFT 28823.85≈内部 C1 的 28723.59 → 疑 R1 实际就是"内部 C1 态"换名,其 16-32K 吞吐 3.64 反而更接近 C4-3 灾难值 3.58。**以终稿为权威,标注存疑,建议恢复期逐轮复测。**
**异常(B)**:官方给定 16-32K baseline=5.38 vs 终稿 4.64(内部实测口径),恢复期按终稿口径(4.64)。
**异常(C)**:终稿五轮数字均无内部出处,且 R4/R5(19.12/19.56、9.73/12.22)超过内部纪录上限(18.26/8.61)——说明终稿测量发生在内部文档记录窗口之外(07-14/15 评测期),属正常,但**无法交叉复核**。

## 5. 限制声明

1. 截图由视觉模型转录 + WinRT OCR 双源交叉;代码文本仍有少量不确定字符(规格表中以 [?] 标注,如 image 16 的 `is_roc m()`、宏名单的 `__gfx938__`),恢复前应以原始截图人工复核。
2. `fla` 依赖:终稿 3.2/5.3 涉及 `fla/ops/*`,本地仅有 **vendored** `vllm/model_executor/layers/fla/ops/`(16 文件);终稿若是按 site-packages `fla` 路径修改,恢复时需二选一(建议:统一改 vendored 树,与本地基线一致)。
3. 内部文档与终稿的"矛盾"多数可由时间线(07-13 内部记录窗口 vs 07-15~09-04 终稿窗口)解释;无法解释的(如 B10 与 4.3/5.2 的对象差异)已在 `02_summary_corrections.md` 中给出具体区别。
4. 本报告不修改任何源码与历史文档(19 号文档的勘误以追加标注形式进行)。
