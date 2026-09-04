# 01 · 终稿改动规格表(按说明文档五轮,共 15 子项)

> 用途:机器回归时逐项套用。每个子项给出:目标文件(本地树路径)、本地当前锚点、终稿代码(来自 `图片和附件/` 截图视觉转录 + WinRT OCR 双源交叉;`[?]`=存疑需以截图复核)。
> **均未上机验证**。套用顺序建议:先 1.1b(HIP 宏,否则空壳)再 1.1a,之后按轮次 1→5 逐轮 A/B。
> 行号锚点基于本地 `vllm_cscc` 07-13 快照。

---

## 一、DCU 瘦 GEMV(LLMM1)与 Prefill 瓦片校准

### 1.1a utils.py 路由(终稿式)　截图:image 8
- 文件:`vllm_cscc/vllm/model_executor/layers/utils.py`
- 当前锚点:`rocm_unquantized_gemm_impl` L170-192(`use_skinny` 于 L170-175,wvSplitK/LLMM1 分支 L184-191)
- 终稿改动:
```python
# L170-175 改为:
use_skinny = (
    envs.VLLM_ROCM_USE_SKINNY_GEMM
    and (on_gfx9() or on_gfx936()) and rocm_skinny_ops_available()
    and x.dtype in [torch.float16, torch.bfloat16]
    and k % 8 == 0
)

# L184 之后新增(替代本地 C9b carve-out;终稿无 m<=20000 上限):
if on_gfx936() and n == 1 and m % 4 == 0 and k <= 8192 and bias is None:
    out = ops.LLMM1(weight, x_view, 4)  # 瘦 GEMV: k>8192 数值不可靠故封顶
    return out.reshape(*x.shape[:-1], weight.shape[0])
```
- 注意:需要 `rocm_skinny_ops_available()`(utils.py 同文件新函数,截图未给出函数体——推断为"平台已加载/可用 _rocm_C skinny ops"的探测,建议实现为 `import vllm._rocm_C` 成功与否,或检查 `hasattr(ops,'LLMM1')`;以截图为准复核)。
- 与本地差异:本地 C9b 是"wvSplitK 分支加 not(…m<=20000…)";终稿是 "on_gfx936() 显式 LLMM1 分支"(允许 m>20000 的 Qwen 大投影也走 LLMM1?截图条件无 m 上限——lm_head 248320 在 n==1/k<=8192 时也会命中;这与内部 C4-3(lm_head 不能动)经验冲突,**恢复期必须 A/B 该点**)。

### 1.1b HIP 侧编译宏(终稿式)　截图:image 5
- 文件:`vllm_cscc/csrc/rocm/skinny_gemms.cu`(L24-31 宏区;文中所指"HIP 头文件"即此宏区)
- 终稿改动:
```c
/* 源码 (baseline) */
#if defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)
  #define __HIP__GFX9__
#endif

/* 优化 */
#if defined(__gfx90a__) || ... || defined(__gfx936__) || ...   /* 纳入 DCU, LLMM1 可编译 */
  #define __HIP__GFX9__
#endif
#if defined(__gfx936__) || ...
  #define __HIP_DCU_GFX93X__ 1                                 /* 改 shfl 归约, 禁用不可靠 mov_dpp */
#endif
```
- `[?]`:截图/OCR 显示条件里除 `__gfx936__` 外还有第 4 个 arch(疑 `__gfx938__`),需以截图复核;`__HIP_DCU_GFX93X__` 生效处(哪些 kernel 换 shfl 归约、哪些禁 `mov_dpp`)截图未展示,需人工按注释补(建议:在 19 号文档 §1.5 的手写算子经验里"shfl 归约"可用、`mov_dpp` 不可靠的结论直接套用——wvSplitK 归约段与 LLMM1 的 `__shfl` 路径)。
- **必须最先套用**:本地态 DCU 构建下 wvSplitK 系为 `assert(false)` 空壳(见 00 §1.3)。

### 1.2 Prefill BLOCK=32　截图:image 4
- 文件:`vllm_cscc/vllm/v1/attention/ops/triton_prefill_attention.py`
- 当前锚点:`get_block_size` L180-188;调用 L210/L244-251
- 终稿改动(替换 `elif is_cuda_alike() and has_device_capability(80): return 128` 分支):
```python
if is_rocm() and head_size is not None and head_size >= 256:
    return 32  # 控 shared, 压 Prefill TTFT
```
- 说明:doc 称"仅 ROCm 且 hd≥256";Qwen3.5-27B FullAttn hd=256 → 命中。

### 1.3 分段 16→32　截图:image 19
- 文件:`vllm_cscc/vllm/v1/attention/backends/triton_attn.py`
- 当前锚点:L43 `NUM_PAR_SOFTMAX_SEGMENTS = 16`;缓冲分配 L171-190 随常量自动变
- 终稿改动:
```python
NUM_PAR_SOFTMAX_SEGMENTS = 32   # 提升长上下文 Decode 并行度
```
- ⚠️ 内部经验:32 段曾使 16-32K 恶化(内部 C5b 反向实验),但内部版无 2.1 的自适应配合;套用 1.3 必须与 2.1 一起评估。

### 1.4 KV evict_last　截图:image 9
- 文件:`vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- 改动:unified attention 内核内 K/V `tl.load`(K_ptrs/V_ptrs 处,约 L1000-1100 区间内核体中)加参数:
```python
tl.load(K_ptrs, mask=..., other=0.0, eviction_policy="evict_last")  # 保复用、减 HBM
tl.load(V_ptrs, mask=..., other=0.0, eviction_policy="evict_last")
```
- 当前本地:该文件 0 处 eviction_policy;要注意加到的是 **unified attention** 的 KV load(非 chunked_prefill_paged_decode 已有的)。

---

## 二、长上下文 Prefill/Decode 调优

### 2.1 自适应 Flash-Decoding　截图:image 7
- 文件:`vllm_cscc/vllm/v1/attention/backends/triton_attn.py`
- 当前锚点:L43 常量、L169 `self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS`
- 终稿改动:
```python
NUM_PAR_SOFTMAX_SEGMENTS = 32  # 长上下文并行上限

def _flash_decode_segments(max_seq_len, max_segments):
    if max_seq_len <= 2048:
        return min(16, max_segments)
    ideal = (max_seq_len + 255) // 256          # ~256 tok/段
    segs = 1 << max(ideal - 1, 0).bit_length()  # 2 次幂
    return max(16, min(max_segments, segs))
```
- 调用点:在 `build()` 中按 `max_seq_len` 调 `_flash_decode_segments(max_seq_len, NUM_PAR_SOFTMAX_SEGMENTS)` 覆盖 `self.num_par_softmax_segments`;**cudagraph 捕获时固定 grid Z**——即捕获路径仍用固定值(建议:capture 分支保持 32,非 capture 用自适应;与 doc 描述一致)。
- 内部对照:内部 C5a 因"与 cudagraph 冲突"判死路——终稿正是用"捕获固定/eager 自适应"化解,可直接采用。

### 2.2 非对称 Prefill 瓦片 + 软件流水　截图:image 20
- 文件:`vllm_cscc/vllm/v1/attention/ops/triton_prefill_attention.py`(主)、`triton_unified_attention.py`、`triton_reshape_and_cache_flash.py`(配套)
- 终稿改动(替换 1.2 的简单 BLOCK 取值,新增函数):
```python
# 源码（baseline）
BLOCK = get_block_size(dtype)  # 易误选 128

# 优化
def get_prefill_tiles(dtype, head_size, max_input_len):
    if is_rocm() and head_size >= 256:
        if max_input_len >= 8192:
            return 32, 16, 8, 3  # Bm,Bn,warps,stages: 非对称 + 深流水线 HBM
        return 32, 32, 8, 2
# Unified: Prefill TILE=16; 长 KV warps=8/stages=3
```
- 配套(文字级):unified attention prefill 侧 `TILE_SIZE_PREFILL=16`、`num_warps=8`、`num_stages=3`(仅长 KV),避免 TILE 与 stages 同时放大触发 shared OOM。
- `[?]`:`get_prefill_tiles` 与 1.2 的 `get_block_size` 并存关系(替换 or 包装)截图未明;建议:保留 1.2 逻辑作为不满足 is_rocm∧hd≥256 时的回退,新增函数在此之上分流。

---

## 三、GDN/FLA 定参与 BLOCK_M 二次幂修复

### 3.1 BLOCK_M 二次幂对齐　截图:image 16
- 文件:`vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- 当前锚点:L942-945
- 终稿改动:
```python
BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)

if is_rocm() and head_size >= 256 and max_seqlen_q > 1 and num_queries_per_kv > 1:
    BLOCK_M = triton.next_power_of_2(max(BLOCK_M, 32))  # 满足 2 次幂约束并摊销长 KV
BLOCK_Q = BLOCK_M // num_queries_per_kv
```
- 注:截图第 5 行 OCR 显示 `is_roc m()`(疑为 `is_rocm()`);`max_seqlen_q>1` vs `max_seqlen_k`(其它图用 `max_seqlen_k`)注意区分。Qwen GQA=6:BLOCK_M=32、BLOCK_Q=5(6 不整除 32 → 5.33?实际 `//` 语义按截图)。

### 3.2 FLA Chunk 定参　截图:image 10
- 文件(本地 vendored 路径;终稿文档写作 `fla/ops/…`):`vllm_cscc/vllm/model_executor/layers/fla/ops/chunk_o.py`、`chunk_delta_h.py`、`utils.py`;配套 `mamba/ops/causal_conv1d.py`、`vllm/v1/attention/backends/utils.py`
- 终稿改动(chunk_o.py 附近):
```python
# 源码（baseline）
BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]    # BKV=128 易 OOM
# stages ∈ [2,3,4]

# 优化
if is_rocm():
    BKV_LIST = [64]        # 禁 BKV=128
    NUM_WARPS = [4, 8]
    _NUM_STAGES = [2, 3]   # 去掉 stages=4
# FLA_GDN_FIX_BT → BT=64
```
- 注:`FLA_GDN_FIX_BT` env 本地已存在于 `fla/ops/utils.py:27`(fork 自带);终稿"BT=64"即该 env 生效(或项目里设环境变量)。
- `chunk_delta_h.py` 自动调优参数(`BV,warps=[2,4]`,`BT=chunk_size`)应同样收窄;`causal_conv1d`/`backends/utils.py` 的改动截图未给细节,以恢复时逐文件对截图复核。

---

## 四、大瓦片 Prefill FA 与 Decode GDN 融合

### 4.1 Prefill 大瓦片 FA(M=128,N=32)　截图:image 18
- 文件:`vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- 当前锚点:BLOCK_M 计算 L942-945、`_get_tile_size` L862-881(返回 TILE_SIZE_PREFILL)
- 终稿改动:
```python
use_large_qwen_prefill_tile = (
    is_rocm() and head_size == 256 and GQA == 6
    and max_seqlen_q > 1 and max_seqlen_k >= 4096  # 仅长 Prefill
)
BLOCK_M = next_power_of_2(max(BLOCK_M, 128 if use_large_qwen_prefill_tile else 32))
if use_large_qwen_prefill_tile:
    TILE_SIZE_PREFILL = 32
    num_warps, num_stages = 8, 1  # stages=1 控大 M×N 的 LDS/VGPR
```
- 注:GQA 变量名以本地上下文为准(本地为 `num_queries_per_kv`);`max_seqlen_k` 变量本地是否存在按 3.1 修订统一。

### 4.2 GDN Decode 值维瓦片(BV=128,warps=4)　截图:image 21
- 文件:`vllm_cscc/vllm/model_executor/layers/fla/ops/fused_recurrent.py`
- 当前锚点:decode 入口 `fused_recurrent_gated_delta_rule_packed_decode` L338-477,其中 L436-438
- 终稿改动:
```python
# 源码（baseline）
BV = min(next_power_of_2(V), 32)  # -> NV=4
num_warps = 1

# 优化
is_qwen35_decode = HV == 48 and V == 128 and K == 128
BV = 128 if is_qwen35_decode else min(next_power_of_2(V), 32)
num_warps = 4 if is_qwen35_decode else 1  # 掩盖 HBM
```
- 注:全 T 循环版(L195/L198)是否同步改,截图未明;建议两处一致(该特化只对 HV=48 生效,不影响其它模型)。

### 4.3 Decode in_proj 融合 + hipBLASLt 旁路　截图:image 12
- 文件:`vllm_cscc/vllm/model_executor/models/qwen3_next.py`(与 `qwen3_5.py` 同构)
- 当前锚点:forward L650-651(两次独立投影)
- 终稿改动:
```python
# 源码（baseline）
qkvz, _ = self.in_proj_qkvz(h)
ba, _ = self.in_proj_ba(h)

# 优化
if self.fused_in_proj_weight is not None and num_tokens == 1:
    fused = rocm_unquantized_gemm(h, self.fused_in_proj_weight, None)  # 单次 GEMV
    qkvz, ba = split(fused)
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"  # M=1 避开 hipBLASLt 劣化
```
- ⚠️ 变量名:x 两张图(4.3/5.2)分别出现 `fused_in_proj_weight` 与 `_fused_in_proj_weight`,以 5.2 截图的 `_fused_in_proj_weight` 为准(私有属性惯例)。`split` 的切分点= qkvz 输出维(2×(QK16+V48)×128=16384?实际以 qkvz 权重行数为准 — 3/5 与 2/5 分法见 5.2)。
- `os.environ` 建议放进程级(env_override.py 或 launch 脚本),避免 per-call 赋值;终稿截图即写在此处,恢复时按截图。

---

## 五、FA TILE_N 扩维与跨阶段 in_proj 融合

### 5.1 TILE_N 32→64(env 可配)　截图:image 30
- 文件:`vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- 终稿改动:
```python
# 优化前
TILE_SIZE_PREFILL = 32

# 优化
tile_n = int(os.getenv("VLLM_ROCM_FA_PREFILL_TILE", "64"))  # 默认 64; 可选 16/32
TILE_SIZE_PREFILL = tile_n if tile_n in (16, 32, 64) else 64
num_warps, num_stages = 8, 1  # 大 N 保持 stages=1, 控 LDS
```
- 与 4.1 的关系:4.1 设 32;5.1 在此基础上默认升 64(可 env 回落)。4.1 与 5.1 是同一段代码的两次演进,恢复时**直接取 5.1 终态**,除非 A/B 需要对比。

### 5.2 跨阶段 in_proj 融合(权重拼接)　截图:image.png
- 文件:`vllm_cscc/vllm/model_executor/models/qwen3_next.py` / `qwen3_5.py`
- 当前锚点:`load_weights`(qwen3_5 L385-505;qwen3_next 顶部 L1564-1565 packed 映射)与 forward(L650-651/qwen3_5 L183-189)
- 终稿改动:
```python
# load_weights 之后:把 qkvz 与 ba 两套权在输出维拼接(不落盘、不改数值):
#   self._fused_in_proj_weight = torch.cat([in_proj_qkvz.weight, in_proj_ba.weight], dim=0)

# forward:
if self._fused_in_proj_weight is not None:   # Prefill + Decode
    fused = rocm_unquantized_gemm(h, self._fused_in_proj_weight, None)
    qkvz, ba = fused[..., :split], fused[..., split:]
    if num_tokens == 1:
        qkvz, ba = qkvz.contiguous(), ba.contiguous()  # Decode 保 stride
    # Prefill: 直接用 view, 避免 T×out 二次 HBM 写
```
- 数学:`Y = X [Wqkvz; Wba]` 与分乘再拼等价;`split`= qkvz 输出维(`key_dim*2+value_dim*2` = 16384 对 GDN 层;具体以 `self.in_proj_qkvz.output_size` 为准)。
- 注:qkvz/ba 均为 `MergedColumnParallelLinear(bias=False)`;按维拼接后单次 GEMM 的 k=5120 不变、m(输出)=16384+96。**注意**:decode 分支该 GEMM 的 n=1 → `rocm_unquantized_gemm` 会走 LLMM1(on_gfx936 路由),需与 1.1a 的 LLMM1 条件匹配(16384≤任意上限;ba=96 也命中 n==1);prefill 分支 n>1 → F.linear/hipBLASLt,`TORCH_BLAS_PREFER_HIPBLASLT=0` 保证不走 hipBLASLt(见 4.3)——即 5.2 的有效前提是 1.1a+4.3 已落地。

### 5.3 融合 Chunk 预处理　截图:image 1
- 文件(新增/修改):fla(`fused_chunk_preprocessing.py` **新文件**、`chunk.py` 改入口)、`env_override.py`(env 门控)
- 终稿:
```python
# 源码（baseline）：多 kernel + 中间 A 写 HBM
# 优化：模块入库；默认关闭，按需门控开启
os.environ.setdefault("VLLM_ROCM_GDN_FUSED_PREPROC", "0")
```
- 内容:cumsum⊕KKT⊕tril⊕WU 融合为单 kernel(概念参考:delta rule chunk 预处理四个算子:累加 cumsum、K^T K、tril 掩码、WU 更新);默认 0 = 落地但默认不启用,故**对默认跑分无贡献**,是"备用优化"入场券。恢复价值最低,放最后。
- `[?]`:具体 kernel 实现截图未含(只有门控行),需按 POV 概念实现或跳过(默认 0,不影响结果)。

---

## 附:本表与本地"已存在改动"的关系

| 本地已有 | 与终稿关系 |
|---|---|
| `rocm.py:149` gfx936(C1) | 终稿 1.1a 的 `on_gfx936()` 谓词**新增**(本地无 `on_gfx936` 函数!);本地只用 `_ON_GFX9` 列表。需按 1.1a 同时补 `on_gfx936()` 谓词与 `rocm_skinny_ops_available()` |
| `utils.py` C9b(m≤20000 carve-out) | 终稿移除该上限、改显式 `on_gfx936()` LLMM1 分支 → **删除 C9b 注释与条件**,按 1.1a 替换 |
| `version.py` hack | 与性能无关,但建议恢复 upstream 版(避免自引用 import 瑕疵) |
| `csrc` 无 gfx936 | 必须按 1.1b 改,否则 wvSplitK 空壳/assert |
