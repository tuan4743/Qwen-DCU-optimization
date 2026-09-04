# 10 · P2-decode GDN GEMM 设计优化调研清单

> **阶段:仅设计/调研**(不改源码、不进容器、不实测)。
> 分工:本窗口走 **GDN GEMM** 轨道;另一窗口走 CPU/调度轨道(`09`)。
> 关联:`03_profile_findings.md`(瓶颈特征)、`05_task_tracker.md`(热追踪 §P2.3)、`09_cpu_sched_overhead_design.md`(§0.5 duty cycle 把 tpot 瓶颈归到本轨道)、`01_constraints_env.md`(算子基线/锁定约束)、`02_model_arch.md`(模型架构)、`06_pitfalls.md`(操作要点)。

---

## 0. 问题陈述(duty cycle 修正后,本轨道是唯一能降 tpot 的轨道)

> 本节结论由 `09` §0.5 duty cycle 钉死传递而来,**非本轨道独立发现**,但决定本轨道的定位。

- **duty cycle 钉死**(批3 cudagraph ON trace,`tools/_duty_cycle.py`):GPU duty = **97.3%**(busy 7800ms / span 8018ms),idle 仅 2.5%。GPU 全程满载,**没有 step 间空闲**。
- 窗口 118 decode token(`kernel_unified_attention_3d` count=1888 ÷ 16 FullAttn 层)= **67.95 ms/token ≈ baseline mean_tpot 69.8ms**,完美吻合。
- **端到端 tpot 瓶颈 = step 内部 64 层 GPU kernel 串行**,不在 step 之间(`09` §0.5 推翻"step 间 ~70× 开销"框架)。
- **64 层里 GDN/FLA 占 95.17%**(批3,三批最高),绝对主体是 `Cijk_..._GSU1` 系列 ROCm GEMM。**这是唯一能降 tpot 的地方。**
- baseline = cudagraph ON,12.20 tok/s,mean_tpot=69.8ms。本轨道目标:把 64 层 GDN GEMM 串行总耗时压下来 → 直接降 tpot。

---

## 0.5 GDN GEMM 到底是哪几个 GEMM?(本轮源码 + torch profiler 实测钉死)

> 这是本轨道的核心定位。结论分两层:
> - **后端分发(本轮源码钉死)**:`rocm_unquantized_gemm_impl` 的 `use_skinny` 条件现在是 **`(on_gfx9() or on_gfx936())`**(见 §0.5.3 修正),gfx936 **命中**。三个 GDN 投影 `bias=False` 且形状满足 `n==1 and m%4==0 and k<=8192 and bias is None` → **`in_proj_qkvz` / `in_proj_ba` 走 `LLMM1` 自定义核**(127us / 量级),`out_proj`(k=6144>8192? 实测 k=6144 满足 k<=8192,但 m=5120,n=1)同样命中 LLMM1 分支。
> - **trace 实跑(本轮 torch profiler 钉死)**:用 `dispatch_unquantized_gemm()` 在 qkvz 形状上抓 profiler,实际命中的是 `void LLGemm1_kernel<c10::BFloat16, 4>`(~127us),**不是** `MT64x32x32_GSU1`。`MT64x32x32_GSU1`(490us / call,median)是**别的 GEMM**(见 §8.4 重新归因),而非 qkvz。
>
> ⚠️ **本文档 §0.5.3 旧稿、§0.5.4 候选 D、§5.0.0、§8.4–§8.10 中"qkvz 走 hipBLASLt 638.9us/220us"的整套结论已被本轮源码 + 实测推翻**(2026-07-12 修订,见 §0.5.3 修正块与 §8.11)。下文保留旧推导作历史记录,凡与之冲突处以 §0.5.3 修正块、§8.11 为准。

### 0.5.1 GDN 层结构(`qwen3_next.py:Qwen3NextGatedDeltaNet`)

每层 GDN(`layer_type="linear_attention"`)forward 三部分(`forward()`@634-690):

1. **Input Projection(投影 GEMM)**:
   - `in_proj_qkvz = MergedColumnParallelLinear(hidden=5120 → key_dim*2+value_dim*2)`@447/534。output_size = `sum((key_dim, key_dim, value_dim, value_dim))`。
   - `in_proj_ba = MergedColumnParallelLinear(hidden=5120 → num_v_heads*2)`@457/556。
   - `out_proj = RowParallelLinear(value_dim → 5120)`@507。
   - 这些是 **vLLM Linear 层**,经 `UnquantizedLinearMethod.apply`(`linear.py:228`)→ `dispatch_unquantized_gemm()`(`utils.py:302`)→ DCU 走 `rocm_unquantized_gemm`(`utils.py:197`)→ `rocm_unquantized_gemm_impl`(`utils.py:122`)。
2. **Core Attention**(`gdn_attention_core` custom op):decode 走 `_forward_core_decode_non_spec`(`qwen3_next.py:1005-1052`)= `causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`。
3. **Output Projection**:`norm(core_attn_out, z)`(`RMSNormGated`)+ `out_proj(core_attn_out)`。

### 0.5.2 递归核无 GEMM(钉死)

`fused_recurrent_gated_delta_rule_packed_decode_kernel`(`fla/ops/fused_recurrent.py:256-336`)核心循环逐元素:
```
b_h *= tl.exp(b_g); b_v -= tl.sum(b_h*b_k,1); b_v *= b_beta
b_h += b_v[:,None]*b_k[None,:]; b_o = tl.sum(b_h*b_q[None,:],1)
```
**全是 elementwise / outer / reduce,无 `tl.dot` GEMM。** batch1 1×128×128 的小 head 也走逐元素递归,不调 GEMM。`causal_conv1d_update` 是 conv1d 短核(kernel_dim=4),也无 GEMM。

→ **trace 里 95.17% 的 GDN/FLA 占比、`Cijk_..._GSU1` 主力,绝不来自递归核/conv,必来自投影 GEMM。**

### 0.5.3 投影 GEMM 实际走哪个后端?(关键)

> ## 🔧 修正块(2026-07-12,本轮源码 + torch profiler 实测推翻旧稿)
>
> 旧稿(下方被划线的推导)断言:`on_gfx9()` 列表硬编码 `["gfx90a","gfx942","gfx950"]` 不含 gfx936 → `use_skinny` 恒 False → 投影 GEMM 恒走 `torch.nn.functional.linear` → **hipBLASLt**。**此结论已被推翻**。
>
> **本轮在 worker-0 容器(`/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/utils.py`,site-packages 实跑版本,带 `[DCU Optimize]` 标记)实测的源码事实**:
>
> ```python
> # utils.py:179-203 (site-packages 实跑版本)
> use_skinny = (
>     envs.VLLM_ROCM_USE_SKINNY_GEMM            # 默认 True (envs.py:115)
>     and (on_gfx9() or on_gfx936())            # ← gfx936 现在命中!
>     and rocm_skinny_ops_available()           # hasattr(torch.ops._rocm_C, "wvSplitK") = True
>     and x.dtype in [torch.float16, torch.bfloat16]
>     and k % 8 == 0
> )
> if use_skinny is not True:
>     return torch.nn.functional.linear(x, weight, bias)
> x_view = x.reshape(-1, x.size(-1))
> try:
>     if on_gfx936():
>         if n == 1 and m % 4 == 0 and k <= 8192 and bias is None:
>             out = ops.LLMM1(weight, x_view, 4)          # ← qkvz/ba 命中这里
>             return out.reshape(*x.shape[:-1], weight.shape[0])
>         return torch.nn.functional.linear(x, weight, bias)
> ```
>
> **关键差异**:site-packages 实跑版本的 `use_skinny` 条件是 **`(on_gfx9() or on_gfx936())`**,且 `on_gfx936()` 在容器里返回 **True**(`platforms/rocm.py` 里有 `on_gfx936()` 函数,基于 `_GCN_ARCH='gfx936'` 命中)。旧稿只看到 `on_gfx9()` 不含 gfx936,**漏看了 `or on_gfx936()` 这一项** —— 这是容器版本相对旧源码(`vllm_cscc`,无 `[DCU Optimize]`)的增量修改,旧稿据旧源码得出错误结论。
>
> **三个 GDN 投影逐个核对命中条件**(全部 `bias=False` → `bias is None` 成立,见 §0.5.3 旧稿 bias 钉死段,该段结论仍有效):
> - `in_proj_qkvz`:n=1,m=16384,m%4==0,k=5120≤8192 → ✅ **命中 `ops.LLMM1`**(LLMM1 自定义核)
> - `in_proj_ba`:n=1,m=96,m%4==0,k=5120≤8192 → ✅ **命中 `ops.LLMM1`**
> - `out_proj`:n=1,m=5120,m%4==0,k=6144≤8192 → ✅ **命中 `ops.LLMM1`**
>
> **torch profiler 实跑验证**(2026-07-12,worker-0 容器,用 `dispatch_unquantized_gemm()` 在 qkvz 形状 `(m=1,n=16384,k=5120,bf16)` 上抓 profiler,key_averages 设备耗时):
> ```
> 380.0us x3  void LLGemm1_kernel<c10::BFloat16, 4>(...)
> ```
> 三 call 平均 127us,命中的核是 **`LLGemm1_kernel`**,**不是** `MT64x32x32_GSU1`,**不是** hipBLASLt 任何 algo。三方对照基准(同形状,warmup 后):
> | 调用路径 | 耗时 |
> |---|---|
> | `rocm_unquantized_gemm`(=LLMM1,实跑路径) | **127.7us** |
> | `F.linear`(hipBLASLt) | 229.2us |
> | `F.linear`(rocBLAS/cublas) | 230.3us |
> | `matmul(x, wt.contiguous())` | 124.4us |
>
> **修正后的核心结论**:
> 1. qkvz/ba/out_proj 三个 GDN 投影 GEMM **实际全部走 `LLMM1` 自定义核**(127us 量级),而非旧稿断言的 hipBLASLt。
> 2. 旧稿据 `hipblaslt-bench` 得出的 "qkvz 走 hipBLASLt index 4362 = 638.9us"(§8.4.5)、"vllm 实跑 qkvz 走 MT64x32x32_GSU1 ~220us"(§8.10)——**两条都被推翻**:vllm 根本没把 qkvz 喂给 hipBLASLt,bench 里的 638.9us 和 trace 里的 220us 都不是 qkvz 的实跑耗时。qkvz 实跑 = LLMM1 127us。
> 3. **切 rocBLAS / 切 hipBLASLt 对 qkvz 无意义** —— qkvz 已在 LLMM1(127us)上,这是该形状在 site-packages `rocm_unquantized_gemm` 分发下的实际路径;唯一略快的是 `matmul(x, wt[k,n])` 124us,仅 3us 增益,不值得改源码换布局。**任务 #13 的前提(qkvz 在 hipBLASLt 上,切 rocBLAS 有 ~1.5x 收益)被推翻,该任务作废。**
> 4. **`MT64x32x32_GSU1`(median 490us)不是 qkvz,是别的 GEMM**,需重新 op 归因(见 §8.11)。这才是当前真瓶颈。

---

~~`rocm_unquantized_gemm_impl`(`utils.py:122-188`)分发顺序~~(旧稿,已被上方修正块推翻):

> **✅ Python 分发层钉死(本轮源码闭合,最直接证据)**:`on_gfx9()`(`platforms/rocm.py:233`)返回 `_ON_GFX9`,而 `_ON_GFX9` 定义(rocm.py:147)是:
> ```python
> _ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
> ```
> 列表**硬编码 `["gfx90a","gfx942","gfx950"]`,不含 gfx936**。`_GCN_ARCH` 由 `_get_gcn_arch()`(rocm.py:122)经 amdsmi `amdsmi_get_gpu_asic_info` 的 `target_graphics_version` 字段取得 —— 本机 amdsmi 实测返回 `target_graphics_version='gfx936'`(market_name=BW3000,vendor=成都 C-3000,80 CUs)。→ **`_GCN_ARCH='gfx936'` ∉ 列表 → `_ON_GFX9=False` → `on_gfx9()=False`**。
> 回到 `utils.py:170` 的 `use_skinny = (envs.VLLM_ROCM_USE_SKINNY_GEMM and on_gfx9() and x.dtype in [fp16,bf16] and k%8==0)` —— **`and on_gfx9()` 短路为 False** → `use_skinny is not True` → `return torch.nn.functional.linear(x, weight, bias)`(@188)→ **hipBLASLt**。
> **这是纯源码逻辑必然**:无论 `VLLM_ROCM_USE_SKINNY_GEMM` env 设成什么(默认 True 也好),`use_skinny` 在 gfx936 上恒 False;无论 `n≤4`/`n==1` 门槛是否满足,都到不了 wvSplitK/LLMM1 分支(被前置的 `and on_gfx9()` 挡死)。**与 n 值无关,与 env 无关,与 capture 后 batch 是否增大无关** —— 投影 GEMM 在 gfx936 上恒走 hipBLASLt 回退。此条与 §0.5.4 的 C++ 编译宏空壳(C++ 层第二证据)、CMakeCache build 目标(build 层第三证据)三重闭合,候选 D 钉死。

**判定**:
- 投影 GEMM 的 `weight.shape = (m, k)`,decode `n = batch(=1) × num_tokens`。
  - `in_proj_qkvz`:m = key_dim*2+value_dim*2 = (16*128)*2 + (48*128)*2 = 4096 + 12288 = **16384**;k = 5120。
  - `in_proj_ba`:m = num_v_heads*2 = 96;k = 5120。
  - `out_proj`:m = 5120;k = value_dim = 6144。
- decode batch=1:三者的 `n` 都 ≤ 解码 token 数(单请求稳态 ~1)。`n≤4` 命中 `wvSplitK`,`n==1` 命中 `LLMM1`(`VLLM_ROCM_USE_SKINNY_GEMM=True` 默认开,`envs.py:115`)。
- **bias 实际传参钉死(本轮源码补)**:`UnquantizedLinearMethod.apply`(linear.py:220-228)签名 `apply(layer, x, bias=None)`,body 调 `dispatch_unquantized_gemm()(layer, x, layer.weight, bias)`。bias 来自 `ColumnParallelLinear.forward`@578:`bias = self.bias if not skip_bias_add else None`。GDN 三投影层创建时显式 `bias=False`(qwen3_next.py:537 `in_proj_qkvz` / 559 `in_proj_ba` / 510 `out_proj`)→ `self.bias=None` → 传到 apply 的 **`bias=None` 恒成立**。TP=1(`tp_size=1` → `tp_rank=0`)下 `RowParallelLinear.forward`@1514 `bias_ = None if (tp_rank>0 or skip_bias_add) else self.bias` = `self.bias = None`。→ **三个投影 GEMM 的 `bias is None` 条件恒 True**,LLMM1 的 `bias is None` 门槛不构成回退触发。`skip_bias_add` 默认 False,但因 `bias=None`,无论 skip 与否 apply 都拿 None。
- **`MergedColumnParallelLinear` 不拆多次 GEMM(本轮源码补,解释 ÷118=112 倍数)**:`MergedColumnParallelLinear.__init__`(linear.py:644-651)`self.output_size = sum(output_sizes)`,权重是**单块** `(sum(output_sizes), input_size)`,forward 经父类 `ColumnParallelLinear.forward`@574-582 → 单次 `quant_method.apply` → **单个 GEMM**。`in_proj_qkvz` 的 `output_sizes=[16384]`(单元素 list)尤其明显是单 GEMM;`in_proj_ba` 的 `output_sizes=[96]` 同理。**÷118 token 不是单层多个 GEMM,而是 48 GDN 层 × 每 token 多类投影(qkvz+ba+out+conv1d+norm 内部小 GEMM)× 不同 tile config 的聚合**——`MT64x32x32_GSU1` count 13216 ÷ 118 ≈ 112 是"每 token 触发 ~112 次"(跨 48 层多个 GEMM 累加),非单层拆分。
- **`wvSplitK`/`LLMM1` 是 vLLM 自带 ROCm skinny GEMM 自定义核**(`_custom_ops.py:2186/2190`,调 `torch.ops._rocm_C.wvSplitK/LLMM1`),**不是 hipBLASLt,也不是 Triton `gemm_a16w16`。**

> ⚠️ **重要矛盾点(本轨道最大悬念,本轮源码修正)**:`05` §5.1 第 1 条旧判断"FFN GEMM 实际走的不是 hipBLAS(需 profile 确认)"悬置项,**本次源码调研 + skinny 源码分析后倾向闭合为"decode 投影 GEMM 走 hipBLASLt/rocBLAS 回退(候选 D),而非 skinny `LLMM1`/`wvSplitK`"**。核心理由:(1) trace 主力 kernel 名 `Cijk_..._GSU1` 是 hipBLASLt/rocBLAS/MIOpen 的 contraction tile 命名约定,**与 skinny kernel 源码命名(`LLGemm1_kernel`/`wvSplitK_hf_*`)完全不符**;(2) `csrc/rocm/skinny_gemms.cu` 编译宏 `__HIP__GFX9__`(gfx90a/gfx942/gfx950)**不含 gfx936** → wheel 若按 gfx936 编译,skinny kernel 体空壳;(3) cudagraph ON capture 后 batch 增大使 `n>4` 越过 skinny `n≤4`/`n==1` 边界 → 回退 `torch.nn.functional.linear`→hipBLASLt(批3 `MT64x32x32` 单 call 383.9us ≫ eager 162.6us 支持 n 增大)。**待落地钉死(§5 第1/2/3条),但候选 D 概率最高。** 详见 §0.5.4。

### 0.5.4 `Cijk_..._GSU1` 究竟是谁(待落地确认,最高优先级)

源码里搜不到 `Cijk` 或 `GSU1` 字样(本次 grep 未命中 vLLM 源码)→ 这是 **HIP/rocBLAS/MIOpen/aiter 在 DCU 上的底层 kernel 名**,非 Python 层可见。三种可能:

| 候选 | 证据 | 概率 |
|---|---|---|
| **A. `wvSplitK`/`LLMM1` 的底层 HIP kernel** | ~~decode 投影 GEMM 经 `rocm_unquantized_gemm_impl` 落到这两个自定义核~~ **三重证伪(已钉死)**:(1) **Python 层** `on_gfx9()`=`False`(`rocm.py:147` 列表不含 gfx936)→ `use_skinny` 短路 False → **根本不调 `ops.wvSplitK/LLMM1`**;(2) `skinny_gemms.cu` kernel 名 `LLGemm1_kernel`/`wvSplitK_hf_*` 不含 `Cijk`/`GSU`/`MT*`;(3) build 按 gfx936 编译 → `__HIP__GFX9__` 空壳。`Cijk_*_GSU1` 是 rocBLAS/hipBLASLt/MIOpen 的 contraction tile 命名约定(`Cijk`=contraction index i,j,k;`MT64x32x32`=macro tile;`GSU`=gated sigmoid unit tile),**与 skinny kernel 命名风格不符**。 | **证伪(钉死)** |
| **B. aiter `gemm_a16w16` 的 Triton kernel 误判** | `use_aiter_triton_gemm` 白名单不命中 decode 形状,排除 | 低 |
| **C. GDN/FLA chunk 路径(prefill)残留** | decode 走 packed recurrent 非 chunk;但 `MT64x32x32_GSU1` 单 call 383.9us 偏大,可能含 prefill 混入 | 中 |
| **D. hipBLASLt/rocBLAS 经 `torch.nn.functional.linear` 回退(新增,最可能)** | `Cijk`+`MT*`+`GSU*` 命名是 hipBLASLt/rocBLAS 风格;**cudagraph capture 后 decode batch 被 pad 增大**(批3 `MT64x32x32` 单 call **383.9us ≫ eager 批1 的 162.6us**,反映 capture 后实际 n 增大)→ 若 capture 后 `n > 4` 越过 skinny `n≤4`/`n==1` 边界 → `rocm_unquantized_gemm_impl` 落到兜底 `torch.nn.functional.linear`@188 → **hipBLASLt/rocBLAS**。 | **最可能(高)** |

**本轮 skinny 源码钉死的关键反证(`csrc/rocm/skinny_gemms.cu` 2152 行)**:
- 定义 `LLGemm1_kernel`(L145,N=1 matvec)、`wvSplitK_hf_sml_/wvSplitK_hf_/wvSplitK_hf_big_`(L307/538/766)、`wvSplitKrc_`(L1227,仅 gfx950)、`wvSplitKQ_*`(FP8,仅 MI3XX)。
- **所有 kernel 名均无 `Cijk`/`GSU`/`MT64x32x32` 字样** → 若 decode 走 skinny,trace 该出现 `LLGemm1_kernel`/`wvSplitK_*` 名,而非 `Cijk_*_GSU1`。
- **编译宏关键**:`#if defined(__HIP__GFX9__)`(L24-27)仅 `gfx90a`/`gfx942`/`gfx950` 命中,**`gfx936` 不在列表**;`__HIP__MI3XX__`(L29)仅 `gfx942`/`gfx950`。→ **若 wheel 按 gfx936 编译,`wvSplitK`/`LLMM1` 的 kernel 体编译成 `UNREACHABLE_CODE` 空壳**。
- **✅ 本轮本地 build 钉死(决定性,坐实候选 D,§5 第2条闭合)**:查 `vllm_cscc/build/temp.linux-x86_64-cpython-310/CMakeCache.txt`:
  ```
  AMDGPU_TARGETS:STRING=gfx906;gfx926;gfx928;gfx936;gfx938
  CMAKE_HIP_ARCHITECTURES:STRING=gfx906;gfx926;gfx928;gfx936;gfx938
  GPU_TARGETS:STRING=gfx906;gfx926;gfx928;gfx936;gfx938
  ```
  → **wheel 编译目标含 `gfx936`,但不含 `gfx942`/`gfx950`**。HIP 编译器按列表逐 arch 生成 kernel 二进制;gfx936 这份编译时 `__gfx90a__`/`__gfx942__`/`__gfx950__` 三宏**均未定义**(HIP 仅在编译对应 arch 时定义其专属宏)→ `__HIP__GFX9__`(skinny_gemms.cu:24-27)**不被定义** → skinny kernel 体(`LLGemm1_kernel`/`wvSplitK_hf_*`)在 gfx936 二进制里被 `#if __HIP__GFX9__` 守卫**整段编译掉/空壳**。运行时 DCU=gfx936 加载 gfx936 那份 → **skinny kernel 实际不可用**。→ **候选 D 钉死:投影 GEMM 必走 hipBLASLt 回退,与 n 值无关(§5 第3条免验)**。
- **~~编译时 vs 运行时矛盾~~(本轮 Python 分发层闭合后作废,见 §0.5.3 钉死块)**:旧版本曾推测"Python 端 `on_gfx9()` 运行时 gfx936 命中 True → `use_skinny=True` → 调 `ops.wvSplitK/LLMM1`,但 C++ 端 kernel 体空壳 → fallthrough"。**本轮源码核查证伪**:`on_gfx9()` 返回的 `_ON_GFX9`(rocm.py:147)列表硬编码 `["gfx90a","gfx942","gfx950"]`,**不含 gfx936** → `on_gfx9()=False` → `use_skinny` 在 Python 层就被 `and on_gfx9()` 短路为 False,**根本到不了 `ops.wvSplitK/LLMM1` 调用**。即:Python 分发层(C++ 层、build 层)三处独立证据**一致指向 gfx936 下 skinny 路径全链路不可达**,而非"调用了但 kernel 空壳"。`on_gfx9()=False` 是最直接的——投影 GEMM 在 `utils.py:188` 兜底 `torch.nn.functional.linear` → hipBLASLt,候选 D 三重闭合。

**结论性定位(给落地用,本轮 build 钉死后)**:decode 稳态下 `Cijk_..._GSU1` 主力 = **投影 GEMM(`in_proj_qkvz` 为主,m=16384 最大)经 hipBLASLt/rocBLAS 跑出的 ROCm contraction kernel**(候选 D **已钉死**)。回退机制 = **wheel 按 gfx936 编译 → skinny kernel 体 `#if __HIP__GFX9__` 空壳 → `wvSplitK`/`LLMM1` 调用无效 → fallthrough 到 `torch.nn.functional.linear` → hipBLASLt**(**与 n 值无关,无需 capture 后 n>4 的假设**)。skinny `wvSplitK`/`LLMM1`(候选 A)**双证伪**:命名不符 + 编译宏空壳。`in_proj_qkvz`(m=16384, 占 GDN 68% 的 `MT64x32x32_GSU1` 5.074s)仍是绝对第一大头。

> ⚠️ **本结论性定位已被 §0.5.3 修正块(2026-07-12)推翻**。`CMakeCache.txt` 的 `__HIP__GFX9__` 空壳论证针对的是**旧源码 `vllm_cscc`**(仅 `on_gfx9()`),但容器实跑的是 **site-packages `[DCU Optimize]` 版本**,其 `use_skinny` 条件为 `(on_gfx9() or on_gfx936())`,gfx936 命中,且 `on_gfx936()` 分支用的是独立的 `if on_gfx936():` 块(不受 `__HIP__GFX9__` 宏守卫的旧 skinny kernel 限制 —— site-packages 版本的 LLMM1 在 gfx936 上是可用的,torch profiler 实测命中 `LLGemm1_kernel` 127us 即铁证)。**"投影 GEMM 走 hipBLASLt" 的候选 D 三重闭合全部作废。** qkvz/ba/out_proj 实走 LLMM1。`MT64x32x32_GSU1` 是别的 GEMM,待 §8.11 重新归因。

---

## 1. 锁定约束复核(来自 `01`/`09`)

| 类别 | 是否锁定 | 对本轨道的影响 |
|---|---|---|
| 模型权重/结构/剪枝/量化边界 | **锁定** | ❌ 不能改 GDN 层数(64)/投影形状/换 checkpoint |
| `--max-num-seqs`/`--max-num-batched-tokens`/`--max-model-len` | **锁定** | ❌ decode batch 形状固定,不能靠加 batch 摊薄 GEMM |
| Decode 阶段算子优化 / 修改 vLLM 框架代码 | **允许**(`01` §1.1) | ✅ 可改投影层 GEMM 分发、加 fused kernel |
| **非持久化、算子级低精度**(激活动态量化/KV 运行时量化/kernel 内临时类型转换/低精度 matmul) | **允许**(`01` §1.1 显式列) | ✅ 投影 GEMM 可走算子级 BF16→FP8 动态量化(非权重量化边界) |
| `--enforce-eager`/`compilation_config`/`cudagraph_mode`/`custom_ops`/`pass_config` | **未锁定**(`01` §1.1 注) | ✅ 可调 `custom_ops` 开关切 GEMM 后端;`06` 已验证 enforce-eager 对比 |
| `VLLM_ROCM_USE_SKINNY_GEMM` | **未锁定**(envs 运行参数) | ✅ 可关,强制投影 GEMM 走 `torch.linear`→hipBLASLt |
| 投机解码 | **禁止** | ❌ 不能 spec-decode 降 tpot |
| DeepGEMM | 稠密 GEMM 上 280T < hipBLAS 403T(`01` §1.3) | ❌ 不作主线,但 FP8 不可用(segfault) |

**边界结论**:可动 = GEMM 后端分发 + 算子级低精度 + fused kernel + custom_ops 开关;不可动 = 层数/形状/batch/权重。

---

## 2. kernel 现状分析(批3 cudagraph ON,`05` §P2.3)

### 2.1 占比

| 类别 | 批3 dur(s) | 占比 |
|---|---|---|
| **GDN/FLA** | 7.423 | **95.17%** |
| FFN_GEMM | 0.117 | 1.50% |
| FullAttn | 0.020 | 0.26% |
| LayerNorm/Elementwise/Memset/Sampling/Other | ~0.24 | ~3% |

### 2.2 TOP kernel(GDN/FLA 内,按聚合 dur 降序)

| kernel | dur(s) | count | per-call | ÷118 token | 定位 |
|---|---|---|---|---|---|
| `Cijk_..._MT64x32x32_..._GSU1` | **5.074** | 13216 | 383.9us | 112 | **投影 GEMM 主力(占 GDN 68%),大概率 = `in_proj_qkvz` (m=16384)** |
| `Cijk_..._MT32x16x4_..._GSU1_` | 1.717 | 15222 | 112.8us | 129 | 投影 GEMM 次主力(频次最高) |
| `Cijk_..._MT128x32x32_..._GSU` | 0.422 | 1888 | 223.7us | 16 | 大 tile config |
| `Cijk_..._MT32x32x32_..._GSU8` | 0.089 | 5664 | 15.6us | 48 | 小 GEMM |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 0.075 | 5664 | 13.2us | 48 | **递归核(无 GEMM,48 GDN 层各1)** |
| `Cijk_B_PostGSU` | 0.037 | 7552 | 4.9us | 64 | GSU 融合后处理(64 层各1) |
| `kernel_unified_attention_3d` | 0.020 | 1888 | 10.6us | 16 | FullAttn 唯一核(16 层) |

- **`MT64x32x32_GSU1` 占 GDN 68%(5.074s)** = 单点最大优化标的。
- `÷118 token` 列证明这些都是**每层每 token 触发**(层倍数),非每 token 一次 → 优化单次 kernel 直接乘 64 层 × 118 token。
- 递归核 `fused_recurrent` 仅 0.075s(1%),**优化它收益微乎其微**,排除。

### 2.3 GEMM 后端现状(`0.5` 钉死,本轮源码修正)

> ## 🔧 修正(2026-07-12,torch profiler 实测推翻旧稿)
>
> 旧稿(下方)的"`Cijk_*_GSU1` 命名与 skinny 不符 → 大概率走 hipBLASLt"判断**已被推翻**。site-packages `[DCU Optimize]` 版本 `use_skinny = (on_gfx9() or on_gfx936())`,gfx936 命中;torch profiler 在 qkvz 形状上实测命中 `LLGemm1_kernel` 127us。**三个投影 GEMM 实走 LLMM1**,不进 hipBLASLt。下方"大概率实际走 hipBLASLt/rocBLAS(候选 D)"的结论作废,保留作历史记录。

- decode 投影 GEMM 经 `rocm_unquantized_gemm_impl`(`utils.py:122`),`VLLM_ROCM_USE_SKINNY_GEMM=True`(默认)下,**理论上**:
  - `n≤4` → `wvSplitK`(自定义 skinny GEMM,`_rocm_C.wvSplitK`)
  - `n==1, k≤8192, bias is None` → `LLMM1`(自定义 skinny GEMM,`_rocm_C.LLMM1`)
  - 否则 → `torch.nn.functional.linear` → **hipBLASLt**
- **bias 钉死**:三投影层 `bias=False`(qwen3_next.py:510/537/559),`apply` 拿到的 `bias=None` 恒成立 → LLMM1 的 `bias is None` 不回退。
- **k 钉死**:`in_proj_qkvz` k=5120,`in_proj_ba` k=5120,`out_proj` k=6144 —— 三者 k 均 ≤ 8192 → LLMM1 的 `k≤8192` 不回退。
- **m 钉定**:`in_proj_qkvz` m=16384,`in_proj_ba` m=96,`out_proj` m=5120。
  - `in_proj_ba`(m=96):`m>8 and n≤4` 命中 `wvSplitK`(n≤4);或 n==1 命中 LLMM1(`m%4==0` ✓ 96%4=0)。
  - `in_proj_qkvz`(m=16384):n==1 命中 LLMM1(`m%4==0` ✓);n∈{2,3,4} 命中 `wvSplitK`(m>8 ✓)。
  - `out_proj`(m=5120):同上,n==1 命中 LLMM1。
- **即 eager decode batch=1 稳态(n=1),三个投影 GEMM 理论上走 `LLMM1`**(n=1 命中 `m%4==0 and n==1 and k≤8192 and bias is None`)。
- **但 trace 的 `Cijk_*_GSU1` 命名与 skinny kernel 源码命名(`LLGemm1_kernel`/`wvSplitK_hf_*`)不符**(§0.5.4 候选 A 证伪倾向)→ 两种解释:
  - (i) **cudagraph ON 路径**(批3,baseline)capture 后 batch 被 pad 增大,实际 `n>4` 越过 skinny 边界 → 回退 `torch.nn.functional.linear` → **hipBLASLt/rocBLAS** → `Cijk_*_GSU1` 命名吻合。批3 `MT64x32x32` 单 call 383.9us ≫ eager 162.6us 支持 n 增大。
  - (ii) wheel 按 gfx936 编译 → skinny kernel 体空壳(`__HIP__GFX9__` 宏不含 gfx936)→ 即便 n=1 调 LLMM1 也无效,fallthrough 到 hipBLASLt。
- **修正后的现状判断**:baseline(cudagraph ON)路径下,投影 GEMM **大概率实际走 hipBLASLt/rocBLAS**(候选 D),而非 skinny LLMM1。这与 §0.5.3 旧判断("走 skinny LLMM1")冲突 —— **以 §0.5.4 候选 D 为准**,旧判断保留作待验证项。

### 2.4 算子性能基线(`01` §1.2,gfx936 实测)

| 后端 | BF16 TFlops | 相对 hipBLAS |
|---|---|---|
| hipBLAS | 403 | 1.00× |
| DeepGEMM | 280 | 0.69× |
| Triton GEMM | 175 | 0.43× |
| CK GEMM | 144.7 | 0.36× |
| FP8 | segfault | 不可用 |

- **hipBLAS BF16 403T 是 BF16 上限**,理论上投影 GEMM 切 hipBLASLt 应最快。但 skinny `LLMM1`/`wvSplitK` 是为 **skinny(n 极小)形状特化**的核,在 n=1 时可能已优于通用 hipBLASLt(否则 vLLM 不会默认开)—— **这是 §3 优化点1 的核心待验证点**。

---

## 3. 可执行优化点(锁定约束内,按预期收益/风险排序)

> 全部**待落地验证**(批3 profile 已就绪)。本节为设计清单,不实施。每点标注 [收益预期]/[风险]/[约束合规]/[验证方法]。

### 优化点 1(最高优先):切换投影 GEMM 后端 —— skinny `LLMM1` vs hipBLASLt vs aiter `gemm_a16w16`

> ## 🔧 修正(2026-07-12,torch profiler 实测推翻)
>
> 旧稿断言"baseline 必走 hipBLASLt,关 skinny env 零 tpot 收益(本来就走 hipBLASLt)"。**已推翻**:baseline 实走 **LLMM1**(qkvz/ba/out_proj 三个投影 GEMM 全部),`MT64x32x32_GSU1` 不是 qkvz。关 skinny env(`VLLM_ROCM_USE_SKINNY_GEMM=0`)的真实效果是 **qkvz 从 LLMM1 127us 退回 hipBLASLt 229us(~1.8x 退化)**,反而是负收益。**优化点1 作废:关 skinny 是退化而非诊断旁证。** 见 §0.5.3 修正块、§8.11。

- **动作**:envs `VLLM_ROCM_USE_SKINNY_GEMM=0` 重启 → 投影 GEMM 全走 `torch.nn.functional.linear` → hipBLASLt。bench 对比 mean_tpot。
- **机制**:**候选 D 已三重闭合钉死**(§0.5.3 Python 层 / §0.5.4 C++ 层 / build 层),baseline(cudagraph ON)路径投影 GEMM **确定已走 hipBLASLt 回退**,且 `on_gfx9()=False` 在 Python 层就挡死 skinny 分支(根本到不了 wvSplitK/LLMM1 调用)。→ 关 skinny env 是**零 tpot 收益**(本来就走 hipBLASLt),仅作反向旁证:
  - 关 skinny 后 `Cijk_*_GSU1` kernel 名/耗时**不变** → 旁证 build/Python 层结论(预期结果)。
  - (旧假说"若 baseline 走 skinny LLMM1"分支**作废** —— 候选 A 已三重证伪,不存在 baseline 走 skinny 的可能。)
- **[收益预期]**:**诊断性,零 tpot 收益**。候选 D 已钉死,baseline 必走 hipBLASLt,关 skinny 不改变任何 GEMM 后端。优化点1 从"切后端降 tpot"降级为"零成本反向确认实验",确认后转优化点2。
- **[风险]**:① envs 级,零代码改动,零成本回滚;② 若 (b) 倒退即回滚。
- **[约束合规]**:✅ `VLLM_ROCM_USE_SKINNY_GEMM` 未锁定(envs 运行参数),`01` §1.1 允许"执行路径优化"。
- **[验证方法]**:关 skinny 重启 → `_decode_only_profile.py` 抓 decode trace → 对比 `Cijk_*_GSU1` kernel 名/单 call 耗时**是否变化**(预期**不变**,旁证候选 D)。
- **优先理由**:**零代码、零编译、纯 envs** 的零成本反向确认。候选 D 已三重闭合,此步仅冗余旁证,优先级从"最高"降为"可选诊断";重心转优化点2(in_proj_qkvz tile)。

### 优化点 2(高优先,归属已闭合):针对性优化 `in_proj_qkvz`(占 GDN 68%)

> ## 🔧 修正(2026-07-12,torch profiler 实测推翻)
>
> 旧稿断言"`MT64x32x32_GSU1`(5.074s)归属 `in_proj_qkvz`,qkvz 是 GDN 单点最大耗时"。**已推翻**:`MT64x32x32_GSU1`(median 490us/call)**不是 qkvz** —— qkvz 实走 LLMM1 127us,在 trace 顶级 kernel 列表里根本排不上(LLMM1 不在 top-25)。优化点2 的对象(qkvz)已被优,真正该优化的是 `MT64x32x32_GSU1` 与 `MT32x16x4_GSU1`(144us),需先 op 归因这两个 kernel 到底对应哪个 GEMM(§8.11)。**本优化点对象待重定。** 见 §8.11。

- **动作**:`MT64x32x32_GSU1`(5.074s)**已由 §5.0.0 邻接 + §0.5.4 候选 D 三方闭合归属 `in_proj_qkvz`**(m=16384, k=5120)。对其单独换后端/换 tile。
- **机制**:`in_proj_qkvz` 是 `MergedColumnParallelLinear` 输出 16384 维,投影 GEMM 形状 = `[n,5120]×[16384,5120]ᵀ`(n 见 §8 capture batch 分析)。这是 GDN 单点最大耗时。
- **[收益预期]**:若该 GEMM 能压 30% → 5.074s×0.3 = 1.5s / 8s 窗口 ≈ tpot 降 ~19%。
- **[风险]**:① 归属已闭合(无需再确认对象);② 单层换后端/换 tile 需改 `dispatch_unquantized_gemm` 加形状特判或调 hipBLASLt heuristic,触及框架代码/env。
- **[约束合规]**:✅ 改框架代码允许(`01` §1.1)。
- **[验证方法]**:`_decode_only_profile.py` + 对比 in_proj_qkvz 的 kernel 名/耗时;或临时在该层加 `torch.profiler` label。

### 优化点 3(中优先):投影 GEMM 算子级 BF16→FP8 动态量化

- **动作**:对投影 GEMM 输入激活做运行时 BF16→FP8 动态量化(权重不变,非量化边界),GEMM 走 FP8 → 输出反量化。
- **机制**:`01` §1.1 显式允许"激活值动态量化/低精度矩阵乘"。FP8 GEMM 理论吞吐 >> BF16。但 `01` §1.2 实测 **FP8 segfault 不可用** → 严重限制。
- **[收益预期]**:若 FP8 可用则高(2× 吞吐);**但实测 segfault → 本点大概率不可行**,除非找到不 segfault 的 FP8 路径(aiter FP8 linear?`rocm_aiter_ops.is_linear_fp8_enabled()`)。
- **[风险]**:① segfault(已实测);② 精度损失(greedy decoding 对 logits 敏感,但投影层中间态可容);③ DCU FP8 fnuz 支持有限。
- **[约束合规]**:✅ 算子级低精度允许;但**禁止量化操作边界**(指权重量化,激活动态量化不触边)。
- **[验证方法]**:先小范围测 aiter FP8 linear 在 gfx936 是否 segfault;若可用再接入投影层。**P2 后期/低优先,先确认 FP8 是否真不可用。**

### 优化点 4(中优先):投影 + 递归核 fused(跨算子融合)

- **动作**:把 `in_proj_qkvz`/`in_proj_ba` 投影 GEMM 与后续 `causal_conv1d_update`+`fused_recurrent` 融合成单 kernel,减少中间 tensor round-trip。
- **机制**:64 层每层当前 = 投影GEMM → cat → conv1d → recurrent → norm → out_proj,多次 HBM 读写。融合减访存。
- **[收益预期]**:中。但递归核本身仅 1%,投影 GEMM 68% 是 compute-bound 不是 memory-bound → 融合对 compute-bound GEMM 收益有限,主要省中间 tensor 访存。
- **[风险]**:① 工程量大(需写 HIP/Triton fused kernel,`01` §1.1 说 docker 依赖全无、无法编译复杂工具 → **可能不可行**);② cudagraph capture 兼容性。
- **[约束合规]**:✅ 允许,但受编译环境限制(`01`:docker 严重简化,只能用预装 rocprof/hipprof)。
- **[验证方法]**:先评估能否在容器内编译 custom kernel;若不能 → 本点作废。**前置:确认编译环境。**

### 优化点 5(低优先):`custom_ops` / `pass_config` 切 GEMM 算子映射

- **动作**:通过 `compilation_config.custom_ops`(未锁定)增删算子映射,如 `+gemm`/`-rocm_unquantized_gemm` 强制走某后端。
- **机制**:vLLM custom_ops 机制可强制特定算子走指定实现。
- **[收益预期]**:低,本质和优化点1 同源(切后端),但更精细。
- **[约束合规]**:✅ `custom_ops` 未锁定。
- **[验证方法]**:对照 `09` §2 的 custom_ops 未锁定结论,逐个试。

---

## 4. 落地路线图(批3 profile 已就绪,等用户确认后执行)

```
[第一步](归属已闭合,免做) `Cijk_..._GSU1` 真身 = in_proj_qkvz(§5.0.0 邻接 + §0.5.4 候选 D 三方闭合)
   └─ 后端 = hipBLASLt 回退(候选 D 三重闭合),非 wvSplitK/LLMM1

[第二步] 优化点1(零成本诊断):VLLM_ROCM_USE_SKINNY_GEMM=0 重启 → 抓 decode trace
   └─ 预期 kernel 名/耗时不变(旁证候选 D),无 tpot 收益 → 转第三步

[第三步] 主线 = 优化点2(in_proj_qkvz tile/heuristic 优化,§8 新增分析)
   ├─ §8.1 hipBLASLt heuristic/tile 调参(HIPBLASLT_HEURISTIC env / 显式 solution)
   └─ §8.2 capture 后实际 n 值确认 → 决定是 skinny GEMM(n 小)还是 mid-size GEMM(n=128)策略

[第四步] 旁支:优化点3(FP8)/优化点4(fused)/优化点5(custom_ops),每次只动一个,bench 对比 mean_tpot
   ├─ 优化点3:先确认 aiter fp8 linear 在 gfx936 是否 segfault
   └─ 优化点4:先确认容器能否编译 custom kernel(01 说不能)

[第五步] 叠加验证,记录到 05 §5.3
```

---

## 5.0 插桩任务(本轮执行:钉死 `MT64x32x32_GSU1` 归属哪个投影 GEMM)

> §0.5.4 候选 D(投影 GEMM 经 hipBLASLt/rocBLAS 回退)已静态闭合,但 trace 主力 `Cijk_..._MT64x32x32_..._GSU1`(5.074s,占 GDN 68%)到底归属 `in_proj_qkvz`/`in_proj_ba`/`out_proj` 哪一个,仍靠 m 值推断(`in_proj_qkvz` m=16384 最大,概率最高)。本节插桩任务用被动 `record_function` 标注实机 1:1 钉死,确定优化点2 的确切对象。

### 5.0.0 ✅ 邻接推断已闭合(eager trace 非插桩,2026-07-11)

> ## ⚠️ 本节邻接推断结论已被 §8.11 / §0.5.3 修正块推翻(2026-07-12)
>
> 本节基于"cudagraph ON 的 `MT64x32x32_GSU1` 大头归 `in_proj_qkvz`,因 §0.5.4 候选 D(全走 hipBLASLt 回退)"。但候选 D 已被推翻(投影 GEMM 实走 LLMM1),故"qkvz = MT64x32x32_GSU1"的衔接推断**前提失效**。torch profiler 实测:qkvz 实跑命中 `LLGemm1_kernel` 127us,**不是** `MT64x32x32_GSU1`(490us)。下文保留作历史记录,**"优化点2 对象 = qkvz = MT64x32x32_GSU1"的结论作废**,真瓶颈 `MT64x32x32_GSU1` 的归属待 §8.11 重新归因。
>
> 注:eager trace 邻接看到的 `MT32x16x128`/`MT32x16x4` 也不可能是 qkvz/ba —— 因为这些 `Cijk_*` 是 hipBLASLt kernel,而 qkvz/ba 走 LLMM1(`LLGemm1_kernel`)。eager trace 里这些 `Cijk_*` 大概率是 FFN / attention 的其他 GEMM(归因错误源于旧稿误判后端)。需 §8.11 重核。

> 插桩卡两天未通(用户在另一窗口并行修)。改用 **eager trace 的 kernel 序列邻接** 推断 GEMM 归属,绕开插桩。结论精度低于插桩的 1:1,但已足够指导优化点2 的对象选择,且与 §0.5.4 候选 D(全走 hipBLASLt 回退)相互独立、相互佐证。

**数据源**:`/tmp/trace_eager.json`(2026-07-11 02:02 抓取,eager 无 cudagraph,层序逐层清晰)。trace 内 **305 个 `Cijk_*` GEMM**,均 `cat=kernel`、单流(pid=0 tid=0),cpu_op args 仅 `['External id','Record function id','Ev Idx']` 无 python stack/层名 → 不能直接读归属,改用**前驱/后继 kernel 邻接 + 层内序列模板**推断。

**层内 cijk 序列模板(按 `MT32x16x128` 切分 48 段,稳定重复 32 次)**:
```
MT32x16x128 → MT32x16x4 → MT16x16x256 → MT32x16x256 → MT16x16x256
                  (变体: 中段多出 MT64x16x128→MT16x16x256→MT32x16x256→MT16x16x256,15 次)
```

**逐 cijk 归属推断(基于 GDN 层 forward 时序 + 前驱/后继 kernel)**:

| MT | 前驱 kernel | 后继 kernel | 推断归属 | 依据 |
|---|---|---|---|---|
| `MT32x16x128` | **RMSNorm**(48/48) | cijk(48/48) | **`in_proj_qkvz`**(m=16384) | 层内第一个 GEMM,前驱=input_layernorm 的 RMSNorm,后继=下一个投影 GEMM(=ba) |
| `MT32x16x4` | **cijk=32x16x128**(48/48) | elementwise | **`in_proj_ba`**(m=96) | 紧跟 qkvz 之后(前驱 cijk 锁定),m=96 最小 → 小 tile;后接 elementwise(cat/reshape) |
| `MT16x16x256` | RMSNorm(48)/ act_and_mul(64)/ elemwise(16) | RMSNorm/ cijk/ elemwise | 混合(out_proj + FFN gate/up/down) | 多前驱 → 横跨 out_proj(后继 RMSNorm=post_attn_layernorm)与 FFN(前驱 act_and_mul) |
| `MT32x16x256` | RMSNorm(64) | — | FFN/投影 | 前驱 RMSNorm,大 tile |
| `MT64x16x128` | RMSNorm(16) | elemwise(16) | FullAttn 层 qkv_proj | 仅 16 个 = 16 FullAttn 层各 1 |

**关键钉死**:
1. **`in_proj_qkvz` = `MT32x16x128`**(eager trace)。前驱恒为 RMSNorm、后继恒为 cijk,完美匹配"input_layernorm→qkvz→ba"时序。**48 个 = 48 GDN 层各 1,无歧义。**
2. **`in_proj_ba` = `MT32x16x4`**(eager trace)。前驱恒为 cijk(=qkvz 的 32x16x128),后继 elementwise,完美匹配"qkvz→ba→cat/reshape"。**48 个 = 48 GDN 层各 1,无歧义。**
3. **out_proj 夹在中段 `MT16x16x256`/`MT32x16x256` 之中**(前驱 act_and_mul 或 fused_recurrent 后),非单一 MT 对应 —— eager 下 out_proj 与 FFN GEMM 共用相近 tile config,邻接法无法干净拆出,但**out_proj 不是 5.074s 大头**(m=5120 < qkvz 的 16384),对优化点2 对象判断无影响。

**与 cudagraph ON trace(批3 `MT64x32x32_GSU1`)的衔接**:
- eager trace 主力是 `MT32x16x4`(频次最高,15222 次)和 `MT16x16x256`;**cudagraph ON trace 主力是 `MT64x32x32_GSU1`(5.074s,68%)** —— 两者 tile config 不同(cudagraph capture 后 batch 增大 → hipBLASLt 选更大 tile `64x32x32`,eager batch=1 → 选小 tile `32x16x4`/`16x16x256`)。
- **但归属不变**:cudagraph ON 的 `MT64x32x32_GSU1` 大头仍归 `in_proj_qkvz`(m=16384 最大,且 §0.5.4 候选 D 已证全走 hipBLASLt 回退 → 大 GEMM 占主导 = 大 m 的 qkvz)。
- **即**:`MT32x16x128`(eager)和 `MT64x32x32_GSU1`(cudagraph ON)都是 `in_proj_qkvz`,只是 batch 不同导致 hipBLASLt 选了不同 tile config。

**结论(给优化点2 用,邻接推断闭合)**:
- **优化点2 的确切对象 = `in_proj_qkvz`**(m=16384, k=5120)。eager 邻接锁定 + cudagraph ON 占 68% 5.074s 大头 + §0.5.4 候选 D(全 hipBLASLt 回退)三方一致。
- `in_proj_ba`(m=96)是 `MT32x16x4`,虽频次最高但 m=96 极小,绝对耗时远低于 qkvz,**非首要优化对象**。
- **此结论不依赖插桩成功**。若另一窗口插桩最终生效,可 1:1 复核;但当前邻接推断已足够支撑优化点2 落地。

**遗留(插桩生效后复核,非阻塞)**:cudagraph ON 下 `MT64x32x32_GSU1` 是否 100% 落 in_proj_qkvz 区间(邻接法只在 eager 闭环,cudagraph ON 因 replay 时序聚簇无法邻接)。

---

### 5.0.1 原理

`torch.profiler.record_function("GEMM_PROBE::in_proj_qkvz")` 在 chrome trace 生成 `cat="user_annotation"` 的 `X` 事件,`ts..ts+dur` 区间覆盖被包住的 `self.in_proj_qkvz(...)` 及其触发的全部底层 GPU kernel。离线看每个 label 区间内落了哪个 `Cijk_GSU1` kernel 即可 1:1 对应。**仅标注元数据,不改算子/输入输出/不引入计算**(profiler 未开启时近乎零开销)。

### 5.0.2 脚本(三件套,均在 `tools/`)

| 脚本 | 角色 | 关键点 |
|---|---|---|
| `tools/_gemm_probe.py` | 被动探针模块 | 三个 contextmanager `label_in_proj_qkvz/label_in_proj_ba/label_out_proj`,各包一层 `record_function("GEMM_PROBE::...")`;torch 不可用时 no-op 兜底;与 fill_alloc_probe v2 独立 |
| `tools/_apply_gemm_probe.py` | 一次性幂等 patch | 注入 `qwen3_next.py:Qwen3NextGatedDeltaNet.forward`(line 650/651/689);精确字符串匹配 + `ast.parse` 校验 + `SYNTAX_OK`;沿用 `_apply_probe_v2.py` 模式 |
| `tools/_parse_gemm_probe.py` | 离线解析 | 双指针匹配落入 `GEMM_PROBE::*` 区间的 kernel/gpu_memcpy/gpu_memset,按名聚合 dur+count;末尾专列 `Cijk_*_GSU*` 系列在三 label 的分布 |

覆盖性:`forward_native` 与 cudagraph replay 都经 `forward`,三处插桩覆盖 decode 路径(capture 时 label 录进图,replay 随图回放)。

### 5.0.3 落地流程(用户要求:插桩前先跑对照基线)

```
[0] 对照基线(已就绪,免跑)
   └─ 批3 无桩干净 trace: profile_traces/rank0.1783570538023481078.pt.trace.json.gz
      已含 §0.5.4 表 MT64x32x32_GSU1=5.074s/13216 calls 等基线数字,作对照:
      插桩后 trace 的 kernel 时序应与之一致(标注不改变执行)。

[1] 插桩 patch(容器内)
   ├─ cp tools/_gemm_probe.py → .../vllm_cscc/vllm/model_executor/models/gemm_probe.py
   ├─ cp tools/_apply_gemm_probe.py 到容器可写目录
   ├─ 确保 gemm_probe.py 所在路径在 PYTHONPATH(或 patch 时改 import 路径)
   ├─ python _apply_gemm_probe.py          # 幂等,输出 Wrapped ×3 / SYNTAX_OK / DONE
   └─ python -c "import ast; ast.parse(open('<qwen3_next.py 全路径>').read())"  # 二次校验

[2] 重装 patched wheel(非 editable!)
   ├─ cd /public/home/xdzs2026_c150/zya/vllm_cscc
   ├─ python setup.py bdist_wheel          # 见 06 流程
   └─ pip install --force-reinstall --no-deps dist/vllm-*.whl

[3] 抓插桩版 decode trace
   └─ python tools/_decode_only_profile.py   # 同批3 参数:streaming→等 TTFT→buffer 2s→/start_profile 8s→/stop_profile
      落 profile_traces/<新>.pt.trace.json.gz
      ★ sanity:trace 应出现 3 种 GEMM_PROBE::* user_annotation(每 step × 64 层 × 3)

[4] 离线钉死
   └─ python tools/_parse_gemm_probe.py profile_traces/<新>.pt.trace.json.gz
      判据(对应 §0.5.4 候选 D 已闭合后,本步只确认归属):
      ├─ MT64x32x32_GSU1 几乎全落 in_proj_qkvz 区间 → 坐实"5.074s 大头 = in_proj_qkvz"(m=16384)
      ├─ 落 in_proj_ba/out_proj 区间 → 修正归属(m=96/5120 反而更大,需复盘 tile config)
      └─ 三 label 都落 Cijk_GSU1 → 二次确认候选 D(投影 GEMM 全经 hipBLASLt/rocBLAS 回退)
```

### 5.0.4 注意 / 风险

- **GPU 时钟域**:profiler 把 user_annotation(CPU 派发)与 kernel(GPU)放同一时间轴,但 GPU kernel `ts` 是 dispatch 后的 GPU 时间。脚本用"kernel 中点 ts 落在 label 区间内"匹配;若全 label 空且 sanity 确认 annotation 存在,需改用 op→kernel 流水关系(`stream`/`device` 字段)重匹配。
- **cudagraph replay 时序**:capture 录进图的 record_function,replay 时 annotation 时间戳可能聚成一簇而非逐层散开,区间被压缩成一点会让 kernel 落入判定失真。备选:enforce-eager 模式抓一份对照 trace(eager 无 replay,label 逐层清晰),交叉确认。
- **不碰运行中服务**:容器可能有别的 vllm 在跑,patch+重装+重启需用户确认时点,禁止擅自启动/重启 vllm。
- **回滚**:桩是加注释,`git checkout -- qwen3_next.py` 还原;重装原 wheel 覆盖。

---

## 5. 待验证清单(汇总,本轮不执行)

1. **[✅ 已闭合·三重] §0.5.4 候选 D 钉死**:`Cijk_..._GSU1` 主力 = 投影 GEMM 经 **hipBLASLt/rocBLAS 回退**(非 LLMM1/wvSplitK/aiter)。三证独立一致:
   - **(a) Python 分发层(本轮源码,最直接)**:`on_gfx9()`(`rocm.py:233`)← `_ON_GFX9`(`rocm.py:147`)列表硬编码 `["gfx90a","gfx942","gfx950"]` **不含 gfx936**;`_GCN_ARCH` 经 amdsmi 实测 = `gfx936`(BW3000,80 CU)→ `on_gfx9()=False` → `utils.py:170` `use_skinny = ... and on_gfx9()` **Python 层短路 False** → `utils.py:188` `return torch.nn.functional.linear` → hipBLASLt。**根本到不了 `ops.wvSplitK/LLMM1` 调用,与 env / n 值 / capture batch 全无关。**
   - (b) C++ 命名层:skinny kernel 名(`LLGemm1_kernel`/`wvSplitK_hf_*`)与 trace `Cijk_*_GSU1` 不符。
   - (c) C++ build 层(见第2条):wheel 按 gfx936 编译 → skinny kernel 体 `#if __HIP__GFX9__` 空壳。
   - **三证互不依赖、独立指向同一结论**:gfx936 下 skinny GEMM 全链路(Python 分发 → C++ 调用 → kernel 体)不可达,投影 GEMM 必走 hipBLASLt 回退。即便(a)单独成立已足够(连调用都不发生),(b)(c)是冗余佐证。
2. **[✅ 已闭合] §0.5.4 编译宏(本轮本地 build 钉死)**:`vllm_cscc/build/.../CMakeCache.txt` 三字段一致 `gfx906;gfx926;gfx928;gfx936;gfx938` —— **含 gfx936,不含 gfx942/gfx950**。gfx936 二进制编译时 `__gfx90a__`/`__gfx942__`/`__gfx950__` 三宏均未定义 → `__HIP__GFX9__`(skinny_gemms.cu:24-27)不定义 → skinny kernel 体空壳。→ decode 必走 hipBLASLt 回退(候选 D 坐实)。
3. **[✅ 已闭合] §2.3 回退门槛**:`bias=False`(三投影层)+ k∈{5120,5120,6144}≤8192 + m∈{96,16384,5120}%4==0 → LLMM1 门槛本就不回退。**但本轮 Python 分发层已证(第1条 a) `on_gfx9()=False` 在 `use_skinny = ... and on_gfx9()` 处直接短路 → `utils.py:188` 兜底 `torch.nn.functional.linear` → hipBLASLt,bias/k/m/n 门槛全部免验**(连 wvSplitK/LLMM1 的调用都到不了,门槛是否满足无意义)。
   - **[✅ 邻接推断闭合,2026-07-11] trace 主力归属**:**eager trace 邻接法钉死** `in_proj_qkvz`=`MT32x16x128`(前驱 RMSNorm/后继 cijk,48 层各1)、`in_proj_ba`=`MT32x16x4`(前驱 cijk=qkvz/后继 elemwise,48 层各1)。cudagraph ON 的 `MT64x32x32_GSU1`(5.074s,68%)仍归 `in_proj_qkvz`(m=16384 最大 + 候选 D 全 hipBLASLt 回退 → 大 GEMM 主导 = qkvz;tile config 不同仅因 batch 差异)。**优化点2 对象 = in_proj_qkvz 已定,非阻塞。** 详见 §5.0.0。插桩生效后可 1:1 复核,非必需。
4. **[降级为诊断性] §3 优化点1**:关 skinny(`VLLM_ROCM_USE_SKINNY_GEMM=0`)切 hipBLASLt。**因候选 D 已坐实(baseline 本就走 hipBLASLt),关 skinny 零 tpot 收益**——但可作为零成本反向确认:关 skinny 后 `Cijk_*_GSU1` kernel 名/耗时不变 = 旁证 build 结论。优先级从"最高"降为"可选诊断"。
5. **§3 优化点3**:aiter `is_linear_fp8_enabled()` 路径在 gfx936 是否 segfault(`01` §1.2 实测 DeepGEMM FP8 segfault,但 aiter FP8 linear 未测)。
6. **§3 优化点4**:docker 简化环境能否编译 custom fused kernel(`01` §1.1 倾向不能)。
7. **§0.5.3 矛盾**:`05` §5.1 第1条旧悬置"FFN GEMM 走的不是 hipBLAS"→ 本次闭合为"投影 GEMM 候选 D 走 hipBLASLt 回退",需在 05 补记修正。
8. **[新增·高] build 重编边界**:若要让 skinny kernel 在 gfx936 生效(优化点1 的反向前提),需重编 wheel 时把 gfx942/gfx950 加入 `AMDGPU_TARGETS` 或改 skinny_gemms.cu 的宏守卫含 gfx936。但 `01` §1.1 docker 严重简化、依赖全无 → **重编可行性待确认**(与第6条同源)。即便能重编,skinny 在 n=1 是否优于 hipBLASLt 仍待 bench(§2.4 待验证点)。

---

## 6. 与另一窗口(CPU/调度轨道 `09`)的边界

- **GDN 轨道(本文件)**:GPU 侧 95.17% 的 GDN GEMM kernel 优化(后端切换/低精度/融合)。
- **CPU/调度轨道(`09`)**:`09` §0.5 duty cycle 钉死后,**该轨道在稳态 decode tpot 上无收益空间**(GPU duty 97.3%,无 step 间空闲可压缩)。本轨道是唯一能降 tpot 的轨道。
- **不重叠**:GPU kernel 内部耗时(compute)归本轨道;GPU 之外(CPU/IPC/调度)归 `09`。但 `09` §0.5 已证稳态无 GPU 空闲 → `09` 的优化点(async_scheduling/stream_interval/.cpu())不降 tpot,**全部 tpot 优化责任在本轨道**。
- **协同点**:批3 cudagraph ON profile 同时服务两轨道 —— 本轨道看 kernel 占比/后端;`09` 看 duty cycle(已确认 97.3%)。
- **传递关系**:`09` §0.5 把"端到端 tpot 瓶颈 = 64 层 GDN GEMM 串行"的结论交给本轨道 §0,本轨道据此定位优化标的(投影 GEMM,`in_proj_qkvz` 占 68%)。

---

## 7. 结论性判断(设计阶段)

1. **本轨道是唯一能降 tpot 的轨道**(`09` §0.5 duty cycle 97.3% 钉死,瓶颈在 64 层 GDN GEMM 串行,不在 step 间)。
2. **GDN GEMM 主力 = 投影 GEMM**(递归核无 GEMM 已钉死,§0.5.2),其中 `in_proj_qkvz`(m=16384)占 GDN 68%(5.074s)是单点最大标的。
3. **投影 GEMM 后端(三重闭合钉死)**:baseline(cudagraph ON)路径下**确定走 hipBLASLt/rocBLAS 回退**(§0.5.4 候选 D)。回退机制 = **Python 分发层 `on_gfx9()=False`(`rocm.py:147` 列表不含 gfx936)在 `use_skinny = ... and on_gfx9()` 处短路 → `utils.py:188` 兜底 `torch.nn.functional.linear` → hipBLASLt**;C++ 编译宏空壳 + CMakeCache build 目标为冗余佐证。**与 n 值 / env / capture batch 全无关,skinny 分支全链路不可达。** 旧判断("走 skinny LLMM1")作废。
4. **bias/k/m/n 门槛全部免验**:三投影层 `bias=False`(qwen3_next.py:510/537/559)→ apply 拿 `bias=None`;k∈{5120,5120,6144}≤8192;m∈{96,16384,5120} 均 %4==0。**但 Python 层 `on_gfx9()=False` 已在 wvSplitK/LLMM1 调用前挡死,门槛是否满足无意义**(连调用都到不了)。
5. **`MergedColumnParallelLinear` 单 GEMM**:`output_sizes=[16384]`/`[96]` 是单元素 list,权重单块,forward 单次 apply → 不拆多次 GEMM;`÷118=112` 倍数是 48 层 × 多类投影 × tile config 聚合,非单层拆分。
6. **优化点1 降级为诊断实验**:候选 D 三重闭合 → baseline 已走 hipBLASLt → 关 skinny 零 tpot 收益,仅冗余旁证。**重心转向优化点2(in_proj_qkvz tile/heuristic 优化,见 §8)+ 优化点3(低精度)**。DeepGEMM/FP8 基本失效(稠密无 MoE + FP8 segfault),fused kernel 受编译环境限制。
7. **落地主线 = 优化点2(§8 新增分析)**:(a) hipBLASLt 自身 heuristic/tile 能否调(`MT64x32x32` 是否次优 tile);(b) capture 后实际 n 值(cudagraph 把 n 撑到多少,决定 skinny vs mid-size GEMM 策略)。**归属问题已由 §5.0.0 闭合,优化点2 对象=in_proj_qkvz 已定,不再阻塞。**

---

## 8. 优化点2 深化分析(hipBLASLt tile 调参 + capture 后 n 值)

> 本节闭合用户提出的两个新分析点。**两点设计阶段均已完成只读核查(未启动 vllm),结论一致指向同一落地实验:抓一次 decode trace**。核查方法:§8.1 = 容器内 `strings libhipblaslt.so` + vLLM 源码 grep;§8.2 = vLLM cudagraph capture 源码逻辑。全部只读,遵守 §5.0.4。

### 8.1 hipBLASLt 自身 tile/heuristic 能否调?(已只读核查闭合)

**核查对象**:海光魔改 hipBLASLt(容器内 `/opt/dtk-26.04-DCC2602-0317/hipblaslt/lib/libhipblaslt.so`,DTK 26.04)。
**核查动作**:`strings` 提取全部 `HIPBLASLT_` 前缀 env;`grep` vLLM 源码是否引用。

#### 8.1.1 海光 hipBLASLt 暴露的全部 env(仅 6 个)

| env | 作用 | 能调 tile? |
|---|---|---|
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | **override 文件**:外部文件强制指定某 problem 形状 → 某 algo/solution(`UserDrivenTuningParser.cpp` / `problem_override_from_file` 机制) | ✅ **唯一真实可调旋钮** |
| `HIPBLASLT_TENSILE_LIBPATH` | Tensile kernel 库搜索路径 | ❌ |
| `HIPBLASLT_EXT_OP_LIBRARY_PATH` | 扩展算子库路径 | ❌ |
| `HIPBLASLT_LOG_FILE` / `HIPBLASLT_LOG_LEVEL` / `HIPBLASLT_LOG_MASK` | 日志(可用来 dump heuristic 实际选了哪个 algo) | ❌(诊断用) |

#### 8.1.2 结论(三个子问题逐一定论)

1. **vLLM 源码层是否介入 tile 选择?→ 不介入(源码闭合)**
   `grep -rnI 'HIPBLASLT_' vllm_cscc/vllm/` 仅命中 2 处注释(`utils.py:110` "use hipblaslt for the larger GEMMs"、`scaled_mm/pytorch.py:110`),**无任何 env 读取 / algo 指定 / solution 选择代码**。投影 GEMM 走 `torch.nn.functional.linear`(`utils.py:188`)→ ATen → hipBLASLt 高层 `hipblasLtMatmul` 接口,ATen 调用**不传 algo 参数**,完全由 hipBLASLt 内置 heuristic 自选 → 命中 `MT64x32x32_GSU1`。

2. **用户原假设的 `HIPBLASLT_HEURISTIC` env 指定 tile?→ 证伪(核查闭合)**
   海光 hipBLASLt 的 env 列表(§8.1.1)**不含 `HIPBLASLT_HEURISTIC`**,也没有 `HIPBLASLT_FORCE_*` / `HIPBLASLT_ALGO` 之类的强制 tile env。heuristic 策略编译进库,不通过 env 选。上游 AMD hipBLASLt 同样无此 env。**该路径不存在。**

3. **有无其他强制指定 tile 的路径?→ 有,但分两类**
   - **(a) `HIPBLASLT_TUNING_OVERRIDE_FILE`(落地可行,不碰源码/不重编译)**:外部 override 文件强制指定 problem→solution 映射,等价于"指定 tile",只是不通过 env 数值而通过文件。**这是落地阶段唯一可行且零侵入的 tile 调参路径。**
   - **(b) hipBLASLt C API(`hipblasLtMatmulAlgoGetHeuristic` / `getAllSolutions` / `getAlgosFromIndex` / `matmulIsTuned`)**:库底层确实有 solution 选择 API,但 `torch.nn.functional.linear` 高层路径调不到 —— 要用必须写自定义 C++ 算子绕过 ATen,触优化点4 编译环境限制,**设计阶段排除**。

#### 8.1.3 §8.1 闭合状态

| 子问题 | 闭合度 | 结论 |
|---|---|---|
| vLLM 是否介入 tile 选择 | ✅ 源码闭合 | 不介入,ATen 高层接口不传 algo |
| `HIPBLASLT_HEURISTIC` env 指定 tile | ✅ 证伪闭合 | 该 env 不存在于海光 hipBLASLt |
| 有无其他强制 tile 路径 | ✅ 核查闭合 | `HIPBLASLT_TUNING_OVERRIDE_FILE` override 文件(可行)+ C API(需自定义算子,排除) |
| `MT64x32x32` 是否次优 / override 能否提升 tpot | ⏳ 降级落地实验 | 设计阶段无法判定,需设 override 文件 + 重启 vllm + 抓 trace 对比(见 §8.3) |

### 8.2 cudagraph capture 后实际 n 值?(已纯源码核查,落地需 trace)

**核查对象**:vLLM v1 cudagraph capture 的 batch 维度 padding 逻辑(纯源码,未启动 vllm)。

#### 8.2.1 源码逻辑核查结论

vLLM v1 cudagraph capture 按 **batch bucket** 分档 capture(`CUDAGraphMode` / `gpu_model_runner.py` 的 capture 区间逻辑):capture 时以 `max_num_seqs`(=128)上限对 batch 维度做 bucket padding,每个 bucket 预录一张 graph。**decode 阶段实际 running batch=1,但若命中某 bucket 的 graph,执行时 GEMM 的 m/batch 维度按该 bucket 的 padding 值跑(而非运行时实际 1)**——这正是 cudagraph 静态形状的固有特性(graph 录制时形状固定,replay 时不能变)。

> 但源码只能确认"capture 时按 bucket padding",**无法纯源码定论 decode batch=1 实际命中哪个 bucket、该 bucket 的 padding n 是多少**。批3 trace `MT64x32x32` 单 call 383.9us ≫ eager batch=1 的 162.6us 强烈暗示 capture 后 n 明显增大(若仍 n=1 耗时应近 162.6us),但精确 n 值需 trace 的 kernel args / 形状维度反推。

#### 8.2.2 策略分叉(取决于 n 值)

| capture 后实际 n | GEMM 性质 | 优化方向 |
|---|---|---|
| n≈1(未撑大) | 极 skinny GEMM / matvec | hipBLASLt heuristic 为大 GEMM 调,在 n=1 可能次优 → §8.1 override 文件指定 n=1 特化 tile |
| n=128(撑到 max_num_seqs) | mid-size GEMM `[128,5120]×[16384,5120]ᵀ` | **完全不是 skinny 问题**,`MT64x32x32` 大 tile 反而合理 → 考虑 split-K / 不同 tile 策略,方向与 skinny 截然不同 |
| n=中间 bucket 值 | small-mid GEMM | 介于两者,需具体 n 判定 |

#### 8.2.3 §8.2 闭合状态

| 子问题 | 闭合度 | 结论 |
|---|---|---|
| capture 是否按 bucket padding batch | ✅ 源码闭合 | 是,vLLM v1 cudagraph 按 `max_num_seqs` 上限 bucket capture |
| decode batch=1 实际命中哪个 bucket / 精确 n 值 | ⏳ 降级落地实验 | 纯源码无法定论,需 decode trace 的 kernel 形状维度反推(见 §8.3) |
| skinny vs mid-size 策略分叉 | ⏳ 待 n 值定论 | n 值出来即定方向 |

### 8.3 落地实验(两点合并:抓一次 decode trace)

§8.1 的"override 能否提升"+ §8.2 的"精确 n 值"**合并为同一次落地实验**——抓一次 cudagraph ON 的 decode trace,一次 trace 同时闭合两点:

**实验目标(一次 trace 双闭合)**:
1. **闭合 §8.2**:从 trace 的 `in_proj_qkvz` GEMM kernel args / 形状维度读出 capture 后实际 n 值 → 定 skinny vs mid-size 策略方向。
2. **闭合 §8.1 前置**:从 trace 确认 `MT64x32x32` 对应的实际 problem 形状(m/n/k),作为后续 `HIPBLASLT_TUNING_OVERRIDE_FILE` override 文件的 problem 键(override 文件按 problem 形状匹配)。

**实验步骤(遵守 §5.0.4,由用户重启/抓 trace)**:
1. 用户确认时点后,设 `HIPBLASLT_LOG_LEVEL=info`(可选,dump heuristic 选的 algo 名)。
2. 抓一次 cudagraph ON、decode batch=1 的 trace(沿用 §5.0.3 流程,`start_profile`/`stop_profile` + `--noproxy "*"`)。
3. 容器内就地分析 trace(§4.2 法):提取 `in_proj_qkvz` 邻接 GEMM kernel 的 `MT64x32x32` 调用,读其 m/n/k 形状维度。
4. **判定**:
   - 若 n≈1 → skinny 方向,进 §8.1 override 文件实验(设 `HIPBLASLT_TUNING_OVERRIDE_FILE` 指定 n=1 特化 solution,重启对比 tpot)。
   - 若 n=128/bucket 值 → mid-size 方向,改考虑 split-K / 不同 tile 策略(可能需 C API 自定义算子,触优化点4)。

**风险/约束**:
- 抓 trace 需 vllm 在线 → 由用户操作,不擅自启动/重启(§5.0.4)。
- trace 分析在容器内 `python3` 就地完成,只回传 n/m/k 数字(§4.2,避免大文件回传)。
- override 文件实验若进行,改的是 env + 外部文件,**不改 vllm 源码、不重编译**,重启即生效。

---

### 8.4 落地实验结果(2026-07-11 trace 已闭合)

**数据源**:`rank0.1783748982487161609.pt.trace.json.gz`(813KB,cudagraph ON、decode batch=1,`record_shapes=true`)。容器内就地分析,只回传数字。

#### 8.4.1 §8.2 闭合 —— capture 后实际 n 值 = **1(未被 pad 撑大)**

trace 的 `cpu_op` 事件**全部 8796 个都带形状**(`Input Dims`/`Input Strides`/`Input type` 键齐全,`record_shapes=true` 生效)。32 个 `vllm::rocm_unquantized_gemm`(及对应的 `aten::linear`/`aten::matmul`/`aten::mm`)的形状**完全一致**:

| 算子 | Input Dims | 含义 |
|---|---|---|
| `aten::linear` 输入1 | `[1, 5120]` | **m = 1**(batch 维) |
| `aten::linear` 输入2(权重) | `[248320, 5120]` | n=248320, k=5120 |
| `aten::mm` 输入1 | `[1, 5120]` | m=1, k=5120 |
| `aten::mm` 输入2 | `[5120, 248320]` | k=5120, n=248320 |

> **关键结论:capture 后这 32 个 GEMM 的 m(batch/n)维 = 1,并未被 bucket padding 撑大**。这与 §8.2 的源码假设("capture 按 max_num_seqs=128 bucket padding,m 被撑大")**矛盾**。即 vLLM v1 cudagraph 在 decode batch=1 时,捕获/回放的图实际跑的就是 **m=1 的 matvec**,而非 m=128 的胖 GEMM。批3 trace 里 `MT64x32x32` 单 call 383.9us ≫ eager 162.6us 的耗时差**不是 batch 撑大导致**,而是别的原因(tile 选择/算子链/replay 排队,见 §8.4.3)。

> ⚠️ **这 32 个 m=1 的 GEMM 经源码核对是 lm_head(vocab 投影),不是 GDN 投影**(详见 §8.4.2 的 248320 维度归属核对)。但 **m=1 这一结论本身对 GDN 投影同样成立**:GDN 投影在 trace 里走 `MT64x32x32` kernel(见 §8.4.2 频次推断),其 cpu_op 形状虽未被 `record_shapes` 单独暴露(被 `MergedColumnParallelLinear`/cudagraph 融合),但 decode batch=1 下 GEMM 的 m 维必然 = 1(lm_head 与 GDN 投影共享同一 `hidden_states` 的 batch=1 输入)。故 **m=1 结论适用于全部 decode GEMM(lm_head + GDN 投影)**,§8.2 闭合有效。

**策略分叉判定(§8.2.2 表)**:m=1 → **skinny/matvec 方向**。hipBLASLt heuristic 为大 GEMM 调优,在 m=1 matvec 上**理论上次优** → §8.1 override 文件指定 m=1 特化 solution 有理论收益空间。

> ⚠️ 但 m=1 同时也意味着"skinny GEMM"本应是 vLLM `wvSplitK`/`LLMM1` 的目标场景(n≤4)。而 §0.5.4 已钉死 skinny 分支在 gfx936 上因 `on_gfx9()=False` + 编译宏空壳**全链路不可达** → m=1 的 matvec 当前只能走 hipBLASLt 回退。

> 🚫 **override 路线已被实测推翻,见 §8.4.5**:`hipblaslt-bench` 实测 `(m=1, n=16384, k=5120, bf16)` 和 `(m=1, n=248320, k=5120, bf16)` 两个 problem **都只有 1 个 algo**(index 4362,`..._WGM1` kernel),heuristic 无可选错,override 无的放矢。本节"override 有理论收益"的措辞**作废**,保留仅作 §8.4.5 实测推翻前的推导记录。

#### 8.4.2 §8.1 闭合前置 —— 248320 维度归属核对 + GDN 投影实际 problem 形状

**(a) 248320 = `vocab_size`,这 32 个 GEMM 是 lm_head 而非 GDN 投影**

回 GDN 模型源码 + Qwen3.5-27B 实际 `config.json` 核对:

- `vocab_size = 248320`(Qwen3.5-27B `config.json` 实测值,`qwen3_5.py:44` 默认亦为 248320)。
- `hidden_size = 5120` ✓(与 trace 的 k=5120 吻合)。
- trace 里 `aten::mm` 的 `[1,5120]×[5120,248320]`,权重第二维 248320 = **`vocab_size`**,这是 **`lm_head`(output embedding)的权重形状 `[vocab_size, hidden_size] = [248320, 5120]`**,不是 GDN 的 `in_proj_qkvz`。

**(b) GDN 三个投影的实际维度(源码 `create_qkvz_proj`/`create_ba_proj`/`out_proj`,tp=1)**

config 实测:`linear_key_head_dim=128, linear_num_key_heads=16, linear_value_head_dim=128, linear_num_value_heads=48`。

| 投影 | output_sizes / 维度公式 | tp=1 输出维(n) | 输入维(k) |
|---|---|---|---|
| `in_proj_qkvz` | `[key_dim+key_dim+value_dim+value_dim]` = 2×(128×16)+2×(128×48) | **16384** | 5120 |
| `in_proj_ba` | `[num_v_heads×2]` = 48×2 | **96** | 5120 |
| `out_proj` | RowParallel(value_dim→hidden) | 5120(出) | 6144(入)=128×48 |

→ **§5.0.0 邻接推断给的 `in_proj_qkvz` 权重输出维 16384 是对的**,trace 里看到的 248320 是 lm_head,两者不矛盾(归属不同算子)。先前 §8.4 旧稿"248320 = in_proj_qkvz"的归属判断**作废**。

**(c) MT64x32x32 频次反推:主力归属 GDN 投影,而非 lm_head**

- trace 共 **MT64x32x32 = 3584 个**,32 个 decode step → **112 个/step**。
- GDN 层:64 层中 48 层 `linear_attention`(layer_types 实测每 4 层 3 个 GDN + 1 个 full attention)→ **48 层/step**。每层 qkvz+ba+out 三个投影,若 qkvz+out 走 MT64x32x32 → 48×2 ≈ **96/step**,与 112 同量级 ✓吻合。
- lm_head:**1 个/step**(仅最后 vocab 投影一次)。若 MT64x32x32 全归 lm_head,32 step 只该有 ~32 个,远小于 3584 ✗不吻合。
- 每个 lm_head cpu_op 邻接窗口(±2.5ms)内只挂 4–5 个 MT64x32x32,远不及其总量。

→ **MT64x32x32 主力(3584 个,1.401s)归属 GDN 投影(以 `in_proj_qkvz` 为主),lm_head 仅占极小份额**。这与 §5.0.0 邻接结论一致,GDN `MT64x32x32` 主力定位不变。

**(d) override 文件 problem 键修正(§8.4.5 已推翻,保留作问题形状记录)**

| 算子 | m | n | k | dtype | 备注 |
|---|---|---|---|---|---|
| **GDN `in_proj_qkvz`(主力大头)** | **1** | **16384** | **5120** | bf16 | 5.074s 大头,override 原首要目标 |
| GDN `out_proj` | 1 | 5120 | 6144 | bf16 | 次大头 |
| GDN `in_proj_ba` | 1 | 96 | 5120 | bf16 | 极小,可忽略 |
| lm_head | 1 | 248320 | 5120 | bf16 | trace 直接可见,但 1/step 非大头 |

> ~~override 文件应优先为 `(m=1, n=16384, k=5120, bf16)`(`in_proj_qkvz`)指定 m=1 特化 solution~~ —— **§8.4.5 实测推翻**:该 problem 在 hipBLASLt 里只有 1 个 algo(index 4362),无第二个可换,override 无的放矢。本表保留仅作 problem 形状的钉死记录(对后续非 override 路线仍有用)。

#### 8.4.3 附带发现 —— hipGraphLaunch 窗口只覆盖 8% 的 MT64x32x32

- trace 共 **32 个 `hipGraphLaunch`**(cat=`cuda_runtime`),每个 dur ≈ 4.5–5.2ms,总 0.150s。
- **MT64x32x32 共 3584 个,总 dur 1.401s**,但**只有 290/3584(8%)落在 hipGraphLaunch 时间窗内**。
- 每个 hipGraphLaunch 窗口内约 9–10 个 MT64x32x32(≈3.5ms),而 3584 个 ÷ 32 次 ≈ 112 个/次 —— **绝大多数 MT64x32x32 在 hipGraphLaunch 窗口外**。

> 解读:`hipGraphLaunch` 的 dur 是 **host 侧 `hipGraphLaunch` API 调用的 wall-clock**(含图提交开销),而 kernel 实际执行是**异步排队**在 stream 上的,kernel 的 `ts` 是 GPU 完成时间戳,常滞后于 host 的 `hipGraphLaunch` 返回。所以"窗口外"不代表"不在图里",而是 **GPU kernel 执行时间轴与 host API 时间轴错位**。32 次 hipGraphLaunch ≈ 32 个 decode step(每 step 1 次 graph replay),每 step 内 ~112 个 MT64x32x32(对应 48 层 × 多个投影,符合 §8.4.2 (c) 频次推断)。非新矛盾。

#### 8.4.4 §8 闭合状态汇总

| 项 | 状态 | 结论 |
|---|---|---|
| §8.1 hipBLASLt tile 能否 env 调 | ✅ 设计闭合 | 仅 `HIPBLASLT_TUNING_OVERRIDE_FILE`(覆盖文件),无 env 数值调 |
| §8.1 problem 形状(m/n/k) | ✅ trace+源码闭合 | **GDN `in_proj_qkvz`: m=1, n=16384, k=5120, bf16**(主力大头);`out_proj`: m=1,n=5120,k=6144。trace 直接可见的 248320 是 lm_head,非 GDN |
| §8.2 capture 是否 bucket pad 撑大 batch | ✅ trace 闭合 | **否,m=1 未撑大**(推翻 §8.2 源码假设,lm_head 与 GDN 投影共享 batch=1) |
| §8.2 skinny vs mid-size 策略方向 | ✅ 定论 | **skinny/matvec 方向(m=1)**,但 skinny 分支 gfx936 不可达 → 原拟走 §8.1 override |
| 248320 维度归属 | ✅ 源码闭合 | = `vocab_size`,属 lm_head,**非** `in_proj_qkvz`(旧稿误判已作废) |
| `MT64x32x32` 是否 m=1 次优 tile | ✅ **实测推翻 override 路线** | 见 §8.4.5:m=1 problem 只有 1 个 algo,override 无的放矢 |

#### 8.4.5 §8.1 override 路线实测推翻(2026-07-11 `hipblaslt-bench` 实测)

**数据源**:worker-0 容器(`/opt/dtk-26.04-DCC2602-0317`,hipBLASLt version 1000 / 0.10.0,git `a6254b89-dirty`,gfx936 BW3000 80CUs),`hipblaslt-bench` 直接枚举两个 m=1 matvec problem。

| problem | heuristic 结果 | 唯一解 index | kernel | 耗时 |
|---|---|---|---|---|
| `in_proj_qkvz`(m=1, n=16384, k=5120, bf16) | `Is supported 1 / Total solutions: 1` | **4362** | `Cijk_Ailk_Bljk_BBS_BH_Bias_AS_SAV_MT16x16x16_..._GSU1_GSUAMB_..._WGM1` | 638.9 us |
| `lm_head`(m=1, n=248320, k=5120, bf16) | `Total solutions: 1` | 同 **4362** | 同上 | 8796 us |

**splitK 手动调优(mix api,splitk=0/2/4/8/16,wgm=0)**:`in_proj_qkvz` 耗时在 **638.9~641.9 us** 之间,**几乎无差异**。GSU 字段虽被接受改成 2/4/8/16,但 matvec M=1 没有输出维度切分空间,splitK 在此退化为无效参数,Gflops/us 不变。

**结论(override 路线死刑)**:
1. 这两个 m=1 problem 在 hipBLASLt Tensile kernel 库里**只有 1 个适用 algo**(index 4362,`..._WGM1`)。`HIPBLASLT_TUNING_OVERRIDE_FILE` 的 problem-keyed solution index override **无意义** —— 没有第二个 index 可换,heuristic 不存在"选错"。
2. splitK/GSU 手动改值对 m=1 matvec **不产生加速** —— M=1 时无输出维度切分空间,splitK 被退化为无效参数。
3. **§8.1 override 路线对 `in_proj_qkvz` / `lm_head` 这两个键应当放弃**。

**机制归因**:override 机制本身有效(git 版本校验 + problem-keyed 表),但前提是 problem 有 ≥2 个 algo 让 heuristic 选错。m=1 matvec 在 Tensile 库里只有 1 个适用 algo → override 无的放矢。这与 `[[duty_cycle_kills_step_gap_theory]]` 一致:批3 trace GPU 占空比 97.3%,瓶颈在 GPU kernel 串行;override/splitK 这类"给同一个 algo 换参数"的路子**本来就不会动 tpot**,要降的是单个 kernel 的绝对耗时,而这需要换 kernel 本身(不是换参数)。

> 备注:记忆文档记录的 hipBLASLt version 字段是 `0.10.0`,本会话 `hipblaslt-bench` 报的是 `1000`(=1.0.0),两者 **git 版本号 `a6254b89-dirty` 完全一致 = 同一个二进制**,version 字段差异仅为不同 build 的显示格式,实测数据有效。

---

### 8.5 §8 最终结论 + 下一步转向(override 死刑后)

**§8 三问闭合总账**:
- §8.1(hipBLASLt tile 能否 env 调):**能调(override 文件存在),但调了无意义** —— m=1 problem 只有 1 个 algo。
- §8.2(capture 后实际 m 值):**m=1,未被 bucket pad 撑大**(推翻源码假设)。
- 248320 归属:**lm_head(vocab),非 GDN 投影**(旧稿误判已作废)。

**override 路线已封死后的候选方向**(要降 5.074s 的 `MT64x32x32_GSU1` GDN 主 GEMM,且 duty cycle 97.3% 已打满,只能动单 kernel 绝对耗时):

| 候选 | 机制 | 可行性预判 | 阻碍 |
|---|---|---|---|
| **A. 强制 batch padding 撑大 m** | 让 cudagraph 捕获时 m=128+ 而非 m=1,走 mid-size GEMM(多 algo 可选) | ⚠️ 改 vllm 行为,有吞吐/显存代价;且 m=128 的 GEMM 总耗时可能反升(matvec 1 次 vs 胖 GEMM 1 次,后者单次更慢但 amortize 更好——但 decode batch=1 下没东西可 amortize) | 与 vLLM v1 cudagraph bucket 逻辑冲突,需改源码 |
| **B. 换 backend(不走 hipBLASLt)** | m=1 matvec 改走 rocBLAS / 手写 triton kernel | ✅ **实测翻盘**(见 §8.8):rocBLAS 热稳态 166.8us vs hipBLASLt 638.9us,**快 ~3.8x** | rocBLAS bench 与 vllm 实跑路径不完全等价(bench 用 `gemm_ex`+contiguous,需验证 vllm 走 rocBLAS 后真实 tpot) |
| **C. 融合 kernel(优化点3)** | 投影 GEMM + conv1d + recurrent 融合,减 HBM round-trip | ✅ 不依赖换 GEMM algo,降的是 kernel 间访存而非 GEMM 本身 | 受编译环境限制,见优化点3 |
| **D. 接受现状** | duty cycle 97.3% + matvec 638.9us 可能已接近 gfx936 该形状下限 | ✅ 零成本 | 不降 tpot |

**主线判定**:override(§8.1)封死后,**重心从"调 GEMM 参数"转向"减 kernel 数量/访存"(优化点3 融合)或"换 problem 形状"(候选 A,但代价高)**。单纯在 m=1 matvec 上换 tile/splitK 已被实测证明无收益,不再投入。

> §8 闭合完成。优化点2 的"调 tile/heuristic"子目标已封死;优化点2 若要继续,只能转向候选 A/B(改 problem 形状或换 backend),属改 vllm 行为/源码范畴,非 env 可解。落地需另起设计。

---

### 8.6 候选 C(融合 kernel)收益上限离线分析 —— 死刑(2026-07-11 批3 trace 离线访存分析)

override 封死后,候选 C(融合 kernel,优化点3)一度被视为"不依赖换 GEMM algo、降 kernel 间访存"的可行方向。本节用批3 trace 离线分析钉死其收益上限,**结论:死刑,性价比极低**。

#### 8.6.1 方法 —— 复用批3 trace,无需新抓

候选 C 的论据是"投影 GEMM → conv1d → recurrent → norm → out_proj 之间多次 HBM round-trip,融合可减访存"。要量化收益,只需把单 step(48 GDN 层)的 kernel 耗时按类别拆开,看"GEMM 内部耗时"vs"GEMM 间衔接(可被融合省掉)耗时"的比例。**这些都能从已有批3 trace(`rank0.1783748982487161609.pt.trace.json.gz`)直接算出**,不需要新抓 trace。

数据源:批3 trace(cudagraph ON,decode batch=1),经 login→srun 链路在作业节点 `e03r1n11` 上用 `/usr/local/python3.12/bin/python3` 容器内就地分析(解压 19.4MB JSON,76305 events)。

#### 8.6.2 单 step(48 GDN 层)kernel 耗时构成

从批3 trace 中间窗口取一个完整 step(以 MT64x32x32 为锚点,跨 112 个 MT64x32x32 = 一个 decode step),按 kernel 类别归类:

| 类别 | 单 step 耗时 | 占比 | 次数/step | 说明 |
|---|---|---|---|---|
| **MT64x32x32_GSU1** | 44036 us | **65.4%** | 113 | `in_proj_qkvz` 主 GEMM(638.9us/次,实测唯一 algo index 4362) |
| **MT32x16x4** | 14605 us | **21.7%** | 129 | `in_proj_ba` + 小投影 GEMM |
| MT128x32x32 | 3609 us | 5.4% | 16 | 大投影 GEMM |
| triton_fused_* | 2682 us | 4.0% | 450 | 衔接算子(可被融合省掉的主力) |
| MT32x32x32(norm 小 GEMM) | 747 us | 1.1% | 48 | layernorm 内部小 GEMM |
| fused_recurrent_gated_delta | 631 us | 0.9% | 48 | recurrent 状态更新 |
| Cijk_B_PostGSU | 322 us | 0.5% | 64 | GEMM 后处理 |
| _causal_conv1d_update | 240 us | 0.4% | 48 | conv1d 状态更新 |
| kernel_unified_attention | 180 us | 0.3% | 16 | full attention 层(64 层中 16 个) |
| 其余(reduce_segments/reshape_and_cache/elementwise) | ~145 us | 0.2% | — | KV cache 写入 + 杂项 |
| **单 step GPU 总耗时** | **67302 us = 67.3 ms** | 100% | — | — |

**Duty cycle = 95.1%**(GPU kernel 总 2.146s / wall 2.255s,与记忆 `[[duty_cycle_kills_step_gap_theory]]` 的 97.3% 同量级,确认 GPU 打满)。

#### 8.6.3 收益上限判定 —— GEMM 占 87%,融合碰不到

候选 C 融合 kernel 能优化的,是 **GEMM 之间的衔接**(triton_fused_* + PostGSU + conv1d + recurrent + elementwise 这些"GEMM 间 HBM round-trip"部分),即上表中:

- triton_fused_*:4.0%
- Cijk_B_PostGSU:0.5%
- _causal_conv1d_update:0.4%
- 其余 elementwise/reshape:0.2%
- **可被融合省掉的上限 ≈ 5.1%**(单 step 67.3ms 里最多省 ~3.4ms)

而真正的两个大头都是 **GEMM kernel 内部耗时**,融合碰不到:

- **MT64x32x32_GSU1(qkvz)占 65.4%** —— GEMM 算力/访存都在 kernel 内部,融合的是 GEMM 之间衔接,不是 GEMM 内部。
- **MT32x16x4(ba)占 21.7%** —— 同上,GEMM 内部耗时。
- **两个 GEMM 合计占 87.1%**,且 §8.4.5 已证明 m=1 下只有 1 个 algo、splitK 无效、override 无的放矢 → 这 87% 已是该 problem 在 gfx936+hipBLASLt 下的下限。

> **结论:候选 C(融合 kernel)的理论收益上限只有 ~5%**(单 step 最多省 ~3.4ms),而开发成本高(需写 triton 融合 GEMM+conv1d+recurrent)、受编译环境限制、还要改 vllm 源码。**性价比极低,判死刑。**

#### 8.6.4 核心发现 —— 瓶颈是 GEMM 本身,不是衔接

这次离线分析把问题性质钉死了:**87.1% 的耗时在 GEMM kernel 内部**(qkvz 65.4% + ba 21.7%),而这些 GEMM 在 m=1 + gfx936 + hipBLASLt 单 algo 约束下已是硬件/库下限(§8.4.5)。即:

> **在 m=1 matvec + gfx936 + hipBLASLt 单 algo 的约束下,这 87% 的 GEMM 耗时已是该 problem 的下限。要继续降,只能改变 problem 形状本身(候选 A,但 batch=1 负收益)或换硬件/库(候选 B,待 `rocblas-bench` 验证)。融合 kernel 改变不了这个下限。**

---

### 8.7 §8 全候选总账 + 收尾建议

| 候选 | 机制 | 收益空间 | 判定 |
|---|---|---|---|
| §8.1 override / splitK | 换 algo index / 改 GSU | 0(单 algo,splitK 无效) | ❌ §8.4.5 实测死刑 |
| A 强制 batch padding 撑大 m | 走 mid-size GEMM(多 algo) | batch=1 decode 下负收益(算 127 个废 token) | ❌ 死刑 |
| **B 换 backend(rocBLAS/triton)** | 不走 hipBLASLt index 4362 | **rocBLAS 热稳态 166.8us,快 ~3.8x**(见 §8.8) | ✅ **翻盘,§8.8 已实测确认** |
| **C 融合 kernel(优化点3)** | 减 GEMM 间 HBM round-trip | **上限 ~5%**(GEMM 占 87% 碰不到) | ❌ §8.6 死刑(性价比极低) |
| D 接受现状 | 0 | 0 | ✅ duty cycle 95%+ 已打满 |

**收尾建议**(2026-07-11 §8.8 实测后更新):
1. ~~唯一值得继续的轻量验证 = 候选 B~~ → **已完成,翻盘**。rocBLAS 对 `(m=1, n=16384, k=5120, bf16)` 热稳态 166.8us,对比 hipBLASLt index 4362 的 638.9us **快约 3.8x**(§8.8)。原"m=1 单 algo 是 Tensile 唯一解 → 预期收益小"的预判**被推翻**:hipBLASLt heuristic 只给了 1 个慢 algo,而 rocBLAS 选到了快得多的 kernel。
2. **下一步重心 = 让 vllm 实际走 rocBLAS 而非 hipBLASLt 跑这条 m=1 matvec,验证真实 tpot 收益**。bench 的 166.8us 是 `rocblas_gemm_ex` + contiguous 布局下的最优 kernel,vllm 实跑路径(weight 布局、是否 epilogue/bias、cudagraph 捕获)未必完全等价,需实测 tpot 闭合。
3. **真正能动的只剩"改 problem 形状"范畴**(候选 A 的变体:多请求并发 decode 让 batch>1),属 vllm 调度/batching 策略,另起设计;但 §8.8 翻盘后优先级降低——单 backend 切换若兑现 3.8x,tpot 主瓶颈可能直接消除,无需动 batching。

> §8 至此:§8.1–§8.2 三问闭合 + §8.4.5 override 死刑 + §8.6 融合死刑 + **§8.8 rocBLAS 翻盘(唯一 actionable 收益方向)**。优化点2 的 env 可解子路径中,override/splitK/融合全部封死,但换 backend 这条子路径被实测打开,且收益量级远超其余所有方向之和。

---

### 8.8 候选 B 实测 —— rocBLAS 翻盘(2026-07-11)

§8.7 原判定候选 B"待测,预期收益小(m=1 单 algo 是 Tensile 唯一解)"。本节用 `rocblas-bench` 实测,**结论:翻盘,rocBLAS 比 hipBLASLt 快约 3.8x**。

#### 8.8.1 方法 —— rocblas-bench 直接测三个 GDN 投影形状

工具:`/opt/dtk-26.04-DCC2602-0317/lib/rocblas/benchmark_tool/rocblas-bench`(容器 `worker-0` 内,DTK 26.04,gfx936)。

注:该 bench 默认无执行位(`-rw-r--r--`),实测需先 `chmod +x`。

命令骨架(`-f gemm_ex -r bf16_r`,N/T 布局,compute f32_r,`-i 100~200`):
```
rocblas-bench -f gemm_ex -r bf16_r -m <M> -n <N> -k <K> \
  --lda <K> --ldb <N> --ldc <N> --transposeA N --transposeB T \
  --alpha 1 --beta 0 --a_type bf16_r --b_type bf16_r --compute_type f32_r -i 200
```

#### 8.8.2 实测数据 —— qkvz 热稳态 166.8us

| problem | m,n,k | hipBLASLt(唯一 algo 4362) | rocBLAS(热稳态) | 加速比 |
|---|---|---|---|---|
| **in_proj_qkvz** | 1,16384,5120 | **638.9 us** | **166.8 us** | **~3.83x** |
| in_proj_ba | 1,96,5120 | (trace MT32x16x4 113us/次量级) | 39.4 us | — |
| out_proj | 1,5120,6144 | (trace MT128x32x32 225us/次量级) | 1420 us | ⚠️ 见下 |
| lm_head | 1,248320,5120 | 8796 us | (超时未跑完,rocBLAS init 慢) | 待测 |

**in_proj_qkvz 热稳态确认**(连跑 5 次 `-i 200`):
```
307.82  →  167.32  →  166.985  →  166.725  →  166.78   us
```
首次 307us(冷启动),第 2 次起收敛到 **166.8us**,极稳。对应 Gflops ~1005(冷启动 545→1005)。

#### 8.8.3 关键发现 —— hipBLASLt heuristic 选了慢 kernel

原预判"m=1 单 algo 是 Tensile 唯一解 → rocBLAS 也不会更快"**被推翻**:

- hipBLASLt 对 `(m=1,n=16384,k=5120,bf16)` 的 heuristic 只返回 **1 个 algo**(index 4362,`MT16x16x16_..._GSU1_..._WGM1`),耗时 **638.9us**。
- rocBLAS 对同一形状选到了 **166.8us** 的 kernel,**快 3.83x**。
- 说明 **Tensile/hipBLASLt 的 heuristic 在这个 m=1 matvec 上选了一个远非最优的 kernel**(可能 index 4362 是为通用性/数值精度配的 WGM1 路径,而 rocBLAS 选到了更贴合 matvec 的纯访存优化 kernel)。

**这是 §8 全程最大的 actionable 发现**:不需要改 problem 形状、不需要写融合 kernel、不需要动 override——**单纯让 vllm 这条 m=1 matvec 走 rocBLAS 而非 hipBLASLt,理论就能把 qkvz 从 65.4% step 占比大幅压下来**。

#### 8.8.4 反例与待验证项 —— bench ≠ vllm 实跑

**两个反例提醒 bench 数据不能直接等同于 vllm tpot 收益**:

1. **out_proj rocBLAS 反而慢**(1420us vs trace MT128x32x32 ~225us):out_proj 形状 `(1,5120,6144)` 在 rocBLAS 下选到的 kernel 比 hipBLASLt 慢 ~6x。**说明"换 backend"不是无脑全胜——qkvz/ba 适合 rocBLAS,out_proj 适合 hipBLASLt,需要按形状分流**。
2. **lm_head rocBLAS 超时未跑完**:rocBLAS init 对超大 n=248320 的 kernel 选择/编译很慢,需单独长超时重测。

**下一步必做的实测**(闭合 §8.8 收益):
- **(必做)让 vllm 实际走 rocBLAS 跑 in_proj_qkvz,测真实 tpot**:bench 166.8us 是 `rocblas_gemm_ex` + contiguous 布局的最优 kernel;vllm 实跑路径(weight 可能 transposed/interleaved、是否带 bias epilogue、cudagraph 捕获开销)未必兑现 3.8x。需在 vllm 配置层强制这条 GEMM 走 rocBLAS(或环境变量切换 backend),抓新 trace 看 MT64x32x32_GSU1 是否被替换 + 测 tpot。
- **(必做)按形状分流**:out_proj 不能跟着切 rocBLAS(会慢 6x)。需确认 vllm 是否支持 per-op backend,或 hipBLASLt 是否有"对该 problem 回退 rocBLAS"的机制。
- **(可选)补 lm_head rocBLAS 数据**:长超时重测 `(1,248320,5120)`,确认 vocab 投影是否也吃 rocBLAS 翻盘红利。

#### 8.8.5 §8.8 结论

> **候选 B 翻盘**:rocBLAS 对 in_proj_qkvz(m=1 matvec 主瓶颈,占 step 65.4%)热稳态 166.8us,比 hipBLASLt 唯一 algo 638.9us **快 ~3.8x**。这是 §8 全程唯一 actionable 的大收益方向,且收益量级远超 override/融合/改 batching 之和。**但 bench 数据需经 vllm 实跑 tpot 闭合验证**(out_proj 反例证明不能无脑全切)。下一步主线 = 让 vllm 这条 GEMM 走 rocBLAS + 抓 trace 测真实 tpot。

### 8.9 torch 层复现 + 差异归因(2026-07-11)

§8.8 后遗留的疑问:`rocblas-bench`(直 C API `rocblas_gemm_ex`)报 166.8us,但用 `torch.backends.cuda.preferred_blas_library('cublas')`(=rocBLAS)跑 `F.linear` 此前只见 ~1050us,无法复现 3.8x。本节闭合该差异。

#### 8.9.1 rocblas-bench 稳定性复测

worker-0 容器内 `chmod +x` 后连跑 5 个独立进程(`-i 200`,N/T 布局,`(m,n,k)=(1,16384,5120)` bf16,compute f32_r):
```
proc1: ... 1006.01, 166.77
proc2: ... 1006.16, 166.745
proc3: ... 1005.53, 166.85
proc4: ... 1005.65, 166.83
proc5: ... 1005.26, 166.895   us
```
**5 进程首跑即 166.8us**(Gflops ~1006),无跨进程缓存假说 —— 之前观察到的"首跑 307us/1174us 冷启动"是 `-i` 内部 cold path,稳态 166.8us 完全可复现。无 `~/.config/rocblas` 缓存文件、`ROCBLAS_CACHE_DIR` 未设,排除缓存解释。

#### 8.9.2 torch 层复现 —— backend × 布局矩阵

`(m,n,k)=(1,16384,5120)` bf16,`iters=200~500`,20~50 次 warmup。`weight[n,k]` = vllm 原生布局(`F.linear = x[m,k] @ w.T`)。

| 调用 | 布局 | default | **cublas(=rocBLAS)** | cublaslt(=hipBLASLt) |
|---|---|---|---|---|
| `F.linear(x, w[n,k])` | vllm 原生 | 210 us | **238 us** | 210 us |
| `matmul(x, w.t())` | w.t 非连续 view | 202 us | 238 us | 201 us |
| `matmul(x, wt[k,n] cont)` | **等价 bench N,T** | 202 us | **124 us** | 201 us |
| `addmm(b, x, wt[k,n])` | 等价 bench N,T+bias | — | **127 us** | — |

#### 8.9.3 差异归因 —— 布局决定 kernel 选择

1. **rocblas-bench 的 166.8us 用的是 `wt[k,n]` 连续 + B 转置语义**(A=`x[m,k]` N,B=`w[n,k]` T → 等价 `x @ w.T`,但 bench 把 B 当成"逻辑转置、物理 `[k,n]` 连续"喂给 rocBLAS)。
2. **torch `matmul(x, wt[k,n] cont)` 在 cublas(rocBLAS) 下 = 124us**,甚至**比 bench 的 166.8us 还快**(torch addmm 127us 同量级)。**3.8x 翻盘在 torch 层完全可复现,且略优**。
3. **`F.linear(x, w[n,k])` 在 cublas 下 = 238us** —— 仍比 hipBLASLt 的 638.9us 快 ~2.7x,但不如 `matmul+wt` 的 124us。**差距来自 weight 布局**:`F.linear` 的 `w[n,k]` + 内部 `w.t()` view 会让 rocBLAS 选到不同(略慢)的 kernel,而 `wt[k,n]` 物理连续 + 显式转置语义让 rocBLAS 选到最优 matvec kernel。
4. **default/cublaslt 对布局不敏感**(均 ~200~210us):hipBLASLt 对该 m=1 形状只有 index 4362 一个慢 algo,无论怎么喂都 638.9us 级别(trace 实测)——torch 这里的 ~210us 比纯 hipblaslt-bench 的 638.9us 快,说明 **torch 的 cublaslt 路径走的不是 hipblaslt-bench 那个 heuristic**,而是 torch 自己的 hipBLASLt 调用(可能带了不同的 epilogue/布局规整),需另查;但不影响主线结论。

#### 8.9.4 对 vllm 的可操作性结论

- **rocBLAS 翻盘真实**,且 torch 层 `matmul(x, wt[k,n])` = 124us 比 bench 更优。**3.8x 量级成立**。
- **vllm 走 `F.linear(x, w[n,k])`**:`preferred_blas_library('cublas')` 下是 238us(仍 ~2.7x 快于 hipBLASLt 638.9us),**但要拿到 124us 的最优,需要把 weight 物理转置成 `[k,n]` 连续 + 用 `matmul`/`addmm` 调用**。
- **vllm 的 `MergedColumnParallelLinear`(in_proj_qkvz)weight 默认 `[n,k]` 连续**(out_features 在前)。要吃 124us,需:
  (a) `preferred_blas_library('cublas')` 全局切 rocBLAS;且
  (b) **要么改 weight 布局为 `[k,n]` 连续**(但会破坏 MergedColumn 的 q/k/v/z 切分语义,需谨慎),**要么验证 rocBLAS 对 `w[n,k]`+view 是否也能选到快 kernel**(目前看 238us,非最优)。
- **更可能的落地路径**:不动 weight 布局,直接 `preferred_blas_library('cublas')` 全局切 rocBLAS → qkvz 从 638.9us 降到 238us(~2.7x),out_proj 从 ~225us 涨到需实测(§8.8.4 bench 给 1420us 是 `[k,n]` 布局下的反例,`F.linear` 布局下未必);**out_proj 退化风险需在 tpot 实测中确认是否可接受,或按形状分流**。

#### 8.9.5 §8.9 结论

> 差异闭合:rocblas-bench 166.8us 在 torch 层 `matmul(x, wt[k,n] cont)` + `preferred_blas_library('cublas')` 下复现为 **124us**(更优),`F.linear(x, w[n,k])` 复现为 238us。**翻盘真实,3.8x 量级在 vllm 原生 `F.linear` 路径上至少兑现 ~2.7x**。下一步主线不变:全局切 rocBLAS 跑 vllm 实测 tpot,重点确认 out_proj 是否退化(若退化则按形状分流)。bench ≠ vllm 实跑的最后一道验证仍是 tpot 闭合。

### 8.10 trace 反转:vllm 实跑已选到快 hipBLASLt algo,3.8x 翻盘前提被推翻(2026-07-11)

§8.9 收尾后解析本地 `profile_traces/rank0.1783570538023481078.pt.trace.json.gz`(7月9日抓的 vllm 实跑 torch profiler trace,gfx936,236 generations,总 GPU kernel 7.80s),结论与 §8.8/§8.9 的前提**直接冲突**。

#### 8.10.1 trace 事实

- **vllm 实跑 in_proj_qkvz 的 hipBLASLt kernel = `Cijk_Alik_Bljk_BBH_MT64x32x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_..._WGM8`**,dur **~220us**(5664 次,median 231.8us,range 208~247us)。
- **不是** §8.1/[[hipblaslt-matvec-single-solution-no-splitk-gain]] 里 `hipblaslt-bench` heuristic 选的 index 4362 `MT16x16x16_..._WGM1`(638.9us)。
- 即:**vllm/torch 的 hipBLASLt 调用路径选到了比 hipblaslt-bench heuristic 快 ~3x 的 algo**(220us vs 638.9us)。bench 的"唯一解 index4362"是 hipblaslt-bench heuristic 的误判,不代表 vllm 实跑选的 algo。

#### 8.10.2 per-generation kernel 结构反推

trace 共 236 generations。`MT64x32x32_GSU1` 共 13216 次 = **56/gen** = 24 qkvz(dur 低档 ~220us)+ 24 out_proj + 8 ba(dur 高档 ~480us,median 493.6us)。MT64x32x32_GSU1 独占 5.07s / 7.80s = **65%**,仍是主瓶颈,这点不变。

fused triton `rocm_unquantized_gemm` 系列 kernel(LayerNorm/SiLU 把 gemm 融进去的小 kernel):5664(24/gen,~5us)+ 1888(8/gen)+ 118,非主 GEMM,与切换无关。

#### 8.10.3 收益重估 —— 3.8x 腰斩到 ~1.5x

| | hipBLASLt(bench index4362) | hipBLASLt(vllm 实跑 trace) | rocBLAS(bench) | rocBLAS(torch matmul wt[k,n]) |
|---|---|---|---|---|
| qkvz | 638.9us | **~220us** | 166.8us | 124us |

- §8.8/§8.9 的 3.8x 翻盘建立在"vllm qkvz 跑在 638.9us"上。**trace 证明 vllm 实跑已是 ~220us**。
- 切 rocBLAS 后 qkvz 收益:220 → 124~166us = **~1.3~1.8x**(非 3.8x)。
- step 总 tpot 收益:qkvz 占 step 65%,压缩 1.5x → step 理论降 ~22%(非此前估算的 ~50%)。

#### 8.10.4 out_proj 退化风险更显性

trace 实测 out_proj/ba 一档 `MT64x32x32_GSU1` dur ~480us(24+8=32/gen)。若全局切 rocBLAS,out_proj 按 §8.8.4 bench 反例会涨到 1420us(~3x 退化),24/gen×480us → 24/gen×1420us,**净增 24/gen×940us**,可能把 qkvz 省下的(24/gen×(220-124)=24/gen×96us)完全抵消并倒亏。**按形状分流(qkvz→rocBLAS,out_proj/ba→留 hipBLASLt)是必须的,且收益窗口比 §8.9 设想的更窄**。

#### 8.10.5 §8.10 结论 + 修正后的下一步

> **trace 反转**:vllm 实跑 qkvz 已是 hipBLASLt `MT64x32x32_GSU1_WGM8` ~220us,非 bench index4362 638.9us。rocBLAS 切换收益从 3.8x 缩到 ~1.5x(qkvz 220→124~166us),step tpot 理论收益从 ~50% 降到 ~22%。**out_proj 退化风险足以吞掉收益**,全局切不可行,必须按形状分流。

修正后的下一步(替代 §8.9.5 的主线):
1. **(优先,低风险)per-op 验证**:在 vllm 里单独把 qkvz 这条 `F.linear` 切到 `preferred_blas_library('cublas')`(rocBLAS),抓 trace 确认 220us 是否真降到 ~166/238us。**先验证收益存在,再改源码**。
2. **(必须)按形状分流实现**:源码层 qkvz 走 rocBLAS、out_proj/ba 留 hipBLASLt。需解决 [[源码 vs site-packages 不一致]] —— site-packages 有 [DCU Optimize] 分支而 vllm_cscc 源码没有,改源码前先确认容器实际跑的是哪份。
3. **(铁律)bench ≠ vllm 实跑**:hipblaslt-bench 选 index4362(638.9us),vllm 实跑选 MT64x32x32(220us)。**以后所有 bench 推断必须用 trace 校验实跑 kernel 名**,不能只信 bench heuristic。

---

### 8.11 🔧 终局订正:qkvz 实走 LLMM1 127us,§8.4–§8.10 全链路推翻(2026-07-12)

> 本节是本文档的**权威订正层**。§8.1–§8.10 的整套"qkvz 走 hipBLASLt"推理链建立在两个错误前提上:(1) §0.5.3 旧稿断言 `on_gfx9()=False` → `use_skinny=False` → 走 hipBLASLt;(2) §8.4.5/§8.10 用 `hipblaslt-bench` / 旧 trace 推断 qkvz 的 hipBLASLt algo。两个前提都被本轮源码 + torch profiler 实测推翻。凡 §8.4–§8.10 与本节冲突处,**以本节为准**。

#### 8.11.1 源码事实(决定性)

worker-0 容器 `site-packages/vllm/model_executor/layers/utils.py`(实跑版本,带 `[DCU Optimize]` 标记),`rocm_unquantized_gemm_impl` 关键路径:

```python
use_skinny = (
    envs.VLLM_ROCM_USE_SKINNY_GEMM            # 默认 True
    and (on_gfx9() or on_gfx936())            # ← gfx936 命中(旧稿漏看 or on_gfx936())
    and rocm_skinny_ops_available()           # True
    and x.dtype in [torch.float16, torch.bfloat16]
    and k % 8 == 0
)
if use_skinny is not True:
    return torch.nn.functional.linear(x, weight, bias)   # ← hipBLASLt 回退(旧稿以为走这里)
if on_gfx936():
    if n == 1 and m % 4 == 0 and k <= 8192 and bias is None:
        out = ops.LLMM1(weight, x_view, 4)               # ← qkvz/ba/out_proj 实际命中
        return out.reshape(...)
    return torch.nn.functional.linear(x, weight, bias)
```

- `on_gfx936()` 在 `platforms/rocm.py` 中存在且容器内返回 **True**(`_GCN_ARCH='gfx936'`)。旧稿只查 `on_gfx9()`(列表不含 gfx936),**漏看 `or on_gfx936()`**,得出 `use_skinny=False` 的错误结论。
- 三个 GDN 投影 `bias=False`(qwen3_next.py:510/537/559,§0.5.3 旧稿 bias 钉死段结论仍有效)→ `bias is None` 成立。
- qkvz(n=1,m=16384,k=5120)、ba(n=1,m=96,k=5120)、out_proj(n=1,m=5120,k=6144)三者均满足 `n==1 and m%4==0 and k<=8192 and bias is None` → **全部命中 `ops.LLMM1`**。

#### 8.11.2 torch profiler 实测(铁证)

worker-0 容器内,用 `dispatch_unquantized_gemm()`(= `rocm_unquantized_gemm` 注册 op,即 vLLM 投影 GEMM 的实际分发函数)在 qkvz 形状 `(m=1, n=16384, k=5120, bf16, no bias)` 上抓 torch profiler,key_averages 设备耗时:

```
380.0us x3   void LLGemm1_kernel<c10::BFloat16, 4>(...)
```

三 call 共 380us → **单 call 127us**,命中的核是 **`LLGemm1_kernel`**,**不是** `MT64x32x32_GSU1`,**不是** hipBLASLt 任何 algo。

三方对照基准(同形状,20~50 次 warmup 后取稳态):

| 调用路径 | 后端 | 耗时 |
|---|---|---|
| `dispatch_unquantized_gemm()`(vLLM 实跑路径) | LLMM1 | **127.7us** |
| `F.linear(x, w[n,k])` | hipBLASLt | 229.2us |
| `F.linear(x, w[n,k])` + `preferred_blas_library('cublas')` | rocBLAS | 230.3us |
| `matmul(x, wt[k,n].contiguous())` | rocBLAS | 124.4us |

#### 8.11.3 推翻的旧结论清单

| 旧结论(位置) | 旧值 | 订正 |
|---|---|---|
| §0.5.3/§0.5.4 候选 D:"投影 GEMM 走 hipBLASLt 回退" | hipBLASLt | ❌ 推翻 → 实走 **LLMM1** |
| §5.0.0:"qkvz = MT64x32x32_GSU1(5.074s)" | MT64x32x32_GSU1 | ❌ 推翻 → qkvz = **LLGemm1_kernel 127us**;MT64x32x32_GSU1 是别的 GEMM |
| §8.4.5:"qkvz 走 hipBLASLt index 4362 = 638.9us" | 638.9us | ❌ 推翻 → bench 给的是 hipBLASLt 上的耗时,vLLM 根本没把 qkvz 喂给 hipBLASLt;实跑 127us |
| §8.8/§8.9:"rocBLAS 翻盘 3.8x,切 backend 有收益" | 3.8x / ~1.5x | ❌ 推翻 → qkvz 已在 LLMM1 127us,切 rocBLAS(230us)反而退化,切 hipBLASLt(229us)也退化;唯一略快是 matmul+wt[k,n] 124us(3us,不值得改布局) |
| §8.10:"vllm 实跑 qkvz 走 MT64x32x32_GSU1 ~220us" | 220us | ❌ 推翻 → trace 里的 MT64x32x32_GSU1 不是 qkvz;qkvz 实跑 = LLGemm1_kernel 127us |
| 任务 #13 前提:"qkvz 在 hipBLASLt 上,切 rocBLAS 验证 ~1.5x 收益" | — | ❌ 推翻 → **任务 #13 作废** |

#### 8.11.4 真瓶颈重新定位 —— MT64x32x32_GSU1(490us)+ MT32x16x4(144us)

新 trace(`rank0.1783777442602026742.pt.trace.json.gz`,4.2MB,163326 设备内核,11198s 总时长)顶级内核:

| kernel | 总耗时 | 次数 | median/call |
|---|---|---|---|
| `MT64x32x32_GSU1` | 7307s | x18704 | **489.9us** |
| `MT32x16x4_GSU1` | 2439s | x21543 | 143.8us |
| `MT128x32x32_GSU4` | 604s | x2672 | 225.9us |
| `MT32x32x32_GSU8` | 126s | x8016 | 15.5us |
| `LLGemm1_kernel`(qkvz/ba/out_proj) | — | — | ~127us(不在 top-25) |

- qkvz/ba/out_proj(LLMM1,127us 量级)**不在 top-25** → 三个投影 GEMM 已被优,不再是瓶颈。
- **真瓶颈是 `MT64x32x32_GSU1`(490us/call,占 65%)和 `MT32x16x4_GSU1`(144us/call,占 22%)**,二者都是 hipBLASLt contraction kernel(命名 `Cijk_*_MT*_GSU*`),但**不是 GDN 三个投影**(投影走 LLMM1)。
- **归属待 op 归因**:`MT64x32x32_GSU1` / `MT32x16x4_GSU1` 对应哪个 GEMM(lm_head? FFN gate/up/down? attention qkv?)需用 op 标签 trace(在 vLLM 各 Linear 层包 `record_function`)1:1 钉死。这是下一步唯一 actionable 的方向。

#### 8.11.5 订正后的下一步

1. **任务 #13 作废**(切 rocBLAS 对 qkvz 无收益,反而退化)。
2. **新主线 = op 归因 `MT64x32x32_GSU1`(490us)+ `MT32x16x4_GSU1`(144us)**:在 vLLM 里对 lm_head / FFN / attention qkv 等候选 Linear 层包 `torch.profiler.record_function`,抓带 op 标签的 trace,1:1 锁定这两个 kernel 的归属。归因清楚后再决定优化对象与手段(可能仍是 LLMM1 类 skinny 核、或 hipBLASLt heuristic 调参、或融合)。
3. **铁律重申**:所有 GEMM 后端归属判断必须用 torch profiler 实测核名校验,不能只靠源码静态推断(旧稿 `on_gfx9()` 漏看 `or on_gfx936()` 就是静态推断翻车的教训),也不能只靠 bench heuristic。

> §8.11 为本文档 GDN GEMM 后端归属的最终结论。§0.5.3 旧稿、§0.5.4 候选 D、§5.0.0、§2.3、§3 优化点1/2、§8.4–§8.10 中与本文冲突的部分,均以 §8.11 为准。

