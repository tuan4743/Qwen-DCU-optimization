# 16 · gfx936 专用 split-K GEMM 算子任务清单(严格按序执行)

> **创建:2026-07-13。**
> **目的**:用户指令"现在需要最短时间获得巨大收益,前面调参浪费巨量时间。改成什么样都能接受。必须登容器,最终编译启动 vllm 验证用户来更快,其它我直接做。先写任务文档严格按它写避免后面打架方向乱套。"
> **背景定性(文档 15 收尾结论)**:源码层只剩 C1(rocm.py:149 加 gfx936 打开 wvSplitK 链)= 80.02 分。剩余第一瓶颈 = FFN gate_up_proj(n=1,m=34816,k=5120,bf16,bias=False),decode 下走 `ops.wvSplitK`(黑盒预编译核,实现不在 vllm_cscc 树,在 `_rocm_C.so`)。调参路径(C4-3 绕过/C5b 段数/C7 cu_count/C8 wvSplitKrc/C9b)全部走完,要么中性要么作废要么灾难。**调参到头,必须写算子。**
> **硬件事实(用户提供,保真)**:
> - gfx936(BW3000),80 CUs,64GB HBM。
> - **Triton 适配相当不好,理论峰值 490T 实际不到 1/3** → 所有 Triton 路线(含 wvSplitK 若是 Triton 实现的话)在 gfx936 上低效。**新算子不能走 Triton,必须走 HIP C++ / 原生 MFMA。**
> - **VGPR 768 个** → 每线程可扛大寄存器压力的 tile,m=1 matvec 应让每 thread 沿 K 维扛一大段(用满 VGPR 换带宽吞吐)。
> **唯一权威源码**:本地 `vllm_optimize_data/vllm_cscc` 副本 + DCU `/public/home/xdzs2026_c150/zya/vllm_cscc`。**绝不看容器 dist-packages。**
> **baseline**:C1(wvSplitK 链打开)实测 4-8K=18.26 / 8-16K=12.30 / 16-32K=8.61 tok/s(文档 15 §C5b 表,加权 80.04;⚠️ 注意 §80.02 订正:C5b 标签作废,16 是出厂值,所以这套数字的真实基线 = C1)。
>
> **本文档铁律**:
> 1. **严格按 T1→Tn 顺序执行,不跳步,不跑偏**。每个任务有进入/退出条件,不满足不进下一步。
> 2. **写算子必须先 bench 验证再集成**——绝不直接改 vllm 分发逻辑跑端到端。微原型单 shape bench 跑通且快于 wvSplitK 后,才进集成阶段。
> 3. **所有 bench 用 torch profiler 实测核名校验**(文档 12 铁律 E),不靠纸面推断。
> 4. **每一步结论写回本文档对应 §,不打架**。方向有变先改文档再动手。

---

## 已确认事实(决策依据,不再重测)

- **目标 GEMM**:FFN gate_up_proj。shape:batch=1(decode)→ 传给底层的是 `(n=1, m=34816, k=5120)`(按 vllm `rocm_unquantized_gemm` 签名:n=x.numel()//x.size(-1)=1=batch, m=weight.shape[0]=34816=out, k=weight.shape[1]=5120=in)。bf16,bias=False。
- **当前路径(文档 15 §C4 闭合)**:gate_up 经 `rocm_unquantized_gemm_impl`(utils.py:122-188),C1 后 `use_skinny=True`,命中 `if m>8 and 0<n<=4`(utils.py:181)→ `ops.wvSplitK(weight, x_view, cu_count, bias)`。**wvSplitK 实现在 `_rocm_C.so`,源码不在 vllm_cscc 树,grep .cu/.hip/.cpp 无源文件,黑盒。**
- **调参死路(文档 15 已闭合)**:
  - C4-3(lm_head 绕过 wvSplitK 走 F.linear)= 灾难,已回滚。gate_up 同理不能绕过走 F.linear。
  - C7(调 wvSplitK 的 cu_count)= 盲目调参,黑盒。
  - C8(wvSplitKrc)= 底层 .so 白名单直接拒 m=34816(`Unsupported N value: 34816,5120,1`)。
  - C5b(段数)= 标签作废,16 是出厂值。
- **硬件**:gfx936,80 CUs,64GB HBM,VGPR 768,Triton 实测不到 1/3 峰值。
- **文档 12 §N6 算过**:gate_up 走 rocBLAS F.linear 时 506us,带宽只用 22%(703/3200 GB/s)→ **非带宽 bound,是 tile/launch/CU 占用问题**。GSU1(GridSplitU=1)→ 单网格没切分,CU 没铺满。**这是 split-K 有空间的理论支撑。**

---

## 任务清单(严格按序)

### T1. 容器环境探明:gfx936 上能否编一个朴素 HIP MFMA bf16 kernel 并跑

> **目的**:写算子前先确认工具链。Triton 不行(用户已确认),得走 HIP C++。但文档 12 §B5 说 skinny C++ 宏在 gfx936 是空壳、§B7 说 FP8 segfault——这些是"特定路径"的问题,不代表"gfx936 编不了朴素 HIP MFMA"。必须实测确认基础编译能力,否则后面全白搭。

- **动作**:
  1. 登容器(三层 ssh,节点/容器 IP 按当前作业查,见文档 08)。
  2. 查 DTK 工具链版本:`hipcc --version`、`rocminfo | grep -i gfx936`、`/opt/dtk-*` 路径。
  3. 写一个 ~50 行的**最朴素** HIP kernel:单个 `__global__` 函数,用 `__builtin_amdgcn_mfma_f32_16x16x16_bf16`(或等价 intrinsic)做一个固定 shape 的 bf16 matvec,`hipLaunchKernel` 跑起来,`hipDeviceSynchronize`,printf 结果正确性。
  4. **不接 vllm,不接 rocBLAS,纯 standalone micro**。只验证"能编、能跑、结果对"。
- **进入条件**:—
- **退出条件**:三选一明确:
  - (a) **能编能跑结果对** → T2 继续,走 HIP C++ 路线。
  - (b) **能编但 MFMA intrinsic 不支持 gfx936** → 降级用普通 `__hfma2` 标量路径(慢但能跑),T2 评估是否仍值得。
  - (c) **根本编不了 gfx936 二进制** → 停,报告用户,改方向(可能只能回去优化 wvSplitK 调参或换别的瓶颈)。
- **禁止**:不接 vllm。不写 split-K。不跑端到端。不改 vllm 源码。
- **产出**:结论 + 编译命令 + 跑通的 kernel 代码贴回 §T1。

### T2. micro-bench 原型:gate_up shape 的朴素 split-K GEMM,对比 wvSplitK

> **目的**:在 standalone micro 里写一个针对 gate_up (n=1,m=34816,k=5120,bf16) 的 split-K GEMM,实测它的耗时,跟当前 wvSplitK 对比。**只有这个数 beat wvSplitK,才值得进集成阶段。** 这是"bench 时间花得值"的核心——先把单 shape 跑通有数,再谈改 vllm。

- **动作**:
  1. 基于 T1 的编译能力,写一个 split-K matvec kernel:
     - 沿 K=5120 切成 `split_k` 段(先试 8/16,每段 640/320)。
     - 沿 N=34816 切块铺 80 CU。
     - 每 thread 沿 K 段内累加,用满 VGPR(768 → 每 thread 扛比如 256 个 bf16 = 128 寄存器)。
     - 段间结果写中间 buffer,第二个 kernel reduce,或单 kernel atomic add(先试哪种简单)。
  2. 生成随机 weight(34816×5120 bf16)+ input(5120 bf16),warmup + 跑 1000 次取 median。
  3. **同环境同 shape 直调 `ops.wvSplitK`(需 vllm import,见文档 15 §C8 op 注册问题)**——如果 op 注册调不出,退而求其次用 rocBLAS F.linear 的 506us(文档 12 §A6)作对比基线,**或用文档 15 §C1 实测的端到端 tpot 反推**。
  4. torch profiler 抓 kernel 名,确认跑的是自己写的核不是被 PyTorch 路由到 rocBLAS。
- **进入条件**:T1 退出 (a) 或 (b)。
- **退出条件**:拿到三个数:
  - 我写的 split-K kernel 的 median 耗时(us)。
  - wvSplitK(或 F.linear fallback)同 shape 的 median 耗时(us)。
  - 我 kernel 的正确性(max abs diff vs torch.matmul ref < 1e-2 bf16 容差)。
  - **判定**:我的 < wvSplitK 且正确 → T3 继续;我的 ≥ wvSplitK → 回 T2 调 split_k/VGPR/tile 参数,3 轮仍不 beat → 停报告。
- **禁止**:不接 vllm。不改 vllm 源码。不跑端到端。
- **产出**:三个数 + kernel 代码 + split_k 取值 + §T2 判定。

### T3. 用户 checkpoint:原型数 beat wvSplitK,确认进集成

- **动作**:把 §T1/§T2 结论(能编能跑 + beat wvSplitK 的数 + 正确性)给用户。
- **进入条件**:T2 判定为"beat"。
- **退出条件**:用户明确"进集成"或"再调原型"或"换方向"。
- **禁止**:用户没确认前不动 vllm 源码。

### T4. 集成:把 split-K kernel 注册成 vllm op,接进 gate_up 分发

> **目的**:原型单 shape 跑通后,把它接进 vllm 的 `rocm_unquantized_gemm` 分发,让 gate_up 走我的核而不是 wvSplitK。

- **动作**:
  1. 把 T2 的 HIP kernel 编进一个独立 .so(或塞进 vllm 的 _rocm_C 构建,看哪个简单)。先试独立 .so + `torch.ops.load_library` 注册成自定义 op,**不动 vllm 原生 _rocm_C.so**(降低风险)。
  2. 改 `vllm_cscc/vllm/model_executor/layers/utils.py` 的 `rocm_unquantized_gemm_impl`:在 `if m>8 and 0<n<=4`(wvSplitK 分支)**之前**,加一条 gate_up 专属分支:`if on_gfx936() and m==34816 and n==1 and k==5120 and bias is None: 走我的 op`。**精确 shape 守卫,只 gate_up 命中,其它 GEMM 不受影响**(lm_head/down/qkvz 仍走 wvSplitK,避免重蹈 C4-3 灾难)。
  3. `cd vllm_cscc && python setup.py bdist_wheel`。
- **进入条件**:T3 用户确认进集成。
- **退出条件**:wheel 构建成功,日志无 error。
- **禁止**:不宽放 shape 守卫(只 gate_up)。不动 wvSplitK 分支本身的条件。不关 cudagraph。

### T5. 安装 + 重启 + 实测(用户做启动验证)

- **动作**:
  1. 我:`pip install --force-reinstall --no-deps dist/vllm-*.whl` → 校验 site-packages 的 utils.py 含我的守卫分支(避免重蹈 §C4-3 site-packages 没更新事故)。
  2. **用户**:kill 旧 vllm → 重启 start_vllm.sh → 跑三段(4-8K/8-16K/16-32K)实测 out_throughput/TPOT/TTFT。
- **进入条件**:T4 wheel 构建成功 + site-packages 校验干净。
- **退出条件**:用户反馈三段实测数。
- **禁止**:site-packages 没校验干净不让用户重启(§C4-3 教训)。

### T6. 对比 baseline,记录,决定下一步

- **动作**:把实测 vs C1 baseline(18.26/12.30/8.61)对比,写回 §结果表。
  - 正收益 → 保留,评估是否扩展到 down/qkvz 等其它 shape(T4 守卫放宽)。
  - 退化/无效 → 回滚守卫分支(一行 if),回 T2 调原型参数或换方向。
- **进入条件**:T5 用户实测数。
- **退出条件**:结果记录 + 用户确认下一步。
- **禁止**:不记录就继续。擅自连测多个不汇报。

---

## §T1 结论(2026-07-13,登容器 e03r2n12 / worker-0 173.0.233.2 实测)

### 工具链事实
- **DTK**:`/opt/dtk` → `dtk-26.04-DCC2602-0317`,HIP 6.2.0-0(注:torch 是 hip 6.3,版本号口径不同,无关)。
- **hipcc**:`/opt/dtk/bin/hipcc`(→ `/opt/dtk/hip/bin/hipcc`,ELF 二进制,非脚本)。
- **clang**:`/opt/dtk/lib/llvm/bin/clang` → clang **18.0.0**。
- **关键坑**:hipcc 默认找 clang 的路径是 `/opt/dtk/hip/lib/llvm/bin/clang++`,**该目录不存在**(`No such file or directory`)。必须在编译前 `export HIP_CLANG_PATH=/opt/dtk/lib/llvm/bin` 才能找到 clang。**未设此环境变量 → 所有 hipcc 编译失败报 `No such file or directory`。已踩。**
  - vllm `setup.py` / torch `cpp_extension` 是否自动设此变量 → T1 退出条件相关,见下。
- **MFMA 硬件支持**:`llc -mcpu=gfx936 -mattr=help` 列出 `mai-insts`(mAI 矩阵指令)、`dot1~dot10-insts`、`fp8-insts`、`mmop-fp8-insts`、`gfx936-mls-insts`(gfx936 专属 matrix load)等。**gfx936 有 MFMA 硬件(`mai-insts`),不是空壳。**

### MFMA intrinsic 实测(关键发现) — ⚠️ 2026-07-13 二次订正:VMFault 根因 = gfx936 不支持任何 MFMA 指令

- **rocBLAS / hipBLASLt 反汇编铁证**(2026-07-13,容器实测,`llvm-objdump --disassemble --mcpu=gfx936 librocblas.so.4.3` 709MB,1.3M 行;`libhipblaslt.so.0.10` 同):
  - **两个库的反汇编里 `v_mfma_*` 指令出现次数 = 0**(grep `v_mfma` 全库零命中)。
  - GEMM 主力乘加指令是 **`v_madmk_f16`**(rocBLAS 101192 次 / hipBLASLt 2947 次,绝对主力)+ `v_madak_f32` / `v_madmk_f32` / `v_fmac_f64` / `v_mac_f32`。**全是标量/向量 FMA,无任何矩阵 MFMA。**
  - 唯一一条 `v_mmac_16x16x4_f64`(f64 矩阵)出现 1 次,非 bf16 GEMM 路径。
  - **bf16 专用指令 0 条**(grep `*bf16*` 零命中)→ bf16 GEMM 是靠 `v_madmk_f16` + bf16→f16 拆位实现的。
- **直接结论:gfx936 这张卡(BW3000)的 MFMA 矩阵指令(`v_mfma_f32_16x16x8bf16` / `v_mfma_f32_16x16x16f16`)在运行时非法**。clang 能编出该指令(`v_mfma_f32_16x16x8bf16 a[0:3], v6, v7, a[0:3]` 汇编生成正常,accvgpr 约束 `=a`/`a` 正确),但 **GPU 执行时报 `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` + KERNEL VMFault**(2026-07-13 `mfma_ok.cpp` 用 `hipLaunchKernelGGL` + 正确 accvgpr 约束仍 VMFault,exit 134)。即:**指令编码合法但硬件不执行**——gfx936 的 `mai-insts` mattr 是 clang 侧的"软支持"(能编),硬件实际未实现该矩阵单元(或被厂商关闭)。
- **rocBLAS 在 gfx936 上能跑 390.7us GEMM,靠的就是 `v_madmk_f16` 标量 FMA 流水**,不是 MFMA。这是这张卡的"真实算力路径"。
- **f16 MFMA builtin 之前能编**(`__builtin_amdgcn_mfma_f32_16x16x16f16` 存在),但**运行同样会 VMFault**(sk3e.cpp 隔离测试已证:无内存读的纯 MFMA+accvgpr kernel 照样 VMFault)。**所有 MFMA 路径在 gfx936 运行时死刑。**

### T1 退出条件判定(二次订正):走 (b) 标量 FMA 路线,放弃 MFMA

- (a) "能编能跑结果对":**朴素 HIP 标量 kernel ✅**(sk2.cpp 459.4us 跑通,diff=0);**MFMA ✗**(编得过但运行 VMFault)。
- (b) "MFMA intrinsic 不支持 gfx936":**确认**——不是 builtin 缺,是**硬件不执行 MFMA**。降级到标量 FMA 路径。
- **判定:进 T2,路线定为 `v_madmk_f16` 标量 split-K(对齐 rocBLAS 同款指令)**。bf16→f16 转换 + 标量 f16 FMA 累加(split-K 沿 K=5120 切段,每 thread 沿 K 段用满 VGPR)。
- **关键策略调整**:既然 rocBLAS 用同款 `v_madmk_f16` 都能跑到 390.7us,我手写 split-K 的优化空间不在"换更猛的指令"(没有更猛的),而在 **split-K 沿 K 切段铺满 80 CU + 用满 VGPR 768 做寄存器内累加,把 rocBLAS 的 launch/tiling 效率问题吃掉**(文档 12 §N6:rocBLAS 只用 22% HBM 带宽,非带宽 bound,是 tile/launch 效率问题,这是手写 split-K 的理论窗口)。
- **T2 前置**:
  1. hipcc 编 `v_madmk_f16` 标量路径**不需要** `+mai-insts`(标量 FMA 是 gfx936 基础指令),`export HIP_CLANG_PATH=/opt/dtk/lib/llvm/bin` 仍必须。
  2. bf16→f16 转换:bf16 与 f16 位布局不同(指数位宽不同),**不能位 reinterpret**,需真转换。但 f16 只有 10 位尾数,bf16 有 7 位尾数+8 位指数——bf16→f16 会丢精度。**需评估**:gate_up 用 f16 累加是否在 bf16 容差内(目标 <1e-2 abs diff)。若不可接受,改走 **f32 累加的标量乘加**(`v_madmk_f32` + bf16→f32 拆位),rocBLAS 也在用 f32 mad(7907 次)。
  3. 标量 split-K 已有原型 sk2.cpp(459.4us),T2 在其基础上调 split_k/VGPR/tile 争取 beat rocBLAS 的 390.7us(或文档 15 的 506us F.linear 基线)。

## §T1b 结论(2026-07-13,登容器 e03r2n11 / worker-0 173.1.15.7 实测)— gfx936 bf16 mmac 可用,推翻 T1"放弃矩阵指令"方向

### 颠覆性发现:v_mfma 与 v_mmac 是两套指令,支持情况相反

T1 的结论"gfx936 拒绝所有 MFMA 矩阵指令"**只对 AMD 标准 `v_mfma_*` 成立,对海光自有 `v_mmac_*`(du_mma.hpp 封装)不成立**。用户搜索发现 `/opt/dtk` 大量 `builtin_amdgcn_mmac` 输出,贴出 du_mma.hpp,工程师确认"mmac 只有三种形状:16 16 4,16 16 8,16 16 16",要求重核查 mmac 路径。重核查后实测:

- **`v_mmac_f32_16x16x16bf16` 在 gfx936 上能编能跑结果对** ✅(2026-07-13 容器实测)。
- 目标特性是 `mmop2-insts`(非 `mai-insts`)。**只有 DCC clang17 `/opt/dtk/dcc/bin/clang++` 有 mmac builtin**,主 clang18 `/opt/dtk/lib/llvm/bin/clang` 无(hipcc 用主 clang18,编 mmac 失败)。这是 T1 当初没发现 mmac 可用的根因——用错了 clang。

### 三验证结果(mmac_official2.cpp,官方 du_load/du_mma/du_store 框架)

- **能编** ✅:DCC clang17 + `--rocm-path=/opt/dtk --rocm-device-lib-path=/opt/dtk/amdgcn/bitcode` + `-I/opt/dtk/include` + 4 个 fp8 宏占位修 host pass 编译坑 → `mmac_off` 二进制 74928 字节。
- **能跑** ✅:`LAUNCH_ERR=0(no error)` — 不 VMFault(与 v_mfma 路径形成铁证对比,后者必 VMFault exit 134)。
- **结果对** ✅:`MAXDIFF=0.0000 FIRST_BAD=-1`,SAMPLE `C[0,0]=0.0` / `C[8,0]=128.0` / `C[15,15]=240.0` 全等于期望(16x16x16 bf16 矩阵乘,A[i,j]=i,B 全 1 → C[i,j]=16*i)。

### micro 代码(mmac_official2.cpp 关键段)

```cpp
#include "du_mma.h"
#include "du_mma.hpp"
using namespace du::dumma;
// host pass 宏占位(device pass 不进入此块):补全全部 14 个 mmac 宏 = (c)
#ifndef __gfx936__
#ifndef __gfx938__
#define __DU_MMA_F32_16x16x16BF16(a,b,c) (c)  // ... 其余 13 个同理
#endif
#endif
__global__ void mmac_kernel(const __hip_bfloat16* A, const __hip_bfloat16* B, float* C, int N){
    DUFragment<matrix_a, 16, 16, 16, __hip_bfloat16, row_major> aFrag;
    DUFragment<matrix_b, 16, 16, 16, __hip_bfloat16, col_major> bFrag;
    DUFragment<accumulator, 16, 16, 16, float> cFrag;
    du_load_matrix_sync(aFrag, A, N);
    du_load_matrix_sync(bFrag, B, N);
    cFrag.x[0]=cFrag.x[1]=cFrag.x[2]=cFrag.x[3]=0.f;
    du_mma_sync(cFrag, aFrag, bFrag, cFrag);
    du_store_matrix_sync(C, cFrag, N, mem_row_major);
}
```

### 编译/运行命令(后续算子复用)

```
# 编译(必须 DCC clang17,不能 hipcc/主 clang18)
/opt/dtk/dcc/bin/clang++ -x hip --offload-arch=gfx936 \
  --rocm-path=/opt/dtk --rocm-device-lib-path=/opt/dtk/amdgcn/bitcode \
  -I/opt/dtk/include -I/opt/dtk/include/hip -D__HIP_PLATFORM_AMD__ \
  -L/opt/dtk/hip/lib -lamdhip64 \
  -Wno-return-type -Wno-unused-result -Wno-unused-variable -Wno-deprecated-declarations \
  mmac_official2.cpp -o mmac_off
# 运行
export LD_LIBRARY_PATH=/opt/dtk/hip/lib:$LD_LIBRARY_PATH; ./mmac_off
```

### host pass 编译坑(关键,后续算子必踩)

du_mma.hpp 模板由 `#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)...` 保护。host pass(__HIP_DEVICE_COMPILE__ 未定义)扫所有形状模板实例化,而 mmac 宏只在 gfx936/938 device 段定义 → host pass 报 "undeclared identifier '__DU_MMA_F32_16x16x16BF16'" 等 8-12 个错。**修复**:源文件顶部 `#ifndef __gfx936__ #ifndef __gfx938__` 块里给全部 14 个 mmac 宏空占位 `#define __DU_MMA_...(a,b,c) (c)`(只满足 host pass,device pass 不进入)。已实测:仅补 4 个 fp8 宏不够(还报 F32_8F32/TF32/F16/BF16/I8/U8/I4/U4/F64/F32_4 等),**必须补全全部 14 个**。

### T1b 退出条件判定:T2 方向改为 mmac 16x16x16 bf16 split-K

- T1 当初"走 (b) 标量 FMA `v_madmk_f16`"的判定**作废**——那是基于"gfx936 无任何矩阵指令"的错误前提。
- **新判定:T2 用 `v_mmac_f32_16x16x16bf16`(du_mma.hpp)做 split-K**。每指令算 16×16×16=4096 个乘加,远快于标量 FMA 流水。split-K 沿 K=5120 切段铺满 80 CU,每段 mmac 矩阵累加。
- **铁律更新**:绝对不用 `__builtin_amdgcn_mfma_*`(VMFault)。手写算子走 `v_mmac_*`(du_mma.hpp)+ DCC clang17。
- v_mfma 死刑结论(memory gfx936_no_mfma_scalar_fma_only)**保留但订正**:只对 v_mfma_* 成立,对 v_mmac_* 不成立。

## §T2 结论(2026-07-13,容器 e03r2n11 / worker-0 173.1.15.7 实测)— 路 B v7 击败 rocBLAS 506us,进 T3

### 微原型结果(scalar_splitk_v7.cpp)

路 B(`v_madmk_f16` 标量 split-K)经 v0→v7 迭代,v7 去 reduce kernel + atomicAdd 直写最终 Y 后击败 rocBLAS 506us 基线:

| splitK | 耗时(us) | HBM BW | 带宽% | MAXDIFF | 判定 |
|---|---|---|---|---|---|
| **16** | **481.2** | 742 GBps | **23.2%** | 0.0000 | **BEAT** ← 最优 |
| 8 | 486.7 | 733 GBps | 22.9% | 0.0000 | BEAT |
| 4 | 506.2 | 706 GBps | 22.0% | 0.0000 | NOT_BEAT(持平基线) |
| 2 | 544.6 | 656 GBps | 20.5% | 0.0000 | NOT_BEAT |

- **rocBLAS 基线(对比)**:506us,22% HBM BW(文档 12 §N6 / §A6,同 shape F.linear)。
- **正确性**:MAXDIFF=0.0000(全 34816 行 vs bf16→f32 标量 ref,BAD=0)。✅
- **kernel 名**:`matvec_v7a`(hipLaunchKernelGGL 直调,非 PyTorch 路由)。

### v7 内核框架(关键决策)

- 1 lane/1 row(blockIdx.x*64+lane = row),grid=(M/64, splitK)。
- 4 路 uint4(8 bf16/路)寄存器累加 a0..a3,沿 K 一次 32 个 bf16。
- **X 塞 shared memory(10KB,全 wave 共享一次)**:消除 X 的重复 HBM 读 → 带宽从 1-2% 升到 21%(v5 关键突破)。
- **去 reduce kernel**:每段 atomicAdd 直写最终 Y(不同 row 不冲突,同 row 跨 splitK=4/8/16 段竞争可接受)+ `hipMemsetAsync(dY,0)` → 省 1 次 launch+barrier(529→481)。

### 演化链(为啥 beat)

v0(sk2)朴素标量 459us(无向量化,文档16 §T1)→ v1=10453us(初版 split-K 铺法错)→ v2=4876us(4lane/row)→ v3=11235us(倒退)→ **v4 uint4 向量化=1344us** → **v5 X塞shared 4路=529us** → v6 2row/8路=1436us(倒退,寄存器压力过大)→ **v7 去reduce+atomicAdd=481us(获胜)**。

三个关键优化:(1) uint4 向量化加载(8 bf16/lane/加载);(2) X 塞 shared 消除重复 HBM 读(带宽 1-2%→21%);(3) 去 reduce kernel atomicAdd 直写(529→481)。

### T2 判定:BEAT,进 T3

- 我 kernel(v7a sk16=481.2us) **<** rocBLAS 506us,且正确性 OK → **满足 T2 退出条件"beat"**,进 T3。
- 注:rocBLAS 506us 是 F.linear 同 shape 基线,非黑盒 wvSplitK(wvSplitK op 在 micro 里调不出,见 §T2 动作3)。506us 是 vllm 走 wvSplitK 的真实耗时吗?——文档15 §C1 trace 实测:gate_up 在 vllm 里跑的 tile = MT64x32x32 big 506us×10688(文档 trace_adjoint_attribution),与 F.linear 506us 吻合。**506us 即 gate_up 在 vllm 实跑的瓶颈 tile 耗时,作对比基线成立。**
- 仍未触及物理上限(111us = 100% HBM),距 3200GB/s 峰值仍有 4-6x 空间。**但 bench 目标"beat 506us"已达成**,先进 T3 集成检查点,是否继续压(481→逼近111)由用户定。

### 待用户决策(T3 检查点)

1. v7a sk16=481us 是否进集成(T4,改 vllm utils.py 加守卫分支 + 编 wheel)?
2. 还是先继续压原型(481→逼近 111us 物理上限,可能 2-3x 收益但需更多 bench 时间)?
3. 路 A(mmac)仍在另一窗口,是否等路 A 出结果再二选一进集成?

**铁律:用户没确认前不动 vllm 源码。**

### §T2 追加(2026-07-13)— v10 VGPR 双缓冲 BEAT v7,新最优 458.8us

用户决策:T3 选 (2) 继续压原型。演化至 v10:

| 版本 | splitK | 耗时(us) | HBM BW% | MAXDIFF | 判定 |
|---|---|---|---|---|---|
| v7a | 16 | 481.2 | 23.2% | 0.0000 | BEAT 506(旧最优) |
| v8 2row | 4-16 | 785-1118 | — | — | 倒退(寄存器压力) |
| v9 sk1 | 1 | 1326.9 | — | — | 倒退(减 wave 假设错) |
| **v10 双缓冲** | **16** | **458.8** | **24.3%** | 0.0000 | **BEAT_V7 ← 新最优** |
| v10 双缓冲 | 8 | 461.5 | 24.1% | 0.0000 | BEAT_V7 |
| v10 双缓冲 | 32 | 459.6 | 24.2% | 0.0000 | BEAT_V7 |
| v10 双缓冲 | 64 | 1163.8 | — | MISMATCH | Kseg=80 尾部 bug(非 target) |

- **v10 框架**:沿用 v7a sk16,新增 **VGPR 双缓冲/软件流水线**(w_cur 本轮算,w_next 下一轮预取,轮末 swap),让 HBM load W 与 ALU FMA 重叠。纯 VGPR,不动 LDS。
- **关键发现**:sk8/sk16/sk32 持平 → **splitK 已饱和**,瓶颈转向单 wave 算力/带宽利用,非占用率。
- **认知订正**:"3 wave/CU 上限"是工程师口头未实测假设(v8/v9 基于它设计全倒退);实测 wave 数与耗时反相关(v7 109 wave/CU 跑最快)。
- **仍距物理上限 111us 有 4x 空间**,但纯标量+双缓冲能挤的重叠有限(只省 22us)。再大幅提速需更强重叠手段(硬件异步拷贝指令,另一窗口在找)或更深 ILP。
- v10 代码:`/public/home/xdzs2026_c150/scalar_splitk_v10.cpp`(本地 `tasks/scalar_splitk_v10.cpp`)。演化/胜出记录详见 `17_dual_path_splitk.md` §7。

**T3 检查点更新**:用户选继续压原型,目前 v12=403.3us(sk8,16x 路径,新最优)。

### §T2 追加(2026-07-14)— v12 16 路 uint4 更深 ILP BEAT v11,新最优 403.3us

| 版本 | splitK | 路径 | 耗时(us) | HBM BW% | MAXDIFF | 判定 |
|---|---|---|---|---|---|---|
| v11 8路ILP | 16 | 8x | 418.5 | 26.6% | 0.0000 | BEAT_V10(旧最优) |
| **v12 16路ILP** | **8** | **16x** | **403.3** | **27.6%** | 0.0000 | **BEAT_V11 ← 新最优** |
| v12 16路ILP | 4 | 16x | 414.2 | 26.9% | 0.0000 | BEAT_V11 |
| v12 8路退化 | 16 | 8x | 423.1 | 26.3% | 0.0000 | NOT_BEAT_V11(Kseg=320 不被128整除走8x) |
| v12 4路退化 | 32 | 4x | 686.6 | 16.2% | 0.0000 | NOT_BEAT_V11(4x路径效率低) |
| v12b 通用尾部 | 8 | 16x+尾部 | 557.0 | 20.0% | 6.9190 | MISMATCH(作废) |

- **v12 框架**:沿用 v11 双缓冲,新增**每轮 16 路 uint4(128 bf16/轮)**,a0..a15 十六路累加。三路径版本按 Kseg 整除性精确匹配(128→16x,64→8x,32→4x)。
- **vs rocBLAS 506us**:v12 sk8 快 102us(20.2%),带宽 22%→27.6%。
- **关键洞察**:16x ILP 红利只在 Kseg 被 128 整除时吃到(sk8 Kseg=640=5×128 甜点)。sk16 被迫走 8x 反而略慢于 v11。
- v12b(通用 16x 主循环+逐元素尾部)实验失败作废:正确性 MISMATCH + 性能反退化,尾部降级路径脆弱不值得。
- **距物理上限 111us 仍 3.6x 空间**,纯标量 ILP 收益放缓(481→459→418.5→403.3,省 22→40→15us)。再大幅提速需硬件异步拷贝指令(仍"没消息")或换思路。
- v12 代码:`/public/home/xdzs2026_c150/scalar_splitk_v12.cpp`(本地 `tasks/scalar_splitk_v12.cpp`)。演化详见 `17_dual_path_splitk.md` §7。

| 版本 | splitK | 耗时(us) | HBM BW% | MAXDIFF | 判定 |
|---|---|---|---|---|---|
| v10 双缓冲 | 16 | 458.8 | 24.3% | 0.0000 | BEAT_V7(旧最优) |
| **v11 8路ILP** | **16** | **418.5** | **26.6%** | 0.0000 | **BEAT_V10 ← 新最优** |
| v11 8路ILP | 8 | 421.3 | 26.4% | 0.0000 | BEAT_V10 |
| v11 退化4x | 32 | 687.2 | 16.2% | 0.0000 | NOT_BEAT(Kseg=160 不整除64走退化路径) |

- **v11 框架**:沿用 v10 双缓冲,新增**每轮 8 路 uint4(64 bf16/轮)**替代 v10 的 4 路,a0..a7 八路累加。单 wave 更多 outstanding load 掩盖 FMA 延迟,用满 VGPR 768。仍 1 lane/1 row(区别于 v6/v8 2-row 倒退)。
- **vs rocBLAS 506us**:v11 sk16 快 87us(17.3%),带宽 22%→26.6%。
- **距物理上限 111us 仍 3.8x 空间**,但 ILP 收益递增(481→459→418.5),方向正确。下一版 v12:让 sk32 也走 8x(32 步整除+尾部余数)确认 splitK 真实饱和点 + 尝 16 路 uint4 更深 ILP。
- v11 代码:`/public/home/xdzs2026_c150/scalar_splitk_v11.cpp`(本地 `tasks/scalar_splitk_v11.cpp`)。演化详见 `17_dual_path_splitk.md` §7。

## §结果表(待填,T6)

| 段 | C1 baseline | split-K kernel | vs C1 |
|---|---|---|---|
| 4-8K | 18.26 | — | — |
| 8-16K | 12.30 | — | — |
| 16-32K | 8.61 | — | — |

---

## 风险与回退

- **最大风险**:gfx936 工具链编不了朴素 HIP MFMA(T1 退出 c)→ 整条路线废。**T1 就是排这个雷,先跑。**
- **集成风险**:守卫分支写错 shape → 误伤其它 GEMM(重蹈 C4-3)。**T4 用精确 `m==34816 and n==1 and k==5120` 守卫,只 gate_up 命中。**
- **回退**:T4 守卫是一行 if,删掉即回滚到纯 C1。T2 原型不接 vllm,失败不影响线上。
