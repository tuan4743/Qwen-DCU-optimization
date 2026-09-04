# 19 · 最终总结:gfx936 vLLM 解码优化经验沉淀

> **创建:2026-07-15。**
> **⚠️【勘误 2026-09——必须优先阅读】**:本文 §0"唯一实质正向收益来自 rocm.py:149"及 §1.6/§2.7 把"算子"整体判死路的结论,**已被后续事实部分推翻**。比赛终稿《基于国产加速卡的千问大模型推理服务优化说明文档.md》(含 `图片和附件/` 40 张截图)显示五轮优化全部落地:4-8K 吞吐 12.20→19.56、8-16K→14.92、16-32K→12.22,P99 TPOT 69.00→45.14ms。**其中"算子融合"(四、五章:大瓦片 Prefill FA(M=128/N=32)、GDN decode BV=128、in_proj 权重拼接融合、TILE_N=64、融合 Chunk 预处理)是单轮收益最大、最有价值的部分**;它们全部是 Python 模型层 + vendored Triton/FLA 改动,**不依赖 `_rocm_C.so` 重编译**,与本文 §1.6 告终的"手写 split-K GEMM 算子进 `_rocm_C`"是两条独立通道——**后者集成失败 ≠ 融合失败/无用**。本文记录的硬事实、方法论、手写算子经验仍全部有效,仅范围性结论(融合=死路、唯一收益=C1)作废;订正论证见 `qwen3_dcu_optimize/docs/02_summary_corrections.md`,本地代码与终稿逐项差距见 `qwen3_dcu_optimize/docs/00_audit_local_vs_final.md`,终稿代码规格见 `qwen3_dcu_optimize/docs/01_final_changes_spec.md`。
> **定位**:本会话全部工作收尾文档。手写算子集成路线最终失败,工作区散乱文件全部清理,经验浓缩进此文档与既有 01-18 编号文档。
> **两大块**:(1)算子开发踩坑(gfx936 手写 split-K GEMM 全套经验);(2)踩坑分析方法论(归因/订正/证伪的方法教训)。
> **读者**:未来接手 gfx936 / 海光 DCU vLLM 优化的人。先读本文,再按需查 01-18 细节。

---

## 0. 一句话结论

**gfx936(BW3000)上 vLLM 解码优化的唯一实质正向收益,来自 `rocm.py:149` 给 `_ON_GFX9` 列表加 `"gfx936"` 这一处改动(打开 wvSplitK 优化链),4-8K 段 +42.6%。**【勘误:此句反映 07-13 窗口认知;终稿五轮另有 14 项改动(含四、五章融合)收益递增至 19.56/14.92/12.22,见顶部勘误与 `qwen3_dcu_optimize/docs/00_audit_local_vs_final.md`】其余所有手写算子、调参、换后端路线全部走完,要么死路要么无法集成。手写算子(v6→v17)在 standalone bench 里 beat 了 rocBLAS(v15c 406us vs 506us),但集成进 vLLM 需改 4-5 个对接点 + 编含 `_rocm_C.so` 的 wheel,接口太复杂总报错,最终放弃集成路线。【勘误:此处"放弃"仅指手写 split-K 算子通道,与 Python 层算子融合无关】

---

## 1. 算子开发踩坑(gfx936 手写 split-K GEMM)

### 1.1 硬件事实底座(所有算子设计的前提)

- **gfx936 = 海光 BW3000**,80 CUs,64GB HBM,HBM 带宽 ~3.2TB/s,VGPR 768/wave,wave64。
- **无 MFMA**:`v_mfma_*`(AMD 标准矩阵指令)**编得过但运行时 VMFault**(`HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`)。clang 能编出指令编码、accvgpr 约束正确,但硬件不执行。rocBLAS/hipBLASLt 反汇编 709MB 库 `v_mfma` 出现次数 = 0。**绝对不用 `__builtin_amdgcn_mfma_*`。**
- **海光自有 `v_mmac_*`(du_mma.hpp)能用**(`v_mmac_f32_16x16x16bf16` 能编能跑结果对),但只能用 DCC clang17(`/opt/dtk/dcc/bin/clang++`)编,主 clang18 编不出。**matvec 下 B 片段利用率 1/16,不是最优。**
- **rocBLAS/hipBLASLt 主力指令 = `v_madmk_f16` 标量 FMA**(rocBLAS 反汇编 101192 次)。bf16 GEMM 靠 `v_madmk_f16` + bf16→f16 拆位。**手写算子对齐这条路径。**
- **occupancy**:实测 6-12 wave/CU(用 `hipOccupancyMaxActiveBlocksPerMultiprocessor`),"3 wave/CU 上限"是未实测的口头假设,曾误导 v8/v9 设计全倒退。

### 1.2 目标 GEMM 与物理天花板

- **FFN gate_up_proj**:batch=1 decode → `(n=1, m=34816, k=5120)`,bf16,bias=False。权重 34816×5120×2B = 356MB/层。
- **算术强度 0.5 FLOP/B ≪ 屋顶线拐点 → 纯 memory-bound matvec**。乘加指令选择被带宽淹没,所有路线物理上限相同 = 357MB / 3200GB/s = **111us**。
- **rocBLAS F.linear 同 shape = 506us,只用 22% HBM 带宽**(703 GB/s)→ 瓶颈是 launch/tiling/CU 占用(GSU1 单网格没切分),**不是带宽天花板**,这是 split-K 有空间的理论窗口。

### 1.3 演化链与每个版本的踩坑(v6→v17)

最终最优 **v15c = 406.7us**(smem=0,无 syncthreads,最稳定),beat rocBLAS 506us(快 19.6%)。演化链:

```
v0(朴素标量 459us,无向量化)
 → v1=10453us(初版 split-K 铺法错)
 → v2=4876us(4lane/row)
 → v3=11235us(倒退)
 → v4 uint4 向量化 = 1344us     ← 关键优化1:8 bf16/lane/加载
 → v5 X 塞 shared memory = 529us  ← 关键优化2:消除 X 重复 HBM 读,带宽 1-2%→21%
 → v6 2row/8路 = 1436us(倒退,寄存器压力过大)
 → v7 去 reduce kernel + atomicAdd = 481us  ← 关键优化3:省 1 次 launch+barrier
 → v10 VGPR 双缓冲/软件流水线 = 458.8us  ← HBM load W 与 ALU FMA 重叠
 → v11 8路 ILP = 418.5us
 → v12 16路 uint4 ILP = 403.3us(sk8 甜点,Kseg=640 被 128 整除)
 → v15c 关 scalarize 后简化 = 406.7us(最稳,smem=0)
 → v16c volatile flat_load = 否决(退化 6.7×)
 → v17 打包点乘 v_pk_fma_f32 = 与 v15c 相同(编译器不发 packed FMA)
```

**三个关键优化(beat rocBLAS 的根本)**:
1. **uint4 向量化加载**(8 bf16/lane/加载)—— 提升带宽利用。
2. **X(input)塞 shared memory 全 wave 共享一次** —— 消除 X 的重复 HBM 读,带宽从 1-2% 升到 21%(v5 决定性突破)。
3. **去 reduce kernel + atomicAdd 直写最终 Y** —— 省掉 split-K 段间 reduce 的一次 launch+barrier(529→481)。

### 1.4 关键踩坑点(每条都是烧出来的,别重蹈)

1. **W 标量提升(`s_load_dwordx8`)是优化非 bug**。scalar 通路 64 lane 共享一份 W,省 64× 带宽。**关 `scalarize-global-loads` 反慢 1.27×(406→515us)**,编译器默认开启是对的。曾误以为是 bug 想关掉,实测打脸。

2. **volatile flat_load 否决**。`volatile` 强制 vector load 退化成 `flat_load_dword`×384,慢 6.7×(406→2748us)。两路强制 vector load 都输给 default。**别碰 volatile。**

3. **packed FMA(`v_pk_fma_f32`)发不出**。v17 想用打包点乘 `__hip_bfloat162 → v_pk_fma_f32`,实测与 v15c 相同 —— 编译器在 matvec 场景不发 packed FMA(利用率问题),手写打包是白费。

4. **raw_buffer_load_lds / MLS-B LDS 路线放弃**。单条小负载搬数据对,但 O3 批量循环+多 lane+syncthreads 下被 DCE(SINK=0/OUT=0,0.44us 假象);且单条最多 4B 无 x4 版指令数爆炸。**LDS 路线放弃,回 v12/v15c 寄存器双缓冲。**

5. **VMFault 真根因 = `__HIP__GFX9__` 宏 gfx936 走空壳**。`skinny_gemms.cu:24` 宏只含 `gfx90a/942/950` 不含 gfx936 → gfx936 走 `UNREACHABLE assert(false)` 空壳 → VMFault。非 v12 引入。修法 = 宏加 gfx936 走标量路径。**这与"v_mfma 运行时非法"是两码事,别混淆。**

6. **ILP 红利只在 Kseg 整除时吃到**。v12 16x 路径 sk8 Kseg=640=5×128 是甜点;sk16 被迫走 8x 反而略慢。通用尾部(v12b 16x 主循环+逐元素尾部)实验失败:正确性 MISMATCH + 性能反退化,尾部降级路径脆弱不值得。**Kseg 整除性精确匹配三路径(128→16x,64→8x,32→4x)。**

7. **splitK 饱和点**。sk8/sk16/sk32 持平 → splitK 已饱和,瓶颈转向单 wave 算力/带宽利用,非占用率。**别再盲目加大 splitK。**

8. **bench 铁律**:warmup 5 次 + 跑 1000 次取 **median**(不是 mean,不是单次)。正确性 vs CPU ref,max abs diff < 1e-2(bf16 容差)。报:median us + 带宽% + maxdiff + grid/block + CU 占用。**绝不纸面推断,只认 bench 数。**

### 1.5 编译/运行工具链(算子复用必读)

```bash
# 标量路(路 B,hipcc 或 clang18):
export HIP_CLANG_PATH=/opt/dtk/lib/llvm/bin   # 必须设,否则 hipcc 找不到 clang
/opt/dtk/bin/hipcc --offload-arch=gfx936 -O3 \
  -I/opt/dtk/include -I/opt/dtk/include/hip \
  scalar_splitk.cpp -o scalar_splitk -L/opt/dtk/hip/lib -lamdhip64

# mmac 路(路 A,必须 DCC clang17):
/opt/dtk/dcc/bin/clang++ -x hip --offload-arch=gfx936 \
  --rocm-path=/opt/dtk --rocm-device-lib-path=/opt/dtk/amdgcn/bitcode \
  -I/opt/dtk/include -I/opt/dtk/include/hip -D__HIP_PLATFORM_AMD__ \
  -L/opt/dtk/hip/lib -lamdhip64 -O3 -Wno-* \
  mmac_splitk.cpp -o mmac_splitk

# 运行(两路通用):
export LD_LIBRARY_PATH=/opt/dtk/hip/lib:$LD_LIBRARY_PATH
```

**关键坑**:
- `HIP_CLANG_PATH` 不设 → hipcc 默认找 `/opt/dtk/hip/lib/llvm/bin/clang++`(不存在)→ 所有编译失败报 `No such file or directory`。
- mmac 必须用 DCC clang17(`/opt/dtk/dcc/bin/clang++`),主 clang18 无 mmac builtin —— 这是 T1 当初没发现 mmac 可用的根因(用错了 clang)。
- du_mma.hpp host pass 编译坑:模板由 `#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)...` 保护,host pass 扫所有形状报 "undeclared identifier"。修复:源文件顶部 `#ifndef __gfx936__ #ifndef __gfx938__` 块给全部 14 个 mmac 宏空占位 `#define __DU_MMA_...(a,b,c) (c)`。**必须补全全部 14 个**(只补 4 个 fp8 不够)。

### 1.6 集成路线失败(本会话最终结论)

**手写算子(v12/v15c)standalone bench 成功,但集成进 vLLM 失败,放弃。**

集成需改 4-5 个对接点 + 编含 `_rocm_C.so` 的 wheel:
1. `csrc/rocm/ops.h` 加 `gate_up_splitk_v12` 声明。
2. `csrc/rocm/torch_bindings.cpp` 加 `.def(...)` + `.impl(..., torch::kCUDA, ...)`。
3. `CMakeLists.txt` 在 `VLLM_ROCM_EXT_SRC` 加 `"csrc/rocm/v12_splitk.cu"`。
4. `vllm/_custom_ops.py` 加 Python 绑定 `gate_up_splitk_v12`。
5. `vllm/model_executor/layers/utils.py` 加 `use_v12_splitk` 精确 shape 守卫(on_gfx9() and bf16 and n==1 and m==34816 and k==5120 and bias is None and contiguous)。

**卡点**:容器内 `import torch` 失败(缺 `libgalaxyhip.so.5`/`librocm_smi64.so.2`),需注入 DTK LD_LIBRARY_PATH 到 setup.py/cmake/hipify.py 顶部。编 `_rocm_C.so` 的 wheel 链路太长,接口太复杂,总报错。**用户最终指示:"算子的接口太复杂了,集成进去总是报错"——承认手写算子集成路线失败。**

**教训**:
- 在 gfx936 这类非主流卡上,改 vLLM 的 C++ 扩展构建链(`_rocm_C.so`)成本极高,远高于改 Python 分发逻辑。
- **真正的正向收益在 Python 层一处改动**(`rocm.py:149` 加 gfx936 打开 wvSplitK 链),不在手写算子。手写算子是"用 4-5 个对接点 + wheel 编译的复杂度,换 19.6% 单 GEMM 提升",ROI 极低且集成失败。
- 详见 `18_v12_integration_buildguide.md`(集成指南)、`16_splitk_gemm_tasklist.md`(算子任务)。

---

## 2. 踩坑分析方法论

### 2.1 铁律:GEMM 后端归属必须 torch profiler 实测核名校验,不能只靠源码静态推断

**翻车点**:旧稿只查 `on_gfx9()` 返回 False(白名单 `["gfx90a","gfx942","gfx950"]` 不含 gfx936),就断定"所有 GEMM 走 F.linear/rocBLAS"。**漏看了 dist-packages 版本是 `or on_gfx936()` 已命中并走 LLMM1**。源码版与线上 dist-packages 版本不同,静态推断源码 ≠ 实跑。

**正确做法**:
- bench 数值必须用 trace 实测核名校验(A2 三方 bench:qkvz 实跑走 LLMM1 188us,不是 hipBLASLt 261us,不是 rocBLAS 267us)。
- **绝不只靠源码静态推断,也不只靠 bench heuristic 或 trace kernel 名臆测归属**。三者交叉闭合。

### 2.2 去污染:正则同名缩写撞车把 GEMM 误吞进 GDN 桶

**翻车点**:旧版 `_parse_profile_trace.py` 的 `CATS` 列表里 `GDN/FLA` 正则排在 `FFN_GEMM` 之前,且含裸 `GSU` 片段。`classify()` 第一个命中即返回 → 所有 `Cijk_Alik_Bljk_BBH_*_GSU1/4/8`(rocBLAS/hipBLASLt GEMM tile,`GSU`=GridSplitU 分块参数)被误吞进 GDN/FLA 桶,得出伪值"GDN/FLA 占 95.17%"。

**订正**:把 `FFN_GEMM` 提到 `GDN/FLA` 之前 + 从 GDN/FLA 正则删掉裸 `GSU` 只留 `PostGSU` 前缀。去污染后真实占比:**FFN_GEMM 95.55%,GDN/FLA 1.06%**(GDN 自身递归核仅 0.9%)。**GSU 是 rocBLAS tile 参数,与 GDN 的 GSU 毫无关系,纯同名缩写撞车。**

**教训**:归因脚本的正则分类顺序和模式精度直接决定结论真伪。任何"X 占比 95%"的结论,先查分类正则有没有同名撞车。

### 2.3 duty cycle 推翻 step 间空闲假说

**假说**:tpot 瓶颈在 step 间 IPC/调度空闲(median_gap=1ms)。

**订正**:8s + 30s 长窗口双闭合,GPU duty = 97.3%,idle 均匀分散无周期性聚集,166912 个 idle gap 全部 <1ms(max 0.644ms)。`median_gap=1ms` 是**同一 token 内相邻层间 kernel 间隔**(64 层 × ~1ms ≈ 64ms ≈ tpot),非跨 token 间隔。**端到端 tpot 瓶颈 = step 内部 64 层 GPU kernel 串行,不在 step 之间。**

**教训**:gap 统计的"中位数 1ms"容易被误读成"step 间空闲"。必须用 duty cycle + idle 分布 + gap 语义(层内 vs 跨 token)三重闭合,才能定位瓶颈在 step 内还是 step 间。这条把优化重心从 CPU/调度轨道归到 GEMM 轨道。

### 2.4 op 标签 trace 归因在 cudagraph ON 下不可行 → shape→tile 正向匹配

**死路**:`torch.profiler.record_function` 是 CPU 侧 op 标签注入,cudagraph 捕获的是 GPU 静态图,重放时只跑已捕获的 kernel 序列,record_function 不进图 → trace 只有 kernel 名没有 op 标签。关 eager 能抓标签,但 eager 下 kernel 选择与 cudagraph ON 不同(不代表实跑)。

**新主线:shape → tile 正向匹配 + 频次/邻接反推归属**:
1. 用各候选 Linear 的 `(m,n,k)` 喂 `rocblas-bench`/`hipblaslt-bench`,看 heuristic 选的 tile 名是否 = 真瓶颈核(命中即归属)。
2. 频次反推:真瓶颈核 `MT64x32x32_GSU1` x18704,用每步该核出现次数 × 64 层 + lm_head(1/step)对齐总次数。
3. trace 内邻接关系:`MT64x32x32_GSU1` 紧邻哪个已知核(如 FFN silu 融合核 → 强提示归属 FFN gate-up)。

**闭合结果**(memory `trace_adjoint_attribution_mttiles`):
- FFN gate_up_proj = `MT64x32x32_GSU1` big 506us × 10688 = 5.407s,占 GPU 时间 48%(最高)。
- lm_head = `MT32x16x4_GSU1` big 1898us。
- attention qkv = `MT128x32x32_GSU4`。
- 167 token 基准闭合所有 tile。

**教训**:cudagraph 静态图挡住 op 标签归因,但 shape→tile 正向匹配 + 频次 + 邻接三重闭合能绕过。归因不一定要 op 标签,tile 名本身就是 shape 的指纹。

### 2.5 "已证实优化"反复订正:标签 / 前提 / 来源三层核查

本项目多次"已证实优化"被后续实测推翻,共性是**只查了一层没交叉**:

| 项 | 原结论 | 订正 | 失误层 |
|---|---|---|---|
| C5b(32→16) | 段数 32→16 提升 +1.90 分 | 16 是源码出厂值,不是从 32 改来的;容器 dist-packages 里是 32 那是旧版 | **标签层**:把"出厂值"误标成"优化" |
| C1 收益来源 | 打开 LLMM1 链 | 实际是打开 wvSplitK 链(decode 下 LLMM1 分支被 `if m>8 and 0<n<=4` 提前截走,永远不可达) | **来源层**:收益归到错误的分支 |
| qkvz 后端 | 走 hipBLASLt 638.9us,切 rocBLAS 快 3.8× | qkvz 实跑走 LLMM1 188us(最快),切 rocBLAS/hipBLASLt 反退化 | **前提层**:建立在"qkvz 跑 hipBLASLt"错误前提上 |
| 80.02 第二个优化 | 源码树有第二个长序列优化 | 穷尽 diff 否定,80.02 源码树相对纯净版只有 C1 | **存在性层**:把"装错版本(site-packages C4-3 残留)"误判成"源码缺优化" |
| skinny gfx936 不可达 | B5 三重闭合说 C++ 宏空壳 | ops.LLMM1 在 gfx936 实测可用(C1 +42.6% 证明),B5 废的是"源码版 skinny 全链路"非 dist-packages 的 LLMM1 | **范围层**:把"源码版"结论误套到"dist-packages 版" |

**教训**:每条"已证实优化"必须标三层 —— **标签**(真是优化还是出厂值)、**前提**(结论依赖的前置事实是否成立)、**来源**(收益归到哪个分支/路径)。任一层错,整条结论翻车。订正时写明"原结论→订正→失误层",别直接覆盖。

### 2.6 改源码生效路径(操作铁律)

- **改 `vllm_optimize_data/` 工作区副本 = 空忙**,不生效。
- **改 `/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/...`** → `cd vllm_cscc && python setup.py bdist_wheel` → `pip install --force-reinstall --no-deps dist/vllm-*.whl` → kill 旧 vllm → 重启 `start_vllm.sh`。
- **dist-packages 是拷贝非 editable**,任何人 `pip install` 都会覆盖,**必须审查源码而非 dist-packages**。
- **改完立即 `ast.parse` 校验语法**(切片移除带缩进代码块易误吃缩进 → IndentationError)。
- **C4-3 事故**:site-packages 没装干净 wheel,机器跑的是 C4-3 灾难版(lm_head 绕过 wvSplitK 走 F.linear,16-32K 3.58)。**装 wheel 后必须校验 site-packages 的 utils.py:181 干净,才让用户重启。**

### 2.7 死路清单(不再追,附证伪证据)

| 死路 | 证伪证据 |
|---|---|
| 消除 256MB fill 提升吞吐 | fill 是 Triton autotune L2-flush buffer,消除后 throughput 7.26 < baseline 8.8 |
| 缩小 Triton cache 至 64MB | throughput 6.58 < 8.8,倒退 |
| CPU/调度 overhead 降 tpot | duty 97.3% 满载,无重叠空间 |
| hipBLASLt override 调 algo | m=1 只返回 1 个 algo(4362,638.9us),override 无的放矢 |
| 切 rocBLAS/hipBLASLt 降 qkvz | qkvz 实跑 LLMM1 188us 最快,切后退化 |
| bucket padding 撑大 m | capture 后 m=1,并发=1 下 N 倍冗余计算纯浪费 |
| FP8 低精度 | gfx936 / DTK 26.04 segfault |
| 投机解码 | 锁定约束明令禁止 |
| 投影+递归核融合 | 递归核核心循环无 tl.dot GEMM,融合碰不到 GEMM 本身 | 【勘误:仅指"GEMM+递归核单 kernel"跨算子融合;终稿四/五章的融合(权重拼接/瓦片/融核)对象不同且已落地有正收益,见 qwen3_dcu_optimize/docs/02】 |
| op 标签 trace 归因 | record_function 不进 cudagraph 静态图 |
| aiter gemm_a16w16 | 白名单只收 5 个 (m,k) 精确组合,gate_up (34816,5120) 不命中 |
| weight 布局优化 | gate_up weight row-major contiguous 无次优,MT64x32x32 非布局导致 |
| rocBLAS algo override | 容器里已有 trace,属错误方向 |
| cudagraph 拆分 | 偏禁区(改 model_runner capture 逻辑),风险高 |
| 手写算子集成 vLLM | 4-5 对接点 + wheel 编译太复杂,总报错,放弃 |

### 2.8 仍开放的方向(未验证也未证伪,留待后续)

- **HBM 带宽天花板判断**:gate_up 走 rocBLAS 506us 只用 22% 带宽,**非物理天花板**,理论上有 4-6× 空间。但纯标量 ILP 收益放缓(481→459→418→403,每版省 22→40→15us),再大幅提速需硬件异步拷贝指令(本会话"没消息")或更深 ILP。
- **site-packages 装回干净 C1 版**:当前装版远低于 80.02 的真因 = site-packages 装了 C4-3 灾难版残留,重装 09:17 干净 wheel 即可恢复,**非源码问题**。这是恢复 80.02 水平的当务之急,不是找新优化。

---

## 3. 文档索引(清理后保留)

| 文档 | 内容 |
|---|---|
| 01-04 | 约束/模型架构/profile 发现/cudagraph 实验 |
| 05 | task tracker(P0→P2 历程) |
| 06 | 关键判断备忘(256MB fill / NO_PROXY / 空 trace 等) |
| 07 | P0 结论 |
| 08 | DCU 访问链路(三层 ssh) |
| 09 | CPU 调度 overhead 设计(已证伪,duty 97.3%) |
| 10 | GDN GEMM 设计(含 archive 大版本) |
| 11 | 占空比/lm_head 区分/m=1 三类质疑审查 |
| 12 | 已证实优化/死路汇总(A 正向/B 死路/C 约束/D 主线/E 铁律/F 候选) |
| 13 | qkvz 后端 bench(LLMM1 188us 最优) |
| 15 | 改源码跑实测任务清单(C1 落地 + 全部 C 系列订正) |
| 16 | split-K GEMM 算子任务(v6→v17 演化) |
| 17 | 双路并行作战(标量路 B + mmac 路 A) |
| 18 | v12 集成编译指南(集成失败卡点) |
| **19** | **本文:最终总结** |

> 散乱的 .cpp 源码(v6-v17)、.sh 脚本、probe/disasm 脚本、_dummy.py/_cfgread.py 等已全部清理。算子最终版 v15c.cpp 和 v12_splitk.cu 若需复现,从 memory 中的演化记录或 git 历史恢复。
