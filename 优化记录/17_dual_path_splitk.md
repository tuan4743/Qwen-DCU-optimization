# 17 · gfx936 gate_up split-K 双路并行作战文档

> **创建:2026-07-13。**
> **目的**:用户指令"你写个文档,我并行窗口两条路都搞"。两条路线各自独立推进,谁先 beat 基线谁先用,结论互不打架。本文档是两路的统一事实源 + 决策表。
> **目标 GEMM**:FFN gate_up_proj。shape: n=1(批1解码), m=34816(out), k=5120(in), bf16, bias=False。
> **baseline**(文档12 §A6):rocBLAS F.linear 同 shape = **506 us**(只用了 22% HBM 带宽)。
> **物理天花板**:357MB 权重 / 3200 GB/s HBM = **111 us**(纯带宽下限,任何路线不可破)。

---

## 0. 共识事实(两路共享,不再争议)

### 0.1 gate_up 是 memory-bound matvec,指令层优势被带宽淹没

- 算术强度 = 1.78e8 FLOP / 3.57e8 B = **0.5 FLOP/B** ≪ 屋顶线拐点 → **纯 memory-bound**。
- **两条路的物理上限相同 = 111 us**(带宽决定,与乘加指令无关)。
- 差别只在"谁更容易逼近 111us",不在"谁的上限高"。
- **rocBLAS 506us 只用 22% 带宽**(文档12 §N6)→ 瓶颈是 launch/tiling/CU 占用(GSU1 单网格没切分),**这正是 split-K 的靶子**,也是两条路共同的收益窗口。

### 0.2 gfx936 矩阵指令事实(2026-07-13 容器实测,详见 memory gfx936_no_mfma_scalar_fma_only)

- **`v_mfma_*`(AMD 标准):运行时 VMFault,死刑。绝对不用 `__builtin_amdgcn_mfma_*`。**
- **`v_mmac_*`(海光自有,du_mma.hpp):能编能跑结果对** ✅。但 matvec 下 B 片段利用率 1/16(只有 col0 有效)。
- **标量 `v_madmk_f16`**:rocBLAS 主力指令(反汇编 101192 次),matvec 下利用率 100%,memory-bound workload 的天然最优指令。

### 0.3 编译/运行工具链(两路通用)

```
# 编译(标量路用 hipcc 或 clang18 均可;mmac 路必须 DCC clang17)
# —— 标量路(路B):
export HIP_CLANG_PATH=/opt/dtk/lib/llvm/bin
/opt/dtk/bin/hipcc --offload-arch=gfx936 -O3 \
  -I/opt/dtk/include -I/opt/dtk/include/hip \
  scalar_splitk.cpp -o scalar_splitk -L/opt/dtk/hip/lib -lamdhip64
# —— mmac 路(路A,必须 DCC clang17):
/opt/dtk/dcc/bin/clang++ -x hip --offload-arch=gfx936 \
  --rocm-path=/opt/dtk --rocm-device-lib-path=/opt/dtk/amdgcn/bitcode \
  -I/opt/dtk/include -I/opt/dtk/include/hip -D__HIP_PLATFORM_AMD__ \
  -L/opt/dtk/hip/lib -lamdhip64 -O3 -Wno-* \
  mmac_splitk.cpp -o mmac_splitk
# 运行(两路通用):
export LD_LIBRARY_PATH=/opt/dtk/hip/lib:$LD_LIBRARY_PATH
```

### 0.4 bench 铁律(两路统一,文档12 §E)

1. warmup 5 次 + 跑 1000 次取 median(不是 mean,不是单次)。
2. 正确性:vs CPU ref(torch.matmul 或自写三重循环),max abs diff < 1e-2(bf16 容差)。
3. bench 时报:median us + 带宽利用率% + maxdiff + grid/block + CU 占用。
4. **绝不纸面推断,只认 bench 数**(用户铁律"bench 时间花的值")。

### 0.5 连接架构(两路通用)

登录节点 `zzeshell.scnet.cn:65032`(`ssh -i ~/.ssh/InstanceKey.txt xdzs2026_c150@...`)→ `srun --jobid=677633` 进计算节点 `e03r2n11` → `ssh root@173.1.15.7` 进容器 worker-0。home `/public/home/xdzs2026_c150` 跨节点共享。源码 base64 传 home → 容器编译运行。

---

## 1. 路 A:mmac 16x16x16 bf16 split-K

### 1.1 路线定性

- 用 `v_mmac_f32_16x16x16bf16`(du_mma.hpp)+ DCC clang17。
- **已验证能编能跑结果对**(mmac_official2.cpp,T1b 三验证 LAUNCH_ERR=0 MAXDIFF=0)。
- **v2 朴素直铺已 bench**:2442 us(慢 4.8x),正确性 OK。暴露两个结构性坑(见 1.3)。

### 1.2 上限分析

- 物理上限 = 111us(带宽,与路B相同)。
- matvec 下 mmac B 片段 1/16 利用率 → 算力层浪费,但 memory-bound 下不致命。
- **现实预期**:乐观 223us(50% 带宽),悲观 318us(35% 带宽)。**难低于 300us**。

### 1.3 v2 已暴露的坑(后续版本必须解决)

1. **每段重建 Btile + 双 `__syncthreads`**:320 段 × 2 sync = 640 次屏障,把 mmac 指令优势吃光。→ 改:host 预处理 Xpad(每段 16 个 X 值 + 240 个 0,col_major 存),kernel 直接 du_load 不重建;或干脆 global 直读 X 塞 register。
2. **CU 严重过载**:grid=2176 wave / 80 CU = 每 CU 27 wave。→ 改 split-K:沿 K 切段,每段一个 wave,grid = 80 量级铺满 CU,段间写中间 buffer 再 reduce。
3. **B 片段 1/16 浪费**:mmac 算 4096 个乘加只 256 个有效。→ matvec 下不可消除,接受(算力本就不是瓶颈)。

### 1.4 路 A 版本规划

| 版本 | 改动 | 目标 |
|---|---|---|
| v2(已完成) | 朴素直铺,每段重建Btile | 2442us,正确性 OK,基线 |
| v3 | Btile 预处理(Xpad)+ 去 sync | < 1000us |
| v4 | split-K 沿 K 切段铺 80 CU + reduce | < 506us(beat rocBLAS) |
| v5 | 调 split_k 段数 + tile + VGPR | 逼近 223us |

### 1.5 路 A 退出条件

- **成功**:v4/v5 beat 506us 且正确 → 进 T3 用户 checkpoint,与路B胜者二选一(或并存评估)。
- **失败**:3 轮调参仍 ≥ 506us → 停,让路B接管。mmac 在 matvec 不成立的结论坐实。

---

## 2. 路 B:标量 `v_madmk_f16` FMA split-K

### 2.1 路线定性

- 用 `v_madmk_f16` 标量 FMA(rocBLAS 同款指令)+ hipcc/clang18。
- **matvec 下指令利用率 100%**(每读一个权重算一次乘加,无浪费)。memory-bound workload 的天然最优指令。
- T1 已有原型 sk2.cpp(459us,朴素标量,未做 split-K)。

### 2.2 上限分析

- 物理上限 = 111us(带宽,与路A相同)。
- **无 shared/sync 开销**(标量累加全在 VGPR 寄存器内),无 B 片段浪费。
- 直接命中 rocBLAS 低效点(launch/CU 占用,文档12 §N6 GSU1 单网格没切分)。
- **现实预期**:乐观 180us(60% 带宽),悲观 280us(40% 带宽)。**更易逼近 111us**。

### 2.3 设计要点

1. **split-K 沿 K=5120 切段**:每段如 512(K/10),每段一个 wave block 算 16 行 × 512 K 的部分和。
2. **铺满 80 CU**:grid = (M/16行块) × (K/splitK 段),让 wave 数 ≥ 80 且每 CU 跑多 wave 复用。
3. **用满 VGPR 768**:每 thread 沿 K 段内扛一大段(如 256 个 bf16 = 128 寄存器),寄存器内累加,减少中间 buffer 写。
4. **段间 reduce**:段间部分和写中间 f32 buffer,第二个 kernel reduce;或单 kernel atomic add(先试简单的)。
5. **bf16→f16 还是 f32 累加**:bf16→f16 会丢精度(尾数位宽不同),需评估 <1e-2 容差。若不行走 f32 累加(`v_madmk_f32` + bf16→f32 拆位,rocBLAS 也在用 f32 mad 7907 次)。

### 2.4 路 B 版本规划

| 版本 | 改动 | 目标 |
|---|---|---|
| v0(sk2,已有) | 朴素标量,无 split-K | 459us,基线 |
| v1 | split-K 沿 K 切段铺 80 CU + reduce | < 300us |
| v2 | 用满 VGPR 寄存器内累加 + 调段数 | < 506us(beat rocBLAS) |
| v3 | bf16/f32 累加精度调优 + tile | 逼近 180us |

### 2.5 路 B 退出条件

- **成功**:v2/v3 beat 506us 且正确 → 进 T3 用户 checkpoint。
- **失败**:3 轮调参仍 ≥ 506us → 停,与路A结论一起汇报,可能需换瓶颈(gate_up 之外)。

---

## 3. 双路决策表(谁先 beat 谁先用)

| 情形 | 决策 |
|---|---|
| 路 B beat 506us,路 A 未 beat | **路 B 进 T3 集成**,路 A 停 |
| 路 A beat 506us,路 B 未 beat | **路 A 进 T3 集成**,路 B 停 |
| 两路都 beat 506us | 取**更快者**进 T3;另一路记录,留作扩展到其它 shape 候选 |
| 两路都未 beat | 汇报用户,gate_up split-K 路线整体存疑,考虑:① 重估文档12 §N6 的 22% 带宽数是否准;② 换瓶颈(lm_head / attn qkv) |
| 一路 beat 111us×1.5≈167us 内 | 物理上限附近,极优,直接进 T3 不再调 |

### 3.1 集成阶段(两路统一,T4-T6 复用)

无论哪条路胜出,集成逻辑相同(文档16 §T4):
- 独立 .so + `torch.ops.load_library` 注册,**不动 vllm 原生 _rocm_C.so**。
- 改 `vllm_cscc/vllm/model_executor/layers/utils.py` `rocm_unquantized_gemm_impl`:在 wvSplitK 分支**之前**加 `if on_gfx936() and m==34816 and n==1 and k==5120 and bias is None: 走我的 op`。**精确 shape 守卫,只 gate_up 命中**。
- 用户重启实测三段(4-8K/8-16K/16-32K)对比 baseline 18.26/12.30/8.61。

---

## 4. 文件清单(两路各自管理)

### 路 A(mmac)
- `mmac_official2.cpp`(home + 容器 /root/mmac_test/)— T1b 三验证 micro,LAUNCH_ERR=0 MAXDIFF=0,已完成。
- `mmac_matvec_v2.cpp`(home + 容器)— 朴素直铺,2442us,已完成。
- `mmac_splitk_v3.cpp` / `v4` / `v5` — 待写。
- 本地草稿:`C:\Users\hp\Desktop\CC-workspace\vllm_optimize_data\tasks\` 下 mmac_splitk_*.cpp。

### 路 B(标量)
- `sk2.cpp`(容器,文档16 §T1 提及)— 朴素标量 459us,已有。
- `scalar_splitk_v1.cpp` / `v2` / `v3` — 待写。
- 本地草稿:同目录下 scalar_splitk_*.cpp。

### 共享
- `16_splitk_gemm_tasklist.md` — T1/T1b/T2 结论源。
- `17_dual_path_splitk.md`(本文档)— 双路作战统一源。

---

## 5. 铁律(两路共同遵守)

1. **绝不看容器 dist-packages 的 vllm**,只看本地 `vllm_cscc` 副本或 `/public/home/xdzs2026_c150/zya/vllm_cscc`。
2. **不宽放 shape 守卫**:集成时只 `m==34816 and n==1 and k==5120`,不误伤 lm_head/down/qkvz(重蹈 C4-3 灾难)。
3. **site-packages 没校验干净不让用户重启**(§C4-3 教训)。
4. **用户没确认前不动 vllm 源码**(T3 checkpoint)。
5. **每步结论写回本文档对应 §,不打架**。两路结论独立记录,不互相覆盖。
6. **bench 只认实测数**,不认纸面推断(用户铁律)。
7. **最终编译启动 vllm 验证用户来更快,其它直接做**(用户指令)。

---

## 6. 风险与回退

- **最大风险(两路共有)**:gate_up 是 memory-bound,split-K 改的是 launch/CU 占用,若 rocBLAS 的 506us 里带宽占比已近极限(22% 是误测),则两路都 beat 不了 → 需重测 §N6 带宽数。
- **集成风险**:守卫分支写错 shape → 误伤其它 GEMM。**T4 精确 shape 守卫兜底。**
- **回退**:T4 守卫是一行 if,删掉即回滚到纯 C1(baseline)。两路原型都不接 vllm,失败不影响线上。
- **双路并行风险**:两窗口同时改 vllm 源码冲突 → **约定:集成阶段(T4)只在一个窗口做,另一窗口停手等结果**。原型阶段(T2)两路独立,不碰 vllm 源码,无冲突。

---

## 7. 进度记录(两路分别填)

### 路 A 进度
- [x] v2 朴素直铺:2442us,正确性 OK(2026-07-13)
- [x] v3 去 shared Btile + 手填 bFrag + 去 sync:2292.9us(仅微降,正确性 OK)— 证明仅去 sync 不够
- [x] v3.5 反汇编铁证(2026-07-14):mmac 路的 load 与标量路同款 `global_load`,无 async/绕寄存器优势 → 见 §8
- [x] global→LDS 绕寄存器指令探测(2026-07-14):`__builtin_amdgcn_raw_buffer_load_lds` gfx936 后端 Cannot select → 见 §9
- [ ] v4 split-K 铺 CU + 普通 global load 双缓冲 + 每 block ≤3 wave/CU(工程师指导,绕寄存器指令不可用后改走普通双缓冲)
- [ ] v5 调参逼近上限

### 路 B 进度
- [x] v0(sk2)朴素标量:459us(文档16 §T1)
- [x] v4 uint4 向量化:1344us(8% BW)
- [x] v5 X 塞 shared + 4 路 uint4 累加:sk4=529us(21% BW)
- [x] v6 2row/8路:1436us(倒退)
- [x] **v7 去 reduce + atomicAdd:sk16=481.2us(23.2% BW,BEAT rocBLAS 506us)**

### 胜出记录
- **路 B v7a 击败 rocBLAS 506us 基线(2026-07-13,容器 e03r2n11 / worker-0 173.1.15.7 实测)**:
  - 内核:`/public/home/xdzs2026_c150/scalar_splitk_v7.cpp`(本地副本 `tasks/scalar_splitk_v7.cpp`)。
  - 框架:1 lane/1 row,4 路 uint4(8 bf16/路)寄存器累加,X 塞 shared(10KB,全 wave 共享一次),去 reduce kernel 改 atomicAdd 直写最终 Y + hipMemsetAsync Y=0。
  - 结果:`v7a sk16=481.2us (23.2% HBM BW, MAXDIFF=0.0000, OK, BEAT)` ← 最优;`v7a sk8=486.7us (22.9%, BEAT)`;`v7a sk4=506.2us (22.0%, NOT_BEAT, 持平基线)`;`v7a sk2=544.6us (NOT_BEAT)`。
  - vs rocBLAS 506us:sk16 快约 25us(4.9%),sk8 快约 19us(3.8%)。带宽利用率 22%→23.2%(微升),主要收益来自去 reduce kernel 省 1 次 launch+barrier。
  - 正确性:MAXDIFF=0.0000(全 34816 行 vs bf16→f32 标量 ref,BAD=0)。
  - 仍未触及物理上限(111us = 100% HBM),距 3200GB/s 峰值仍有 4-6x 空间,但已达成"beat 506us"决策表成功条件。
- **路 B v10 双缓冲 BEAT v7(2026-07-13,容器 e03r2n11 / worker-0 173.1.15.7 实测)** — 新最优:
  - 内核:`/public/home/xdzs2026_c150/scalar_splitk_v10.cpp`(本地副本 `tasks/scalar_splitk_v10.cpp`)。
  - 框架:沿用 v7a sk16(1 lane/1 row,4 路 uint4,X 塞 shared,atomic 写 Y,hipMemsetAsync Y=0)。**新增:VGPR 双缓冲/软件流水线**——两组 uint4 缓冲(w_cur 本轮算,w_next 下一轮预取),轮末 swap,让 HBM load W 与 ALU FMA 时间重叠(GPU 硬件级内存流水线支持数十个 outstanding load)。纯 VGPR,不动 LDS 布局。
  - 结果:`v10 sk16=458.8us (24.3% HBM BW, MAXDIFF=0.0000, OK, BEAT_V7)` ← 新最优;`v10 sk8=461.5us (24.1%, BEAT_V7)`;`v10 sk32=459.6us (24.2%, BEAT_V7)`;`v10 sk64=1163.8us MISMATCH(Kseg=80 不能被 32 整除,尾部 bug,非 target)`。
  - vs v7 sk16=481.2us:v10 快 22us(4.6%),带宽 23.2%→24.3%。双缓冲生效,纯 VGPR 预取 swap 实现 load/compute 重叠。
  - **splitK 已饱和**:sk8/sk16/sk32 持平(459-462us),说明再增 splitK 无益,瓶颈转向单 wave 算力/带宽利用。
  - 正确性:sk8/sk16/sk32 MAXDIFF=0.0000(全 34816 行 vs bf16→f32 标量 ref,BAD=0)。
- 演化链:v0(sk2)=459us(文档16 §T1,朴素无向量化)→ v1=10453us → v2=4876us → v3=11235us(倒退)→ v4 uint4=1344us → v5 X-in-shared 4路=529us → v6 2row/8路=1436us(倒退)→ v7 去reduce+atomicAdd=481us(获胜)→ v8 2row=785us(倒退)→ v9 sk1=1327us(倒退)→ **v10 VGPR双缓冲=458.8us(新最优)**。
- 四个关键优化:(1) v4 uint4 向量化加载(8 bf16/lane/加载);(2) v5 X 塞 shared 消除 X 重复 HBM 读(带宽 1-2%→21%);(3) v7 去 reduce kernel atomicAdd 直写(529→481);(4) **v10 VGPR 双缓冲 load/compute 重叠(481→459)**。
- **路 B v11 深度 ILP BEAT v10(2026-07-14,容器 e03r2n04 实测)** — 旧最优:
  - 内核:`/public/home/xdzs2026_c150/scalar_splitk_v11.cpp`(本地副本 `tasks/scalar_splitk_v11.cpp`)。
  - 框架:沿用 v10(1 lane/1 row,X 塞 shared,atomic 写 Y,VGPR 双缓冲)。**新增:每轮 8 路 uint4(64 bf16/轮)替代 v10 的 4 路(32 bf16/轮)**,a0..a7 八路累加。目的:单 wave 内更多 outstanding HBM load 掩盖 FMA 延迟,用满 VGPR 768。关键区别 v6/v8(2 row/lane 倒退):v11 仍 1 row/lane,只深展开单行 K,寄存器压力可控(8 float 累加器 + 双缓冲 16 uint4)。
  - 结果:`v11 sk16=418.5us (26.6% HBM BW, MAXDIFF=0.0000, OK, BEAT_V10)` ← 新最优;`v11 sk8=421.3us (26.4%, BEAT_V10)`;`v11 sk32=687.2us (16.2%, NOT_BEAT_V10, Kseg=160 不能被 64 整除走 4x 退化路径)`。
  - vs v10 sk16=458.8us:v11 快 40us(8.8%),带宽 24.3%→26.6%。深度 ILP 生效。
  - 正确性:sk8/sk16 MAXDIFF=0.0000(BAD=0)。
  - **sk32 倒退根因**:Kseg=160 不能被 64 整除,v11b 退化到 4 路 uint4(32 步)。640→687us 说明 4x 退化路径本身效率低,而非 8x 在大 splitK 失效。需 v12 让 sk32 也走 8x(32 步整除 + 尾部余数处理)确认。
- 演化链(全):v0(sk2)=459us → v1=10453us → v2=4876us → v3=11235us(倒退)→ v4 uint4=1344us → v5 X-in-shared 4路=529us → v6 2row/8路=1436us(倒退)→ v7 去reduce+atomicAdd=481us(获胜)→ v8 2row=785us(倒退)→ v9 sk1=1327us(倒退)→ v10 VGPR双缓冲=458.8us → v11 8路深度ILP=418.5us → **v12 16路深度ILP=403.3us(新最优)**。
- 六个关键优化:(1) v4 uint4 向量化加载;(2) v5 X 塞 shared 消除 X 重复 HBM 读(1-2%→21%);(3) v7 去 reduce kernel atomicAdd 直写(529→481);(4) v10 VGPR 双缓冲 load/compute 重叠(481→459);(5) v11 8 路 uint4 深度 ILP 用满 VGPR(459→418.5);(6) **v12 16 路 uint4 更深 ILP + 双缓冲(418.5→403.3)**。

- **路 B v12 16 路 uint4 更深 ILP BEAT v11(2026-07-14,容器 e03r2n04 实测)** — 新最优:
  - 内核:`/public/home/xdzs2026_c150/scalar_splitk_v12.cpp`(本地副本 `tasks/scalar_splitk_v12.cpp`)。
  - 框架:沿用 v11(1 lane/1 row,X 塞 shared,atomic 写 Y,VGPR 双缓冲)。**新增:每轮 16 路 uint4(128 bf16/轮)替代 v11 的 8 路(64 bf16/轮)**,a0..a15 十六路累加 + 双缓冲 16 个 w_cur/w_next。单 wave 内 16 个 outstanding HBM load 掩盖 FMA 延迟,用满 VGPR 768。三路径版本(`matvec_v12_16x`/`_8x`/`_4x`)按 Kseg 整除性选择(128→16x,64→8x,32→4x)。
  - 结果:`v12 sk8=403.3us (27.6% HBM BW, MAXDIFF=0.0000, OK, BEAT_V11)` ← 新最优(16x 路径);`v12 sk4=414.2us (26.9%, BEAT_V11, 16x)`;`v12 sk16=423.1us (26.3%, NOT_BEAT_V11, 8x 路径 Kseg=320 不被 128 整除)`;`v12 sk32=686.6us (16.2%, NOT_BEAT_V11, 4x 路径)`。
  - vs v11 sk16=418.5us:v12 sk8 快 15.2us(3.6%),带宽 26.6%→27.6%。16 路 ILP 仍递增但放缓(481→459→418.5→403.3,省 22→40→15us)。
  - **关键洞察**:16x 路径在 sk8(waves=4352, 54.4/CU)最优;sk4(16x, 27.2/CU)次之;sk16 被迫走 8x(Kseg=320 不被 128 整除)反而略慢于 v11 的 8x——说明**16x ILP 红利只在能整除 128 的 splitK 上吃到**,sk8 是甜点(Kseg=640=5×128,wave 数适中)。
  - 正确性:sk4/sk8/sk16/sk32 全 MAXDIFF=0.0000(BAD=0)。
  - **距物理上限 111us 仍 3.6x 空间**,纯标量+双缓冲 ILP 收益已放缓。再大幅提速需硬件异步拷贝指令(另一窗口在找,但仍"异步还没消息")或换思路。

- **路 B v12b 通用 16x 主循环实验(2026-07-14,容器 e03r2n04 实测)** — 失败作废:
  - 动机:让 sk16/sk32 也吃 16x 主循环(主循环按 128 步走能整除部分,尾部逐元素降级),消除"sk16 被迫走 8x 退化"问题。
  - 结果:**全部 MISMATCH/倒退**。sk4=562us(NOT),sk8=557us MAXDIFF=6.9(原 v12 sk8=403us),sk16=1340us MAXDIFF=5.8e30,sk32=1413us。尾部逐元素循环 + `kMainEnd2` 边界处理引入正确性 bug(尾部 acc 没加回主累加器路径)且性能反退化。
  - 结论:**v12b 作废删除**,v12 三路径版本(按 Kseg 整除性精确匹配)保持为最优。教训:通用尾部降级路径比精确整除路径更脆弱,不值得为 sk16 单点换 16x。

---

## 附录 A:gate_up 物理参数速查

| 参数 | 值 |
|---|---|
| shape | n=1, m=34816, k=5120 |
| dtype | bf16, bias=False |
| 权重大小 | 34816×5120×2B = 357 MB |
| 输入大小 | 5120×2B = 10 KB |
| FLOP | 2×34816×5120 = 1.78e8 |
| 算术强度 | 0.5 FLOP/B(memory-bound) |
| HBM 带宽 | 3200 GB/s |
| 带宽下限 | 357MB/3200 = 111 us |
| rocBLAS 现状 | 506 us(22% 带宽) |
| CUs | 80 |
| VGPR | 768 |

## 8. 反汇编铁证:mmac 路 load 与标量路同款 global_load,无异步优势(2026-07-14 容器实测)

### 动机

v3 实测 2292.9us(仅比 v2 2442us 微降),未达 <1000us。需定位:mmac 路的 load 到底编出什么指令?是否真有"矩阵指令省 load"的红利?

### 方法

DCC clang17 `--cuda-device-only -S` 出 gfx936 纯 device 汇编(`du_mma.hpp` 14 宏 host 占位 + `--cuda-device-only`)。

### 铁证 1:du_load_matrix_sync 编出标量 global_load

单 `du_load_matrix_sync(aFrag, W, 5120)` kernel(s5)device 汇编指令清单:
```
global_load_ushort   ×1    ← 每 lane 读 1 个 bf16(标量,非向量、非异步)
s_waitcnt            ×2
global_store_dword   ×1
s_endpgm             ×1
(无 buffer_load、无 ds_read、无 v_mmac)
```
**结论**:`du_load_matrix_sync` 编出的是**标量 `global_load_ushort`**(每 lane 一次读 1 个 bf16),与路 B 标量 `v_madmk_f16` 用的 global load **同款**。mmac 路在 load 层**没有任何优势**——load 指令形态、带宽消耗、时延特征与标量路完全相同。

### 铁证 2:mmac 指令只在 MMA 本身

`du_load_matrix_sync(aFrag) + du_load_matrix_sync(bFrag) + du_mma_sync` kernel(s6)device 汇编:
```
global_load_dwordx2  ×2    ← du_load B(bFrag) 用 dwordx2 向量读
v_mmac_f32_16x16x16_bf16 ×1 ← MMA 本体,每指令算 4096 个 MAC
(du_load A 仍是 global_load 标量)
```
**结论**:mmac 的全部价值集中在**单条 `v_mmac` 指令算 4096 MAC**这一点。但 matvec 下 B 片段 1/16 有效 → 4096 MAC 里只 256 有效,**算力红利在 matvec 下被 1/16 稀释**。而 load 层(带宽瓶颈所在)mmac 与标量**完全无差异**。

### 铁证 3:gate_up 是 memory-bound,瓶颈在 load 不在 MMA

- 算术强度 0.5 FLOP/B ≪ 屋顶线拐点 → **纯 memory-bound**。
- 506us 里 22% HBM 带宽 → 瓶颈是 launch/CU 占用/load 效率,**不是 MMA 吞吐**。
- mmac 路 load 与标量路同款 → **mmac 路无法在带宽瓶颈上胜过标量路**。
- 标量路 MMA 用 `v_madmk_f16`(matvec 100% 利用率),mmac 路 MMA 用 `v_mmac`(matvec 1/16 利用率)→ **算力层 mmac 反而更浪费**。

### 路 A 重定性(基于反汇编铁证)

mmac 路在 matvec 下:
- load 层:与标量路同款 `global_load`,**无优势**。
- MMA 层:`v_mmac` 1/16 利用率,**比标量 `v_madmk_f16` 100% 更浪费**。
- shared/sync 层:mmac 需构造 B 片段(col_major 16×16),**比标量全寄存器更重**。

**三条全输**。路 B v7a 已 481us BEAT rocBLAS 506us;路 A v3 仍 2292us(4.8x 慢于标量胜者)。**路 A 在 gate_up matvec 下不成立,符合 §0.2 / 附录 B 预期**。

### v4 决策:路 A 暂停,路 B 接管集成

- 路工程师指导(3 wave/CU + 小流水线 + TMA)的核心是**用 async load 重叠搬运/计算**。但 §9 证明 global→LDS 绕寄存器指令在 gfx936 不可用,且本节证明 mmac load 与标量同款无异步优势 → **路 A 即便做双缓冲,load 指令层也不比标量路强**。
- 路 B 标量路同样可做双缓冲(X 已塞 shared),且无 mmac 的 B 片段浪费。
- **决策:路 A 暂停于 v3,不进 v4。路 B v7a(481us BEAT)进 T3 集成检查点。**
- 路 A 的 mmac 资产(`du_mma.hpp` 编译链 + lane 映射 + 反汇编方法)保留,留作未来 m≥16 满 GEMM 场景(那时 B 片段不浪费)。

---

## 9. global→LDS 绕寄存器指令探测结论:gfx936 不可用(2026-07-14 容器实测)

### 工程师指导原文

> "每个 CU 最多 3 个 wave…需要做很多的小流水线,然后搭配 TMA。需要边搬数据边计算,有一条指令,可以直接让你的数据从 global memory 不通过寄存器到 LDS,就可以一边算一边搬数据。"

### 探测目标

定位"数据从 global memory 不通过寄存器到 LDS"的指令 = global→LDS async load(buffer_load_lds 类)。

### 探测链(逐 builtin 验证)

| builtin(DCC clang17) | 签名 | 前端类型检查 | 后端 codegen |
|---|---|---|---|
| `__builtin_amdgcn_raw_buffer_load_lds` | 7 参(rsrc int4 向量 + lds as(3) ptr + 5 int) | ✅ 通过 | ✗ **Cannot select: intrinsic %llvm.amdgcn.raw.buffer.load.lds** |
| `__builtin_amdgcn_struct_buffer_load_lds` | 8 参 | ✅ 通过 | ✗ 同上 |
| `__builtin_hcu_raw_buffer_load_lds` | 7 参 | ✅ 通过 | ✗ 同上 |
| `__builtin_hcu_struct_buffer_load_lds` | 8 参 | ✅ 通过 | ✗ 同上 |
| 内联汇编 `buffer_load_lds_dword` | — | — | ✗ **invalid instruction** |

### 关键签名细节(供后续工具链升级后复用)

```cpp
typedef int __attribute__((vector_size(16))) rsrc_t;  // buffer resource {base_lo, base_hi, stride_fmt, swizzle}
int __attribute__((address_space(3)))* sp = (int __attribute__((address_space(3)))*)s;  // __shared__ int s[] 强转
__builtin_amdgcn_raw_buffer_load_lds(rsrc, sp, size, offset, soffset, aux, policy);
```
rsrc 必须 `int __attribute__((vector_size(16)))`(4 int 向量),不是 `int4`/`float[16]`;lds ptr 必须 `address_space(3)`。

### 结论

- **builtin 前端存在但 gfx936 LLVM 后端无 selection pattern** → 4 个 builtin 全部 "Cannot select intrinsic"。
- 内联汇编助记符 `buffer_load_lds_dword` 在 gfx936 报 invalid instruction。
- `du_mma.hpp`(1359 行)无 async load / prefetch / global-to-lds 接口,只有 mmac MMA builtin。
- 全 `/opt/dtk/include` grep 无 `prefetch_lds`/`load_to_lds`/`global_to_lds` 等封装。

**工程师说的"global→LDS 绕寄存器"指令在当前 gfx936 DTK 26.04 DCC clang17 工具链下无法生成。** 可能是:① 该指令在 gfx936 硬件上不存在(海光未实现该 buffer_load_lds 变体);② 工具链后端 selection 缺失(待 DTK 升级);③ 该指令以另一名字/builtin 暴露(未找到)。

### 对 v4 的影响

原 v4 计划用该指令做 global→LDS 双缓冲边搬边算。**该指令不可用 → v4 改走普通 `global_load` 双缓冲**(数据 global→寄存器→LDS,仍可重叠搬运/计算,但不绕寄存器,多一跳)。但 §8 铁证表明 mmac 路 load 与标量路同款无优势,故路 A 整体暂停(见 §8 v4 决策)。

---

## 附录 C:本会话(2026-07-14)反汇编探测产物清单

| 产物 | 位置 | 用途 |
|---|---|---|
| s5.cpp / s5.dev.s | 容器 /tmp | 单 du_load_matrix_sync device 汇编,证 load=标量 global_load_ushort |
| s6.cpp / s6.dev.s | 容器 /tmp | du_load+du_mma device 汇编,证 v_mmac_f32_16x16x16_bf16 存在 + B load=global_load_dwordx2 |
| _probeA~Z.sh / _probeAA.sh | home + 本地 tasks/ | builtin 探测 + 反汇编脚本链 |
| 编译命令(反汇编专用) | 见 §0.3 + `--cuda-device-only -S` | 出 gfx936 纯 device 汇编 |

**反汇编方法学(供后续复用)**:`/opt/dtk/dcc/bin/clang++ -x hip --offload-arch=gfx936 --rocm-path=/opt/dtk --rocm-device-lib-path=/opt/dtk/amdgcn/bitcode -I/opt/dtk/include -I/opt/dtk/include/hip -D__HIP_PLATFORM_AMD__ -O3 -Wno-everything --cuda-device-only -S <src>.cpp -o <src>.dev.s`。注意:host pass 仍需 14 mmac 宏占位(`--cuda-device-only` 不跳过 host pass 类型检查)。`llvm-objdump -d --mcpu=gfx936` 对 gfx936 报 "not a recognized processor"(AOMP clang18 不认 gfx936),故用 `--cuda-device-only -S` 直接出汇编,绕过 objdump。

| 维度 | mmac 16x16x16 | 标量 v_madmk_f16 |
|---|---|---|
| 指令利用率(matvec) | 1/16(B片段只col0有效) | 100% |
| shared/sync 开销 | 高(B片段构造) | 无(全寄存器) |
| 物理上限 | 111us(带宽) | 111us(带宽) |
| 逼近上限难度 | 高 | 低 |
| 实现复杂度 | 高 | 低 |
| 适用场景 | 满GEMM(m≥16) | matvec/任意 |
