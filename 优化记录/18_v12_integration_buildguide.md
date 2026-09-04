# 18 · v12 split-K GEMM 集成 vLLM 编译指南

> **创建:2026-07-14。**
> **目的**:把路 B v12 split-K kernel(403.3us,beat rocBLAS 506us)集成进 vLLM,作为 FFN gate_up_proj 的 op。
> **目标 shape**:n=1(批1解码), m=34816(out), k=5120(in), bf16, bias=False。
> **集成方式**(文档17 §3.1):独立 .so + torch op 注册,**不动 vllm 原生 `_rocm_C.so`**;改 `utils.py` 加精确 shape 守卫。
> **当前状态**:卡在容器内 `import torch` 失败(lib 找不到),需先解决环境,才能用 `torch.utils.cpp_extension` 编 .so。

---

## 1. 集成总览(T4-T6)

集成分 5 步,顺序依赖:

| 步 | 动作 | 关键文件 | 状态 |
|---|---|---|---|
| S1 | 容器内 `import torch` 跑通(环境) | — | ❌ **当前卡点**(见 §2) |
| S2 | v12 kernel 编成 torch op `.so` | `scalar_splitk_v12.cpp` → `v12_op.cu` + `torch_bindings` | ⬜ 待 S1 |
| S3 | 改 `utils.py` 加 gate_up 精确守卫分支 | `vllm_cscc/.../layers/utils.py` | ⬜ 待 S2 |
| S4 | 编 vllm wheel | `vllm_cscc/` | ⬜ 待 S3 |
| S5 | 装 wheel + 校验 site-packages 干净 | 容器 `/usr/local/lib/python3.10/dist-packages/vllm` | ⬜ 待 S4 |

**铁律(文档17 §5)**:S3 守卫必须精确 `m==34816 and n==1 and k==5120 and bias is None`,不误伤 lm_head/down/qkvz(重蹈 C4-3 灾难)。S5 校验干净前不让用户重启。最终编译启动 vllm 验证用户来更快,其它直接做。

---

## 2. 卡点:容器内 `import torch` 失败(S1)

### 2.1 现象

容器 `root@173.0.58.3` 内裸 `python -c "import torch"`:
```
ImportError: libgalaxyhip.so.5: cannot open shared object file: No such file or directory
```
`source /root/.bashrc` 后改报:
```
ImportError: librocm_smi64.so.2: cannot open shared object file: No such file or directory
```
但 **vllm 实际在跑**(`pgrep` = PID 49783/49784),说明正确环境存在,只是我的 shell 没 source 对。

### 2.2 已知:运行中 vllm 进程的环境(从 /proc/PID/environ 抓到)

这是**已知能跑**的环境,可以直接复制:

```bash
# —— 运行中 vllm (PID 49783) 的真实环境 ——
export HIP_PATH=/opt/dtk/hip
export HIP_VISIBLE_DEVICES=0
export DTKROOT=/opt/dtk
export ROCM_PATH=/opt/dtk
export PATH=/opt/ucx/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/dtk/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk/opencl/bin:/opt/mpi/bin:/opt/hwloc/bin/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib:
```

⚠️ **注意**:`HIP_VISIBLE_DEVICES_IDX=5` 是 vllm 进程的设备绑定,**编译/bench 时不要照抄**,改用 `HIP_VISIBLE_DEVICES=0` 或不设(让 hipSetDevice 选 0)。

### 2.3 待用户确认/排查项(我来手动找)

1. **为什么 `source /root/.bashrc` 后还缺 librocm_smi64.so.2**:
   - `/root/.bashrc` 里有 PATH/LD_LIBRARY_PATH 导出(见下),但 LD 链里 `/opt/dtk/hip/lib` 应含 librocm_smi —— 需确认该 .so 实际在哪个目录。
   - `/root/.bashrc` 导出的 LD(从 grep 拿到):
     ```
     /opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib:
     ```
     —— 与 §2.2 进程环境**完全一致**。所以 `source /root/.bashrc` 理论上应该能 import torch。
   - **疑点**:`source /root/.bashrc` 时可能因 `~/.bashrc` 顶部 `[ -z "$PS1" ] && return`(非交互直接 return)提前退出,没执行到 export。**需用 `bash -lc` 或显式 source 全程**。

2. **libgalaxyhip.so.5 实体位置**(find 已确认):
   ```
   /opt/dtk-26.04-DCC2602-0317/hip/lib/libgalaxyhip.so.5   ← 实体
   ```
   `/opt/dtk` 是 `/opt/dtk-26.04-DCC2602-0317` 的软链?需 `ls -la /opt/dtk` 确认。若是软链,§2.2 的 LD 已覆盖(`/opt/dtk/hip/lib`)。

3. **librocm_smi64.so.2 位置**:未 find。可能在 `/opt/dtk/rocm_smi/lib` 或 `/opt/dtk/lib`。需 `find /opt -name "librocm_smi64.so*"`。

### 2.4 推荐验证命令(S1 通过判据)

```bash
# 容器内,逐条验证
source /root/.bashrc   # 若提前 return,改 bash -lic 或直接 export §2.2 那段
python -c "import torch; print(torch.__version__); p=torch.cuda.get_device_properties(0); print(p.name, p.major, p.minor)"
# 期望:torch 版本 + gfx936 设备名
python -c "import vllm; print(vllm.__file__)"
# 期望:/usr/local/lib/python3.10/dist-packages/vllm/__init__.py
python -c "import torch.utils.cpp_extension as c; print('CUDA_HOME=', c.CUDA_HOME)"
# 期望:CUDA_HOME 指向 /opt/dtk(hipcc 工具链)
```

S1 通过后,S2 才能用 `torch.utils.cpp_extension.load` 编 .so。

---

## 3. S2:v12 kernel 编 torch op `.so`

### 3.1 集成策略:独立 .so,不动原生 `_rocm_C.so`

- vLLM 原生 ROCm op 在 `csrc/rocm/skinny_gemms.cu` + `csrc/rocm/torch_bindings.cpp`,编译进 `vllm._rocm_C`。
- **不往里面塞 v12**(避免重编整个 vllm + 风险)。
- 改用**独立 PyTorch extension**:`vllm_cscc/csrc/rocm/v12_splitk.cu` + 独立 binding,编成独立 `.so`,`torch.ops.load_library` 加载。

### 3.2 v12 op 签名设计

参考原生 `LLMM1` 签名(`csrc/rocm/ops.h:5`):
```cpp
torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b, const int64_t rows_per_block);
// in_a = weight [M,K], in_b = x [N,K], N==1, 返回 [N,M]
```
v12 op 签名(对齐):
```cpp
// in_weight: [M,K] bf16, in_x: [N,K] bf16 (N==1), 返回 [N,M] bf16
torch::Tensor gate_up_splitk_v12(
    const at::Tensor& in_weight,
    const at::Tensor& in_x);
```
**注意 dtype 转换**:v12 kernel 输出是 **float**(atomicAdd 累加),需在 host 端:
1. `hipMemsetAsync(Y_float, 0, ...)` 清零;
2. launch `matvec_v12_16x`(Kseg%128==0 时)等三路径;
3. kernel 跑完把 float Y 转 bf16 写进 out_c(M 个元素,bf16)。
4. splitK 选 **sk8**(Kseg=640, 16x 路径, 403.3us 最优),硬编码 splitK=8。

### 3.3 v12 三路径选择(从 bench_v12 逻辑搬)

```cpp
int splitK = 8;               // 硬编码最优 splitK
int Kseg = K / splitK;        // K=5120, Kseg=640
const char* path;
if (Kseg % 128 == 0)      path = "16x";   // 640%128==0 → 16x ✓ (最优)
else if (Kseg % 64 == 0)  path = "8x";
else if (Kseg % 32 == 0)  path = "4x";
// K=5120, splitK=8 → Kseg=640 → 16x 路径
```

### 3.4 编译命令(草稿,S1 通过后用)

```bash
# 容器内,source §2.2 环境后
cd /public/home/xdzs2026_c150/zya/vllm_cscc/csrc/rocm

# 方式A:torch.utils.cpp_extension.load(运行时编,自动缓存)
python build_v12_op.py   # 见 §3.5

# 方式B:hipcc 直接编 .so(更可控,需手写 binding)
/opt/dtk/bin/hipcc --offload-arch=gfx936 -O3 -shared -fPIC \
  -I/opt/dtk/include -I/opt/dtk/include/hip \
  -I$(python -c "import torch; print(torch.__path__[0])")/include \
  -I$(python -c "import torch; print(torch.__path__[0])")/include/torch/csrc/api/include \
  v12_splitk.cu -o v12_splitk.so \
  -L/opt/dtk/hip/lib -lamdhip64 \
  -L$(python -c "import torch; print(torch.__path__[0])")/lib -ltorch_python
```

### 3.5 build_v12_op.py(草稿)

```python
# 用 torch.utils.cpp_extension 编 v12 .so 并注册 op
import os
from torch.utils.cpp_extension import load
os.environ["HIPCC_FLAGS"] = "--offload-arch=gfx936 -O3"
ext = load(
    name="v12_splitk",
    sources=["csrc/rocm/v12_splitk.cu"],
    extra_include_paths=["/opt/dtk/include", "/opt/dtk/include/hip"],
    extra_ldflags=["-L/opt/dtk/hip/lib", "-lamdhip64"],
    verbose=True,
)
print("LOADED:", ext)
# 验证 op 存在
print(torch.ops.v12_splitk)
```

### 3.6 v12_splitk.cu 结构(待写,基于 scalar_splitk_v12.cpp)

```
1. #include <torch/extension.h> + <hip/hip_runtime.h> + <hip/hip_bf16.h>
2. 搬 matvec_v12_16x kernel(原样,改 W/X/Y 参数类型适配 at::Tensor)
3. host 函数 gate_up_splitk_v12_impl(weight, x):
   - TORCH_CHECK shape/dtype/device
   - M=weight.size(0), K=weight.size(1), N=x.size(0)==1
   - 分配 float* dY(M), hipMemsetAsync 0
   - launch matvec_v12_16x(grid=(M/64, 8), block=64, smem=K*2, ...)
   - float→bf16 转 out_c [N,M]
4. TORCH_LIBRARY 注册 op "gate_up_splitk_v12"
```

---

## 4. S3:改 utils.py 加 gate_up 守卫

### 4.1 守卫位置(`rocm_unquantized_gemm_impl`,utils.py ~122-208)

当前 gate_up(n=1,m=34816,k=5120,bias=None)命中:
```python
elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:
    out = ops.LLMM1(weight, x_view, 4)   # ← 当前走这里(单网格 GSU1, CU 占用差)
```
要在 `use_skinny` 守卫**之前**(更早分支)插入 v12 守卫,精确 shape 命中走 v12 op。

### 4.2 守卫代码(草稿)

```python
def rocm_unquantized_gemm_impl(x, weight, bias=None):
    from vllm.platforms.rocm import on_gfx9, on_gfx950
    n = x.numel() // x.size(-1); m = weight.shape[0]; k = weight.shape[1]
    # ... 原有 N_p2 / rndup_cus / fits_wvsplitkrc 计算 ...

    # === v12 split-K gate_up 守卫(新增,精确 shape,只 gate_up 命中)===
    use_v12_splitk = (
        on_gfx9()                       # gfx936 在 on_gfx9 内(rocm.py:149 含 gfx936)
        and x.dtype == torch.bfloat16
        and n == 1
        and m == 34816                  # 精确 gate_up out
        and k == 5120                   # 精确 gate_up in
        and bias is None
        and weight.is_contiguous()
    )
    if use_v12_splitk:
        x_view = x.reshape(-1, x.size(-1))
        out = ops.gate_up_splitk_v12(weight, x_view)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    # === 原有分支不动 ===
    use_skinny_reduce_counting = (...)
    if use_skinny_reduce_counting: ...
    ...
```

⚠️ **shape 精确守卫铁律**:只 `m==34816 and n==1 and k==5120`,不宽放。lm_head(M=151936)/down_proj(qkvz)/其它 GEMM 一律不命中,回退原路径。

⚠️ **回退**:删掉这一段 if 即回滚到纯 baseline(C1)。

### 4.3 op 注册对接

`ops.gate_up_splitk_v12` 要在 vllm 的 `_custom_ops.py` 里加一个 thin wrapper 指向 `torch.ops.vllm.gate_up_splitk_v12`(若用 vllm 命名空间)或 `torch.ops.v12_splitk.gate_up_splitk_v12`(独立 .so 命名空间)。**待定**:走 vllm 命名空间需编进 `_rocm_C.so`;走独立 .so 命名空间需 `torch.ops.load_library` 加载(推荐,不动原生)。

---

## 5. S4:编 vllm wheel

```bash
cd /public/home/xdzs2026_c150/zya/vllm_cscc
source /root/.bashrc   # §2.2 环境
# 编译(走 vllm 自带 build,产 dist/*.whl)
python setup.py bdist_wheel  # 或 pip wheel .
```
产出:`dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl`(覆盖现有 50786809 字节 Jul 13 版)。

⚠️ 若 S2 走独立 .so 路线(不编进 _rocm_C),S4 **可能只需改 utils.py 不重编 wheel** —— 但 utils.py 在 wheel 里,改了必须重编/重装。需确认:改 utils.py 后是重编 wheel 还是直接改 site-packages 的 utils.py(后者更快但脏,违反"site-packages 干净"铁律)。

---

## 6. S5:装 wheel + 校验 site-packages 干净

```bash
# 容器内
pip install --force-reinstall --no-deps /public/home/xdzs2026_c150/zya/vllm_cscc/dist/vllm-*.whl
# 校验
python -c "import vllm; print(vllm.__file__)"
# 期望:/usr/local/lib/python3.10/dist-packages/vllm/__init__.py
# 校验 utils.py 含 v12 守卫:
grep -n "gate_up_splitk_v12" /usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/utils.py
# 期望:命中守卫 if 块
# 校验无残留旧 patch:
python -c "from vllm.model_executor.layers.utils import rocm_unquantized_gemm; print('OK')"
```
校验通过才让用户重启 vllm 实测三段(4-8K/8-16K/16-32K)对比 baseline 18.26/12.30/8.61。

---

## 7. 文件清单

| 文件 | 位置 | 状态 |
|---|---|---|
| v12 kernel 源 | `/public/home/xdzs2026_c150/scalar_splitk_v12.cpp`(md5=8eca51ec) | ✅ 已有(standalone bench) |
| v12 cu 移植 | `/public/home/xdzs2026_c150/zya/vllm_cscc/csrc/rocm/v12_splitk.cu` | ⬜ 待写(S2) |
| build 脚本 | `/public/home/xdzs2026_c150/zya/vllm_cscc/csrc/rocm/build_v12_op.py` | ⬜ 待写(S2) |
| 守卫 | `/public/home/xdzs2026_c150/zya/vllm_cscc/vllm/model_executor/layers/utils.py` | ⬜ 待改(S3) |
| wheel | `/public/home/xdzs2026_c150/zya/vllm_cscc/dist/*.whl` | ⬜ 待编(S4) |
| 原生 ops.h | `csrc/rocm/ops.h`(LLMM1/wvSplitK 声明) | 参考 |
| 原生 binding | `csrc/rocm/torch_bindings.cpp`(TORCH_LIBRARY_EXPAND) | 参考 |
| 原生 LLMM1 host | `csrc/rocm/skinny_gemms.cu:242-300`(host 模板) | 参考 |

---

## 8. 关键事实速查(防遗忘)

- **gfx936 在 `on_gfx9()` 内**:`rocm.py:149` `_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950", "gfx936"])` → 守卫用 `on_gfx9()` 即可命中 gfx936(无 `on_gfx936` 函数)。
- **v12 输出是 float 非 bf16**:atomicAdd 累加到 float Y,host 端转 bf16。
- **v12 最优 splitK=8**:Kseg=640, 16x 路径, 403.3us, 27.6% HBM BW, MAXDIFF=0.0000。
- **物理上限 111us**(357MB/3200GB/s),v12 距上限 3.6x 空间,但已 beat rocBLAS 506us。
- **gfx936 硬件约束**(memory):无 MFMA(VMFault),每 CU 最多 3 wave,VGPR 768 不降 occupancy 直到跨 128/256 档。
- **当前容器 vllm 在跑**:PID 49783/49784, port 8888(`vllm serve`,不是 start_vllm.sh 的 8001)。
- **DTK 实体**:`/opt/dtk-26.04-DCC2602-0317`,`/opt/dtk` 疑似软链(待确认)。
