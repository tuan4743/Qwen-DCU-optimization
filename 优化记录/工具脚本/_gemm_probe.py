#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GDN 投影 GEMM 钉死探针(被动 record_function label)。

目标(对应 `10_gdn_gemm_design.md` §0.5.4 / §5 待验证第1条):
  钉死 decode 稳态下 trace 里的 `Cijk_..._GSU1` 主力 kernel 到底是哪个投影 GEMM。

设计:
  在 `Qwen3NextGatedDeltaNet.forward`(qwen3_next.py:634)的三个投影 GEMM
  调用点各包一层 `torch.profiler.record_function(...)`。torch profiler 会在
  chrome trace 里生成 cat="user_annotation" 的 X 事件,其 ts 区间正好覆盖该
  GEMM 及其触发的底层 kernel。离线看每个 label 区间内落了哪个 `Cijk_GSU1`
  kernel,即可 1:1 对应。

  仅做 record_function 标注 —— 不改算子、不改输入输出、不引入额外计算。
  标注本身是元数据,不影响 kernel 实际执行(开销可忽略,仅 profiler 开启时
  产生 annotation 事件)。

  全程轻量:不分配大 buffer、不做 .cpu() 同步、不记录 alloc 栈。
  与 fill_alloc_probe v2 完全独立(不动 alloc 记录路径)。

接口(由 _apply_gemm_probe.py patch 注入的桩调用):
  label_in_proj_qkvz()   -> contextmanager, 包住 self.in_proj_qkvz(hidden_states)
  label_in_proj_ba()     -> contextmanager, 包住 self.in_proj_ba(hidden_states)
  label_out_proj()       -> contextmanager, 包住 self.out_proj(core_attn_out)

用法:
  不直接运行。由 _apply_gemm_probe.py 把调用桩 patch 进 qwen3_next.py 后,
  随 vllm 启动自动 import 本模块(gemm_probe 需在 PYTHONPATH 上,见 §落地)。
  产物:无独立日志文件,label 直接进 torch profiler trace。
"""
from contextlib import contextmanager

# record_function 在 profiler 未开启时几乎零开销(只构造一个 no-op 上下文)。
try:
    from torch.profiler import record_function  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover - 极端环境兜底
    _HAS_TORCH = False

    @contextmanager
    def record_function(name):  # type: ignore[no-redef]
        yield


@contextmanager
def label_in_proj_qkvz():
    """包住 in_proj_qkvz 投影 GEMM(最大,m=16384)。"""
    with record_function("GEMM_PROBE::in_proj_qkvz"):
        yield


@contextmanager
def label_in_proj_ba():
    """包住 in_proj_ba 投影 GEMM(m=96,最小)。"""
    with record_function("GEMM_PROBE::in_proj_ba"):
        yield


@contextmanager
def label_out_proj():
    """包住 out_proj 投影 GEMM(m=5120)。"""
    with record_function("GEMM_PROBE::out_proj"):
        yield
