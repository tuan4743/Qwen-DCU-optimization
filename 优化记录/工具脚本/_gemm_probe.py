#!/usr/bin/env python3
"""Passive record_function probe that labels projection GEMM call sites.

Purpose
-------
Nail down, from a torch-profiler Chrome trace, *which* projection GEMM the
dominant ``Cijk_..._GSU1`` kernel belongs to. The probe wraps each projection
GEMM call with ``torch.profiler.record_function(...)``; the profiler emits a
cat="user_annotation" X event whose ts interval brackets the GEMM and the
underlying kernels, so the offline parser (_parse_gemm_probe.py) can
attribute kernels 1:1 to labels.

Design notes
------------
- Metadata only: nothing is computed, nothing is moved, no big buffers, no
  .cpu() sync, no alloc records. record_function is a near-zero-cost no-op
  when the profiler is off.
- Fully independent from fill_alloc_probe (does not touch the alloc path).

Interface (called by stubs injected with _apply_gemm_probe.py)
--------------------------------------------------------------
    label_in_proj_qkvz()  -> contextmanager wrapping in_proj_qkvz(x)
    label_in_proj_ba()    -> contextmanager wrapping in_proj_ba(x)
    label_out_proj()      -> contextmanager wrapping out_proj(x)

Usage
-----
    Not run directly: _apply_gemm_probe.py patches the call stubs into the
    model file; this module is imported at vLLM startup (put it on
    PYTHONPATH). Output goes straight into the torch profiler trace, no
    separate log file.

Generalization notes
--------------------
- Change ``PREFIX`` and the labels below to probe any operator of any model;
  the offline parser just matches ``PREFIX + label``.
- The per-GEMM shape comments (m=16384 / m=96 / m=5120) are illustrative
  for Qwen3.5-27B GDN layers; the labels are generic.
"""
from contextlib import contextmanager

PREFIX = "GEMM_PROBE::"
LABEL_QKVZ = "in_proj_qkvz"
LABEL_BA = "in_proj_ba"
LABEL_OUT = "out_proj"

# record_function is a near-zero-cost no-op when the profiler is off.
try:
    from torch.profiler import record_function  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover - minimal-env fallback
    _HAS_TORCH = False

    @contextmanager
    def record_function(name):  # type: ignore[no-redef]
        yield


@contextmanager
def label_in_proj_qkvz():
    """Bracket the in_proj_qkvz projection GEMM (largest, m=16384)."""
    with record_function(PREFIX + LABEL_QKVZ):
        yield


@contextmanager
def label_in_proj_ba():
    """Bracket the in_proj_ba projection GEMM (smallest, m=96)."""
    with record_function(PREFIX + LABEL_BA):
        yield


@contextmanager
def label_out_proj():
    """Bracket the out_proj projection GEMM (m=5120)."""
    with record_function(PREFIX + LABEL_OUT):
        yield
