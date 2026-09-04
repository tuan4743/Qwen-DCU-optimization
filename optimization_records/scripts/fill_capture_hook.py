"""Capture-time instrumentation to locate the 256MB int32 fill source.

Purpose
----
The baseline (cudagraph ON) run spends 62.4% of GPU time on
`at::native::vectorized_elementwise_kernel<..., FillFunctor<int>, ...>`,
i.e. an eager `at::native::fill` of a ~256MB int32 buffer
(shape [4096, 16384], 67,108,864 int32 elems). It is captured into the
cudagraph during capture, so a *runtime* hook never sees it (replay does
not re-enter Python). But during the *capture* pass every `zero_/fill_/
zeros` call genuinely dispatches an eager kernel, so a hook active during
capture DOES see it.

This module installs a `TorchDispatchMode` that is **only enabled while a
cudagraph capture is in progress**, and prints the Python call stack for
any int32 fill whose shape matches the target signature.

WHY TorchDispatchMode (not monkey-patching)
-------------------------------------------
vLLM uses torch.compile / dynamo AOT. Tracing runs ops through FakeTensor
mode. Two things break dynamo and must BOTH be avoided:

1. Monkey-patching `torch.zeros` / `Tensor.zero_` — the patched fn
   interferes with fake-tensor device propagation, producing
     "Tensor on device cpu is not on the expected device cuda:0"
   during tracing (see log/ERROR.txt, the v1 failure).
2. Replacing `torch.cuda.graph` at import time — dynamo's AOT tracing
   inspects / depends on `torch.cuda.graph`, and substituting a plain
   function for it corrupts stream-capture bookkeeping and triggers the
   *same* `mul ... cpu` fake-tensor crash during profile_run -> aot_compile
   (the v2 failure). So we NEVER touch `torch.cuda.graph`.

A `TorchDispatchMode` is transparent to dynamo tracing (dynamo ignores
it) and only fires on real eager dispatch — which is exactly the capture
pass. It is entered/exited by wrapping `GPUModelRunner.capture_model`
(vLLM's single capture entry point, run strictly AFTER the dynamo compile
in profile_run), so the mode is never even on the dispatch stack while
dynamo is tracing.

USAGE (DCU box, inside the vLLM serving process)
------------------------------------------------
1. Make sure this file is importable (drop next to the vLLM entry point
   or add its dir to PYTHONPATH).
2. Enable via env var (no vLLM code change):

       export VLLM_TRACE_FILL=1
       # optional tunables (defaults shown):
       export VLLM_TRACE_FILL_DTYPE=int32
       export VLLM_TRACE_FILL_NELEMS=67108864   # 4096*16384
       export VLLM_TRACE_FILL_TOL=0.05          # +/-5% on elem count
       export VLLM_TRACE_FILL_LOG=/tmp/fill_trace.log
       export VLLM_TRACE_FILL_ALL=0             # 1 => log every fill

   PowerShell (if prepping env on Windows side):
       $env:VLLM_TRACE_FILL=1

3. Import early via a sitecustomize.py on PYTHONPATH:
       try:
           import fill_capture_hook  # noqa: F401
       except Exception as e:
           print("[fill_capture_hook] failed to load:", e)

4. Start vLLM normally (cudagraph ON). Capture happens once at startup
   ("Graph capturing finished in N secs..."). Inspect the log afterwards.

5. Once located, unset VLLM_TRACE_FILL and restart — zero runtime cost.

WHY IT MAY HAVE STAYED SILENT BEFORE (fixed here)
-------------------------------------------------
- WRONG MODULE PATH: previous version imported
  `vllm.v1.worker.gpu_model_runner`, but `GPUModelRunner.capture_model`
  actually lives in `vllm.v1.worker.gpu.model_runner` (model_runner.py:515).
  The import failed at sitecustomize time and was swallowed by the
  try/except — so the wrap was never installed and nothing printed.
  Fix: probe BOTH module paths and a lazy-import fallback (see
  `_resolve_capture_model`), and emit a loud self-check on install.
- EAGER IMPORT TIME: wrapping at module import can race with vLLM's own
  import / hit partial init. Fix: wrap is deferred — we monkeypatch the
  first time `capture_model` is looked up, via a guard that retries if
  the runner module isn't importable yet.
- NARROW OP SET: the previous op set missed `aten.full` / `aten.copy_`
  / `aten.empty_like` / `aten.clone` (some "fills" are implemented as
  `copy_` from a zero buffer or `full`). Fix: broaden `_FILL_OP_NAMES`.
- SILENT FAILURE: there was no positive confirmation the hook armed.
  Fix: an explicit "INSTALLED / CAPTURE ENTERED / CAPTURE EXITED /
  n hits" trail so an empty result is diagnosable (no entry line =
  capture_model never ran / wrap never installed; entry but 0 hits =
  the fill is not one of the matched ops or wrong dtype/shape).

NOTES
-----
- The dispatch mode is entered only while capture is active. The capture
  window is bracketed by wrapping `GPUModelRunner.capture_model` (vLLM's
  single cudagraph-capture entry point, run after dynamo compile).
- We never execute the op ourselves: the mode calls `func` to let the
  real dispatch proceed, then inspects the result/args.
- Only the FIRST 8 hits per unique (shape,dtype) are logged to avoid spam.
- Stack is trimmed to remove this module + torch internals, keeping
  vllm + user code, giving a direct `file:line`.
- We deliberately do NOT patch `torch.cuda.graph`: doing so breaks dynamo
  AOT tracing (see WHY section).
"""

from __future__ import annotations

import os
import traceback
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
_ENABLED = os.environ.get("VLLM_TRACE_FILL", "0") not in ("0", "", "false", "False")
_TARGET_DTYPE = os.environ.get("VLLM_TRACE_FILL_DTYPE", "int32")
# 4096 * 16384 = 67108864 int32 elems == 256 MiB
_TARGET_NELEMS = int(os.environ.get("VLLM_TRACE_FILL_NELEMS", str(4096 * 16384)))
_TOL = float(os.environ.get("VLLM_TRACE_FILL_TOL", "0.05"))
_LOG_ALL = os.environ.get("VLLM_TRACE_FILL_ALL", "0") not in ("0", "", "false", "False")
_LOG_PATH = os.environ.get("VLLM_TRACE_FILL_LOG", "")

# state
_capture_depth = 0
_in_capture = False
_seen: dict[tuple, int] = {}
_hit_total = 0

_dtypes_by_name = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "int8": torch.int8,
}
_TARGET_DT = _dtypes_by_name.get(_TARGET_DTYPE, torch.int32)

# fill-like aten ops we care about. aten.zeros / aten.zero_ / aten.fill_ /
# aten.new_zeros all map to the FillFunctor<int> kernel for int32. We also
# include aten.full / aten.copy_ / aten.empty_like / aten.clone: some
# "fills" are materialized as `copy_` from a zero source or `full`. Match
# by canonical name (overloadpacket-qualified), which is stable across
# PyTorch versions and avoids OpOverload-vs-OpOverloadPacket identity
# pitfalls in __torch_dispatch__.
_FILL_OP_NAMES = {
    "aten.zeros.default",
    "aten.zeros.out",
    "aten.zero_.default",
    "aten.fill_.Scalar",
    "aten.fill_.Tensor",
    "aten.new_zeros.default",
    "aten.full.default",
    "aten.full.out",
    "aten.copy_.default",
    "aten.empty_like.default",
    "aten.clone.default",
    "aten.arange.start",
    "aten.arange.default",
}


def _log(msg: str) -> None:
    line = f"[fill_capture_hook] {msg}"
    print(line, flush=True)
    if _LOG_PATH:
        try:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def _dt_name(dt: Any) -> str:
    try:
        return str(dt).removeprefix("torch.")
    except Exception:
        return repr(dt)


def _tensor_meta(x: Any):
    """Return (shape, dtype, numel) for a tensor or fake-tensor, else None."""
    try:
        shape = tuple(int(s) for s in x.shape)
        dt = x.dtype
        numel = 1
        for s in shape:
            numel *= s
        return shape, dt, numel
    except Exception:
        return None


def _matches(shape: tuple, dtype: torch.dtype, numel: int) -> bool:
    if _LOG_ALL:
        return True
    if dtype != _TARGET_DT:
        return False
    if _TARGET_NELEMS > 0:
        lo = _TARGET_NELEMS * (1 - _TOL)
        hi = _TARGET_NELEMS * (1 + _TOL)
        if not (lo <= numel <= hi):
            return False
    if 16384 in shape or 4096 in shape:
        return True
    return _TARGET_NELEMS > 0 and numel == _TARGET_NELEMS


def _record(opname: str, shape: tuple, dtype: torch.dtype, numel: int) -> None:
    global _hit_total
    if not _matches(shape, dtype, numel):
        return
    key = (shape, _dt_name(dtype))
    n = _seen.get(key, 0)
    if n >= 8:
        _hit_total += 1
        return
    _seen[key] = n + 1
    _hit_total += 1
    bytes_ = numel * _dtype_bytes(dtype)
    _log(
        f"HIT #{n + 1} op={opname} dtype={_dt_name(dtype)} "
        f"shape={shape} numel={numel} bytes={bytes_}"
    )
    _log("CALL STACK:\n" + _clean_stack())


def _dtype_bytes(dtype: torch.dtype) -> int:
    try:
        return int(torch.tensor([], dtype=dtype).element_size())
    except Exception:
        return 4


def _clean_stack() -> str:
    frames = traceback.extract_stack()
    out = []
    skip = True
    for fr in frames:
        if skip:
            if "fill_capture_hook" in (fr.filename or ""):
                continue
            skip = False
        fn = fr.filename or ""
        # normalize separators so the filter works on Windows too
        fn_norm = fn.replace("\\", "/")
        if "/torch/_ops" in fn_norm or "/torch/_subclasses" in fn_norm or \
           "/torch/utils/_contextlib" in fn_norm or "/torch/_dynamo" in fn_norm:
            continue
        out.append(
            f'  File "{fn}", line {fr.lineno}, in {fr.name}\n'
            f'    {fr.line or ""}'.rstrip()
        )
    return "\n".join(out) if out else "  <no frames>"


# --------------------------------------------------------------------------- #
# the dispatch mode
# --------------------------------------------------------------------------- #
class FillTraceMode(TorchDispatchMode):
    """Inspect fill-like aten ops while capture is active; otherwise no-op."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if _capture_depth > 0:
            try:
                self._maybe_record(func, args, kwargs)
            except Exception:
                # never let instrumentation break the model
                pass
        return func(*args, **kwargs)

    def _maybe_record(self, func, args, kwargs):
        opname = self._op_name(func)
        if opname not in _FILL_OP_NAMES:
            return
        # locate the subject tensor(s)
        candidates = []
        for a in args:
            if isinstance(a, torch.Tensor):
                candidates.append(a)
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                candidates.append(v)
        # new_zeros: first arg is the base tensor, sizes in args[1]
        for c in candidates:
            meta = _tensor_meta(c)
            if meta is None:
                continue
            shape, dt, numel = meta
            _record(opname, shape, dt, numel)
        # zeros.default / zeros.out / full / arange: shape comes from args,
        # not a tensor.
        if opname.startswith("aten.zeros") or opname.startswith("aten.full") \
                or opname.startswith("aten.arange"):
            try:
                if opname.startswith("aten.arange"):
                    end = int(args[0])
                    start = int(args[1]) if len(args) > 1 else 0
                    step = int(args[2]) if len(args) > 2 else 1
                    n = max(0, (end - start + (step - (1 if step > 0 else -1))) // step)
                    shape = (n,)
                else:  # zeros / full: first arg is the size tuple
                    size = args[0]
                    shape = tuple(int(s) for s in size)
                numel = 1
                for s in shape:
                    numel *= s
                dt = kwargs.get("dtype", torch.get_default_dtype())
                _record(opname, shape, dt, numel)
            except Exception:
                pass

    @staticmethod
    def _op_name(func) -> str:
        # __torch_dispatch__ hands us an OpOverload / OpOverloadPacket.
        # Normalize to "aten.<name>.<overload>" (or "aten.<name>" for packets).
        ns = getattr(func, "namespace", "")
        nm = getattr(func, "name", None)
        ov = getattr(func, "overload_name", "")
        if nm:
            return f"{ns}.{nm}" + (f".{ov}" if ov else "")
        return str(func)


_mode: FillTraceMode | None = None


def _enable_mode() -> None:
    global _mode
    if _mode is None:
        _mode = FillTraceMode()
    _mode.__enter__()  # idempotent-ish; ok if entered multiple times


def _disable_mode() -> None:
    global _mode
    if _mode is not None:
        try:
            _mode.__exit__(None, None, None)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# bracket the capture window by wrapping GPUModelRunner.capture_model
# --------------------------------------------------------------------------- #
# We do NOT touch `torch.cuda.graph`: replacing it with a plain function
# breaks dynamo AOT tracing (the profile_run -> aot_compile path inspects
# stream-capture state and crashes with the same `mul ... cpu` fake-tensor
# error). capture_model() is vLLM's single entry point for cudagraph
# capture and runs strictly AFTER the dynamo compile completes, so wrapping
# it is both safe and exactly brackets the eager pass we want to observe.
#
# Module path note: `GPUModelRunner` lives in
#   vllm.v1.worker.gpu.model_runner  (current vLLM, model_runner.py:102/515)
# Older layouts used `vllm.v1.worker.gpu_model_runner`. We probe both plus a
# lazy retry, so a version mismatch no longer silently disables the hook.
_RUNNER_CANDIDATES = (
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.model_runner",
)


def _resolve_runner_module():
    """Import and return (module, GPUModelRunner) or (None, None).

    Tries each candidate module path; returns the first that has a
    `GPUModelRunner` with `capture_model`.
    """
    import importlib
    last_err = None
    for modpath in _RUNNER_CANDIDATES:
        try:
            mod = importlib.import_module(modpath)
        except Exception as e:  # ImportError / partial init
            last_err = e
            continue
        runner = getattr(mod, "GPUModelRunner", None)
        if runner is not None and hasattr(runner, "capture_model"):
            return mod, runner
    if last_err is not None:
        _log(f"could not import any runner module (last err: {last_err!r})")
    return None, None


def _install_capture_wrap() -> bool:
    mod, runner = _resolve_runner_module()
    if runner is None:
        return False
    target = runner.capture_model
    if getattr(target, "_fill_hook_wrapped", False):
        return True
    _orig_capture = target

    def _wrapped_capture(self, *args, **kwargs):
        global _in_capture, _capture_depth, _hit_total
        _in_capture = True
        _capture_depth = 1
        _hit_total = 0
        _enable_mode()
        _log("capture_model ENTERED — dispatch hook armed "
             "(filter dtype=%s numel~=%d tol=%s)"
             % (_TARGET_DTYPE, _TARGET_NELEMS, _TOL))
        try:
            return _orig_capture(self, *args, **kwargs)
        finally:
            _capture_depth = 0
            _in_capture = False
            _log("capture_model EXITED — dispatch hook disarmed, "
                 "total matched hits=%d" % _hit_total)
            _disable_mode()

    _wrapped_capture._fill_hook_wrapped = True  # type: ignore[attr-defined]
    _wrapped_capture.__wrapped__ = _orig_capture  # type: ignore[attr-defined]
    runner.capture_model = _wrapped_capture  # type: ignore[assignment]
    _log(f"wrapped {runner.__module__}.GPUModelRunner.capture_model")
    return True


def _install_with_retry(max_attempts: int = 60, delay: float = 1.0) -> None:
    """Defer the wrap until the runner module is importable.

    vLLM may not have imported the worker module yet when sitecustomize
    runs. We try once immediately; if that fails we keep retrying via a
    background thread until the module shows up (or we give up). The
    thread only runs at startup and self-terminates once installed.
    """
    if _install_capture_wrap():
        return

    import threading

    def _retry():
        for i in range(max_attempts):
            if _install_capture_wrap():
                return
            try:
                import time
                time.sleep(delay)
            except Exception:
                return
        _log("GIVE UP: runner module not importable after %d retries — "
             "hook NOT installed. Check that vllm.v1.worker.gpu.model_runner "
             "exists in this vLLM build." % max_attempts)

    t = threading.Thread(target=_retry, name="fill_capture_hook_install",
                         daemon=True)
    t.start()


if _ENABLED:
    _install_with_retry()
    _log(
        f"loaded. target dtype={_TARGET_DTYPE} numel={_TARGET_NELEMS} "
        f"tol={_TOL} log_all={_LOG_ALL} log_path={_LOG_PATH or '<stdout>'}"
    )
else:
    # not enabled — silently no-op
    pass
