#!/usr/bin/env python3
"""Environment introspection probe: vLLM location + GPUModelRunner capture_model.

Purpose
-------
One-shot diagnostics for the vLLM process inside a container:
  - which python, which sys.path entries,
  - where vllm is installed and its version,
  - which module path provides ``GPUModelRunner`` and whether it has
    ``capture_model`` (with file + starting line).

This is the standard "where should my hook go?" first step for instrumentation
and patch scripts (see _patch_*.py / fill_capture_hook.py).

Usage
-----
    python __probe_dcu.py            # run inside the worker container

Generalization notes
--------------------
- ``_RUNNER_CANDIDATES`` covers the current module layouts (vllm.v1.worker.
  gpu.model_runner / vllm.v1.worker.gpu_model_runner / vllm.v1.worker.
  model_runner); edit the list for other vLLM versions.
- Replace ``GPUModelRunner`` / ``capture_model`` with any class+method you
  want to locate; the probing pattern is generic.
"""
import sys, os, importlib, inspect

_RUNNER_CANDIDATES = [
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.model_runner",
]

print("PYEXE", sys.executable)
for p in sys.path:
    if p:
        print("PATH", p)
try:
    import vllm
    print("VLLMFILE", vllm.__file__)
    print("VLLMVER", getattr(vllm, "__version__", "?"))
except Exception as e:
    print("VLLMERR", repr(e))

# locate GPUModelRunner capture_model
for mp in _RUNNER_CANDIDATES:
    try:
        m = importlib.import_module(mp)
        R = getattr(m, "GPUModelRunner", None)
        if R is None:
            print("NORUNNER", mp)
            continue
        print("RUNNER_OK", mp, m.__file__)
        cm = getattr(R, "capture_model", None)
        print("HAS_CAPTURE", cm is not None)
        if cm is not None:
            print("CAPTURE_FILE", inspect.getsourcefile(cm))
            src, start = inspect.getsourcelines(cm)
            print("CAPTURE_LINE", start)
        break
    except Exception as e:
        print("IMPERR", mp, repr(e))
