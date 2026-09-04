#!/usr/bin/env python3
"""Reorder _RUNNER_CANDIDATES so V1 (gpu_model_runner) is probed first,
and log which runner module actually got wrapped."""
import sys

p = "/usr/local/lib/python3.10/dist-packages/vllm/fill_capture_hook.py"
s = open(p).read()

old = (
    '_RUNNER_CANDIDATES = (\n'
    '    "vllm.v1.worker.gpu.model_runner",\n'
    '    "vllm.v1.worker.gpu_model_runner",\n'
    '    "vllm.v1.worker.model_runner",\n'
    ')'
)
new = (
    '# P0-1 fix (2026-06-29): V1 runner (gpu_model_runner.py) is the one\n'
    '# actually instantiated when use_v2_model_runner=False (the default).\n'
    '# Probing gpu.model_runner (V2) first caused the wrap to land on the\n'
    '# V2 class capture_model, while V1 capture_model ran un-wrapped ->\n'
    '# 0 ENTERED. Probe V1 FIRST.\n'
    '_RUNNER_CANDIDATES = (\n'
    '    "vllm.v1.worker.gpu_model_runner",\n'
    '    "vllm.v1.worker.gpu.model_runner",\n'
    '    "vllm.v1.worker.model_runner",\n'
    ')'
)

if "Probing gpu.model_runner (V2) first caused" in s:
    print("ALREADY_PATCHED")
    sys.exit(0)

cnt = s.count(old)
assert cnt == 1, ("old marker count", cnt)
s2 = s.replace(old, new)
assert s2 != s, "replace did nothing"

open(p, "w").write(s2)
print("PATCHED", p)
