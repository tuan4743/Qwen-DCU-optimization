#!/usr/bin/env python3
"""Patch the V1 GPUModelRunner.__init__ to import fill_capture_hook."""
import sys

p = "/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_model_runner.py"
s = open(p).read()

marker = (
    "    ):\n"
    "        self.vllm_config = vllm_config\n"
    "        self.model_config = vllm_config.model_config\n"
    "        self.cache_config = vllm_config.cache_config\n"
    "        self.offload_config = vllm_config.offload_config\n"
)

block = (
    "    ):\n"
    "        # --- fill_capture_hook (P0-1, V1 runner) ------------------------\n"
    "        # GPUModelRunner (V1, gpu_model_runner.py) is the runner actually\n"
    "        # instantiated when use_v2_model_runner=False (the default). The\n"
    "        # spawn child does NOT inherit VLLM_TRACE_FILL/PYTHONPATH, so import\n"
    "        # unconditionally; hook self-gates on its own _ENABLED.\n"
    "        try:\n"
    "            from vllm import fill_capture_hook  # noqa: F401\n"
    "            print(\"[fill_capture_hook] FILL_HOOK_IMPORT_OK_V1\", flush=True)\n"
    "        except Exception as _e:\n"
    "            print(\"[fill_capture_hook] load failed in GPUModelRunner(V1).__init__:\", _e, flush=True)\n"
    "        # ------------------------------------------------------------------\n"
    "        self.vllm_config = vllm_config\n"
    "        self.model_config = vllm_config.model_config\n"
    "        self.cache_config = vllm_config.cache_config\n"
    "        self.offload_config = vllm_config.offload_config\n"
)

# Guard against double-patch
if "FILL_HOOK_IMPORT_OK_V1" in s:
    print("ALREADY_PATCHED")
    sys.exit(0)

cnt = s.count(marker)
assert cnt == 1, ("marker count", cnt)
s2 = s.replace(marker, block)
assert s2 != s, "replace did nothing"
open(p, "w").write(s2)
print("PATCHED", p)
