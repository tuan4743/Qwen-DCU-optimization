#!/usr/bin/env bash
set -u
set -o pipefail

# 模型路径改为工作区下的真实路径（禁止使用 /root）
# --- 保证 EngineCore 子进程能 import fill_alloc_probe (zya/ 在 path) ---
export PYTHONPATH="/public/home/xdzs2026_c150/zya:/usr/local/:${PYTHONPATH}"
export ZYA_HOME="/public/home/xdzs2026_c150/zya"
# --- Triton autotune cache 持久化(P1 候选1:治本减次)---
export TRITON_CACHE_AUTOTUNING=1
mkdir -p /public/home/xdzs2026_c150/zya/triton_autotune_cache
export TRITON_CACHE_DIR=/public/home/xdzs2026_c150/zya/triton_autotune_cache
MODEL_DIR="${MODEL_DIR:-../Qwen3.5-27B}"

# --- 日志：所有输出（含 EngineCore 子进程的 print）都落盘 ---
LOG_DIR="/public/home/xdzs2026_c150/zya/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_start.log"
: > "$LOG_FILE"   # 每次启动清空，避免和旧日志混淆

# --- 把 stdout/stderr 同时打到终端和日志文件（tee） ---
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== start_vllm.sh @ $(date) ==="
echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "VLLM_TRACE_FILL=${VLLM_TRACE_FILL:-<unset>} VLLM_TRACE_FILL_ALL=${VLLM_TRACE_FILL_ALL:-<unset>} VLLM_TRACE_FILL_DISABLE=${VLLM_TRACE_FILL_DISABLE:-<unset>}"

# --- 显式自检：fill_alloc_probe 能否在当前 Python 里 import（结果也会进日志） ---
python - <<'PYCHECK'
try:
    import fill_alloc_probe as f
    print('[selfcheck] import fill_alloc_probe OK  supported=', f._SUPPORTED, 'log=', f._LOG_DIR)
except Exception as e:
    import traceback
    print('[selfcheck] fill_alloc_probe FAILED:', repr(e))
    traceback.print_exc()
PYCHECK

# --- torch profiler 输出目录 ---
PROF_DIR="/public/home/xdzs2026_c150/zya/profile_traces"
mkdir -p "$PROF_DIR"

# 使用 python -m 启动，因为 vllm 命令不可用
# --profiler-config: 让 /start_profile 路由挂载(profile 路由仅在 profiler!=None 时 attach)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name Qwen3.5-27B \
    --port 8001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/public/home/xdzs2026_c150/zya/profile_traces"}'
