#!/usr/bin/env python3
"""P2 decode-only profile: 用 vLLM /start_profile + /stop_profile HTTP 端点,
   在纯 decode 阶段(首请求 prefill 完成后)抓 torch profiler trace,
   拆出单步 decode 内各算子 GPU 耗时分布。

策略:
1. 发一个长输出 streaming 请求(temperature=0, max_tokens=200)。
2. 等 TTFT(首 token 到达)确认已进入 decode。
3. POST /start_profile 抓 ~8s(覆盖 ~110 个 decode step, tpot~70ms)。
4. POST /stop_profile。
5. 打印 trace 落盘路径(profile_traces/),后续用 _parse_profile_trace.py 离线解析。

依赖 vllm 以 --profiler-config 启动(/start_profile 路由仅在 profiler!=None 时挂载)。
在 worker 容器内跑(贴近 vllm, 127.0.0.1:8001)。
"""
import os
# 绕过容器 squid 代理, 让 127.0.0.1 直连 vLLM(否则 urllib 走 http_proxy 返回 squid 错误页)
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
import sys, time, json, urllib.request, threading

HOST = "127.0.0.1"
PORT = 8001
MODEL = "Qwen3.5-27B"
PROFILE_SECONDS = 30  # 长窗口复核占空比代表性: 覆盖 ~400 decode step, 看 idle 分布不只看平均数

def http(method, path, body=None, timeout=30):
    url = f"http://{HOST}:{PORT}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except Exception as e:
        return None, repr(e)

def stream_decode(prompt, max_tokens, ttft_holder):
    """streaming decode, 返回 token 数; ttft_holder[0]=首token到达时间。"""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True}
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    n = 0
    try:
        r = urllib.request.urlopen(req, timeout=180)
        for line in r:
            line = line.decode(errors="replace").strip()
            if line.startswith("data:") and line != "data: [DONE]":
                n += 1
                if ttft_holder[0] is None:
                    ttft_holder[0] = time.time()
    except Exception as e:
        print(f"[!] stream err: {e}", flush=True)
    return n

if __name__ == "__main__":
    t0 = time.time()
    ttft = [None]

    # 1. 触发 decode 请求(streaming, 不阻塞主线程)
    print("[*] starting streaming decode request...", flush=True)
    th = threading.Thread(target=stream_decode,
                          args=("请从1数到200,每个数字单独一行。", 200, ttft), daemon=True)
    th.start()

    # 2. 等到首 token 到达 = 已过 prefill, 进入 decode
    #    TTFT 放宽到 180s: 首请求含 cudagraph capture + Triton autotune warmup, 实测可 >60s。
    print("[*] waiting for first token (TTFT) to confirm decode phase...", flush=True)
    deadline = t0 + 180
    while ttft[0] is None and time.time() < deadline:
        time.sleep(0.2)
    if ttft[0] is None:
        print("[!] no first token in 180s, abort profile", flush=True)
        sys.exit(1)
    ttft_ms = (ttft[0] - t0) * 1000
    print(f"[*] first token at +{ttft_ms:.0f}ms (TTFT). now in decode.", flush=True)

    # 多 buffer 2s 确保 decode 稳态(scheduler 进稳态)
    time.sleep(2)

    # 3. start profile (decode 阶段)
    print(f"[*] POST /start_profile  (will capture ~{PROFILE_SECONDS}s)", flush=True)
    st, resp = http("POST", "/start_profile")
    print(f"    start_profile -> {st} {resp[:200]}", flush=True)
    if st != 200:
        print("[!] start_profile failed, abort", flush=True)
        sys.exit(1)

    # 4. profile N 秒
    time.sleep(PROFILE_SECONDS)

    # 5. stop profile (trace 落盘到 torch_profiler_dir)
    print("[*] POST /stop_profile", flush=True)
    st2, resp2 = http("POST", "/stop_profile", timeout=120)
    print(f"    stop_profile -> {st2} {resp2[:300]}", flush=True)

    # 6. 等请求结束
    th.join(timeout=120)

    # 7. 提示 trace 路径
    print(f"\n[*] done. elapsed={time.time()-t0:.1f}s", flush=True)
    print("[*] trace 落盘在 profile_traces/ (文件名形如 dp0_pp0_tp0_*_rank0.pt.trace.json[.gz])", flush=True)
    print("[*] 离线解析: python _parse_profile_trace.py <trace路径>", flush=True)
