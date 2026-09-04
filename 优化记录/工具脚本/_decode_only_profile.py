#!/usr/bin/env python3
"""Decode-only torch profiler capture via vLLM HTTP endpoints.

Purpose
-------
Capture a torch profiler Chrome trace covering **pure decode phase** (after
the first request's prefill finished), so downstream analysis separates
per-step operator GPU cost from prefill / TTFT effects.

Strategy
--------
1. Start a long streaming request (temperature=0, max_tokens=N).
2. Wait for the first token (TTFT) => we are in decode now.
3. POST /start_profile (start_time/end_time are irrelevant; capture runs for
   ``--profile-seconds``), then /stop_profile.
4. The trace lands in the vLLM profiler dir; parse offline with
   ``_parse_profile_trace.py``.

Requirements
------------
- vLLM must be started with ``--profiler-config`` so that the
  ``/start_profile`` + ``/stop_profile`` routes exist (profiler != None).
- Run near the server (127.0.0.1) inside the worker container.

Usage
-----
    python _decode_only_profile.py [--host 127.0.0.1] [--port 8001]
                                   [--model Qwen3.5-27B] [--max-tokens 200]
                                   [--profile-seconds 30] [--prompt "..."]

Generalization notes
--------------------
- Any OpenAI-compatible vLLM endpoint works: model name, port, prompt and
  window length are all CLI parameters.
- TTFT timeout is generous (180s) because the FIRST request includes
  cudagraph capture + Triton autotune warmup (>60s observed); for a
  warm server reduce ``timeout`` in ``http()`` if you like.
- Proxy bypass (no_proxy) is set before urllib import to keep 127.0.0.1
  direct (corporate squid proxies return error pages otherwise).
"""
import os
# Bypass container squid proxy so 127.0.0.1 connects directly (urllib would
# otherwise honor http_proxy and get squid error pages back).
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
import sys, time, json, urllib.request, threading, argparse


def http(host, port, method, path, body=None, timeout=30):
    url = f"http://{host}:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except Exception as e:
        return None, repr(e)


def stream_decode(host, port, model, prompt, max_tokens, ttft_holder):
    """streaming decode; returns token count; ttft_holder[0] = first-token time."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True}
    url = f"http://{host}:{port}/v1/chat/completions"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="Qwen3.5-27B")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--profile-seconds", type=int, default=30,
                    help="capture window in seconds (30s covers ~400 decode steps for TPOT~70ms)")
    ap.add_argument("--prompt", default="Count from 1 to 200, one number per line.")
    args = ap.parse_args()

    t0 = time.time()
    ttft = [None]

    # 1. kick off the streaming decode request (non-blocking on the main thread)
    print("[*] starting streaming decode request...", flush=True)
    th = threading.Thread(target=stream_decode,
                          args=(args.host, args.port, args.model, args.prompt,
                                args.max_tokens, ttft), daemon=True)
    th.start()

    # 2. wait for the first token => prefill done, we are in decode
    #    (TTFT is generous: the first request includes cudagraph capture +
    #     Triton autotune warmup, >60s observed)
    print("[*] waiting for first token (TTFT) to confirm decode phase...", flush=True)
    deadline = t0 + 180
    while ttft[0] is None and time.time() < deadline:
        time.sleep(0.2)
    if ttft[0] is None:
        print("[!] no first token in 180s, abort profile", flush=True)
        sys.exit(1)
    ttft_ms = (ttft[0] - t0) * 1000
    print(f"[*] first token at +{ttft_ms:.0f}ms (TTFT). now in decode.", flush=True)

    # small extra buffer so the scheduler settles into steady state
    time.sleep(2)

    # 3. start profile (decode phase)
    print(f"[*] POST /start_profile  (will capture ~{args.profile_seconds}s)", flush=True)
    st, resp = http(args.host, args.port, "POST", "/start_profile")
    print(f"    start_profile -> {st} {resp[:200]}", flush=True)
    if st != 200:
        print("[!] start_profile failed, abort", flush=True)
        sys.exit(1)

    # 4. profile N seconds
    time.sleep(args.profile_seconds)

    # 5. stop profile (trace lands in torch_profiler_dir)
    print("[*] POST /stop_profile", flush=True)
    st2, resp2 = http(args.host, args.port, "POST", "/stop_profile", timeout=120)
    print(f"    stop_profile -> {st2} {resp2[:300]}", flush=True)

    # 6. wait for the request to finish
    th.join(timeout=120)

    # 7. reminder: where the trace landed
    print(f"\n[*] done. elapsed={time.time()-t0:.1f}s", flush=True)
    print("[*] trace will be under the vLLM profiler dir "
          "(e.g. profile_traces/ as dp0_pp0_tp0_*_rank0.pt.trace.json[.gz])", flush=True)
    print("[*] offline parse: python _parse_profile_trace.py <trace path>", flush=True)


if __name__ == "__main__":
    main()
