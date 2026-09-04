#!/usr/bin/env python3
"""Streaming probe: does the server stream token-by-token? What is TTFT?

Purpose
-------
A zero-profile HTTP probe that connects to a running OpenAI-compatible
endpoint, sends one chat request, then records:
  - latency until HTTP connection returns (first byte of the response),
  - latency until the FIRST "data:" chunk (TTFT proxy),
  - total chunk count / total elapsed.
It answers "is streaming incremental?" and "how long to first token?" without
touching the profiler or the server internals — works against any vLLM /
OpenAI-compatible server.

Usage
-----
    python _stream_probe.py [--host 127.0.0.1] [--port 8001] [--model Qwen3.5-27B]
                            [--prompt "..."] [--max-tokens 30]

Notes
-----
- The proxy bypass (no_proxy/NO_PROXY) is set BEFORE urllib import to avoid
  corporate proxies on 127.0.0.1 (squid returns error pages otherwise).
- The default prompt is a trivial counting task; anything works.
- ``max_tokens`` keeps the stream short for field tests.

Generalization notes
--------------------
- Fully generic: works with any endpoint, model name, port and prompt.
- If the server streams partial tokens per chunk, first-chunk latency is an
  optimistic TTFT; use ``--max-tokens`` >= 2 to observe chunk granularity.
"""
import os
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
import time, json, urllib.request, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="Qwen3.5-27B")
    ap.add_argument("--prompt", default="Count from 1 to 10, one number per line.")
    ap.add_argument("--max-tokens", type=int, default=30)
    args = ap.parse_args()

    body = {"model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0, "max_tokens": args.max_tokens, "stream": True}
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})

    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=120)
    except Exception as e:
        print(f"[!] urlopen failed after +{(time.time()-t0)*1000:.0f}ms: {e}")
        raise

    print(f"[*] urlopen returned (HTTP connected) at +{(time.time()-t0)*1000:.0f}ms", flush=True)

    n = 0
    first_data_ts = None
    for raw in r:
        line = raw.decode(errors="replace").strip()
        if line.startswith("data:") and line != "data: [DONE]":
            n += 1
            if n == 1:
                first_data_ts = time.time()
                print(f"[*] FIRST data: token at +{(first_data_ts-t0)*1000:.0f}ms", flush=True)
            if n <= 5:
                print(f"  [{n}] {line[:80]}", flush=True)

    print(f"[*] total data chunks={n}, elapsed={(time.time()-t0)*1000:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
