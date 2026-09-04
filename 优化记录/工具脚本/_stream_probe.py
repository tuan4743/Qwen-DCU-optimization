#!/usr/bin/env python3
"""诊断 streaming 是否逐 token 返回 + 测 TTFT。非 profile,纯探针。"""
import os
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
import time, json, urllib.request

HOST, PORT, MODEL = "127.0.0.1", 8001, "Qwen3.5-27B"
body = {"model": MODEL,
        "messages": [{"role": "user", "content": "请从1数到10,每个数字单独一行。"}],
        "temperature": 0, "max_tokens": 30, "stream": True}
url = f"http://{HOST}:{PORT}/v1/chat/completions"
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
