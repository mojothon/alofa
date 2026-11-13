"""TTFT/延迟 SLO 测量（账本一百四十六）：p50/p90/p99 分位 + 长 prompt 场景。

压测（139）只报均值——分位才是延迟画像。场景：短 prompt（~10 tok）
与长 prompt（~200 tok）× solo/conc4/conc8/conc16；流式（temp0/0.8
各半）+ 每 10 个一个非流式。输出每场景的 TTFT 分位、端到端延迟分位、
吞吐。
"""
import json
import statistics
import threading
import time
import urllib.request

BASE = "http://localhost:8139"

SHORT = [
    "Once upon a time, in a land far away,",
    "给我讲一个简短的睡前故事。",
    "The little robot looked up at the stars and",
    "春天来了，公园里的花",
]
# 长 prompt：重复拼接近 ~200 token（混合中英）。
LONG = [
    " ".join(SHORT) * 6,
    ("给我讲一个简短的睡前故事。" * 12),
]


def req_stream(prompt, temp, ttfts, lats):
    body = json.dumps({
        "model": "qwen2.5-0.5b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": temp,
        "stream": True,
    }).encode()
    r = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=120) as resp:
        first = None
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if first is None and line.startswith("data:") and line != "data: [DONE]":
                first = time.perf_counter() - t0
            if line == "data: [DONE]":
                break
        ttfts.append(first if first is not None else -1.0)
        lats.append(time.perf_counter() - t0)


def req_plain(prompt, temp, ttfts, lats):
    body = json.dumps({
        "model": "qwen2.5-0.5b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": temp,
        "stream": False,
    }).encode()
    r = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=120) as resp:
        resp.read()
    ttfts.append(-1.0)
    lats.append(time.perf_counter() - t0)


def pct(xs, p):
    if not xs:
        return 0.0
    ys = sorted(x for x in xs if x >= 0)
    if not ys:
        return 0.0
    k = min(len(ys) - 1, int(len(ys) * p / 100))
    return ys[k]


def scene(name, nthreads, nreq, prompts):
    ttfts = []
    lats = []
    lock = threading.Lock()
    errors = []

    def worker(k):
        for ji in range(k, nreq, nthreads):
            prompt = prompts[ji % len(prompts)]
            temp = 0.0 if (ji % 2 == 0) else 0.8
            fn = req_plain if (ji % 10 == 9) else req_stream
            try:
                fn(prompt, temp, ttfts, lats)
            except Exception as e:
                with lock:
                    errors.append(f"{type(e).__name__}: {e}")

    ths = []
    t0 = time.perf_counter()
    for k in range(nthreads):
        th = threading.Thread(target=worker, args=(k,))
        th.start()
        ths.append(th)
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0
    tps = nreq * 16 / wall if wall > 0 else 0
    print(f"[{name}] thr={nthreads} reqs={nreq} tps={tps:.1f} "
          f"err={len(errors)}", flush=True)
    if errors:
        for e in errors[:3]:
            print("   ERR:", e, flush=True)
    print(f"   TTFT ms  p50={pct(ttfts,50)*1000:.0f} "
          f"p90={pct(ttfts,90)*1000:.0f} p99={pct(ttfts,99)*1000:.0f}",
          flush=True)
    print(f"   E2E   s  p50={pct(lats,50):.2f} "
          f"p90={pct(lats,90):.2f} p99={pct(lats,99):.2f}", flush=True)
    return tps


def main():
    print("== short prompt (~10 tok) ==", flush=True)
    scene("short-solo", 1, 24, SHORT)
    scene("short-c4", 4, 48, SHORT)
    scene("short-c8", 8, 64, SHORT)
    scene("short-c16", 16, 96, SHORT)
    print("== long prompt (~200 tok) ==", flush=True)
    scene("long-solo", 1, 16, LONG)
    scene("long-c8", 8, 48, LONG)
    scene("long-c16", 16, 64, LONG)
    print("SLO GATE DONE", flush=True)


if __name__ == "__main__":
    main()
