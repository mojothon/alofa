"""HTTP/OpenAI 层并发压测（账本一百三十九）：合批收益穿透验收。

场景：solo（1 并发基线）→ conc2 → conc4 → conc8（满池）→ mixed
（4 流式 temp0 + 4 非流式 temp0.8，137 共存语义验收）。
口径：tokens/s = 完成 token 总数 / 场景墙钟；流式记 TTFT。
"""
import json
import threading
import time
import urllib.request

BASE = "http://localhost:8139"
PROMPTS = [
    "Once upon a time, in a land far away,",
    "给我讲一个简短的睡前故事。",
    "The little robot looked up at the stars and",
    "春天来了，公园里的花",
    "In the middle of winter, the old lighthouse",
    "小明推开家门，发现",
    "Deep in the forest, a hidden door appeared",
    "夜空下，老人缓缓说起",
]
MAXTOK = 16


def req_stream(prompt, temp, out, idx):
    body = json.dumps({
        "model": "qwen2.5-0.5b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": temp,
        "stream": True,
    }).encode()
    r = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ntok = 0
    ttft = None
    with urllib.request.urlopen(r, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
                delta = obj["choices"][0].get("delta", {})
                if delta.get("content"):
                    ntok += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
            except Exception:
                pass
    out[idx] = (ntok, ttft, time.perf_counter() - t0)


def req_plain(prompt, temp, out, idx):
    body = json.dumps({
        "model": "qwen2.5-0.5b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": temp,
        "stream": False,
    }).encode()
    r = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ntok = 0
    with urllib.request.urlopen(r, timeout=120) as resp:
        obj = json.loads(resp.read().decode())
        u = obj.get("usage", {})
        ntok = int(u.get("completion_tokens", 0)) or 1
    out[idx] = (ntok, None, time.perf_counter() - t0)


def run_scene(name, nthreads, nreq_each, temp, nonstream_frac=0.0):
    total_jobs = nthreads * nreq_each
    results = [None] * total_jobs
    threads = []

    def worker(k):
        for ji in range(k, total_jobs, nthreads):
            prompt = PROMPTS[ji % len(PROMPTS)]
            nonstream = (ji % 100) < int(nonstream_frac * 100)
            fn = req_plain if nonstream else req_stream
            try:
                fn(prompt, temp, results, ji)
            except Exception as e:
                results[ji] = (0, None, -1.0)
                print("  req", ji, "FAILED:", e, flush=True)

    t0 = time.perf_counter()
    for k in range(nthreads):
        th = threading.Thread(target=worker, args=(k,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    ntok = sum(r[0] for r in results if r)
    ttfts = [r[1] for r in results if r and r[1] is not None]
    lats = [r[2] for r in results if r and r[2] >= 0]
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    mean_lat = sum(lats) / len(lats) if lats else 0
    tps = ntok / wall if wall > 0 else 0
    print(f"{name:8s} thr={nthreads} reqs={total_jobs} tok={ntok} "
          f"wall={wall:.2f}s tps={tps:.1f} ttft={mean_ttft * 1000:.0f}ms "
          f"lat={mean_lat:.2f}s", flush=True)
    return tps


def main():
    # 一百四十二：档位可从命令行覆盖（默认原五场景）——池上限复测爬坡时
    # 传自定义档，如 `python3 pressure_139.py 1 8 16`。
    import sys
    levels = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]
    base = run_scene("solo", 1, 8, 0.0)
    tps = {}
    for n in levels[1:]:
        tps[n] = run_scene(f"conc{n}", n, 8, 0.0)
    print("---- speedup vs solo ----")
    print("  ".join(f"conc{n} {v / base:.2f}x" for n, v in tps.items()))
    run_scene("mixed", levels[-1], 4, 0.0, nonstream_frac=0.5)
    print("PRESSURE GATE DONE")


if __name__ == "__main__":
    main()
