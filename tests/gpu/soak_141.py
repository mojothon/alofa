"""长期跑验收（账本一百四十一）：混合负载 + 泄漏观测。

负载面：4 持续线程轮转请求——流式 temp0 / 流式 temp0.8 / 非流式 temp0.8 /
非法请求（应 4xx 且服务不死），连接 keep-alive 与短连接混合。
观测面：每 60s 采样——完成数、失败数、/health 计数器增量、
进程 RSS、fd 数、GPU 显存（nvidia-smi）。

验收：无死锁（各窗口完成数 > 0）、无泄漏（RSS/fd/GPU 稳定或回落）、
错误路径不影响后续请求。
"""
import json
import subprocess
import threading
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8139"
PID_FILE = "/tmp/soak_pid"
PROMPTS = [
    "Once upon a time, in a land far away,",
    "给我讲一个简短的睡前故事。",
    "The little robot looked up at the stars and",
    "春天来了，公园里的花",
]
MAXTOK = 16

stats = {"ok": 0, "bad": 0, "err": 0}
lock = threading.Lock()
stop = False


def req_stream(prompt, temp):
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
    with urllib.request.urlopen(r, timeout=60) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line == "data: [DONE]":
                break


def req_plain(prompt, temp):
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
    with urllib.request.urlopen(r, timeout=60) as resp:
        obj = json.loads(resp.read().decode())
        assert "choices" in obj


def req_bad():
    # 非法：messages 缺失——应 400/4xx 且服务继续活着
    body = json.dumps({"model": "qwen2.5-0.5b", "max_tokens": 4}).encode()
    r = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
    if 400 <= code < 500:
        with lock:
            stats["bad"] += 1
    else:
        with lock:
            stats["err"] += 1


def worker(k):
    i = 0
    while not stop:
        try:
            kind = i % 4
            if kind == 3:
                req_bad()
            else:
                prompt = PROMPTS[i % len(PROMPTS)]
                if kind == 0:
                    req_stream(prompt, 0.0)
                elif kind == 1:
                    req_stream(prompt, 0.8)
                else:
                    req_plain(prompt, 0.8)
            with lock:
                stats["ok"] += 1
        except Exception as e:
            with lock:
                stats["err"] += 1
            print("  worker", k, "failed:", type(e).__name__, e, flush=True)
        i += 1


def snapshot():
    # 进程 RSS（KB）与 fd 数——PID_FILE 记的是 bash 壳，真进程用 pgrep 找
    rss = "?"
    fds = "?"
    try:
        import os
        import subprocess as sp
        pid = sp.check_output(["pgrep", "-x", "mojo"], timeout=10
                              ).decode().split()[0]
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = line.split()[1] + "KB"
                    break
        fds = str(len(os.listdir(f"/proc/{pid}/fd")))
    except Exception:
        pass
    # GPU 显存（GPU3）
    gpu = "?"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "3"],
            timeout=10).decode().strip()
        gpu = out + "MiB"
    except Exception:
        pass
    h = json.loads(urllib.request.urlopen(BASE + "/health", timeout=10).read())
    return h, rss, fds, gpu


def main():
    dur = 1500  # 25 分钟
    for k in range(4):
        threading.Thread(target=worker, args=(k,), daemon=True).start()
    t0 = time.time()
    last = snapshot()
    win = {"ok": 0, "bad": 0, "err": 0}
    prev_ok = stats["ok"]
    print(f"[soak] start rss={last[1]} fds={last[2]} gpu={last[3]} "
          f"counters={last[0].get('forwards')}/{last[0].get('ticks')}",
          flush=True)
    while time.time() - t0 < dur:
        time.sleep(60)
        cur = snapshot()
        ok = stats["ok"] - prev_ok
        prev_ok = stats["ok"]
        hf, ht = cur[0].get("forwards"), cur[0].get("ticks")
        pf, pt = last[0].get("forwards"), last[0].get("ticks")
        print(f"[soak {int(time.time()-t0)}s] ok/min={ok} err={stats['err']} "
              f"bad={stats['bad']} rss={cur[1]} fds={cur[2]} gpu={cur[3]} "
              f"fwd+={max(0,(hf or 0)-(pf or 0))} tick+={max(0,(ht or 0)-(pt or 0))}",
              flush=True)
        # 死锁检测：连续两窗口零完成
        if ok == 0:
            print("[soak] ZERO completions in this window - deadlock?",
                  flush=True)
        last = cur
    print(f"[soak] done ok={stats['ok']} bad={stats['bad']} err={stats['err']}",
          flush=True)
    print("SOAK GATE DONE", flush=True)


if __name__ == "__main__":
    main()
