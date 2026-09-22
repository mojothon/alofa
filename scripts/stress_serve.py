#!/usr/bin/env python3
"""alofa 服务的并发压测 harness（只观测、不参与判据）。

这是什么
--------
多线程 keep-alive 客户端，对着一个**已经在跑**的 alofa 服务打
`/v1/chat/completions`，量：QPS、p50/p90/p95/p99/max 延迟、错误分类计数、
服务端 fd 数与 RSS 的前后对比。报告头带环境字段。

铁律（与账本一致）
------------------
本机长期过载（runq 6–27），这里的墙钟数字**只能作同轮相对比较**
（1 worker vs 4 worker 这类），不得当作绝对吞吐声明外引。报告头里
写了这句话，引用时请原样带走。

用法
----
    # 服务自己起（另一个终端或 tmux）：
    ALOFA_WORKERS=4 ALOFA_PORT=8000 ALOFA_MAX_TOKENS=8 pixi run serve
    # 客户端：
    pixi run stress -- --url http://127.0.0.1:8000 --concurrency 8 --duration 60
    # 要观测服务端 fd/RSS，给 master 的 pid（fd 差值会算全部子孙进程之和）：
    pixi run stress -- --url http://127.0.0.1:8000 --duration 60 --master-pid 12345

错误分类
--------
connect_failed / send_failed / timeout / non_200 / bad_body / incomplete —— 
「零错误」必须是分类后的零，不是「没统计」。
"""

import argparse
import datetime
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


def percentiles(latencies_ms):
    if not latencies_ms:
        return {}
    xs = sorted(latencies_ms)
    def pick(p):
        return xs[min(len(xs) - 1, int(len(xs) * p))]
    return {
        "p50": round(pick(0.50), 1),
        "p90": round(pick(0.90), 1),
        "p95": round(pick(0.95), 1),
        "p99": round(pick(0.99), 1),
        "max": round(xs[-1], 1),
    }


class Counter:
    def __init__(self):
        self.lock = threading.Lock()
        self.latencies = []
        self.errors = {
            "connect_failed": 0,
            "send_failed": 0,
            "timeout": 0,
            "non_200": 0,
            "bad_body": 0,
            "incomplete": 0,
        }

    def ok(self, ms):
        with self.lock:
            self.latencies.append(ms)

    def err(self, kind):
        with self.lock:
            self.errors[kind] += 1


def descendant_pids(master_pid):
    """master 的全部存活子孙（worker），外加 master 自己。"""
    pids = [master_pid]
    try:
        with open(f"/proc/{master_pid}/task/{master_pid}/children") as f:
            kids = [int(x) for x in f.read().split()]
    except OSError:
        return pids
    for k in kids:
        pids.extend(descendant_pids(k))
    return pids


def server_snapshot(master_pid):
    """服务端进程树的 fd / RSS / PSS 之和。

    RSS 会把 worker 间写时复制的共享页**每个进程都计一遍**；要证「fork 前加载
    → 权重共享」靠 PSS（按比例分摊共享页）—— 4 个 worker 若各自真加载一份，
    PSS 也会是 4×。
    """
    total_fd = 0
    total_rss = 0
    total_pss = 0
    for pid in descendant_pids(master_pid):
        try:
            total_fd += len(os.listdir(f"/proc/{pid}/fd"))
        except OSError:
            pass
        try:
            with open(f"/proc/{pid}/statm") as f:
                total_rss += int(f.read().split()[1]) * 4  # pages -> KB
        except OSError:
            pass
        try:
            with open(f"/proc/{pid}/smaps_rollup") as f:
                for line in f:
                    if line.startswith("Pss:"):
                        total_pss += int(line.split()[1])
                        break
        except OSError:
            pass
    return {
        "fd": total_fd,
        "rss_kb": total_rss,
        "pss_kb": total_pss,
        "processes": len(descendant_pids(master_pid)),
    }


def one_request(host, port, body, timeout_s, counter):
    """一条连接上打一个请求；错误分类后计入。"""
    t0 = time.monotonic()
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
    except OSError:
        counter.err("connect_failed")
        return
    try:
        sock.settimeout(timeout_s)
        payload = (
            "POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "\r\n"
        ).encode() + body
        sock.sendall(payload)
        # 读到 Content-Length 声明的字节数或 EOF。
        buf = b""
        want = None
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
            head_end = buf.find(b"\r\n\r\n")
            if want is None and head_end > 0:
                for line in buf[:head_end].split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        want = int(line.split(b":")[1])
                if want is not None and len(buf) < head_end + 4 + want:
                    continue
                break
            if want is not None and len(buf) >= head_end + 4 + want:
                break
        ms = (time.monotonic() - t0) * 1000.0
        if not buf.startswith(b"HTTP/1.1 200"):
            if b"HTTP/1.1" in buf[:32]:
                counter.err("non_200")
            else:
                counter.err("incomplete")
        elif b'"content":"' not in buf:
            counter.err("bad_body")
        else:
            counter.ok(ms)
    except socket.timeout:
        counter.err("timeout")
    except OSError:
        counter.err("send_failed")
    finally:
        try:
            sock.close()
        except OSError:
            pass


def worker_loop(host, port, body, deadline, timeout_s, counter):
    while time.monotonic() < deadline:
        one_request(host, port, body, timeout_s, counter)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    ap.add_argument("--master-pid", type=int, default=0,
                    help="服务端 master 的 pid：观测 fd/RSS 前后差（不给就跳过）")
    ap.add_argument("--out", default="", help="把报告追加写进这个 md 文件")
    args = ap.parse_args()

    host = args.url.split("//")[1].split(":")[0]
    port = int(args.url.split(":")[-1].split("/")[0])
    body = json.dumps({
        "model": "qwen2.5-0.5b",
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }).encode()

    loadavg = ""
    try:
        with open("/proc/loadavg") as f:
            loadavg = f.read().split()[:3]
    except OSError:
        pass

    before = server_snapshot(args.master_pid) if args.master_pid else None
    counter = Counter()
    threads = []
    t_start = time.monotonic()
    deadline = t_start + args.duration
    for _ in range(args.concurrency):
        th = threading.Thread(
            target=worker_loop,
            args=(host, port, body, deadline, args.request_timeout, counter),
            daemon=True,
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.monotonic() - t_start
    after = server_snapshot(args.master_pid) if args.master_pid else None

    n_ok = len(counter.latencies)
    qps = n_ok / wall if wall > 0 else 0.0
    p = percentiles(counter.latencies)
    total_err = sum(counter.errors.values())

    lines = []
    lines.append(f"# stress report {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- date: {datetime.date.today().isoformat()} host: {os.uname().nodename} ({os.uname().machine})")
    lines.append(f"- loadavg(1/5/15) at start: {' '.join(loadavg)} — 本机长期过载，**以下数字只能作同轮相对比较，不得当绝对吞吐外引**")
    lines.append(f"- target: {args.url} concurrency={args.concurrency} duration={args.duration:.0f}s max_tokens={args.max_tokens} temperature={args.temperature} prompt_tokens≈{len(args.prompt.split())}")
    if args.master_pid:
        lines.append(
            f"- server pid tree: {before['processes']} procs; "
            f"fd {before['fd']} → {after['fd']} (Δ{after['fd']-before['fd']}); "
            f"RSS {before['rss_kb']//1024}MB → {after['rss_kb']//1024}MB "
            f"(共享页逐进程重复计入); "
            f"PSS {before['pss_kb']//1024}MB → {after['pss_kb']//1024}MB "
            f"(Δ{(after['pss_kb']-before['pss_kb'])//1024}MB, COW 共享分摊后)"
        )
    lines.append("")
    lines.append(f"- requests ok: {n_ok}, errors: {total_err} ({counter.errors})")
    lines.append(f"- QPS: {qps:.2f} | latency ms: p50={p.get('p50')} p90={p.get('p90')} p95={p.get('p95')} p99={p.get('p99')} max={p.get('max')} (mean {round(statistics.fmean(counter.latencies),1) if counter.latencies else '-'})")
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "a") as f:
            f.write(report + "\n\n")

    # 非零退出：有错误就该让 CI/脚本看见，而不是埋在文本里。
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
