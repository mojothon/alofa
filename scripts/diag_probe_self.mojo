"""探针自检：不碰模型，只验缓冲/记录/dump 三个动作不越界。"""
from alofa.model.probe_barrier import (
    PROBE_KIND_BIAS,
    PROBE_KIND_HEAD,
    PROBE_KIND_ROWS,
    probe_alloc,
    probe_dispatch,
    probe_dump,
    probe_mark_end,
    probe_mark_start,
    probe_wait_done,
)


def main() raises:
    var probe = probe_alloc()
    for r in range(4):
        var kind = r % 3
        var n = 8
        var rec = probe_dispatch(probe, kind, n)
        print("rec=", String(rec))
        for k in range(n):
            probe_mark_start(probe, rec, k)
            probe_mark_end(probe, rec, k)
        probe_wait_done(probe, rec)
    probe_dump(probe)
    print("自检完成")
