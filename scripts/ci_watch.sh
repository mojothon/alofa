#!/usr/bin/env bash
# 监视当前 HEAD 对应的那一次 CI 运行。
#
# `gh run watch` 在非交互模式下报 "run ID required when not running interactively"
# —— 它不会替你猜要看哪一次。所以必须自己取 run-id，而且要绑定到 HEAD，不能取列表
# 第一条：推送刚落地的那几秒里，新 run 还没排到列表顶部，取第一条会监视到上一次运行
# （往往是上一次的失败），于是得出完全相反的结论 —— 这条脚本就是为了不再吃这个亏。
set -euo pipefail

sha="$(git rev-parse HEAD)"
run=""
for _ in $(seq 1 24); do
    run="$(gh run list --commit "$sha" --limit 1 --json databaseId -q '.[0].databaseId' || true)"
    [ -n "$run" ] && break
    sleep 5
done

if [ -z "$run" ]; then
    echo "找不到 commit ${sha} 对应的 workflow run" >&2
    exit 1
fi

echo "watching run ${run} for commit ${sha}"
exec gh run watch "$run" --exit-status --interval 10
