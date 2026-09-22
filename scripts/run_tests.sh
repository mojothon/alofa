#!/usr/bin/env bash
# `pixi run test` 的实现：逐个执行"测试套件清单"，并产出「文件 → 通过条数」清单。
#
# 为什么要把条数落盘
# ------------------
# docs/plan/capability-ledger.md 里形如「test_engine_core.mojo（14/14）」的 N/M 只是
# 当时手写的快照，`check-ledger` 又只校验"文件存在"不看数字 —— 于是这些快照会安静
# 腐烂：同一个 test_engine_core.mojo 在账本里曾同时出现 10/10、14/14、15/15 三个
# 互不兼容的数字，而 CI 一直是绿的。这类"越来越不真的话"正是账本要消灭的东西。
#
# 所以这里顺便把每份套件**自报的通过条数**记成 `target/test_counts.tsv`，交给
# `pixi run check-counts` 去核对账本里的 `?count=N`。清单由本次真实运行产出，
# 因此不需要额外跑一遍测试。
#
# 清单在这里单一维护
# ------------------
# 原本这份清单是 pixi.toml 里一串 `&&` 拼起来的长命令（还会嵌套 `pixi run`，每个
# 套件多付一次环境解析开销）。搬到这里后：增删套件只改一处，可读，且能对每份套件
# 单独标注编译参数。
#
# 参数说明：`套件路径|额外参数`（额外参数可为空，空字段别加空格）
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="target/test_counts.tsv"
mkdir -p target
: >"$OUT"

SUITES=(
  "tests/capability/test_libc_ffi.mojo|"
  "tests/capability/test_deps.mojo|"
  "tests/capability/test_layering.mojo|-I src"
  # reactor 能力门（P3.2）：非阻塞 accept 报 EAGAIN、poll 超时返回、跨线程 wakeup、
  # 一个循环盯住多条连接。它 import flare，所以要 -I src 之外的环境依赖 flare
  # （pixi 已装）；不 fork、不加载权重，因此留在 `pixi run test` 里。
  "tests/capability/test_reactor.mojo|-I src"
  # 账本门自己也参与计数：账本里的「能力账本 CI 校验」一条的 evidence 就是它，
  # 少了它这一行就永远核验不到 —— 自证循环会缺一环。
  "tests/capability/test_ledger.mojo|"
  "tests/unit/test_core_error.mojo|-I src"
  "tests/unit/test_core_log.mojo|-I src"
  "tests/unit/test_core_ffi.mojo|-I src"
  "tests/unit/test_core_dtype.mojo|-I src"
  "tests/unit/test_core_tensor.mojo|-I src"
  "tests/unit/test_core_memory.mojo|-I src"
  "tests/unit/test_core_mmap.mojo|-I src"
  "tests/unit/test_verify_roofline.mojo|-I src"
  # 模型格式：config.json / safetensors（含 bf16 就地放宽与 fp16 指名拒绝）。
  "tests/unit/test_model_formats.mojo|-I src"
  "tests/unit/test_tokenizer_parity.mojo|-O0 -I src"
  "tests/unit/test_tokenizer_json.mojo|-O0 -I src"
  "tests/unit/test_layer0_parity.mojo|-I src"
  "tests/unit/test_sampler_parity.mojo|-O0 -I src"
  "tests/unit/test_q4_parity.mojo|-O0 -I src"
  "tests/unit/test_q4_matmul_vec.mojo|-O0 -I src"
  "tests/unit/test_parallel_shards.mojo|-O2 -I src"
  "tests/unit/test_gemm_batch_layout.mojo|-O0 -I src"
  "tests/unit/test_avx2_parity.mojo|-I src"
  "tests/unit/test_scheduler.mojo|-O0 -I src"
  "tests/unit/test_kv_pool.mojo|-O0 -I src"
  "tests/unit/test_paged_attention.mojo|-O0 -I src"
  "tests/unit/test_paged_scatter.mojo|-O0 -I src"
  "tests/unit/test_batch_pool.mojo|-O0 -I src"
  "tests/unit/test_batch_executor.mojo|-O0 -I src"
  "tests/unit/test_engine_core.mojo|-O0 -I src"
  "tests/unit/test_kv_room.mojo|-O0 -I src"
  # HTTP（srv）：线格式与 OpenAI 非流式契约。两块都是纯函数，不需要权重，所以能进
  # `pixi run test`。
  # ⚠️ 真 socket 的那条端到端门（test_http_server.mojo）**不在这里**：它 fork，
  # 而 `mojo run`（JIT）下 fork 会崩编译器 —— 那条走 `pixi run test-http`。
  "tests/unit/test_http.mojo|-I src"
  "tests/unit/test_sse.mojo|-I src"
  "tests/unit/test_openai.mojo|-I src"
  # 连接机（P3.2 reactor 第一步）：入站攒字节 / 请求切分 / 出站队列上限。它没有
  # socket，所以能逐字节钉住 —— 事件循环里最难复现的那几类错在这里是确定的。
  # （真 socket 的那条 reactor 端到端门走 `pixi run test-loop`：它 fork。engine 独占
  # 线程那条端到端门走 `pixi run test-engine`：它 fork + 起线程。）
  "tests/unit/test_conn.mojo|-I src"
)

failed=0
for entry in "${SUITES[@]}"; do
  suite="${entry%%|*}"
  args="${entry#*|}"

  printf '\n===== %s =====\n' "$suite"
  # shellcheck disable=SC2086
  out="$(mojo run $args "$suite" 2>&1)"
  status=$?
  printf '%s\n' "$out"

  if [ "$status" -ne 0 ]; then
    failed=$((failed + 1))
  fi

  # 取 Summary 行里的 "N tests run"；取不到记 -1（由 check-counts 报错）
  n="$(printf '%s\n' "$out" |
    sed -n 's/^Summary \[[^]]*\][[:space:]]*\([0-9][0-9]*\) tests run:.*/\1/p' |
    tail -1)"
  [ -n "$n" ] || n=-1
  printf '%s\t%s\n' "$suite" "$n" >>"$OUT"
done

printf '\n===== counts -> %s =====\n' "$OUT"
cat "$OUT"

if [ "$failed" -ne 0 ]; then
  printf '\nFAILED: %s 份套件未通过\n' "$failed"
  exit 1
fi
