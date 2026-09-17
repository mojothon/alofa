#!/usr/bin/env bash
# 编译期红测：未知的 CPU 后端必须**编译失败**。
#
# 为什么需要一道"必须编译不过"的门：QwenForward 的算子分发写成
# `comptime if uses_vector_backend[backend]()`，于是**任何**不是 BACKEND_AVX2
# 的取值都会走进标量分支。拼错一个常量得到的不是报错，而是一个安静跑在标量
# 后端上、所有门全绿的"向量后端" —— 这类失败最该被挡住：它不红，它只是把要
# 验的东西换掉了。
#
# 因此这里编译 tests/fixtures/bad_backend.mojo（里面写了 prefill[7]），要求：
#   1. 编译**失败**（退出码非 0）；
#   2. 失败原因里出现 `unknown cpu backend` —— 否则"编译不过"可能是因为别的
#      毛病（比如缺 main），那这道门就成了恒真。
#
# 用法：pixi run test-backend-guard

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p target
log="target/bad_backend.log"

echo "编译 tests/fixtures/bad_backend.mojo（它应当编译失败）……"
pixi run mojo build -I src tests/fixtures/bad_backend.mojo -o target/bad_backend >"$log" 2>&1
status=$?

if [ $status -eq 0 ]; then
    echo "失败：未知后端编译通过了 —— 它会静默退化成标量后端，整网向量门会照绿"
    exit 1
fi

if ! grep -q "unknown cpu backend" "$log"; then
    echo "失败：编译确实失败了，但不是因为 unknown cpu backend —— 这道门没验到它想验的东西"
    tail -20 "$log"
    exit 1
fi

echo "观测：未知后端如期被编译期拒绝，门有效"
