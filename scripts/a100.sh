#!/usr/bin/env bash
# alofa —— A100 远程验证机的同步与运行助手
#
# 背景（2026-09-16 实测，见 docs/plan/02-architecture.md §11）：
#   开发机 GPU 是 Maxwell sm_52，现代 CUDA 栈与 MAX 均不支持。
#   远程验证机 lcl@10.107.6.60 有 6×A100（280GB，CC 8.0）。
#
# 三个必须知道的坑：
#   1. 远程机【无外网】→ 不能 pixi install，必须本地装好后连同 .pixi 一起 rsync
#   2. MAX 26.5 要求驱动 ≥580，该机是 560.35.03
#      → 必须 export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas 绕过
#   3. 这是【共享机】：GPU 0-3 常满载。务必先用 `./scripts/a100.sh gpu` 看空闲卡，
#      再用 CUDA_VISIBLE_DEVICES=<空闲卡> 限定，不要占用他人资源。
#   4. 【SSH 端口是 3389，不是 22】—— 22 与 2222 实测全关，只有 3389 开。
#      2026-09-17 记下的"连接超时"就是脚本一直敲 22 端口造成的，机器本身没坏。
#      （3389 通常被当作 RDP 端口，这里拿来跑 SSH，属于该机的运维约定。）
#
# 用法：
#   ./scripts/a100.sh sync            # 同步项目（含 .pixi）到远程机
#   ./scripts/a100.sh gpu             # 查看远程 GPU 占用，挑一张空闲卡
#   ./scripts/a100.sh run 4 FILE.mojo # 在 GPU 4 上运行
#   ./scripts/a100.sh shell           # 打开已配好环境的远程 shell
#   ./scripts/a100.sh probe [GPU_ID]  # 跑 GPU kernel 冒烟测试（默认卡 4）
#
set -euo pipefail

REMOTE_HOST="${ALOFA_A100_HOST:-lcl@10.107.6.60}"
# ⚠️ 见文件头坑 4：该机 SSH 监听 3389，22/2222 均关闭。
REMOTE_PORT="${ALOFA_A100_PORT:-3389}"
REMOTE_ROOT="${ALOFA_A100_ROOT:-/app/lcl/mojo-projects/alofa}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI_ENV_REL=".pixi/envs/default"

# 远程侧必须设置的环境
REMOTE_ENV='
export PIXI_ENV="__ROOT__/'"$PIXI_ENV_REL"'"
export PATH="$PIXI_ENV/bin:$PATH"
export MODULAR_HOME="$PIXI_ENV/share/max"
export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
'

remote_env() {
  echo "$REMOTE_ENV" | sed "s|__ROOT__|$REMOTE_ROOT|g"
}

cmd_sync() {
  echo "==> 同步 $LOCAL_ROOT -> $REMOTE_HOST:$REMOTE_ROOT"
  echo "    （含 $PIXI_ENV_REL；远程机无外网，不能在那里 pixi install）"
  rsync -a --info=progress2 -e "ssh -p $REMOTE_PORT" \
    --exclude '.git/' \
    --exclude 'target/' \
    --exclude '__pycache__/' \
    --exclude '*.mojopkg' \
    "$LOCAL_ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/"

  # ⚠️ pixi 环境【不可重定位】：`.pixi/envs/default/share/max/modular.cfg` 里硬编码着
  #    **本机**的绝对路径（package_root / cache_dir / path）。rsync 到远程后路径对不上，
  #    mojo 会报 `unable to locate module 'std'`（连 print / range 都找不到，看起来像
  #    语法错误，其实是环境问题）。所以每次同步完都必须按远程实际路径重写一遍 ——
  #    否则「同步成功但环境被改坏」，比不同步还难查。
  echo "==> 重写 modular.cfg 里的本机绝对路径 -> 远程路径"
  # shellcheck disable=SC2029
  ssh -p "$REMOTE_PORT" -o BatchMode=yes "$REMOTE_HOST" \
    "sed -i 's|$LOCAL_ROOT|$REMOTE_ROOT|g' '$REMOTE_ROOT/.pixi/envs/default/share/max/modular.cfg'"
  echo "==> 同步完成"
}

cmd_gpu() {
  echo "==> 远程 GPU 状态（挑一张 used 小、util 低的卡）"
  # shellcheck disable=SC2029
  ssh -p "$REMOTE_PORT" -o BatchMode=yes "$REMOTE_HOST" \
    'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv'
}

cmd_run() {
  local gpu="${1:?用法: a100.sh run <GPU_ID> <FILE.mojo> [args...]}"
  shift
  local file="${1:?用法: a100.sh run <GPU_ID> <FILE.mojo> [args...]}"
  shift || true
  echo "==> GPU $gpu 上运行 $file"
  # shellcheck disable=SC2029
  ssh -p "$REMOTE_PORT" -o BatchMode=yes "$REMOTE_HOST" \
    "$(remote_env)
     cd $REMOTE_ROOT || exit 1
     export CUDA_VISIBLE_DEVICES=$gpu
     mojo run -I src $file $*"
}

cmd_shell() {
  echo "==> 打开远程 shell（已设好 PIXI_ENV / MODULAR_HOME / PTXAS）"
  echo "    记得自己 export CUDA_VISIBLE_DEVICES=<空闲卡>"
  # shellcheck disable=SC2029
  ssh -t -p "$REMOTE_PORT" -o BatchMode=yes "$REMOTE_HOST" \
    "$(remote_env)
     cd $REMOTE_ROOT || exit 1
     exec \$SHELL -l"
}

cmd_probe() {
  local gpu="${1:-4}"
  echo "==> GPU kernel 冒烟测试（卡 $gpu）"
  # shellcheck disable=SC2029
  ssh -p "$REMOTE_PORT" -o BatchMode=yes "$REMOTE_HOST" \
    "$(remote_env)
     cd $REMOTE_ROOT || exit 1
     export CUDA_VISIBLE_DEVICES=$gpu
     echo '--- gpu-query（设备识别）---'
     gpu-query 2>&1 | grep -E 'name|compute_capability|api_version|driver' | head -8
     echo '--- vector-add kernel（数值校验）---'
     if [ -f tests/gpu/vecadd.mojo ]; then
       mojo run tests/gpu/vecadd.mojo 2>&1 | tail -3
     else
       echo '  (未找到 tests/gpu/vecadd.mojo；P1 阶段请把 kernel 冒烟测试放到该路径)'
     fi"
}

case "${1:-help}" in
  sync)  shift; cmd_sync "$@" ;;
  gpu)   shift; cmd_gpu "$@" ;;
  run)   shift; cmd_run "$@" ;;
  shell) shift; cmd_shell "$@" ;;
  probe) shift; cmd_probe "$@" ;;
  *)
    sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
