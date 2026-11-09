"""GPU (CUDA) backends for the L1 compute layer.

**只能在 A100 远程验证机上运行**（`scripts/a100.sh`）；本地开发机是 Maxwell
sm_52，现代 CUDA 栈与 MAX 均不支持。因此本包**不被任何本地套件导入** —— 与
`tests/gpu/` 同一条纪律：本地跑必然失败的代码不进 `pixi run test`，免得污染
能力账本的信号。

正确性判据与 CPU 后端共用一条：`tests/gpu/test_cuda_diff.mojo` 把同一个输入喂给
GPU kernel 与标量后端，逐值比 fp32 相对容差 1e-5。每个 GPU kernel 落地时都要
在这台机器上过一次差分（账本四十九起：rmsnorm `diff=0.0` 逐位一致）。

Device kernel 的写法约束（实测踩坑，详见 `docs/plan/02-architecture.md` §11.3）：
kernel 标量参数必须 `Int32`/`Int64`/`Float32`（`Int`/`UInt` 不 conform
`DevicePassable`）；指针用 `Pointer[T, MutUntrackedOrigin]`。

Submodules are imported directly by callers
(`from alofa.kernels.cuda.basic import ...`).
"""
