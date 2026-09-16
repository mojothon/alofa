# alofa 架构设计

> 本文给出 alofa 的整体架构、核心数据结构、并发模型、后端抽象与验证体系。
>
> **重要前提**：`01-ecosystem-analysis.md` 的结论约束了本文的每一个设计选择。文中标注"设计草图"的代码为**未经编译验证**的形态说明，不是可直接使用的实现。

---

## 1. 定位与设计公理

**定位一句话**：alofa 是一个**可被验证**的纯 Mojo LLM 推理引擎 —— 用编译期后端特化换可移植性，把服务层建立在生态已有的 `flare` reactor 之上（不重复造轮子），用能力账本把"生产级"从形容词变成可执行的门，**并把所有创新工时投在无人做过的引擎内核（L0–L4）**。

### 设计公理（每条都直接来自生态分析）

| # | 公理 | 出处 |
|---|---|---|
| A1 | **证据优先于性能主张**。任何 `verified` 能力必须指向存在的测试与可执行命令 | 14 个参考项目中 13 个陷入"宣传 >> 实现"；`A.E.S.I.R.` 的 `CAPABILITY_LEDGER` 是唯一有效反制 |
| A2 | **async 不是必需品，事件循环才是。但优先依赖已有的 `flare`，而非自研 epoll** | Mojo 1.0 无 async（Phase 2 未开始）；本机实测 libc FFI 全通（自研**可行**）；但 `flare` 已是纯 Mojo 的生产级 reactor → 自研属重复劳动 |
| A3 | **调度器是唯一允许复杂的地方**，且必须零分配、纯状态机、可重放 | vLLM 的复杂度来自 Python 层与多进程 IPC；SGLang 的复杂度来自树治理 |
| A4 | **后端可移植性必须在编译期解决**，运行期只做一次派发 | Mojo 无运行时多态（动态 trait 未开始），但 `comptime` + trait 是强项 |
| A5 | **KV 复用与请求分页必须共享同一物理池** | vLLM 的 paged allocator 与 APC 耦合、SGLang 的树与分页分离，两套视图互相打架 |
| A6 | **不做未经本机验证的性能声明**，但"本机"包含**可复现的远程验证机** | 开发机 GPU 为 Maxwell sm_52（不可用）；**已打通 A100 远程验证机（6×A100，sm_80）** → CUDA 从 `hardware-blocked` 升级为可验证 |
| A7 | **不重复造生态已有的轮子**。`flare` / `nova` / `json` / `mojo.core` 能解决的，直接依赖 | 生态已有纯 Mojo 的成熟基础设施；alofa 的稀缺工时必须投在**无人做过的引擎内核（L0–L4）** |

---

## 2. 分层架构

### 2.1 依赖方向：严格单向，禁止反向引用与跨层跳跃

```
外部依赖（纯 Mojo，非 alofa 所有，但**直接依赖而非自研**）：
  flare 0.2.0   reactor / scheduler / timer_wheel / watchdog / mutex / reuseport
                / buffer_pool / io_uring + HTTP1-3 / QUIC / TLS / WS / gRPC / DNS   ← L5/L6 地基
  json 0.1.2    SIMD 两遍解析 + tape Document + 编译期反射序列化                    ← config / OpenAI API
  mojo.core     Go stdlib 移植：log / flag / time / os / sync / atomic / chan
                / context / encoding.json / regexp / template                       ← 补 stdlib 缺口

L6  接口适配     srv/http, srv/openai, srv/sse, cli          （基于 flare）
L5  服务运行时   srv/loop(flare reactor), srv/conn, srv/master(SO_REUSEPORT)
L4  执行编排     engine/scheduler, engine/batch, engine/core, engine/trace
L3  运行时       runtime/kv/*, runtime/executor, runtime/sampler, runtime/backend
L2  模型层       model/arch/*, model/config, model/weights
L1  算子层       kernels/traits, kernels/cpu/*, kernels/cuda/*, kernels/wgpu/*, kernels/quant
L0  基础         core/dtype, core/tensor, core/memory, core/mmap, core/ffi/*, core/error

正交支柱（不进入上面的阶梯，被各层依赖）：
  tokenizer/   编解码（BPE / Unigram / WordPiece + 预分词器 + chat_template）
  verify/      oracle 差分、roofline、能力账本校验
  quant/       量化格式描述与权重布局（内核实现在 kernels/quant）
```

**硬规则**：
- L0 不得依赖任何上层；`core/` 中不允许出现 model/kernel 概念。
- 上层可以依赖下层的**接口 trait**，不得依赖具体实现类型。
- `verify/` 与 `tokenizer/` 可以向任意层取数，但**没有任何层可以依赖它们**（否则验证代码会进入生产路径）。

### 2.2 目录布局（对齐现有 `src/alofa/`）

```
src/alofa/
├── __init__.mojo
├── version.mojo
├── core/
│   ├── dtype.mojo          # DType 描述、块量化格式元数据
│   ├── tensor.mojo         # 张量视图（不拥有数据）：shape/stride/dtype/offset
│   ├── memory.mojo         # arena 分配器、对齐分配、move-only 所有权
│   ├── mmap.mojo           # mmap/madvise/mlock 权重映射与页缓存提示
│   ├── ffi/                # libc / posix 绑定：epoll, socket, timerfd, eventfd, mmap
│   ├── error.mojo          # 具名错误（禁止用字符串当错误码）
│   └── log.mojo            # 结构化日志（JSONL，可被 verify 消费）
├── kernels/
│   ├── traits.mojo         # KernelSpec trait：量化/后端/形状参数的编译期契约
│   ├── cpu/                # scalar, avx2, avx512, neon
│   └── quant/              # dequant: q4_0, q4_k, q8_0, int8, fp8
├── model/
│   ├── config.mojo         # HF config 解析（含 GGUF 元数据）
│   ├── weights.mojo        # safetensors / GGUF 读取器 + 布局重排
│   └── arch/               # llama.mojo, qwen.mojo, gemma.mojo ...
├── runtime/
│   ├── kv/                 # 统一 KV 寻址（见 §3）
│   ├── executor.mojo       # 批前向驱动
│   ├── sampler.mojo        # top-k/top-p/min-p/温度/重复惩罚/logit bias
│   └── backend.mojo        # 运行期单次派发
├── engine/
│   ├── scheduler.mojo      # 纯状态机
│   ├── batch.mojo          # 批组装与张量池
│   ├── core.mojo           # 忙循环：调度 → 执行 → 采样 → 回写
│   └── trace.mojo          # 调度 trace：录制与重放
├── tokenizer/
│   ├── pretoken.mojo       # 预分词器（无正则依赖，手工实现）
│   ├── bpe.mojo, unigram.mojo, wordpiece.mojo
│   ├── vocab.mojo          # tokenizer.json / GGUF vocab 加载
│   └── chat_template.mojo  # 极简 Jinja 子集
├── srv/
│   ├── loop.mojo           # epoll 事件循环
│   ├── conn.mojo           # 连接状态机（解析、发送队列、背压）
│   ├── http.mojo, sse.mojo, openai.mojo
│   └── master.mojo         # SO_REUSEPORT supervisor
└── verify/
    ├── oracle.mojo         # 与 HF / llama.cpp 的差分
    ├── diff.mojo           # logits 相关性、argmax、top-k 集合、分布检验
    ├── roofline.mojo       # 带宽/算力利用率报告
    └── ledger.mojo         # 能力账本解析与 CI 校验
```

### 2.3 外部依赖决策（A7：不重复造轮子）

调研本地 `/home/rontom/mojo_project` 与三个新参考库后，**服务层与工具层的结论发生了反转**：原计划自研 epoll 事件循环，但生态里已经有纯 Mojo 的生产级实现。

| 依赖 | 版本 / 来源 | 提供什么 | alofa 的用法 | 风险与对策 |
|---|---|---|---|---|
| **`flare`** | 0.2.0，已发布到 `https://prefix.dev/mojo-force`（conda 可装） | 纯 Mojo 网络栈：`runtime/`(reactor 30KB、scheduler 42KB、**timer_wheel**、**watchdog**、mutex、`_thread`、**reuseport**、blocking、buffer_pool、pool、handoff、`io_uring` ~120KB) + `http/` `http2/` `http3/` `quic/` `tls/` `tcp/` `udp/` `dns/` `grpc/` `ws/` `crypto/` | **L5/L6 直接依赖**。`EngineCore` 作为 reactor 的一个长期任务；SSE 用其流式写 + 背压 | 0.x API 未冻 → 锁版本；`flare` 是同作者生态，改动可控；若 API 不稳则只取 `runtime/` 子系统（reactor + reuseport + timer_wheel），HTTP 自建 |
| **`nova`** | 本地，基于 flare | FastAPI 风格路由/中间件；**实测 31k QPS 单 worker / 117k QPS 四 worker** | **可选**：仅当需要复杂路由时；否则直接用 flare 的 HTTP 更薄 | 框架约束可能不适合长连接 SSE → 默认不用 |
| **`json`** | 0.1.2，同频道 | 纯 Mojo 高性能 JSON：64 字节无分支 SIMD 两遍扫描、tape-backed `Document`、编译期反射 `serialize_json` / `deserialize_json`、JSONPath / JSON Patch / JSON Schema、支持 GPU | `config.json` 解析、OpenAI API 请求/响应、结构化输出 | 已发布 conda 包，风险低 |
| **`mojo.core`** | `tamnd/mojo.core`，Apache-2.0，纯 Mojo，2030 符号 / 23.1% 对齐度 | Go 标准库移植：`log` `flag`(CLI) `time` `os` `exec` `signal` `sync` `atomic` `chan` `context` `encoding/json` `regexp` `template` | 补 Mojo stdlib 缺口（结构化日志、CLI 参数、时间、进程/信号） | **无 conda 包** → 以源码/pixi 集成；对齐度仅 23%，`regexp` 是否完整**必须实测**才能依赖 |
| **`mojo-mpi`** | `BenWibking/mojo-mpi` | MPI 绑定，**但只有点对点** | 多机通信的**最底层**；集合通信须自研 | ⚠️ **无 allreduce / allgather** → TP 的 reduce-scatter / allgather 要么自研，要么走 NCCL FFI（A100 机上 `libcudart`/NCCL 可用） |
| **`mojo-embree`** | `lee101/mojo-embree` | 图形学专用，**无直接复用价值**；但三项技法通用 | 借鉴：① **调用方拥有的固定容量工作栈**（arena，零分配遍历）② **`地址 + 计数` 的 C ABI 参数形态**（比传结构体稳定）③ **与 C 参考实现逐像素差分**的验证组织法 | 只取技法，不引依赖 |

**明确不采用**：
- **`fast-tokenizer`（本地）** —— 其 BPE 用**贪心 trie 而非 rank 优先合并**，已在其仓库提交的基准数据中出现与参考实现的**已知分歧**；且 `decode` 路径混入 Python FFI、golden 测试为空。**它是反面教材：证明"看起来能跑"与"数值正确"是两件事**。alofa 的 P1 tokenizer 门（4560 条差分用例 0 失败）正是为此而设。可借鉴其数据结构，不借鉴其合并算法。

---

## 3. 核心设计一：统一 KV 寻址空间（创新点 1）

### 3.1 问题

| 系统 | KV 管理 | 缺陷 |
|---|---|---|
| vLLM | 分页分配器 + block table（block 粒度）；APC 用增量哈希链做前缀匹配 | 前缀匹配**必须 block 对齐**；分页器与前缀缓存耦合，无法独立演进 |
| SGLang | RadixAttention 基数树（token 粒度、任意分支）；节点映射到分页 KV | 树与分页是两套视图，淘汰决策只看树；**纯 LRU 对 agentic 工作流次优**（论文自认） |

两者都维护**两套内存视图**，导致前缀复用的淘汰决策与请求分页的分配决策互相打架。

### 3.2 方案：一个物理池，两个索引视图，共享 refcount

```mojo
# 设计草图 - 形态说明，未经编译验证
comptime INVALID_BLOCK = UInt32(0xFFFF_FFFF)

struct BlockPool:
    """单一物理分配；所有 KV 数据只在这里存在一次。"""
    var data: Pointer[UInt8, MutUntrackedOrigin]   # 单块大分配
    var block_bytes: Int                            # block_size × layers × kv_heads × head_dim × 2B
    var free_head: UInt32                           # 空闲链（O(1) 分配/回收）
    var refcnt: Pointer[UInt16, MutUntrackedOrigin] # 共享计数，两个视图共用

struct PageTable:
    """视图 A：请求 → 物理块序列。追加 O(1)，无匹配开销。"""
    var blocks: List[UInt32]

struct RadixNode:
    """视图 B：token 前缀 → 块区间。前缀匹配 O(prefix)，支持任意分支。"""
    var first_block: UInt32
    var start_offset: UInt16       # ┐ 节点在块内的 token 偏移与长度：
    var token_len: UInt16          # ┘ 让 token 粒度映射到 block 粒度存储
    var parent: UInt32
    var children: Dict[UInt32, UInt32]
    var freq: UInt32               # 频率状态（视图 C 用）
    var last_access: UInt64
```

**关键技巧**：`RadixNode` 保存 `(first_block, start_offset, token_len)`。于是**节点分裂只是调整这三个数，不拷贝任何 KV 数据**，且 token 粒度的树可以直接落在 block 粒度的物理池上。这消除了 SGLang 树节点与分页块之间的阻抗失配。

**收益**：
- 一次 `insert` 同时更新两个视图，二者用同一个 `refcnt` 决定回收，**永不出现"树说可回收、分页器说在用"**。
- 前缀匹配不再受 block 对齐限制（对比 vLLM APC）。
- 淘汰策略可以在**块粒度**执行，但依据**树粒度的语义信息**（子树大小、访问频率、前缀深度）。

### 3.3 视图 C：频率感知淘汰（创新点 2）

SGLang 用纯 LRU。alofa 用 2Q 风格的准入队列 + 复合评分：

```mojo
# 淘汰评分：先保 protected，再按分值驱逐 candidate
# score = w_freq * log2(freq) + w_recency * decay(last_access)
#         + w_depth * depth + w_subtree * (1 / subtree_size)
#
# 淘汰顺序：
#   1. 从 candidate_queue 取低分节点
#   2. 优先驱逐叶子；驱逐父节点前必须先驱逐/提升所有子节点
#   3. 驱逐后若父节点只剩一个子节点，执行节点合并（与 RadixAttention 一致）
```

**自我约束**：该功能**必须附一组可复现的 workload 回放**（多轮对话 + 合成 agent trace + ShareGPT），并证明命中率优于 LRU 基线，否则不进入主线。这是 A1 公理的直接应用 —— 不允许出现"看起来更聪明但无数据"的算法。

---

## 4. 核心设计二：可重放的纯函数调度器（创新点 3）

### 4.1 为什么必须纯

vLLM 调度器与 Worker 同进程、与 Python 对象纠缠，导致边界 bug 多、难以单测。alofa 把调度器做成**零分配、无 I/O 的纯状态机**：

```mojo
# 设计草图
struct SchedInput:              # 本步新到/新完成的事件，由服务层注入
    var arrived: List[ReqId]
    var cancelled: List[ReqId]
    var freed_blocks: Int

struct Action:                  # 纯数据、可序列化 → 因此可重放
    var prefill: List[PrefillSlice]   # (req, token_start, token_end)
    var decode: List[ReqId]
    var preempted: List[ReqId]
    var finished: List[ReqId]
    var tick_seq: UInt64

struct Scheduler:
    var waiting: Deque[ReqId]
    var running: List[ReqId]
    var budget: List[Int]            # 每请求剩余 token 预算
    var kv_watermark: Float32

    def step(mut self, inm: Span[SchedInput]) -> Action:
        """唯一入口。无 I/O、无分配（复用预分配 buffer）。"""
        ...
```

### 4.2 调度策略

- **单一 token 预算**，prefill 与 decode 不分区（对标 vLLM V1 的统一调度）→ 天然支持 chunked prefill、prefix caching、spec decode 共存。
- **chunked prefill 默认开**：长 prompt 切片与 decode 混批，平滑 TTFT/ITL。
- **抢占只做重计算**（与 V1 一致，不做 GPU↔CPU KV swap），并**把抢占次数作为容量告警指标暴露**（生产上这是"KV 不够用"的最早信号）。
- **延迟护栏**：默认配置必须带**每请求最大等待拍数**上限，防止重演 MAX `--max-batch-size` 造成的"等批次"延迟陷阱。
- 调度优先级：默认 FCFS，但**预留 `Priority` 与 `PrefixAffinity` 两种策略的 trait 位置**（后者用于 cache-aware 路由）。

### 4.3 可重放：让调度器可以脱离模型被测试

`Action` 是纯数据 → 把它写成 JSONL trace。于是：

```
录制：engine/trace.mojo 记录每一步 (SchedInput, Action, kv_state_digest)
重放：verify/ 直接用 trace 驱动调度器，断言 Action 逐字节一致
构造：手工构造极端 SchedInput（超长 prompt、并发抢占风暴、预算耗尽、0 预算）
```

**这解决了 14 个参考项目共有的一个痛点**：它们全部只能在"有模型、有 GPU"的情况下测调度边界，导致边界分支实际上从未被验证过。

---

## 5. 核心设计三：编译期后端特化 + 运行期单次派发（创新点 4）

Mojo 没有运行时多态（动态 trait / existentials 在 roadmap 上**未开始**），但 `comptime` + trait 极强。因此：

```mojo
# 设计草图
trait KernelSpec[quant: Quant, isa: ISA, block: Int]:
    """一个算子在不同 (量化格式, 指令集, 块形状) 下的编译期契约。"""
    ...

# 同一份源码，comptime 参数化生成多个变体
struct MatMulDequant[q: Quant, i: ISA, b: Int](KernelSpec[q, i, b]):
    ...

# 运行期只派发一次（每个算子选一次，而不是每层每 op 虚调用）
struct DispatchTable:
    var matmul_q4_avx2: MatMulDequant[Quant.q4_0, ISA.avx2, 128]
    var matmul_q4_scalar: MatMulDequant[Quant.q4_0, ISA.scalar, 128]
    var chosen: Int    # 由 core/cpu probe 在启动时决定
```

**对比收益**：
- vLLM 的 Python 层逐 op 派发在 decode 阶段占显著 CPU 开销（这是 V1 削减 CPU 开销换 1.7× 吞吐的同一个杠杆）。alofa 从终点开始：派发在启动时一次完成。
- LM Studio 通过多后端覆盖广度，但每个后端是独立编译的 C++ 实现。alofa 用一份源码 + comptime 参数化覆盖多后端，**避免维护 N 份实现**。

**ISA 分级（对齐"开发机 + A100 远程验证机"的现实）**：

| 级别 | 状态 | 依据 |
|---|---|---|
| `scalar` | verified（P1） | 参考实现，用于正确性 oracle |
| `avx2` (8×fp32) | verified（P1） | 开发机 i7-9700K 唯一可端到端验证的向量路径（无 AVX-512） |
| `avx512` (16×fp32) | designed-only | 开发机无此指令集；**A100 机为 Xeon Platinum 8358，待实测确认是否可用** |
| `neon` (4×fp32) | designed-only | 无 ARM 硬件 |
| `cuda` | **`verified-capable`（sm_80）** | 开发机 Maxwell sm_52 不可用；但 **A100 远程验证机已实测打通**（`gpu-query` 识别 A100 / CC 8.0 / api 12060，端到端 kernel 数值正确）→ **可按 `verified` 目标推进，性能数字须在该机测得** |
| `wgpu` | designed-only | 复用 `wgpu-mojo`，作为跨平台（Vulkan/Metal/DX12）路线 |

---

## 6. 核心设计四：服务层（依赖 `flare`，不是自研 epoll）

> **本节在 2026-09-16 被推翻重写。** 原方案是"纯 Mojo 自研 epoll 事件循环"（依据：本机实测 libc FFI 全通）。调研本地 `flare` 后结论反转：**`flare` 已经是纯 Mojo 的生产级 reactor，还带 timer_wheel / watchdog / reuseport / buffer_pool / io_uring 与 HTTP1-3+QUIC+TLS**。自研它属于重复劳动，与 A7 冲突。
>
> 原实测仍然有效且重要 —— 它证明了**即使没有 `flare`，这条路也走得通**，因此我们对 flare 的依赖是**可回退的依赖**，不是单点。

### 6.1 线程模型：reactor 与 GPU 执行分离

`flare` 的 reactor 是单线程的。但 LLM 服务有一个特殊约束：**一次 decode step 会占住 GPU 数十毫秒，绝不能阻塞 I/O 线程**。因此：

```
线程 A（flare reactor，单线程，零锁）
  ├── 新连接 / HTTP 解析（flare http）
  ├── 请求入 pending 队列
  ├── timer_wheel 到期 → 触发调度 tick → 提交一个 step 任务到线程 B
  ├── 线程 B 回传 token → 推进对应连接的 SSE 发送队列（flare 背压）
  └── eventfd → master 指令（优雅退出 / 重载）

线程 B（engine worker，独占）
  └── engine.core 忙循环：调度 → 批组装 → GPU/CPU 前向 → 采样 → 回传
      与线程 A 之间只通过 handoff 队列交换 `Action` / `Token`，不共享可变状态
```

**关键收益**：调度决策（纯函数，§4）留在 reactor 线程 → 保持**可重放、零锁**；重活（前向）在线程 B。这与 vLLM V1 "overlap scheduling" 的动机一致，但**靠线程分离而非 Python 协程**实现。

**已实测的 libc 原语**（`tests/capability/test_libc_ffi.mojo`，5/5 通过，`socket` / `epoll_create1` / `timerfd_create` / `eventfd` / `setsockopt(SO_REUSEPORT)`）：即使未来弃用 `flare`，自研路径已被证明可行。

### 6.2 横向扩展：多进程而非多线程

```
master (不监听，只做进程管理)
  ├── worker 0 .. N-1   各自 bind(SO_REUSEPORT) 同一端口   ← flare 的 reuseport.mojo
  │     内核级连接分发，无用户态负载均衡器
  └── eventfd + pipe：健康探测与优雅退出
       SIGTERM → 停止 accept → 排空在途请求 → 退出（超时后 SIGKILL）
```

每个 worker 是**独立的 reactor + 独立模型副本 + 独立 engine 线程**。这避开 Mojo 缺 async 与线程同步原语不成熟的现实，并获得近线性扩展（`nova` 实测 4 worker 达 117k QPS）。

**多卡**：每张卡一个 worker 进程，`CUDA_VISIBLE_DEVICES` 隔离。TP（张量并行）暂不做 —— `mojo-mpi` **只有点对点、没有集合通信**，要么自研 allreduce 要么走 NCCL FFI，属 P5 之后的独立议题。

### 6.3 赌注与明确回退

**赌注已从"纯 Mojo 能否写出事件循环"降级为"`flare` 的 API 是否稳定、是否适配长连接 SSE"** —— 风险量级显著变小。

| 判据 | 若通过 | 若失败 |
|---|---|---|
| P3：100 并发长连接 SSE，连续 1 小时无错、无 fd 泄漏、P99 无劣化 | flare 方案成立 | ① 先尝试**只取 `flare.runtime` 子系统**、HTTP 层自建；② 仍失败则切 sidecar |

**回退形态**：`EngineCore`（L4 及以下，全 Mojo）编译为独立二进制，通过 **stdio RPC**（借鉴 `hyf`）暴露给极薄的 Rust/Python 边车，边车只负责 HTTP/SSE/连接管理。边界由 `provider` trait 定义，含 **fallback taxonomy + deadline 预算 + 健康探测**。**这保证赌注失败时只损失 L5/L6，不损失任何引擎价值。**

**回退形态**：`EngineCore`（L4 及以下，全 Mojo）编译为独立二进制，通过 **stdio RPC**（借鉴 `hyf` 的 stdio 守护进程模式）暴露给一个极薄的 Rust/Python 边车，边车只负责 HTTP/SSE/连接管理。边界由一个 `provider` trait 定义，包含 **fallback taxonomy + deadline 预算 + 健康探测**。这保证赌注失败时只损失 L5/L6，不损失任何引擎价值。

---

## 7. 验证体系（这是 alofa 真正的差异化，创新点 5）

`verify/` 是与各层平级的**一等模块**，且被 CI 强制。

### 7.1 三层验证

| 层 | 机制 | 判据 |
|---|---|---|
| **数值** | 与 HF transformers / llama.cpp 做 oracle 差分 | logits：余弦相似度 + argmax 一致 + top-k 集合一致；greedy 输出 token 序列**逐 token 相等** |
| **分布** | 采样正确性检验 | 固定 seed，用**卡方/总变差距离**验证采样分布（直击 `llm-mojo` 用"精确匹配"替代拒绝采样导致分布错误的问题） |
| **性能** | roofline 报告 + 回归门 | 不只报 tok/s，报**达到内存带宽的 x%**、prefill 达算力的 y%；每 PR 跑固定 prompt，波动超阈值即失败 |

### 7.2 能力账本由 CI 校验

`capability-ledger.md` 中每个 `verified` 条目**必须**指向一个存在的测试文件与一条可执行命令。`verify/ledger.mojo` 解析账本并校验：

```
- 若 status == verified  → 对应测试文件必须存在，且命令必须能跑通
- 若 status == scaffold  → 必须标注"未接入"，禁止出现在 README 的功能列表里
- 若 status == designed-only → 必须标注原因（如"本机无对应硬件"）
```

这条规则的价值在于：它**在机制上**使得"宣传 >> 实现"无法发生。

### 7.3 证据分级（写进文档与 README 的强制约定）

| 标签 | 含义 |
|---|---|
| `verified` | 本机跑通，有测试与命令，可复现 |
| `scaffold` | 代码在，但未接入主路径（**必须在 README 中排除**） |
| `designed-only` | 设计完成、编译通过，但本机无硬件验证（**禁止给出性能数字**） |
| `missing` | 未开始 |

---

## 8. 创新点汇总（相对现状的净增量）

| # | 创新 | 相对谁 | 可验证方式 |
|---|---|---|---|
| 1 | **统一 KV 寻址空间**：分页与基数树共享同一物理池与 refcount，token 粒度映射到 block 粒度存储（节点分裂零拷贝） | vLLM（两视图耦合）、SGLang（两视图分离） | 内存占用 + 前缀命中率对比测试 |
| 2 | **频率感知淘汰**（2Q + 复合评分）替代纯 LRU | SGLang LRU 的自认短板 | workload 回放命中率曲线 |
| 3 | **可重放的纯函数调度器**：`Action` 可序列化 → 调度边界可脱离模型被测试 | vLLM（调度与 Python 对象纠缠） | trace 重放一致性测试 |
| 4 | **编译期后端特化 + 运行期单次派发**：一份源码参数化多后端多 ISA | LM Studio（N 份后端实现）、vLLM（逐 op Python 派发） | 启动派发开销基准 + 后端切换测试 |
| 5 | **可执行的能力账本**：CI 校验 `verified` 必须指向真实测试 | 14 个参考项目中 13 个的失败模式 | CI 门失败即证明有效 |
| 6 | **服务层基于 `flare` reactor 且与 GPU 执行线程分离**（I/O 与调度同线程、零锁；重活在独占线程） | MAX（服务层用 Python）、纯 Mojo 社区无先例 | 压测门（100 并发 SSE）；且 libc 原语已实测，可自研回退 |
| 7 | **roofline 优先的性能报告**：以带宽/算力利用率为主指标 | 所有只报 tok/s 的项目 | 报告模板 |

---

## 9. Non-goals（防止范围爆炸）

- ❌ 训练 / 微调 / RLHF
- ❌ 多机分布式（P5 之前）；TP 只留接口，不实现
- ❌ GUI / 桌面应用（LM Studio 的地盘，不做）
- ❌ Python 或 Rust 作为主体语言（sidecar 是**回退**，不是路线）
- ❌ 在没有对应硬件的机器上给出性能数字
- ❌ 语义模糊的 KV 匹配（保持 token 级精确，与 SGLang 一致）

---

## 10. 已知风险与设计侧的对策

| 风险 | 对策 |
|---|---|
| Mojo 破坏性变更（`fn` 移除、`mojo package` → `mojo precompile`、`Pointer` 家族合并均已在 1.0 发生过） | 固定 Mojo 版本下限；语法集中在小体量 adapter；CI 锁版本 |
| MAX 导入路径迁移（`max.` 前缀 vs 顶层 `linalg`/`layout`） | 自建 `kernels/traits.mojo` 作为隔离层，MAX 内核只在实现侧引用 |
| 无 async 导致服务层不稳 | 已实测 libc 原语可行；且改依赖 `flare`（§6）后风险大幅下降；仍有 sidecar 回退 |
| 开发机无可用 GPU → CUDA 是"未验证的代码" | **已缓解**：A100 远程验证机打通（§11）；性能数字必须在该机测得并标注环境 |
| `flare` 0.x API 未冻结 | 锁版本；优先只取 `runtime/` 子系统以缩小耦合面；最坏可自研 epoll（原语已实测） |
| 远程 A100 为**共享机**（GPU 0–3 常被占满） | 只用空闲卡；验证脚本**必须接受 `CUDA_VISIBLE_DEVICES`**，禁止长时占卡 |
| 远程机**无外网**，无法 `pixi install` | 同步工作流：本地安装后**连同 `.pixi/envs/default` 一起 rsync**（已验证可行，§11） |
| 共享环境下的性能数字不可信（他人负载干扰） | 性能基准报 **roofline 利用率**而非裸 tok/s，并记录同机其他卡占用；正确性结论与性能结论分开标注 |
| 频率感知淘汰无收益 | 回放数据不达标则不上线（§3.3 自我约束） |
| 调度器复杂度失控 | 纯函数 + 零分配 + trace 重放；复杂度只允许存在于 `engine/` |

---

## 11. 验证环境（A100 远程验证机，2026-09-16 实测打通）

开发机 GPU（Maxwell sm_52）不可用，因此**专门打通了一台远程验证机**。以下均为实测，不是推断。

### 11.1 硬件与资源

| 项 | 实测 |
|---|---|
| 主机 | `lcl@10.107.6.60`（**SSH 免密已配置**，无需密码） |
| 工作目录 | `/app/lcl/mojo-projects` → 软链至 `/shared/lcl/mojo-projects` |
| GPU | **6× A100**（1× A100-80GB PCIe + 5× A100-PCIE-40GB，**合计 280 GB 显存**） |
| 驱动 / CUDA | `560.35.03` / `CUDA 12.1`（`nvcc` 位于 `/usr/local/cuda/bin`） |
| 设备属性 | `compute_capability = 8.0`、`api_version = 12060`、`max_threads_per_block = 1024` |
| CPU / 内存 / 磁盘 | 2× Xeon Platinum 8358（**128 vCPU**）/ 755 GB（可用 ~521 GB）/ 可用 ~252 GB |
| 模型库 | `/app/lcl/models/` 已有 44 个模型（含 DeepSeek-R1-Distill-Llama-70B、Qwen-32B、BERT、bge 等） |
| ⚠️ 网络 | **无外网**（无 DNS、ping 与 curl 均不通） |
| ⚠️ 共享 | **共享机器**：GPU 0–3 常满载（实测 92–99%），**只能用空闲卡（当前为 4、5）** |

### 11.2 打通步骤（可复现）

```bash
# 1) 离线同步：本地安装后，连同 pixi 环境一起 rsync（远程机无外网，不能 pixi install）
rsync -a --info=progress2 \
  /home/rontom/mojo_project/alofa/ \
  lcl@10.107.6.60:/app/lcl/mojo-projects/alofa/      # 需包含 .pixi/envs/default

# 2) 远程运行所需的三个环境变量（缺一不可）
export PIXI_ENV=<项目>/.pixi/envs/default
export PATH=$PIXI_ENV/bin:$PATH
export MODULAR_HOME=$PIXI_ENV/share/max              # 不设则找不到标准库
export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas   # 见下方关键坑

# 3) 指定空闲卡
CUDA_VISIBLE_DEVICES=4 mojo run <file>.mojo
```

**关键坑（值得单独记录）**：MAX 26.5 要求 NVIDIA 驱动 ≥ 580（CUDA ≥ 13.0），而该机是 `560.35.03`。直接跑 `gpu-query` 会报驱动过旧。**官方给出的绕过方式是指定系统 `ptxas`**：

```bash
export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
```

设置后 `gpu-query` 正确输出 A100 与 CC 8.0，**端到端 kernel 数值校验通过**：

```
DeviceContext created
*** GPU VECADD PASS on A100 *** c[0]= 0.0  c[999]= 2997.0     # 999 + 2×999 = 2997 ✓
```

### 11.3 Mojo 1.0 的 GPU 语法要点（实测踩坑记录）

这些是查文档不易得到、但每个都会卡住半小时的事实：

| 项 | 正确写法 | 说明 |
|---|---|---|
| 线程/块索引 | `from std.gpu import block_idx, block_dim, thread_idx, global_idx` | 是 `std.gpu`，与 `std.sys` 同构 |
| `DeviceContext` | `from max.gpu.host import DeviceContext` | ⚠️ **不是** `std.gpu.host`（后者不含 `DeviceContext`） |
| `LayoutTensor` | `from layout import Layout, LayoutTensor` | ⚠️ 顶层 `layout`，**不是** `max.layout` |
| 指针 | `Pointer[T, _]` 或 `Pointer[T, MutUntrackedOrigin]` | `UnsafePointer` **已废弃**；不写 origin 参数会报错 |
| kernel 标量参数 | 用 `Int32` / `Int64` | ⚠️ `Int` / `UInt` **不 conform `DevicePassable`** |
| 启动 | `ctx.enqueue_function[k](args..., grid_dim=(x,1,1), block_dim=(BS,))` | — |
| `LayoutTensor` 参数 | `[mut: Bool, dtype, layout, origin]` 四参 | origin 名字难确定 → **优先用裸 `Pointer` 参数**规避 |

### 11.4 环境对路线图的影响

| 原计划 | 修订后 |
|---|---|
| CUDA 后端 `hardware-blocked`，P5 之后才能碰 | **CUDA 可在 P1 末同步开始验证**（正确性），P2 起可出性能数字 |
| 性能基准只有 llama.cpp CPU | 可在 A100 上做**同机** GPU 对比基准（仍须报 roofline） |
| 担心"GPU 代码从未运行过" | 已运行且数值正确 |

---

**相关文档**
- 生态与竞品分析 → [`01-ecosystem-analysis.md`](01-ecosystem-analysis.md)
- 路线图与验收门 → [`03-roadmap.md`](03-roadmap.md)
- 能力账本 → [`capability-ledger.md`](capability-ledger.md)
