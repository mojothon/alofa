# alofa GPU 服务层战报（账本一百一十九 → 一百二十三）

2026-11-12 收束。本阶段把 tokenizer + GPU 推理底座推向可服务的完整形态：
从「能算」到「能并发地、确定地、可采样地生成」。

## 一、性能终局（Qwen2.5-0.5B，f32 权重，GPU4，钟态漂移内同 run 交错口径）

| 能力 | 数字 | 备注 |
|---|---|---|
| prefill（128 token） | 8.75ms | 起点 23ms，-62% |
| 单序列 decode | 2.23ms/token | 低延迟场景保留 |
| 跨序列交错 | 1.44× | span 翻译案 |
| 批解码 B=8 | 0.80ms/token | **5.3×** |
| 批解码 B=16 / 32 | 0.539 / 0.434 | 8.2× / 10.1× |
| 批解码 B=64 | 0.392ms/token | **~11.2×，吞吐封顶** |
| generate_batch B=8 / 16 | 1173 / 1675 有效 tok/s | API 层零开销（步成本与裸批完全吻合） |
| sample 模式 | -0.8~1.6ms/步 | 去 4.9MB logits D2H，只回 4B×B |
| 温度采样 | +0 开销 | 与 argmax 同结构 |

**拐点判读**：B=32 后每 token 成本硬摊平——GEMM 侧早已摊薄（prefill t 1→8 仅 +4%），
剩余成本是 attention 真 KV 流量 + tail。服务甜点 **B=16-32**；B=64 的 25ms 步延迟
会拉大每序列 token 间隔，吞吐增益（+10%）不抵延迟损失。

## 二、API 面

```
GpuSession
├─ prefill_step[20 comptime](tokens, profile, sync_tail, copy_logits,
│                            sample_temp, sample_seed)   # 单序列/批解码双态（comptime batch）
├─ decode_step[...]                                         # 单序列专用快路径
└─ generate_batch[20 comptime](flat_prompts, lens, max_new, eos,
                               join_flat, join_lens,
                               temperature=0.0, seed=0) -> List[Int]  # 扁平 [B+jb][max_new]
```

- **逐行 token 流**：每行独立 prompt→生成链，gen[0] = 末 prompt token 步的
  tokout；无 padding 行（idle 尾 ~9%）
- **槽位回收零 KV 搬运**：完成序列让槽，`h_posarr` 归零，新序列按 pos 序覆写——
  attention 只读 `[0..pos]`，此为 119-b pos_arr 语义的直接红利
- **EOS 早停 + join 队列**：常驻批 + 动态进出 = 真·连续批处理
- **温度采样**：`sample_rb_gpu`，Gumbel-max 恒等式
  `argmax(logit + T·(-log(-log(u))))` ≡ softmax(logits/T) 精确采样

## 三、核库存（本阶段新增）

| 核 | 职责 |
|---|---|
| `kv_store_bb_gpu` | 每序列独立 pos_arr rope + `[B][max_kv]` KV 基址 |
| `attn_rs_h_bb_gpu` | 批 scores（行距 max_kv，GQA head 复用） |
| `attn_vt2_h_bb_gpu` | 批 V 加权（FUSED 尾写同构） |
| `argmax_rb_gpu` | 行 argmax（首现并列语义） |
| `sample_rb_gpu` | 行温度采样（LCG + Gumbel-max，temp=0 退化 argmax） |

GEMM ×5 / gather / rmsnorm / LMHEAD 全部 row 通用零改动直通 t=B。

## 四、正确性门（全绿矩阵）

| 门 | 判据 |
|---|---|
| f32 差分门 | GPU vs 标量后端逐位 |
| decode 门 | 参照 parity |
| 批门 | 8 序列×3 步 vs 逐序列重放——**worst diff 0.0（逐位恒等）** |
| generate 门 | parity 零失配 + 确定性 + **join parity 零失配**（无 KV 污染） |
| 采样门 | temp=0 恒等 + 种子确定性 + N=400 分布对齐 ±0.10 |

## 五、坑册（本阶段七枚，按杀伤力排序）

1. **签名藏参错位**（120-a）：`prefill_step` 在 wpack/tile 间藏 `mma_q/mma_o`，
   19 位置参错位后 `batch=True` 静默落到 `prefill_spl`——且 B=1 时 else 分支
   算术等价，backbone 看起来全对。**破案手段：方法头加印**。
2. **参照读错缓冲**（123）：批 tail logits 在 `h_logits_b`，测试读 `h_logits`
   （陈旧垃圾）→ 伪造整层假分布。**教训：参照先自洽断言（kernel argmax ==
   host top-1）**。
3. **参照覆写 KV**（120-a）：单参照 session 顺序跑多序列互相污染——step 0
   diff=0.0 早已证明批侧正确，MISMATCH 全是测试伪影。
4. **公式的诚实性**（122）：「70% padding 浪费」把生成 token 误计入浪费，
   真实 ~10%。错账差点把 chunked prefill 的优先级抬高一倍。
5. **pk/pv 跨 List origin**（120-a）：批分支直传 store_b 指针。
6. **`out` 保留字**（×2）与 **Tuple 字面量不可迭代**（×3）——Mojo 惯性坑。
7. **bitcast 参数是 DType**、non-copyable 返回要 `out^`、`raises` 上下文传染。

## 六、遗留（按优先级）

1. ~~3-session 堆损坏审计~~ **已收案（124）**：非堆损坏非显存——
   MAX 运行时第 3 个 DeviceContext 硬上限（2 恒好 / 3 恒坏，
   尺寸无关）。多实例路线改为**单 ctx 多路复用**或**每 session 一进程**
2. top-p 采样（排序阈值 pass）
3. chunked prefill（~10% 级，已降级）
4. KV store 复用/压缩（9GB/session 之坑）
5. 采样进 top-k、重复惩罚等生产标配

## 七、阶段判词

从 23ms prefill + 4.4ms/token 起步，到 prefill **-62%**、批解码 **~11×**、
服务层四件套（prefill/批解码/连续批处理/采样）全部带逐位级或分布级正确性门。
吞吐上限已被实验钉死在 ~11×（119-b 预测 9× 吻合），下一阶段的收益不在核内，
而在调度形态与资源复用。
