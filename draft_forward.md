# DFlash Draft Model Forward 过程详解

## 1. 整体架构概览

DFlash 是一种面向投机解码（Speculative Decoding）的 Draft 模型训练方法。其核心思想是：给定目标模型（Target Model）在上下文序列上产生的中间隐状态，Draft 模型通过"闪现（Flash）"方式在训练阶段**并行**预测多个 block 内的 token，而在推理阶段则**串行**地以块（block）为单位投机生成候选 token，由 Target 模型验证。

### 主要组件

| 组件 | 类/文件 | 作用 |
|------|---------|------|
| 目标模型封装 | `DFlashTargetModel` / `dflash_target_model.py` | 前向传播，输出多层隐状态 |
| Draft 模型主体 | `DFlashDraftModel` / `modeling/draft/dflash.py` | 实际 draft 模型，基于 Qwen3 架构改造 |
| 训练包装器 | `OnlineDFlashModel` / `core/dflash.py` | block-wise 并行训练逻辑、loss 计算 |
| 目标嵌入与 LM Head | `TargetEmbeddingsAndHead` / `target_utils.py` | 从目标模型加载 `embed_tokens` 和 `lm_head`，冻结 |

---

## 2. DFlashDraftModel 模型架构

```
输入:
  noise_embedding  [B, N×bs, H]   # draft block 的 token 嵌入（含 MASK）
  target_hidden    [B, S, K×H]    # 目标模型多层隐状态拼接
  position_ids     [B, S + N×bs]  # 全序列位置 ID

内部结构:
  fc           : Linear(K×H → H)       # 压缩多层隐状态
  hidden_norm  : RMSNorm(H)            # 归一化压缩后的 context
  rotary_emb   : Qwen3RotaryEmbedding  # 旋转位置编码
  layers       : num_hidden_layers × Qwen3DFlashDecoderLayer
  norm         : RMSNorm(H)            # 最终输出归一化

其中:
  B  = batch size
  S  = context 序列长度（原始序列）
  N  = anchor（block）数量
  bs = block_size（每块 token 数，默认 16）
  H  = hidden_size
  K  = 捕获的目标模型层数（= len(target_layer_ids)）
```

### 2.1 Qwen3DFlashDecoderLayer

每个 Decoder Layer 包含：
- `input_layernorm`: RMSNorm
- `self_attn`: `Qwen3DFlashAttention`（改造后的注意力，见下）
- `post_attention_layernorm`: RMSNorm
- `mlp`: 标准 Qwen3MLP（SwiGLU）

### 2.2 Qwen3DFlashAttention（关键改造）

这是 DFlash 的核心创新：注意力的 **Q、K、V 来源不同**：

```
Q  ← q_proj(hidden_states)        # 仅来自 draft noise tokens [B, N×bs, H]
K  ← cat( k_proj(target_hidden),  # 来自目标模型压缩后的 context [B, S, ...]
           k_proj(hidden_states) ) # 来自 draft noise tokens [B, N×bs, ...]
V  ← cat( v_proj(target_hidden),
           v_proj(hidden_states) )
```

即 **Q 只来自 draft tokens**，而 **K/V 同时包含 context（目标模型隐状态）和 draft 本身**，形成"上文交叉注意力 + 块内自注意力"的混合结构。

最终 KV 序列布局：
```
KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{N-1}]
Q:  [Block_0 | Block_1 | ... | Block_{N-1}]
```

---

## 3. 训练阶段 Forward 过程（OnlineDFlashModel.forward）

### 输入

```python
input_ids    : Tensor[B, S]   # 原始 token ID（完整序列）
hidden_states: Tensor[B, S, K×H]  # 目标模型多层隐状态拼接
loss_mask    : Tensor[B, S]   # 1=assistant token（需要预测），0=user/padding
```

### Step 1：目标模型生成隐状态（训练循环外）

```python
target_output = target_model.generate_dflash_data(input_ids, attention_mask, loss_mask)
hidden_states = target_output.hidden_states  # [B, S, K×H]
```

目标模型做一次完整前向传播，捕获 `target_layer_ids` 指定的多个中间层隐状态并拼接：

```python
# build_target_layer_ids 选层策略（以 num_draft_layers=1 为例）
target_layer_ids = [num_target_layers // 2]   # 中间层

# 多层时均匀分布在 [1, num_target_layers-3] 范围内
# 隐状态拼接: hidden_states[layer_id + 1] (offset=1 因为 index 0 是 embedding 输出)
target_hidden = cat([hidden_states[lid+1] for lid in target_layer_ids], dim=-1)  # [B, S, K×H]
```

### Step 2：采样 Anchor 位置

```python
anchor_positions, block_keep_mask = _sample_anchor_positions(seq_len, loss_mask, device)
# anchor_positions: [B, N]  每个样本随机采样 N 个有效位置（loss_mask=1 的位置）
# block_keep_mask:  [B, N]  标记哪些 anchor 有效（用于过滤不足的样本）
```

- 从 `loss_mask=1` 的位置中随机采样，最多 `num_anchors` 个
- 每个 anchor 是一个 block 的起始位置

### Step 3：构造 Noise Embedding

每个 block 的 noise 输入：
- **位置 0（block 起始）**：anchor token 的真实 embedding
- **位置 1~(bs-1)**：MASK token 的 embedding

```python
noise_ids: [B, N×bs]
# noise_ids[:, block_start + 0] = input_ids[:, anchor_pos]  (真实 token)
# noise_ids[:, block_start + 1:] = mask_token_id            (MASK token)

noise_embedding = embed_tokens(noise_ids)  # [B, N×bs, H]
```

这模拟了推理时 draft 模型"只知道块首 token、后续均待预测"的场景。

### Step 4：构造 Position IDs

```python
context_position_ids = arange(S)          # [B, S]
draft_position_ids   = anchor + [0,1,...,bs-1]  # [B, N×bs]
full_position_ids    = cat([context_position_ids, draft_position_ids], dim=1)  # [B, S+N×bs]
```

### Step 5：构造 DFlash Attention Mask

注意力遮罩保证：
1. **每个 block 只能看到其 anchor 位置之前的 context**（严格小于，不含 anchor 位置本身）
2. **块内 bidirectional**（Block 内所有位置互相可见）
3. **不同 block 之间不可见**
4. **无效 block（block_keep_mask=False）看不到任何内容**

```
Q[block_i, pos_k] → KV 范围:
  - context[0 .. anchor_i - 1]         （上文，不含 anchor 本身）
  - Block_i[0 .. bs-1]                  （本块内全部，双向）
  - 其他 Block:  ✗ 不可见
```

支持两种实现：
- `flex_attention`：`BlockMask`（稀疏高效）
- `sdpa/eager`：稠密 4D float mask（`[B, 1, N×bs, S+N×bs]`）

### Step 6：DFlashDraftModel 前向传播

```python
# 1. 压缩目标隐状态
target_hidden_compressed = hidden_norm(fc(hidden_states))  # [B, S, H]

# 2. 计算旋转位置编码（覆盖全部 S+N×bs 位置）
position_embeddings = rotary_emb(noise_embedding, full_position_ids)

# 3. 逐层 Decoder
hidden = noise_embedding
for layer in layers:
    hidden = layer(
        hidden_states=hidden,            # Q 来源
        target_hidden=target_hidden_compressed,  # K/V 上下文部分来源
        attention_mask=dflash_attn_mask,
        position_embeddings=position_embeddings,
    )

# 4. 最终归一化
output = norm(hidden)  # [B, N×bs, H]
```

**注意**：`target_hidden` 在每一层的注意力中都作为 KV 的 context 部分，但并不是残差连接（与 EAGLE3 的隐状态融合方式不同）。

### Step 7：计算 Logits 与 Loss

```python
logits = lm_head(output)  # [B, N×bs, vocab_size]

# 标签：位置 k 处预测 input_ids[anchor + k]
# 即：Block[k] 预测序列位置 (anchor+k) 处的 token
label_indices = anchor_positions.unsqueeze(-1) + arange(block_size)  # [B, N, bs]
target_ids = input_ids[..., label_indices]

# 权重 mask（联合多个条件）：
weight_mask = block_keep_mask               # 有效 block
            * (label_indices < seq_len)     # 不越界
            * (pos_in_block > 0)            # 排除 block 第 0 位（已知 anchor token）
            * loss_mask[label_indices]      # 只对 assistant token 计 loss

# 可选指数衰减（越靠后预测难度越高，权重越小）
if loss_decay_gamma:
    decay = exp(-(k-1) / gamma)   # k=1 时权重=1，之后指数衰减
    weight_mask *= decay

# 交叉熵 loss（加权平均）
loss = sum(CE(logits, labels) * weight_mask) / sum(weight_mask)
```

**关键设计**：
- 第 0 位（anchor token）已知，不纳入 loss
- 第 1 位预测下一个 token（最容易），之后难度递增
- 论文建议：`gamma=7`（bs=16），`gamma=5`（bs=10），`gamma=4`（bs=8）

---

## 4. 推理阶段 Forward 过程（spec_generate）

### Prefill 阶段

```
1. target_model(input_ids) → logits, hidden_states
2. 采样第一个新 token: token_0 = sample(logits[-1])
3. target_hidden = extract(hidden_states, target_layer_ids)  # [1, S, K×H]
```

### Decoding 循环（每次迭代处理一个 block）

```
while not done:

  ┌─ Draft Step ──────────────────────────────────────────────────┐
  │ block_input = [anchor_token, MASK, MASK, ..., MASK]  (bs 个)  │
  │ noise_emb = embed_tokens(block_input)                         │
  │                                                               │
  │ draft_output = draft_model(                                   │
  │     noise_embedding=noise_emb,                                │
  │     target_hidden=target_hidden,   ← 来自上一轮 target 输出   │
  │     position_ids=[start..start+bs],                           │
  │     use_cache=True,                                           │
  │ )                                                             │
  │                                                               │
  │ draft_logits = lm_head(draft_output[:, 1:, :])               │
  │ # 跳过第 0 位（已知 anchor），取后 bs-1 个位置的预测          │
  │ draft_tokens = sample(draft_logits)   # 生成 bs-1 个候选     │
  │ block = [anchor_token] + draft_tokens                         │
  └───────────────────────────────────────────────────────────────┘

  ┌─ Verify Step ─────────────────────────────────────────────────┐
  │ target_output = target_model(                                 │
  │     input_ids=block,              ← bs 个 token               │
  │     use_cache=True,               ← KV Cache 续接             │
  │ )                                                             │
  │ posterior = sample(target_output.logits)                      │
  │                                                               │
  │ # 贪婪接受：找最长匹配前缀                                    │
  │ acceptance = cumprod(block[1:] == posterior[:-1]).sum()        │
  │ 接受 acceptance+1 个 token（含 anchor + 匹配部分 + 一个修正）  │
  └───────────────────────────────────────────────────────────────┘

  target_hidden = extract(target_output.hidden_states)[:acceptance+1]
  start += acceptance + 1
```

**注意 KV Cache 管理**：
- `past_key_values_target`：目标模型的 KV Cache，每轮 crop 到已接受的长度
- `past_key_values_draft`：draft 模型的 KV Cache，每轮 crop 到 `start` 位置

---

## 5. 输入输出汇总

### 训练阶段

| 阶段 | 输入 | 输出 |
|------|------|------|
| **目标模型前向** | `input_ids [B,S]`, `attention_mask [B,S]` | `hidden_states [B, S, K×H]` |
| **OnlineDFlashModel.forward** | `input_ids [B,S]`, `hidden_states [B,S,K×H]`, `loss_mask [B,S]` | `loss (scalar)`, `accuracy (scalar)` |
| **DFlashDraftModel.forward** | `noise_embedding [B,N×bs,H]`, `target_hidden [B,S,K×H]`, `position_ids [B,S+N×bs]`, `attention_mask` | `hidden [B, N×bs, H]` |

### 推理阶段

| 阶段 | 输入 | 输出 |
|------|------|------|
| **Prefill** | `input_ids [1,S]` | `token_0`, `target_hidden [1,S,K×H]` |
| **Draft** | `block_input [1,bs]`, `target_hidden [1,accept,K×H]` | `draft_tokens [bs-1]` |
| **Verify** | `block [1,bs]` | `posterior [bs]`, 新 `target_hidden` |

---

## 6. 与 EAGLE3 的主要区别

| 对比维度 | EAGLE3 | DFlash |
|----------|--------|--------|
| **Draft 模型输入** | 3层隐状态拼接 + token embedding | 多层隐状态作为 KV context，token embedding 作为 Q |
| **注意力机制** | 标准因果注意力 | 混合：context 交叉注意力 + 块内双向自注意力 |
| **训练并行性** | 顺序展开（autoregressive roll-out） | **并行 block**（所有 anchor block 并行前向） |
| **推理粒度** | 逐 token 预测树 | 以 block 为单位批量预测 |
| **目标隐状态使用** | 融入 draft embedding（残差） | 作为全局 context（K/V） |
| **预训练基座** | 自定义轻量层 | 完整 Qwen3 Decoder Layer（改造注意力） |
