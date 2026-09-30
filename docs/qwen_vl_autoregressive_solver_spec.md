# Qwen VL 自回归排序求解器方案

## 1. 任务定义

给定 `K` 个漫画分镜 panel 的图像（以及可选的 OCR 对话文本），预测这 K 个 panel 的正确阅读顺序。

输出形式与 `mangazero_panel_ordering_plan.md` 中的 set-to-sequence solver 保持一致：

```text
target_order[t] = 第 t 个正确 panel 在乱序输入中的位置
```

因此 solver 输出的是“逐步选出的 panel 指针序列”，而不是全排列类别。

---

## 2. 总体结构

```text
输入: [B, K, 3, H, W] 图像 + [B, K] OCR 文本
          |
          v
  Qwen2.5-VL Processor (chat template + 图像预处理)
          |
          v
  Qwen2.5-VL Backbone (VLM, LoRA 微调，输出 last hidden states)
          |
          v
  per-panel 池化: <|image_pad|> token 切段 + mean-pool
          |
          v
  panel_projection + panel_norm -> memory [B, K, decoder_dim]
          |
          v
  自回归 pointer decoder:
    当前步对“剩余候选 panel”打分数
    -> mask 已选 panel
    -> argmax 选出下一张 panel
          |
          v
  重复 K 次 -> target_order [B, K]
```

核心区别：不再枚举 `K!` 个全排列类别，而是每一轮只预测“剩余 panel 中最可能是下一张的 panel”，重复 `K` 次完成排序。

---

## 3. VLM 输入

### 图像输入

- `panel_images`: `[B, K, 3, H, W]`，默认 `224x224`
- 每个 panel 的图像 tensor 先转成 `PIL.Image`
- 一张图一条对话消息，按 panel 0、panel 1、... 顺序构成多图像 user 消息

```python
conversations = [{
    "role": "user",
    "content": [
        {"type": "image", "image": <panel0>},
        {"type": "image", "image": <panel1>},
        ...
        {"type": "text", "text": <排序 prompt>},
    ],
}]
```

### 文本输入

`build_ordering_prompt()` 生成排序 prompt：

```text
You are solving a manga panel ordering puzzle.
There are K input panels numbered 0, 1, ..., K-1.
Use visual continuity, reading flow, characters, dialog bubbles,
and scene transitions to infer the correct chronological reading order.
Do not generate the order; provide an internal representation for classification.
```

可选地把每个 panel 的 OCR 文本按编号追加为 `OCR text:` 段落：

```text
panel 0: ...
panel 1: ...
```

实际文本格式为：

```text
panel 0: ...
panel 1: ...
```

### Processor 编码

`AutoProcessor.apply_chat_template()` 产出：

- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`（若 processor 返回）

这些 tensor 会移动到 backbone 所在设备。

兼容处理：`_call_processor()` 对 transformers 新旧版本做回退。新版把 `return_tensors/padding` 作为顶层 processing kwargs 传入；旧版不允许时降级为不带这些参数的调用。

---

## 4. 编码：把每个 panel 变成一个向量

VLM 的最后一层 `hidden_states` 形状为 `[B, seq_len, hidden_size]`。

实现上（不是可选的抽象方案）：

1. 通过 `input_ids` 找到 `<|image_pad|>` token 的连续区间，每个连续 run 对应一张 panel 的图像 token。
2. 对每个 run 的 hidden state 做 mean-pool，得到 `[B, K, hidden_size]`。
3. 经过 `panel_projection`（`Linear(hidden_size, decoder_dim)`）和 `panel_norm`（`LayerNorm`）得到最终 memory。

```text
per-panel 表示 [B, K, hidden_size]
-> panel_projection + panel_norm
-> memory [B, K, decoder_dim]
```

代码位置：`encode_panels()` / `_pool_per_panel()`。

注意：代码里声明了 `type_embedding` 参数，但目前没有把它加回 memory 前向路径，属于未启用的预留 token。

---

## 5. 自回归头实现细节（输入、结构、输出）

### 5.1 头部的模块组成

自回归头是指 backbone 之外的部分：

```python
self.panel_projection = nn.Linear(hidden_size, decoder_dim)
self.panel_norm = nn.LayerNorm(decoder_dim)
self.type_embedding = nn.Parameter(...)   # 已声明，未参与前向
self.bos_token = nn.Parameter(...)        # decoder 起始向量

decoder_layer = nn.TransformerDecoderLayer(
    d_model=decoder_dim,
    nhead=num_heads,
    dim_feedforward=decoder_dim * 4,
    dropout=decoder_dropout,
    activation="gelu",
    batch_first=True,
    norm_first=True,
)
self.pointer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

self.output_norm = nn.LayerNorm(decoder_dim)
self.query_projection = nn.Linear(decoder_dim, decoder_dim)
self.key_projection = nn.Linear(decoder_dim, decoder_dim)
```

默认构造参数：

| 参数 | 默认值 |
| --- | --- |
| `decoder_layers` | 2 |
| `num_heads` | 8 |
| `decoder_dim` | 256 |
| `decoder_dropout` | 0.1 |

### 5.2 输入

头部有两个输入：

1. `memory [B, K, decoder_dim]`：全部候选 panel 的编码，作为 pointer 的 key/value，也是解码历史中“已选 panel 的表示”来源。
2. 解码历史：`[BOS] + memory[selected]`，即 `[B, 1 + t, decoder_dim]`。

- 推理第 0 步只有 `bos_token`
- 推理第 t 步为 `[B, 1 + t, decoder_dim]`
- 训练 teacher forcing 时用 `target_order[:, :-1]` 代替已选序列

### 5.3 输出

打分公式：

```python
queries = query_projection(output_norm(decoder_state))   # [B, 1, decoder_dim]
keys = key_projection(memory)                             # [B, K, decoder_dim]
logits = queries @ keys.T / sqrt(decoder_dim)             # [B, 1, K]
```

- 训练输出 `[B, K, K]`：`logits[b, t, j]` 是第 b 个样本第 t 步选择第 j 个输入 panel 的倾向。
- 推理输出 `[B, K]`：整个 greedy 循环结束后得到逐 step 选中的 panel index 序列。

### 5.4 训练/推理差异

| 阶段 | decoder 输入 | mask 方式 | 输出 |
| --- | --- | --- | --- |
| 训练（teacher forcing） | `[BOS] + memory[target_order[:, :-1]]` | 按 `target_order` 构造上三角 mask；pointer logits 再 mask 已选 | `[B, K, K]` |
| 推理（greedy） | `[BOS] + memory[selected]` | 每步把已选 panel logits 置 `-inf` | `[B, K]` |

`memory` 可复用：`forward(..., memory=...)` / `predict_order(..., memory=...)` 支持传入已编码 memory，避免评测时重复跑 VLM。

---

## 6. 自回归排序流程

### 6.1 每一轮做什么

模型不是一次输出整个排序，而是逐步输出“下一张应该选哪个 panel”：

1. 当前已选历史为 `selected`（乱序输入中的 panel index）。
2. 对剩余候选 panel 计算分数。
3. 把 `selected` 中已选 panel 的分数设为 `-inf`，确保不会重复选择。
4. `argmax` 得到当前步选择的 panel index。
5. 把它追加到 `selected`。
6. 重复 K 次，得到完整排序。

### 6.2 推理 greedy decode

```text
selected = []
for t in 0..K-1:
    decoder_inputs = [BOS] + memory[selected]
    decoder_states = pointer_decoder(decoder_inputs, memory, causal_mask)
    logits = pointer_logits(decoder_states[:, -1:], memory)   # [B, 1, K]
    logits[selected] = -inf
    next_idx = argmax(logits)
    selected.append(next_idx)
```

示例：

```text
输入乱序 panel: [p3, p0, p5, p1, p4, p2]
模型逐步输出:   [1, 3, 5, 0, 4, 2]

含义:
第 0 步选 p0
第 1 步选 p1
第 2 步选 p2
第 3 步选 p3
第 4 步选 p4
第 5 步选 p5
```

---

## 7. 训练流程

### 数据准备

- 数据集从 manifest/jsonl 读取，样本含：
  - `panel_images: [K,3,H,W]`
  - `target_order: [K]`
  - `dialog_texts: [K]`
- `collate_panel_ordering_batch` 将样本堆成 batch：`[B,K,3,H,W]`

### 在线重排增强

`randomize_training_batch()`：

1. 按 `target_order` 恢复 canonical 顺序
2. 随机生成一个新排列 `perm`
3. 按 `perm` 重排 panel 图像、OCR 文本和 `target_order`

效果是同一个 puzzle 每次看到不同的面板摆放顺序，避免模型依赖固定输入顺序。

### 损失

```python
def pointer_cross_entropy(pointer_logits, target_order):
    # pointer_logits: [B, K, K]
    # target_order:   [B, K]
    return F.cross_entropy(
        pointer_logits.reshape(-1, K),
        target_order[:, :K].reshape(-1),
    )
```

对每一步的 panel 指针分类做交叉熵。

### 优化

```python
AdamW([
    {"params": backbone_params, "lr": 1e-4},  # LoRA adapter
    {"params": decoder_params,  "lr": 1e-3},  # 自回归 decoder + 投影层
])
```

默认关键参数：

| 参数 | 值 |
| --- | --- |
| backbone | Qwen2.5-VL-3B-Instruct |
| LoRA | r=16, alpha=32, dropout=0.05，q/k/v/o/gate/up/down_proj |
| dtype | bfloat16 |
| batch_size | 1 |
| grad_accum_steps | 8 |
| 梯度裁剪 | 1.0 |
| decoder | 从零初始化 |

### 训练循环

```text
for epoch:
    for batch in train_loader:
        batch = randomize_training_batch(batch)
        logits = model(panel_images, target_order=target_order, dialog_texts=...)
        loss = pointer_cross_entropy(logits, target_order) / grad_accum_steps
        loss.backward()
        每 grad_accum_steps 步: clip -> optimizer.step -> zero_grad
    在 val / test 上 evaluate（greedy decode）
    按 best/last/epoch 保存 checkpoint
```

### 评测指标

- `loss`: pointer 交叉熵
- `exact_match`: 完整排列正确比例
- `position_accuracy`: 每个位置正确比例
- `pairwise_accuracy`: 所有 panel 对相对顺序正确比例

---

## 8. 量化与 checkpoint

- 支持 4-bit NF4 QLoRA（`bitsandbytes`），加载时 `prepare_model_for_kbit_training`
- 默认只保存 LoRA adapter + decoder 相关参数（不含 `class_orders`，自回归实现没有该 buffer）
- `--save-full-state` 可保存完整状态

---

## 9. 与全排列分类方案的差异

| 维度 | 全排列分类 | 自回归 pointer 排序（本方案） |
| --- | --- | --- |
| 输出 | `[B, K!]` logits，argmax 查表 | 逐步输出 `[B, K]` panel 指针 |
| 类别数 | `K!`（K=8 即 40320 类） | 每步最多 K 个候选，总量线性 |
| 解码 | 一次前向 | K 次 decoder 前向，每次选下一张 |
| 训练 loss | 全排列分类交叉熵 | 每步 pointer 交叉熵（teacher forcing） |
| 是否允许逐张修正 | 否 | 是，greedy/beam 可按步修正 |
| 复杂度 | `K!` 类别随 K 指数增长 | 每步 K 选 1，适合更大 K |

---

## 10. 核心代码文件

| 文件 | 职责 |
| --- | --- |
| `Solver/agent/mangazero_qwen_vl_autoregressive.py` | VLM + LoRA 编码器、per-panel 池化、自回归 pointer decoder、teacher forcing、greedy decode |
| `Solver/train_mangazero_qwen_vl_autoregressive.py` | 训练循环、评测、checkpoint |
| `Solver/test_mangazero_qwen_vl_autoregressive.py` | 从 checkpoint 重建模型并评估指定 split |
| `Solver/env/mangazero_panel_env.py` | 数据集和 batch 组装 |

---

## 11. 实现状态与使用说明

### 11.1 已实现功能

| 功能 | 实现位置 | 说明 |
| --- | --- | --- |
| VLM + LoRA 主干 | `MangaZeroQwenVLAutoregressiveSolver.__init__` | Qwen2.5-VL-3B + PEFT LoRA，支持 4-bit NF4 QLoRA |
| 输入编码 | `prepare_inputs()` | K 张 panel 转 PIL，多图像 user 消息 + 排序 prompt，经 AutoProcessor 编码 |
| per-panel 池化 | `encode_panels()` / `_pool_per_panel()` | 按 `<|image_pad|>` token 的连续 run 切分出 K 个 panel 段，分别 mean-pool |
| 投影到 decoder 空间 | `panel_projection` + `panel_norm` | VLM hidden 映射到 `decoder_dim` |
| pointer decoder | `pointer_decoder` + `query_projection` + `key_projection` | TransformerDecoder 逐层解码，query 与 memory 的 key 做点积打分 |
| teacher forcing 训练 | `forward(target_order=...)` + `pointer_cross_entropy()` | 输出 `[B, K, K]` pointer logits；训练 loss 是每一步的 panel 指针交叉熵 |
| greedy decode 推理 | `predict_order()` / `greedy_decode()` | 每步把已选 panel 掩码为 `-inf`，argmax 选出下一张，重复 K 次得到 `[B, K]` 排序 |
| 内存复用 | `forward(memory=...)` / `predict_order(memory=...)` | 已有 memory 时跳过 VLM 前向 |
| 训练脚本 | `Solver/train_mangazero_qwen_vl_autoregressive.py` | 在线重排增强、梯度累积、梯度裁剪、双学习率、best/last/epoch checkpoint、train/val/test 评估 |
| 测试脚本 | `Solver/test_mangazero_qwen_vl_autoregressive.py` | 从 checkpoint 重建模型，评估指定 split 并可输出 JSON |

### 11.2 快速调用

基础训练（6 panel）：

```bash
python Solver/train_mangazero_qwen_vl_autoregressive.py \
  --dataset-dir Data/Mangazero/ordering_dataset \
  --panel-count 6
```

多 epoch + 更大等效 batch：

```bash
python Solver/train_mangazero_qwen_vl_autoregressive.py \
  --dataset-dir Data/comix/6panel-ordering \
  --panel-count 6 \
  --epoch 10 \
  --batch-size 2 \
  --grad-accum-steps 4
```

从 checkpoint 继续训练：

```bash
python Solver/train_mangazero_qwen_vl_autoregressive.py \
  --dataset-dir Data/Mangazero/ordering_dataset \
  --panel-count 6 \
  --epoch 10 \
  --load Solver/checkpoints_mangazero_qwen_vl_autoregressive/last.pt
```

测试 checkpoint：

```bash
python Solver/test_mangazero_qwen_vl_autoregressive.py \
  --dataset-dir Data/Mangazero/ordering_dataset \
  --panel-count 6 \
  --checkpoint Solver/checkpoints_mangazero_qwen_vl_autoregressive/best.pt \
  --split test
```

输出保存为 JSON：

```bash
python Solver/test_mangazero_qwen_vl_autoregressive.py \
  --dataset-dir Data/Mangazero/ordering_dataset \
  --panel-count 6 \
  --checkpoint Solver/checkpoints_mangazero_qwen_vl_autoregressive/best.pt \
  --split test \
  --output-json Solver/checkpoints_mangazero_qwen_vl_autoregressive/test.json
```

### 11.3 训练脚本参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset-dir` | （必填） | manifest/jsonl 数据集目录 |
| `--panel-count` | （必填） | 每个 puzzle 的 panel 数，必须与数据集一致 |
| `--model-name` | `Qwen/Qwen2.5-VL-3B-Instruct` | Hugging Face 模型名 |
| `--epoch` / `--epochs` | `3` | 训练 epoch 数；同时设置时以 `--epochs` 为准 |
| `--split-ratio` | `0.8,0.1,0.1` | train/val/test 切分比例 |
| `--test-per-epoch` | `1` | 每多少 epoch 跑一次 test |
| `--batch-size` | `1` | 微 batch 大小；VLM 显存大，默认 1 |
| `--grad-accum-steps` | `8` | 梯度累积步数，等效 batch = batch-size × 梯度累积 |
| `--lr-backbone` | `1e-4` | LoRA adapter 学习率 |
| `--lr-decoder` | `1e-3` | decoder + 投影层学习率 |
| `--weight-decay` | `0.0` | AdamW 权重衰减 |
| `--num-workers` | `0` | DataLoader worker 数 |
| `--image-width` / `--image-height` | `224` / `224` | panel 输入分辨率 |
| `--torch-dtype` | `bfloat16` | 可选 auto / float32 / fp32 / float16 / fp16 / bfloat16 / bf16 |
| `--attn-implementation` | `sdpa` | transformer 注意力实现 |
| `--min-pixels` / `--max-pixels` | `None` / `512*28*28` | processor 图像像素上下限 |
| `--load-in-4bit` | 关闭 | 开启 4-bit NF4 QLoRA |
| `--device-map` | `None` | 可选 HF device_map，如 `auto` |
| `--no-lora` | 关闭 | 打开后冻结 backbone，不使用 LoRA |
| `--lora-r` | `16` | LoRA 秩 |
| `--lora-alpha` | `32` | LoRA 缩放系数 |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--lora-target-modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | LoRA 注入模块 |
| `--decoder-layers` | `2` | pointer decoder 层数 |
| `--num-heads` | `8` | decoder 注意力头数 |
| `--decoder-dim` | `256` | decoder 隐藏维度 |
| `--decoder-dropout` | `0.1` | decoder dropout |
| `--grad-clip-norm` | `1.0` | 梯度裁剪阈值 |
| `--save-dir` | `Solver/checkpoints_mangazero_qwen_vl_autoregressive` | checkpoint 保存目录 |
| `--load` | `None` | 续训 checkpoint 路径 |
| `--save-every` | `1` | 每多少 epoch 保存 `epoch_XXXX.pt` |
| `--save-full-state` | 关闭 | 打开后保存完整模型状态，否则只保存 LoRA + decoder |
| `--device` | 自动 `cuda` / `cpu` | 训练设备 |
| `--seed` | `0` | 随机种子，影响 split 与在线重排 |

### 11.4 测试脚本参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset-dir` | （必填） | 数据集目录 |
| `--panel-count` | （必填） | 每个 puzzle 的 panel 数，必须与 checkpoint 一致 |
| `--checkpoint` | （必填） | 要评估的 checkpoint 路径，如 `best.pt` / `last.pt` |
| `--split` | （必填） | `train` / `valid` / `val` / `test` |
| `--seed` | `0` | split 随机种子，必须与训练时一致 |
| `--split-ratio` | checkpoint 配置优先，否则 `0.8,0.1,0.1` | split 比例；不传时优先用 checkpoint 保存的配置 |
| `--batch-size` | checkpoint 配置优先，否则 `1` | 测试 batch 大小 |
| `--num-workers` | checkpoint 配置优先，否则 `0` | 测试 worker 数 |
| `--device` | 自动 `cuda` / `cpu` | 测试设备 |
| `--output-json` | `None` | 可选，把指标保存为 JSON |

注意 `--seed` 与 `--split-ratio` 必须与训练时一致，否则 train/val/test 样本划分会变化。

### 11.5 checkpoint 与评估

保存内容（`Solver/checkpoints_mangazero_qwen_vl_autoregressive/`）：

```text
config.json
epoch_XXXX.pt
best.pt
last.pt
test_epoch_XXXX.json
test.json
```

checkpoint 内部：

```python
{
    "model": ...,        # 默认 LoRA adapter + decoder，full_state 时是完整 state_dict
    "optimizer": ...,
    "epoch": ...,
    "config": ...,       # 训练参数，用于测试脚本重建模型
    "solver_family": "qwen_vl_autoregressive",
    "full_state": ...,
}
```

评估指标：`loss`、`exact_match`、`position_accuracy`、`pairwise_accuracy`。

### 11.6 实现细节与已知注意点

- 自回归头是轻量模块（2 层 TransformerDecoder，默认 256 维），主要开销在 VLM 编码；推理 K 次 decoder 前向成本很低。
- `type_embedding` 已声明但未加入前向，当前属于预留 token；如果启用，需要在 `encode_panels()` 中把它加到 memory 上。
- `_pool_per_panel` 依赖 processor 输出 `input_ids` 中的 `<|image_pad|>` token 位置；若后续升级 processor 导致图像 token 标记变化，需要同步调整。
- 推理是 greedy decode（argmax），暂无 beam search。
- `_call_processor` 已兼容 transformers 新旧版本的 `return_tensors/padding` 传参方式。
- 本机未跑完整训练前向；需在配置好 torch/transformers/peft 的环境中运行验证。
