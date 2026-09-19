# Qwen VL LoRA 排序求解器方案

## 1. 问题定义

给定 `K` 个漫画分镜 panel 的图像（以及可选的 OCR 对话文本），预测这 K 个 panel 的正确阅读顺序。实现上把“排列预测”转成“排列分类”：所有合法排列共有 `K!` 种，模型直接输出 `K!` 个类别的得分。

---

## 2. 总体结构

```text
输入: [B, K, 3, H, W] 图像 + [B, K] OCR 文本
          |
          v
  Qwen2.5-VL Processor (chat template + 图像预处理)
          |
          v
  Qwen2.5-VL Backbone (VLM, LoRA 微调)
          |
          v
  最后一层 hidden_states -> last-token pooling
          |
          v
  分类头 MLP -> logits [B, K!]
          |
          v
  argmax -> 排列类别 id -> class_orders 查表 -> [B, K] 排序
```

核心文件：
- `Solver/agent/mangazero_qwen_vl_lora_classifier.py`：模型、输入编码、推理、解码
- `Solver/train_mangazero_qwen_vl_lora.py`：训练、评测、checkpoint
- `Solver/env/mangazero_panel_env.py`：数据集和 batch 组装

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

`build_ordering_prompt()` 生成排序 prompt，明确给出 panel 编号，要求模型根据视觉连续性、阅读流、角色、对话气泡、场景转换推断阅读顺序；明确“不要生成顺序，只做内部表示用于分类”。

可选地把每个 panel 的 OCR 文本按编号追加为 `OCR text:` 段落：

```text
panel 0: ... 
panel 1: ...
```

### Processor 编码

`AutoProcessor.apply_chat_template()` 产出：
- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`（Qwen2.5-VL 的图像网格）

这些 tensor 会移动到 backbone 所在设备。

---

## 4. VLM 输出与表示

### backbone 输出

```python
outputs = self.backbone(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden_size]
```

### pooling

取 attention_mask 中最后一个真实 token：

```python
indices = attention_mask.long().sum(dim=1).clamp(min=1) - 1
pooled = hidden[range(B), indices]  # [B, hidden_size]
```

### 分类头

- 默认单层：`LayerNorm -> Dropout -> Linear(hidden_size, K!)`
- 可选 MLP：`LayerNorm -> Dropout -> Linear(hidden, hidden) -> GELU -> Dropout -> Linear(hidden, K!)`
- 输出 `logits [B, K!]`

---

## 5. 排列索引与解码

初始化时枚举所有排列：

```python
class_orders = list(itertools.permutations(range(K)))  # [K!, K]
_class_index = {tuple(order): index ...}
```

- `class_orders` 是 `[K!, K]` 的 buffer
- `_class_index` 是排列 -> 类别 id 的反向映射

解码：

```python
class_ids = logits.argmax(dim=-1)          # [B]
pred_orders = self.class_orders[class_ids]  # [B, K]
```

`pred_orders[i] = [p0, p1, ..., pK-1]` 表示：第 0 位放 panel `p0`，第 1 位放 panel `p1`，依此类推。

K 需要满足 `K! <= max_permutation_classes`（默认 50000）。`K=5` 为 120 类，`K=7` 为 5040 类，`K=8` 为 40320 类。

---

## 6. 训练流程

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
class_targets = model.target_to_class_indices(target_order)  # 排列 -> 类 id
loss = F.cross_entropy(logits, class_targets)
```

### 优化

```python
AdamW([
    {"params": backbone_params, "lr": 1e-4},  # LoRA adapter
    {"params": head_params,     "lr": 1e-3},  # 分类头
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
| 分类头 | 从零初始化 |

### 训练循环

```text
for epoch:
    for batch in train_loader:
        batch = randomize_training_batch(batch)
        logits = model(panel_images, dialog_texts)
        loss = cross_entropy(logits, targets) / grad_accum_steps
        loss.backward()
        每 grad_accum_steps 步: clip -> optimizer.step -> zero_grad
    在 val / test 上 evaluate
    按 best/last/epoch 保存 checkpoint
```

### 评测指标

- `loss`: 分类交叉熵
- `class_accuracy`: 类别 id 预测正确率
- `exact_match`: 完整排列正确比例
- `position_accuracy`: 每个位置正确比例
- `pairwise_accuracy`: 所有 panel 对相对顺序正确比例

---

## 7. 量化与 checkpoint

- 支持 4-bit NF4 QLoRA（`bitsandbytes`），加载时 `prepare_model_for_kbit_training`
- 默认只保存 LoRA adapter + 分类头 + `class_orders`，不保存完整 3B 权重
- `--save-full-state` 可保存完整状态

---

## 8. 方案取舍

- 优点：直接利用预训练 VLM 的视觉-语言理解和漫画/文本语义，比自训练小模型更容易对齐任务。
- 代价：显存/推理开销大，默认 batch=1；`K!` 分类类别数随 K 快速增长；解码只做 argmax，不会评估次优排列或做 beam search。
