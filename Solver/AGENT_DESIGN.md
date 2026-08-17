# Solver Agent Design

本文档说明当前 Solver 中三种 DQN agent 方案：纯视觉 solver、纯文字 solver、多模态 solver。三者共享同一个固定中心块环境、动作空间、replay buffer 和 DQN 训练逻辑，只替换策略网络的输入模态和特征提取方式。

## 1. 环境约束

当前任务是固定中心块的 3x3 拼图：

- board position 共 9 个，编号 `0..8`。
- 中心位置固定为 `CENTER_INDEX = 4`。
- `reset()` 只打乱其余 8 个位置：`0, 1, 2, 3, 5, 6, 7, 8`。
- 合法 action 是两个非中心位置的 swap。
- `legal_action_mask` shape 为 `[9, 9]`，对角线和中心行/列均为 `False`。
- 模型仍输出 81 个 Q 值，非法 action 在选择和 target 计算时被 mask 到 `-inf`。

reward 保持 shaped reward：

- `pairwise_reward = correct_pairs / 12`
- `category_reward = correct_movable_positions / 8`
- solved 时额外加 `done_reward`

中心块固定后，绝对位置奖励只统计 8 个可动位置，避免固定中心块天然贡献奖励。

## 2. 统一训练入口

训练入口仍是：

```bash
python Solver/train_rl.py --dataset-dir <dataset>
```

新增核心参数：

```bash
--solver-type visual|text|multimodal
```

默认值是 `visual`，保持现有 FEN 视觉 solver 行为。

通用训练参数包括：

- `--epochs`
- `--episodes-per-epoch`
- `--max-steps`
- `--batch-size`
- `--buffer-size`
- `--gamma`
- `--lr`
- `--epsilon-start`
- `--epsilon-end`
- `--epsilon-decay-steps`
- `--target-update-interval`
- `--save-dir`

每个 epoch 内会显示采样步数进度条，进度条按 `episodes_per_epoch * max_steps` 估计总采样步数，每次 `env.step()` 更新一次。

## 3. 共享 DQN 外壳

实现文件：

- `Solver/agent/q_learning_agent.py`

共享类：

- `DQNJigsawAgent`
- `ReplayBuffer`
- `CharTokenizer`
- `QHead`

所有 solver 都复用同一套 DQN 流程：

1. `env.reset()` 获取 observation。
2. `select_action()` 按 epsilon-greedy 选择合法 swap。
3. `env.step(action)` 执行动作。
4. transition 写入 replay buffer。
5. `optimize()` 采样 batch 并更新 policy network。
6. 定期同步 target network。

Q-learning target：

```python
target = reward + gamma * max_a target_net(next_obs, a) * (1 - done)
```

其中 `max_a` 会使用 `legal_action_mask` 排除非法 action。

## 4. 纯视觉 Solver

参数：

```bash
--solver-type visual
```

结构：

```text
current puzzle image
  -> FEN visual feature extractor
  -> QHead
  -> 81 swap Q values
```

实现类：

- `FenFeaturePolicy`

输入：

- `obs["image"]`

不使用：

- `obs["texts"]`

可选 FEN backbone：

```bash
--fen-model-name ef
--fen-model-name modulator
--fen-model-name central
--fen-model-name attention
--fen-model-name dualstem_ef
--fen-model-name dualstem_modulator
```

相关参数：

- `--image-size`：默认 288。FEN 当前按 3x3 patch 切分，每块 96。
- `--image-feature-dim`：FEN 输出特征维度。
- `--fen-hidden-size1`
- `--fen-feature-hidden`
- `--hidden-dim`：Q head 隐层维度。

适用场景：

- OCR 文本质量差或希望先验证视觉拼图能力。
- 训练速度通常比多模态更快。
- 默认 `ef` backbone 可能需要 torchvision 预训练权重缓存；离线环境可用 `central` 或已有缓存模型做 smoke test。

示例：

```bash
python Solver/train_rl.py ^
  --dataset-dir Data/newspaper-navigator/output ^
  --solver-type visual ^
  --fen-model-name central ^
  --image-feature-dim 256 ^
  --hidden-dim 256
```

## 5. 纯文字 Solver

参数：

```bash
--solver-type text
```

结构：

```text
OCR piece texts
  -> byte-level tokenizer
  -> ByteTransformerTextEncoder
  -> QHead
  -> 81 swap Q values
```

实现类：

- `TextFeaturePolicy`
- `ByteTransformerTextEncoder`

输入：

- `obs["texts"]`

不使用：

- `obs["image"]` 的视觉特征

文本来源：

- 默认：每个 piece 一段文本，形如 `<P0> ...`
- 启用 `--reconstruct-text-by-line`：按 OCR 字符框重建 board row 的行文本，形如 `<R0-L0> ...`

相关参数：

- `--text-feature-dim`
- `--text-num-heads`
- `--text-num-layers`
- `--text-max-length`
- `--hidden-dim`

适用场景：

- 验证 OCR 阅读顺序是否足以推断拼图排列。
- 对版面图像干扰不敏感。
- OCR 噪声较强时可能不稳定。

示例：

```bash
python Solver/train_rl.py ^
  --dataset-dir Data/newspaper-navigator/output ^
  --solver-type text ^
  --text-feature-dim 256 ^
  --text-num-heads 8 ^
  --text-num-layers 2 ^
  --text-max-length 512
```

按行重建文本：

```bash
python Solver/train_rl.py ^
  --dataset-dir Data/newspaper-navigator/output ^
  --solver-type text ^
  --reconstruct-text-by-line
```

## 6. 多模态 Solver

参数：

```bash
--solver-type multimodal
```

结构：

```text
current puzzle image -> FEN visual feature extractor -> image projection
OCR piece texts      -> ByteTransformerTextEncoder   -> text projection
projected image/text features -> concat -> QHead -> 81 swap Q values
```

实现类：

- `MultimodalFeaturePolicy`
- FEN visual feature extractor
- `ByteTransformerTextEncoder`

输入：

- `obs["image"]`
- `obs["texts"]`

相关参数：

- 视觉侧：`--image-size`、`--image-feature-dim`、`--fen-model-name`、`--fen-hidden-size1`、`--fen-feature-hidden`
- 文本侧：`--text-feature-dim`、`--text-num-heads`、`--text-num-layers`、`--text-max-length`
- 融合/Q head：`--hidden-dim`

适用场景：

- 同时利用视觉纹理、版面连续性和 OCR 文字上下文。
- 通常表达力最强，但训练和显存开销也最大。
- 建议先分别跑通 visual/text，再跑 multimodal。

示例：

```bash
python Solver/train_rl.py ^
  --dataset-dir Data/newspaper-navigator/output ^
  --solver-type multimodal ^
  --fen-model-name central ^
  --image-feature-dim 256 ^
  --text-feature-dim 256 ^
  --hidden-dim 256
```

## 7. Batch Collate

`DQNJigsawAgent._collate_observations()` 会统一准备 action mask，并按 `solver_type` 只准备当前策略网络需要的模态：

- `images`：仅 `visual` 和 `multimodal` 准备，float tensor，shape `[B, 3, image_size, image_size]`，保持 0..255 像素值。
- `text_inputs`：仅 `text` 和 `multimodal` 准备，byte tokenizer 输出的 `input_ids` 和 `attention_mask`。
- `legal_action_masks`：bool tensor，shape `[B, 9, 9]`。

不同 solver 在 `_forward_policy()` 中选择需要的输入：

- `visual`：只传 `images`
- `text`：只传 `text_inputs`
- `multimodal`：传 `images` 和 `text_inputs`

## 8. Checkpoint 兼容

checkpoint 保存：

- `policy_net`
- `target_net`
- `optimizer`
- `global_step`
- `epoch`
- `config`

由于三种 solver 的网络结构不同，checkpoint 只能加载到相同 `--solver-type` 和相同关键结构参数的 agent 中。

关键结构参数包括：

- `solver_type`
- `image_feature_dim`
- `text_feature_dim`
- `hidden_dim`
- `fen_model_name`
- `fen_hidden_size1`
- `fen_feature_hidden`
- `text_num_heads`
- `text_num_layers`
- `text_max_length`

## 9. 推荐实验顺序

1. 环境 smoke test：确认中心块固定、mask 正确。
2. `visual + central` 小 batch 跑通训练。
3. `text` 小 batch 跑通训练，比较默认文本和 `--reconstruct-text-by-line`。
4. `multimodal + central` 跑通融合模型。
5. 换更强 FEN backbone，如 `ef` 或 `dualstem_ef`。
6. 扩大数据集和训练轮数，比较 solved rate、avg reward、pairwise/category reward。

## 10. 后续扩展

- 增加 `--freeze-visual-backbone`，只训练 Q head。
- 增加 `--freeze-text-backbone`，只训练融合层和 Q head。
- 增加 evaluation-only 脚本，固定 epsilon=0。
- 增加 per-solver checkpoint 命名和自动恢复校验。
- 增加 piece-level multimodal transformer，用 8 个可动 piece token 和 1 个中心 anchor token 显式建模。
