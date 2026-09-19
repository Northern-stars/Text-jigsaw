# Text Jigsaw 强化学习需求文档与实现方案

## 1. 目标

在 `Solver/` 下实现一个基于强化学习的 3x3 文本拼图求解流程。

系统由三部分组成：

1. `env`：读取 `DataGenerator.py` 生成的数据，构造 3x3 拼图环境。
2. `agent`：使用 `Solver/model_code/fen_model.py` 中的多模态模型，输入当前拼图图像和 piece 文本，输出交换动作。
3. `train`：训练入口，使用 Q-learning / DQN 训练 agent。

## 2. 数据格式

参考 `Data/DataGenerator.py`，每个 puzzle 样本由一个 `.json` 标签文件和 9 张 piece 图片组成。

示例文件结构：

```text
Data/PuzzleData/
  test-puzzle-576.json
  test-puzzle-576_0.png
  test-puzzle-576_1.png
  ...
  test-puzzle-576_8.png
```

JSON 样本格式：

```json
{
  "grid_size": 3,
  "pieces": [
    {
      "piece_id": 0,
      "row": 0,
      "col": 0,
      "text": "<SEG>...",
      "segments": ["..."],
      "image": "test-puzzle-576_0.png"
    }
  ]
}
```

实现要求：

- env 初始化时遍历数据集文件夹，收集所有 `.json` 文件路径。
- 每次 `reset()` 时随机抽取一个 JSON 样本。
- 读取该样本的全部 9 个 piece 图片和 label。
- 按 `piece_id` 建立原始正确顺序：`0..8`。
- 随机打乱当前顺序，作为 episode 初始状态。

## 3. 环境设计

建议文件：

```text
Solver/env/jigsaw_env.py
```

核心类：

```python
class TextJigsawEnv:
    def __init__(
        self,
        dataset_dir: str,
        pairwise_weight: float = 1.0,
        category_weight: float = 1.0,
        done_reward: float = 10.0,
        max_steps: int = 50,
        reconstruct_text_by_line: bool = False,
        image_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ): ...
```

### 3.1 状态定义

env 内部维护：

```python
self.pieces: list[Piece]
self.current_order: list[int]
self.step_count: int
```

其中：

- `current_order[position] = piece_id`
- `position` 是当前棋盘位置，范围 `0..8`
- `piece_id` 是原始正确位置编号，范围 `0..8`

每一步返回 observation：

```python
{
    "image": current_image,
    "texts": current_texts,
    "order": current_order,
    "legal_action_mask": mask,
}
```

字段说明：

- `image`：将 9 块 piece 按 `current_order` 顺序拼接成当前 3x3 图片。
- `texts`：当前棋盘对应的文本信息，具体格式由 `reconstruct_text_by_line` 控制。
- `order`：长度为 9 的 int 列表，用于训练、debug 和 reward 计算。
- `legal_action_mask`：`[9, 9]` bool mask，对角线为 `False`，其余为 `True`。

### 3.2 当前文本输出

数据生成时每个 piece 有两个文本字段：

- `text`：使用 `<SEG>` 标记 segment 边界的字符串。
- `segments`：segment 列表，每个 segment 对应该 piece 内的一行或一个换行片段。

env 文本输出必须优先读取 `segments`，不应通过手写 split 解析 `<SEG>`。`<SEG>` 只作为兼容旧字段的标记。

#### 默认模式：简单拼接

当 `reconstruct_text_by_line=False` 时，保持简单输出。

推荐输出：

```python
texts = [
    f"<P{pos}> {piece_text}"
    for pos, piece_text in enumerate(current_piece_texts)
]
```

其中 `piece_text` 优先由 `segments` 生成：

```python
piece_text = " ".join(piece["segments"])
```

如果某个旧样本没有 `segments` 字段，再 fallback 到 `piece["text"]`。

该模式的语义：

- `texts` 长度为 9。
- `texts[pos]` 对应当前棋盘位置 `pos` 的 piece 文本。
- 仅保留 piece 级位置信息，不尝试恢复跨 piece 的同行阅读顺序。

#### 按行重组模式

当 `reconstruct_text_by_line=True` 时，env 需要读取每个 piece 的每个 `segment`，标记行 id，并将当前棋盘中同一大行内、同一 segment 行号的 piece 文本拼接到一起。

3x3 棋盘位置定义：

```python
board_row = position // 3
board_col = position % 3
```

对每个当前棋盘大行 `board_row`：

1. 取该大行的三个位置：

```python
row_positions = [
    board_row * 3 + 0,
    board_row * 3 + 1,
    board_row * 3 + 2,
]
```

2. 找到这三个 piece 的最大 segment 数：

```python
max_line_count = max(len(piece["segments"]) for piece in row_pieces)
```

3. 对每个 `line_id in range(max_line_count)`，按 `board_col=0..2` 读取对应 segment，并拼成一行：

```python
line_text = " ".join(
    piece["segments"][line_id].rstrip("\n")
    for piece in row_pieces
    if line_id < len(piece["segments"])
)
```

4. 输出时必须显式标记行 id。

推荐输出格式：

```text
<R0-L0> text_from_col0 text_from_col1 text_from_col2
<R0-L1> text_from_col0 text_from_col1 text_from_col2
...
<R1-L0> ...
...
<R2-L0> ...
```

其中：

- `R{board_row}` 表示当前棋盘大行。
- `L{line_id}` 表示该大行内部的 segment 行号。
- 拼接顺序必须使用当前棋盘顺序，而不是原始 label 顺序。

该模式的语义：

- `texts` 可以是一个字符串，也可以是按行字符串列表。
- 推荐 env 输出 `texts: list[str]`，每个元素是一行重组文本。
- agent collate 时再将该 list 合并为一个文本序列。

示例：

```python
texts = [
    "<R0-L0> row0_col0_line0 row0_col1_line0 row0_col2_line0",
    "<R0-L1> row0_col0_line1 row0_col1_line1 row0_col2_line1",
    "<R1-L0> row1_col0_line0 row1_col1_line0 row1_col2_line0",
]
```

该模式用于让文本 encoder 看到跨 piece 的同行关系，适合文本连续性较强的数据。

### 3.3 当前图片拼接

拼接规则：

- 按棋盘位置 `0..8` 顺序放置。
- `row = position // 3`
- `col = position % 3`
- 使用 `current_order[position]` 找到对应 piece 图片。
- 拼接结果尺寸为：
  - 高度：`piece_h * 3`
  - 宽度：`piece_w * 3`

建议内部使用 PIL 或 numpy，最终可按训练需要输出：

- PIL Image
- numpy array
- torch tensor

优先方案：env 输出 PIL/numpy，训练代码统一做 transform，避免 env 绑定 PyTorch。

### 3.4 动作定义

动作为 `permute`：

```python
action = (index1, index2)
```

执行效果：

```python
current_order[index1], current_order[index2] = (
    current_order[index2],
    current_order[index1],
)
```

约束：

- `index1` 范围：`0..8`
- `index2` 范围：`0..8`
- `index1 != index2`
- 使用 mask 屏蔽 `index1 == index2`

### 3.5 Reward 定义

最终 reward：

```python
reward = (
    pairwise_reward * pairwise_weight
    + category_reward * category_weight
    + done_reward_if_solved
)
```

#### Pairwise reward

3x3 棋盘中需要计算 6 组水平邻接和 6 组垂直邻接。

水平位置对：

```python
[(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]
```

垂直位置对：

```python
[(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8)]
```

判断规则：

- 对水平位置对 `(left_pos, right_pos)`：
  - `left_piece = current_order[left_pos]`
  - `right_piece = current_order[right_pos]`
  - 当 `right_piece == left_piece + 1` 且两者原始 row 相同，记 1 分。
- 对垂直位置对 `(top_pos, bottom_pos)`：
  - `top_piece = current_order[top_pos]`
  - `bottom_piece = current_order[bottom_pos]`
  - 当 `bottom_piece == top_piece + 3` 且两者原始 col 相同，记 1 分。

推荐归一化：

```python
pairwise_reward = correct_pair_count / 12.0
```

如需更强信号，也可使用未归一化分数 `0..12`，但训练超参需要同步调整。

#### Category reward

category reward 衡量每个 piece 是否位于绝对正确位置：

```python
correct_position_count = sum(
    current_order[pos] == pos
    for pos in range(9)
)
category_reward = correct_position_count / 9.0
```

#### Done reward

done 条件：

```python
current_order == [0, 1, 2, 3, 4, 5, 6, 7, 8]
```

当 done 为真时：

```python
done_reward_if_solved = done_reward
```

否则：

```python
done_reward_if_solved = 0.0
```

### 3.6 done / truncated

建议返回 Gymnasium 风格：

```python
obs, reward, done, truncated, info = env.step(action)
```

规则：

- `done=True`：拼图完全复原。
- `truncated=True`：达到 `max_steps` 但未复原。
- `info` 包含 reward 分解：

```python
{
    "pairwise_reward": float,
    "category_reward": float,
    "done_reward": float,
    "correct_pairs": int,
    "correct_positions": int,
    "order": list[int],
    "sample_path": str,
}
```

## 4. Agent 设计

建议文件：

```text
Solver/agent/q_learning_agent.py
```

### 4.1 网络输出

使用 `fen_model.py` 中的 `FenModel` / `MultimodalFenModel`：

```python
q_model = FenModel(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    image_feature_dim=image_feature_dim,
    text_feature_dim=text_feature_dim,
    num_outputs=81,
)
```

输出解释：

```python
q_values = model(image_inputs, text_inputs)
q_values = q_values.view(batch_size, 9, 9)
```

动作选择：

- 对角线 `q_values[:, i, i]` 设为 `-inf`
- epsilon-greedy：
  - 随机探索时从合法动作中采样。
  - 利用时选择 masked Q 最大的 `(index1, index2)`。

说明：

- `(i, j)` 和 `(j, i)` 在 swap 语义上等价。
- 第一版可保留 72 个有向合法动作，降低实现复杂度。
- 后续如需减少动作空间，可只保留上三角 36 个无向动作。

### 4.2 文本输入编码

env 给出的 `texts` 有两种格式，由 `reconstruct_text_by_line` 控制：

- `False`：长度为 9 的 piece 文本列表，顺序与当前棋盘一致。
- `True`：按当前棋盘大行和 segment 行号重组后的行文本列表。

推荐第一版做法：

```text
默认模式：
<P0> piece_text_at_position_0
<P1> piece_text_at_position_1
...
<P8> piece_text_at_position_8

按行重组模式：
<R0-L0> row0_col0_line0 row0_col1_line0 row0_col2_line0
<R0-L1> row0_col0_line1 row0_col1_line1 row0_col2_line1
...
```

agent collate 阶段将 `texts` 列表拼成一个序列，再交给 `BasicTransformerTextEncoder`。

原因：

- 当前 `FenModel` 是全局图像 + 全局文本融合结构。
- 默认模式能保留 piece 级当前位置提示。
- 按行重组模式能显式暴露同行 piece 的文本连续性。
- 实现简单，适合先打通 Q-learning 流程。

后续增强方案：

- 对每个 piece 单独 text encode，得到 `[batch, 9, dim]`。
- 加入 position embedding。
- 与 image patch features 做 cross-attention。
- 输出 pair-wise action Q。

### 4.3 图像输入编码

env 输出当前拼接图像。

第一版建议使用轻量 CNN：

```python
class SmallCNNImageEncoder(nn.Module):
    ...
```

输出单个全局 image feature。

后续增强方案：

- 使用 ResNet / ViT。
- 输入 9 个 piece tensor 而不是拼接图。
- 对每个 piece 单独 encode，再构造 pair-wise Q。

### 4.4 Replay Buffer

需要实现经验回放：

```python
Transition = namedtuple(
    "Transition",
    ["obs", "action", "reward", "next_obs", "done"]
)
```

ReplayBuffer 接口：

```python
class ReplayBuffer:
    def push(...): ...
    def sample(batch_size: int): ...
    def __len__(self): ...
```

注意：

- 图像建议存 CPU tensor 或压缩后的 numpy，避免显存占用过大。
- 文本可以存原始 string，在 batch collate 时 tokenize。

### 4.5 Q-learning 目标

采用 DQN 形式：

```python
q_sa = q_model(obs).gather(action)
target = reward + gamma * max_a target_q_model(next_obs, a) * (1 - done)
loss = smooth_l1_loss(q_sa, target)
```

动作索引转换：

```python
flat_action = index1 * 9 + index2
```

计算 `max_a` 时必须应用 legal action mask：

```python
next_q[~next_legal_action_mask] = -inf
```

目标网络：

- `policy_net`：训练网络。
- `target_net`：周期性同步参数。

推荐同步方式：

```python
if global_step % target_update_interval == 0:
    target_net.load_state_dict(policy_net.state_dict())
```

## 5. 训练代码设计

建议文件：

```text
Solver/train_rl.py
```

### 5.1 命令行参数

建议参数：

```text
--dataset-dir
--epochs
--episodes-per-epoch
--max-steps
--batch-size
--buffer-size
--gamma
--lr
--epsilon-start
--epsilon-end
--epsilon-decay-steps
--pairwise-weight
--category-weight
--done-reward
--reconstruct-text-by-line
--target-update-interval
--save-dir
--device
```

### 5.2 训练流程

伪代码：

```python
env = TextJigsawEnv(...)
agent = DQNJigsawAgent(...)

for epoch in range(epochs):
    for episode in range(episodes_per_epoch):
        obs = env.reset()

        for t in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            agent.optimize()

            obs = next_obs
            if done or truncated:
                break

    evaluate_and_save_checkpoint()
```

### 5.3 日志指标

每个 epoch 输出：

- 平均 episode reward
- 平均 step 数
- solved rate
- 平均 pairwise reward
- 平均 category reward
- epsilon
- loss 均值

checkpoint 内容：

```python
{
    "policy_net": policy_net.state_dict(),
    "target_net": target_net.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
    "config": vars(args),
}
```

## 6. 文件结构方案

目标结构：

```text
Solver/
  RL_REQUIREMENTS.md
  train_rl.py
  env/
    __init__.py
    jigsaw_env.py
  agent/
    __init__.py
    q_learning_agent.py
  model_code/
    fen_model.py
```

## 7. 实现顺序

推荐按以下顺序实现：

1. `Solver/env/jigsaw_env.py`
   - JSON 遍历
   - 样本读取
   - 随机 shuffle
   - 图片拼接
   - 文本输出模式切换
   - reward 计算
   - `reset()` / `step()`

2. `Solver/agent/q_learning_agent.py`
   - tokenizer / text collate
   - image transform
   - Q 网络 wrapper
   - action mask
   - replay buffer
   - DQN optimize

3. `Solver/train_rl.py`
   - CLI 参数
   - env / agent 初始化
   - 训练循环
   - 日志
   - checkpoint 保存

4. 最小验证
   - reset 后 observation 字段完整。
   - step 后 order 交换正确。
   - 正确 order 的 reward 达到最大且 done=True。
   - Q 输出 shape 为 `[batch, 9, 9]`。
   - diagonal mask 生效。

## 8. 关键边界条件

必须处理：

- 数据集目录下没有 `.json` 文件。
- JSON 中 `grid_size != 3`。
- JSON 中 pieces 数量不是 9。
- piece 图片缺失。
- piece 图片尺寸不一致。
- JSON piece 缺少 `segments` 字段。
- 同一棋盘大行内三个 piece 的 segment 数不一致。
- action 越界。
- `index1 == index2`。
- episode 初始 shuffle 恰好为正确顺序。

建议处理策略：

- 初始化或 reset 阶段对样本做校验。
- 图片尺寸不一致时 resize 到首张 piece 的尺寸。
- 缺少 `segments` 时 fallback 到 `text`。
- 按行重组时，对缺失的 line_id 跳过该 piece 的文本，不补伪文本。
- 非法 action 直接 raise `ValueError`，训练侧通过 mask 避免产生非法 action。
- 如果初始 shuffle 已经 solved，则重新 shuffle，最多重试若干次。

## 9. 设计取舍

第一版采用全局拼接图像 + 全局拼接文本 + 81 维 Q 输出。

优点：

- 与当前 `FenModel` 兼容。
- 实现成本低。
- 可以快速验证 reward、env 和训练流程。

限制：

- 模型对单个 piece 与 pair action 的结构归纳较弱。
- swap 动作空间有对称冗余。
- 文本 piece 的局部关系没有显式建模。

后续优化方向：

- 使用 9 个 piece 级 image embedding 和 9 个 piece 级 text embedding。
- 加入位置 embedding。
- 直接构造 `[9, 9]` pair-wise Q head。
- 只输出上三角 36 个合法 swap 动作。
- reward 改为 delta reward：使用交换前后 reward 差值，减少状态绝对分数带来的学习偏置。
