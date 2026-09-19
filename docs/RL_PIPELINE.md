# Text Jigsaw RL Pipeline

本文档整理当前拼图强化学习环境和 Solver 相关代码的实现方案，覆盖数据生成、环境读取、状态/动作/奖励定义、DQN agent、模型结构、训练流程和后续可扩展点。

## 1. 总体目标

当前系统把 OCR 报纸页面构造成 3x3 文本-图像拼图任务。每个 episode 中，中心 piece 固定在位置 4，环境只随机打乱其余 8 个 piece，agent 通过 swap action 交换两个非中心位置，目标是在有限步数内恢复原始拼图顺序。

系统使用多模态观测：

- 图像：当前拼图顺序拼接后的整张 puzzle image。
- 文本：当前每个 piece 的 OCR 文本，或按 OCR 字符框重建后的行文本。
- 位置信息：当前 `order`，即每个 board position 上的真实 piece id。
- action mask：合法 swap action 的 `[9, 9]` bool mask，中心行/列和对角线均为非法。

训练侧目前实现为 DQN，并支持三类 solver：纯视觉、纯文字、多模态。三者共享固定中心块环境和 DQN 训练外壳，只替换策略网络输入模态。

## 2. 代码结构

核心文件：

- `Data/newspaper-navigator/create_jigsaw_ocr_dataset.py`
  - 从 Newspaper Navigator OCR 数据生成拼图数据集。
  - 输出 `manifest.json`、每页 puzzle 的 `label.json` 和 `pieces/*.jpg`。

- `Solver/env/jigsaw_env.py`
  - `TextJigsawEnv` 强化学习环境。
  - 读取生成的数据集，执行 reset/step/render，计算 reward。

- `Solver/agent/q_learning_agent.py`
  - `DQNJigsawAgent`、replay buffer、visual/text/multimodal policy、byte-level tokenizer。
  - 负责 epsilon-greedy 选动作、batch collate、DQN 优化。

- `Solver/model_code/fen_model.py`
  - FEN 系列视觉特征提取器，包括 `fen_model`、`dualstem_fen_model`、`central_fen_model`、`attention_fen_model`。
  - 当前 DQN 使用 FEN 输出的特征张量作为策略网络输入。

- `Solver/train_rl.py`
  - 训练入口。
  - 构建环境和 agent，执行 epoch/episode 循环，保存 checkpoint。

## 3. 数据生成 Pipeline

入口脚本：

```bash
python Data/newspaper-navigator/create_jigsaw_ocr_dataset.py ^
  --dataset-dir Data/newspaper-navigator/ocr_dataset ^
  --output-dir Data/newspaper-navigator/output ^
  --rows 3 ^
  --cols 3 ^
  --piece-resolution 384 ^
  --max-images 1000 ^
  --seed 0
```

### 3.1 输入

脚本依赖 `create_ocr_dataset.create_image_xml_pairs(dataset_dir)` 获取图像和 ALTO XML 的配对：

- page image：报纸页面图像。
- ALTO XML：OCR word box 信息。

### 3.2 生成逻辑

当前生成逻辑是两层裁剪：

1. 对原始页面按 `rows x cols` 切成网格 cell。
2. 在每个 cell 内随机浮动选取一个固定分辨率正方形区域作为最终 piece。
3. 确认最终正方形 `bbox` 后，再从 XML 解析出的字符框中过滤属于该 piece 的字符。
4. 保存最终正方形图像到 `pieces/rXXX_cXXX.jpg`。
5. 写入每个 piece 的文字、字符框和坐标信息。

关键参数：

- `PIECE_RESOLUTION = 384`：脚本顶部默认 piece 边长，单位是原图像素。
- `--piece-resolution`：覆盖默认 piece 边长。
- `--limit`：最多扫描多少个输入 image/XML pair。
- `--max-images`：最多成功生成多少张 puzzle page，跳过的页面不计入。
- `--min-char-overlap`：字符框至少有多少比例落入 piece 才归属该 piece。
- `--seed`：控制正方形浮动 crop 的随机偏移。

如果任意 grid cell 无法容纳目标 `piece_resolution`，该 page 会被跳过，脚本不会中断。跳过信息写入 manifest 的 `skipped_pages`。

### 3.3 输出格式

默认输出结构：

```text
output/
  manifest.json
  puzzles/
    <page_id>/
      label.json
      pieces/
        r000_c000.jpg
        r000_c001.jpg
        ...
```

`manifest.json` 记录全局信息：

- `meta.rows / meta.cols`
- `meta.piece_resolution`
- `meta.input_limit`
- `meta.max_images`
- `meta.page_count`
- `meta.skipped_page_count`
- `puzzles[]`
- `skipped_pages[]`

每个 `label.json` 中：

- `meta`
  - 数据集路径、输出路径、网格大小、piece 数量、piece 分辨率。
- `page`
  - 原图路径、XML 路径、原图宽高、字符数。
- `pieces[]`
  - `piece_id`：例如 `r000_c000`。
  - `piece_path`：最终 piece 图片路径。
  - `row / col`：原始正确网格位置。
  - `grid_bbox`：原始网格 cell 在页面坐标系中的范围。
  - `bbox`：最终正方形 piece 在页面坐标系中的范围。
  - `width / height`：最终 piece 宽高，应等于 `piece_resolution`。
  - `text`：归属该 piece 的 OCR 文本。
  - `chars[]`：字符级标注，`bbox` 是相对最终 piece 的坐标，`page_bbox` 是页面坐标。

## 4. 环境读取与任务定义

环境类：`Solver/env/jigsaw_env.py::TextJigsawEnv`

### 4.1 数据集发现

`TextJigsawEnv(dataset_dir=...)` 支持几种输入：

- 直接传 `manifest.json`
- 传包含 `manifest.json` 的输出目录
- 传 `puzzles/*/label.json` 所在目录
- 传单个 `label.json`
- 保留旧版 `Data/PuzzleData` JSON 格式兼容

环境会跳过 `manifest.json` 本身，最终得到可采样的 `label.json` 列表。

### 4.2 格式读取

读取 `label.json` 时：

- 从 `meta.rows / meta.cols` 或旧版 `grid_size` 获取网格大小。
- 当前 agent 固定支持 3x3，因此环境要求 `rows == cols == 3` 且 piece 数量为 9。
- 将 `r000_c000` 形式的 `piece_id` 映射为整数 id：`row * cols + col`。
- 解析 `piece_path` 或旧版 `image` 字段，加载 PIL RGB 图片。
- 所有 piece 会按 id `0..8` 组织为 solved order。
- 中心位置固定为 `CENTER_INDEX = 4`，该位置始终放置 piece 4。

### 4.3 Observation

`reset()` 和 `step()` 返回的 observation：

```python
{
    "image": PIL.Image.Image | transformed_image,
    "texts": list[str],
    "order": list[int],
    "legal_action_mask": np.ndarray,  # shape [9, 9], bool
}
```

`image` 是按照当前 `order` 拼接出的整张 3x3 puzzle 图像。若传入 `image_transform`，环境会对图像做外部转换。

`texts` 有两种模式：

- 默认模式：每个 board position 输出一段 piece 文本，格式为 `<P{position}> ...`。
- `reconstruct_text_by_line=True`：按 OCR 字符框重建行文本，再按当前 board row 合并，格式为 `<R{row}-L{line}> ...`。

默认模式更快，适合训练；按行重建模式可以给文本模型更接近页面阅读顺序的信息。

### 4.4 State 和 Order

环境内部维护：

- `pieces`：当前样本的 9 个 `Piece`，按 solved id 排列。
- `current_order`：长度为 9 的列表，表示每个 board position 上放的是哪个 solved piece id。
- `SOLVED_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8]`
- `CENTER_INDEX = 4`
- `MOVABLE_POSITIONS = (0, 1, 2, 3, 5, 6, 7, 8)`

`reset()` 只会打乱 `MOVABLE_POSITIONS` 上的 8 个 piece，中心位置永远保持 `current_order[4] == 4`，并确保初始状态不是 solved state。

### 4.5 Action

动作是一个二元组：

```python
action = (index1, index2)
```

含义：交换 board position `index1` 和 `index2` 上的 piece。`index1` 和 `index2` 都不能是中心位置 4。

动作空间大小：

- 模型仍输出按 `[9, 9]` 展开的 81 个 Q 值。
- 对角线 action 即 `(i, i)` 非法。
- 涉及中心位置 4 的 action 非法。
- `legal_action_mask` 对角线和中心行/列均为 `False`，其它非中心 swap 为 `True`。

### 4.6 Reward

当前 reward 由三部分组成：

```python
reward = pairwise_reward * pairwise_weight
       + category_reward * category_weight
       + done_reward
```

`pairwise_reward`：

- 统计当前棋盘上正确相邻 pair 的数量。
- 横向 pair 共 6 个，纵向 pair 共 6 个，总计 12 个。
- 分数为 `correct_pairs / 12.0`。

`category_reward`：

- 统计在正确绝对位置上的 piece 数量。
- 只统计 8 个可动位置，不给固定中心块白送分数。
- 分数为 `correct_positions / 8.0`。

`done_reward`：

- 若 `current_order == SOLVED_ORDER`，额外奖励 `done_reward_value`，默认 10。

episode 结束条件：

- `done=True`：拼图完全复原。
- `truncated=True`：达到 `max_steps` 且未完成。

## 5. DQN Agent 实现

Agent 类：`Solver/agent/q_learning_agent.py::DQNJigsawAgent`

### 5.1 动作值输出

当前固定：

```python
PIECE_COUNT = 9
ACTION_COUNT = 81
```

模型输出 shape 为 `[batch, 81]`，在选动作和 target 计算时 reshape 为 `[batch, 9, 9]`，并用 `legal_action_mask` 将非法 action 置为 `-inf`。

### 5.2 Epsilon-greedy 策略

训练时：

- 以 epsilon 概率从 legal actions 中随机选。
- 否则用 policy network 选 Q 值最大的合法 action。

epsilon 线性衰减：

```python
epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)
progress = min(1.0, global_step / epsilon_decay_steps)
```

### 5.3 Replay Buffer

`ReplayBuffer` 保存 transition：

```python
Transition(obs, action, reward, next_obs, done)
```

每次优化从 buffer 随机采样 batch。

### 5.4 Observation Collate

`_collate_observations()` 按 `solver_type` 准备 batch：

1. 对 `visual` / `multimodal` 做图像转 tensor：
   - PIL/np.ndarray 转 RGB。
   - resize 到 `image_size x image_size`，默认 288。
   - 转成 float tensor，保持 0..255 像素值，交给 FEN backbone 内部处理归一化。

2. 对 `text` / `multimodal` 做文本编码：
   - 将 observation 中的 `texts` 拼接为单条 OCR 序列。
   - 使用 byte-level tokenizer 得到 `input_ids` 和 `attention_mask`。

3. 对所有 solver 准备 action mask：
   - stack 为 bool tensor `[batch, 9, 9]`。

### 5.5 DQN 优化

优化逻辑：

```python
q_sa = policy_net(obs).gather(action)
target = reward + gamma * max_a target_net(next_obs, a) * (1 - done)
loss = smooth_l1_loss(q_sa, target)
```

关键点：

- target network 用于 bootstrapping。
- next Q 计算时应用 legal action mask。
- 每 `target_update_interval` 次优化同步一次 target network。
- 使用 AdamW 优化器和 gradient clipping。

## 6. 模型实现方案

模型主要由可选输入模态的 policy 组成，详见 `Solver/AGENT_DESIGN.md`。当前支持三条路线：

- `visual`：FEN 视觉特征提取器 + Q head。
- `text`：byte-level OCR 文本 Transformer + Q head。
- `multimodal`：FEN 视觉特征 + OCR 文本特征 concat 融合 + Q head。

### 6.1 FEN 特征提取器

视觉和多模态 solver 通过 `--fen-model-name` 选择 FEN backbone：

- `ef`：使用 `fen_model`，横向/纵向各一套 EfficientNet-B3 patch encoder。
- `modulator`：使用 `fen_model(model_name="modulator")`。
- `central`：使用 `central_fen_model`，以中心块和其他块的关系构建特征。
- `attention`：使用 `attention_fen_model(single_output=True)`，用 patch Transformer 输出 board summary。
- `dualstem_ef` / `dualstem_modulator`：使用 `dualstem_fen_model`，融合局部 FEN 和全局 ViT 特征。

FEN 输入是当前拼接后的棋盘图像，默认 resize 到 `288x288`。FEN 内部按 `96x96` patch 切为 3x3，因此当前训练仍要求 3x3 puzzle。

### 6.2 策略 Q Head

`FenFeaturePolicy` 封装纯视觉 solver：

```python
features = feature_extractor(images)      # [B, image_feature_dim]
q_values = q_head(features)               # [B, 81]
```

Q head 是一个 MLP：

1. `LayerNorm(feature_dim)`
2. `Linear(feature_dim, hidden_dim)`
3. `GELU + Dropout`
4. `Linear(hidden_dim, hidden_dim)`
5. `GELU + Dropout`
6. `Linear(hidden_dim, ACTION_COUNT)`

`image_feature_dim` 现在表示 FEN 输出特征维度，默认 256；`hidden_dim` 表示 Q head 隐层维度。

### 6.3 文本与多模态

纯文字 solver 使用 `TextFeaturePolicy`：

```text
texts -> CharTokenizer -> ByteTransformerTextEncoder -> QHead -> 81 Q values
```

多模态 solver 使用 `MultimodalFeaturePolicy`：

```text
image -> FEN -> image projection
texts -> text encoder -> text projection
concat -> QHead -> 81 Q values
```

训练时用 `--solver-type visual|text|multimodal` 选择。

## 7. 训练 Pipeline

入口：

```bash
python Solver/train_rl.py ^
  --dataset-dir Data/newspaper-navigator/output ^
  --solver-type visual ^
  --epochs 10 ^
  --episodes-per-epoch 100 ^
  --max-steps 50 ^
  --batch-size 32 ^
  --save-dir Solver/checkpoints
```

若希望文本按 OCR 行重建：

```bash
python Solver/train_rl.py ^
  --dataset-dir Data/newspaper-navigator/output ^
  --reconstruct-text-by-line
```

### 7.1 初始化

`train_rl.py`：

1. 解析训练超参。
2. `torch.manual_seed(args.seed)`。
3. 创建 `TextJigsawEnv`。
4. 创建 `DQNJigsawAgent`。
5. 保存 config 到 `save_dir/config.json`。

### 7.2 Episode 循环

每个 episode：

1. `obs = env.reset()`
2. agent 根据 obs 选择 swap action。
3. `next_obs, reward, done, truncated, info = env.step(action)`
4. transition 写入 replay buffer。
5. agent 尝试 optimize。
6. done 或 truncated 时结束。

### 7.3 Epoch 指标

每个 epoch 聚合输出：

- `avg_reward`
- `avg_steps`
- `solved_rate`
- `avg_pairwise_reward`
- `avg_category_reward`
- `epsilon`
- `avg_loss`

### 7.4 Checkpoint

按 `--save-every` 保存：

- `epoch_XXXX.pt`
- 训练结束保存 `last.pt`

checkpoint 内容：

- policy network state
- target network state
- optimizer state
- global step
- epoch
- config

## 8. 当前约束与实现取舍

### 8.1 固定 3x3

环境和 agent 目前固定 3x3：

- `GRID_SIZE = 3`
- `PIECE_COUNT = 9`
- `ACTION_COUNT = 81`

虽然数据生成脚本支持 `--rows/--cols`，但当前 RL 训练只支持固定中心块的 3x3。若要支持 NxM 或不同固定锚点，需要同步修改：

- 环境中的 grid 常量、pair 统计、mask 构造。
- 固定位置集合和可动位置集合。
- agent 的 `PIECE_COUNT/ACTION_COUNT`。
- 模型输出维度。
- 训练和 checkpoint config。

### 8.2 Reward 是监督式 shaped reward

环境知道 solved order，因此 reward 直接使用正确相邻关系和正确绝对位置。这适合训练验证，但不是无监督拼图求解。

### 8.3 观测没有显式 piece-level embedding

当前图像输入是整张拼接后的 puzzle，文本输入是拼接后的文本序列。模型没有显式对每个 piece 建模，因此 pair-wise action 的结构归纳较弱。后续可考虑：

- 对 9 个 piece 单独编码。
- 用 Transformer/GNN 建模 piece 之间关系。
- 为 action `(i, j)` 构造 pair-specific Q head。

### 8.4 OCR 文本质量不稳定

Newspaper OCR 有大量噪声。默认 piece text 是基于字符框排序得到的文本，行重建模式会更接近版面，但也更慢。训练时可先用默认模式跑通，再尝试 `--reconstruct-text-by-line`。

## 9. 建议执行顺序

1. 生成小规模数据集做 smoke test：

```bash
python Data/newspaper-navigator/create_jigsaw_ocr_dataset.py ^
  --dataset-dir Data/newspaper-navigator/ocr_dataset ^
  --output-dir Data/newspaper-navigator/output_smoke ^
  --rows 3 ^
  --cols 3 ^
  --piece-resolution 384 ^
  --max-images 10 ^
  --seed 0
```

2. 验证环境 reset/step：

```python
from Solver.env.jigsaw_env import TextJigsawEnv

env = TextJigsawEnv("Data/newspaper-navigator/output_smoke", seed=0)
obs = env.reset()
next_obs, reward, done, truncated, info = env.step((0, 1))
image = env.render()
```

3. 启动短训练：

```bash
python Solver/train_rl.py ^
  --dataset-dir Data/newspaper-navigator/output_smoke ^
  --epochs 1 ^
  --episodes-per-epoch 5 ^
  --max-steps 20 ^
  --batch-size 4 ^
  --save-dir Solver/checkpoints_smoke
```

4. 扩大数据规模和训练步数。

## 10. 后续优化方向

- 支持可变 grid size，使数据生成和 RL 训练完全一致。
- 增加环境单元测试，覆盖 manifest 读取、label 读取、路径解析、reward、action mask。
- 增加 evaluation 脚本，固定 epsilon=0 评估 solved rate。
- 增加可视化脚本，保存每步拼图图像和 action 序列。
- 改造模型为 piece-level encoder，提高 swap action 的结构归纳。
- 将 reward 改为 delta reward，奖励每一步相对上一步的 pair/position 改善，减少 reward 绝对值偏置。
