# MangaZero Panel Ordering Pipeline

## 使用说明

### 1. 生成数据集

先下载原始 MangaZero 数据：

```bash
python Data/Mangazero/download_raw.py --raw-dir Data/Mangazero/raw --split train
```

再从本地 raw 数据构建 puzzle：

```bash
python Data/Mangazero/build_dataset.py --raw-dir Data/Mangazero/raw --output-dir Data/Mangazero/ordering_dataset --panel-count 6 --stride 3 --shuffle-per-window 1 --target-width 224 --target-height 224
```

快速无 OCR 检查：

```bash
python Data/Mangazero/build_dataset.py --raw-dir Data/Mangazero/raw --output-dir Data/Mangazero/ordering_smoke --max-samples 2 --disable-ocr
```

构建阶段默认使用 GPU OCR，即 `--ocr-device gpu:0`。如果需要强制使用 CPU，可显式传入 `--ocr-device cpu`：

```bat
set FLAGS_enable_pir_api=0
set FLAGS_enable_pir_in_executor=0
set FLAGS_use_mkldnn=0
set FLAGS_use_onednn=0
python Data/Mangazero/build_dataset.py --raw-dir Data/Mangazero/raw --output-dir Data/Mangazero/ordering_dataset --panel-count 6 --stride 3 --shuffle-per-window 1 --target-width 224 --target-height 224 --ocr-device cpu
```

说明：

- 下载阶段需要 `datasets`、`requests`、`Pillow`。
- 构建阶段需要 `Pillow`，可选 `paddleocr`、`paddlepaddle`。
- `--disable-ocr` 只保留 dialog bbox，`dialog_text` 为空，适合先检查切分和样本格式。
- 下载阶段默认读取 Hugging Face 数据集 `jianzongwu/MangaZero` 的 `train` split。

### 2.1 Debug 模式

先单独检查 OCR 是否能初始化并完成一次最小推理：

```bash
python Data/Mangazero/build_dataset.py --ocr-debug-probe --debug-ocr --ocr-device gpu:0 --ocr-version PP-OCRv6 --ocr-timeout 120
```

如果要看完整 puzzle 构建时的 OCR / 下载 / 切分步骤日志，在正常构建命令后加上 `--debug-ocr`：

```bash
python Data/Mangazero/build_dataset.py --raw-dir Data/Mangazero/raw --output-dir Data/Mangazero/ordering_dataset --panel-count 6 --stride 3 --shuffle-per-window 1 --target-width 224 --target-height 224 --debug-ocr
```

两个参数的区别：

- `--debug-ocr`：打印 OCR worker、Paddle 初始化、请求发送、响应返回等细粒度日志。
- `--ocr-debug-probe`：只初始化 OCR 并做一次最小图片探测，不进入完整数据集构建流程。

下载 raw 数据参数：

- `--dataset-name`：Hugging Face 数据集名，默认 `jianzongwu/MangaZero`。
- `--split`：读取的数据 split，默认 `train`。
- `--raw-dir`：原始 MangaZero 图片和标注输出目录。
- `--max-pages`：最多下载多少条 MangaZero page sample；不设置则处理全部。
- `--streaming`：使用 streaming 方式读取 Hugging Face 数据集；当前默认开启。
- `--no-streaming`：关闭 streaming，改为普通加载。
- `--trust-remote-code`：允许 Hugging Face 数据集加载远程代码。
- `--download-timeout`：单张页面图片下载超时时间，单位秒。
- `--download-retries`：单张页面图片下载失败后的重试次数。
- `--retry-sleep`：下载重试之间的等待时间，单位秒。
- `--skip-network-errors`：页面图片下载失败时输出 log 并跳过当前 page sample；当前默认开启。
- `--no-skip-network-errors`：网络下载失败时直接中断，便于调试 URL 或网络环境。
- `--overwrite`：覆盖已经下载过的 raw sample。

构建 puzzle 参数：

- `--raw-dir`：读取第一阶段下载得到的 raw 数据目录。
- `--output-dir`：生成后的数据集输出目录。
- `--panel-count`：每个 puzzle 包含的连续 panel 数量。
- `--stride`：连续 panel 窗口的滑动步长。
- `--shuffle-per-window`：每个连续窗口生成多少个随机乱序样本。
- `--target-width`：padding 后 panel 图片的目标宽度。
- `--target-height`：padding 后 panel 图片的目标高度。
- `--max-samples`：最多处理多少条本地 raw page sample；不设置则处理全部。
- `--puzzle-num`：最多生成多少组 puzzle，默认 `500`。
- `--seed`：随机种子，影响 puzzle 输入乱序。
- `--disable-ocr`：关闭 PaddleOCR，dialog 文本写为空。
- `--skip-network-errors`：本地 raw 页面图片缺失时输出 log 并跳过当前 page sample；当前默认开启。
- `--no-skip-network-errors`：本地 raw 页面图片缺失时直接中断。
- `--skip-ocr-errors`：OCR 单个 dialog 失败时写空字符串并继续生成；当前默认开启。
- `--no-skip-ocr-errors`：OCR 报错时直接中断，便于调试 PaddleOCR 环境。
- `--ocr-lang`：PaddleOCR 语言，默认 `ch`。
- `--ocr-version`：PaddleOCR 模型版本，可选 `PP-OCRv3`、`PP-OCRv4`、`PP-OCRv5`、`PP-OCRv6`；默认 `PP-OCRv6`。
- `--ocr-device`：PaddleOCR 推理设备，默认 `gpu:0`；如需 CPU 可手动设为 `cpu`。
- `--ocr-mode`：OCR 运行模式，`subprocess` 或 `inline`；默认 `subprocess`，用于隔离 PaddleOCR 原生崩溃。
- `--ocr-timeout`：子进程 OCR 单个 dialog 的超时时间，单位秒。
- `--ocr-max-restarts`：OCR 子进程崩溃或超时后的最大重启次数。
- `--ocr-use-angle-cls`：启用 PaddleOCR 角度分类。
- `--ocr-enable-pir`：允许 PaddleOCR 使用 Paddle PIR 执行路径；默认关闭以规避部分 Paddle/PaddleX 版本兼容问题。
- `--ocr-enable-mkldnn`：允许 PaddleOCR 使用 MKL-DNN/oneDNN；默认关闭以规避部分 CPU 推理兼容问题。
- `--image-format`：保存 panel 图片的格式，可选 `jpg` 或 `png`。
- `--overwrite`：保留的构建参数；当前同名目录会按 puzzle index 写出。

### 2. 训练 solver

纯视觉 solver：

```bash
python Solver/train_mangazero_panel_ordering.py --dataset-dir Data/Mangazero/ordering_dataset --solver-type visual --panel-count 6
```

纯文本 solver：

```bash
python Solver/train_mangazero_panel_ordering.py --dataset-dir Data/Mangazero/ordering_dataset --solver-type text --panel-count 6
```

多模态 solver：

```bash
python Solver/train_mangazero_panel_ordering.py --dataset-dir Data/Mangazero/ordering_dataset --solver-type multimodal --panel-count 6
```

多模态 + layout：

```bash
python Solver/train_mangazero_panel_ordering.py --dataset-dir Data/Mangazero/ordering_dataset --solver-type multimodal --panel-count 6 --use-layout
```

从 checkpoint 继续训练：

```bash
python Solver/train_mangazero_panel_ordering.py --dataset-dir Data/Mangazero/ordering_dataset --solver-type visual --panel-count 6 --epoch 20 --load Solver/checkpoints_mangazero/last.pt
```

checkpoint 默认保存到：

```text
Solver/checkpoints_mangazero/
```

训练参数：

- `--dataset-dir`：读取的数据集目录，需包含 `manifest.jsonl` 或数字编号 puzzle 目录。
- `--solver-type`：solver 类型，可选 `visual`、`text`、`multimodal`。
- `--epoch`：总训练 epoch 数。
- `--epochs`：`--epoch` 的兼容别名；如果同时设置，以 `--epochs` 为准。
- `--split-ratio`：训练时将完整 puzzle 数据随机划分为 train/val/test 的比例，格式如 `0.8,0.1,0.1`。
- `--test-per-epoch`：每训练多少个 epoch 后跑一次 test，默认 `5`。
- `--batch-size`：训练 batch size。
- `--lr`：AdamW 学习率。
- `--weight-decay`：AdamW 权重衰减。
- `--num-workers`：DataLoader worker 数。
- `--panel-count`：每个样本的 panel 数，必须和生成数据集时一致。
- `--image-width`：训练时读取 panel 后的图像宽度。
- `--image-height`：训练时读取 panel 后的图像高度。
- `--image-feature-dim`：视觉 ViT encoder 输出维度。
- `--vit-backbone`：视觉骨干，默认 `pretrained`，可切 `lightweight`。
- `--vit-pretrained` / `--no-vit-pretrained`：是否加载 torchvision ViT 预训练权重。
- `--vit-freeze`：是否冻结 ViT backbone 参数。
- `--vit-patch-size`：ViT patch 大小。
- `--vit-layers`：ViT encoder 层数。
- `--vit-num-heads`：ViT encoder attention heads 数。
- `--text-feature-dim`：dialog 文本 encoder 输出维度。
- `--text-vocab-size`：hash 文本 encoder 的词表大小。
- `--d-model`：Transformer token 维度。
- `--encoder-layers`：Transformer set encoder 层数。
- `--decoder-layers`：Transformer pointer decoder 层数。
- `--num-heads`：Transformer multi-head attention 的 head 数。
- `--dropout`：Transformer dropout。
- `--use-layout`：额外加入 panel bbox/page 位置特征。
- `--grad-clip-norm`：梯度裁剪阈值。
- `--save-dir`：checkpoint 保存目录。
- `--load`：checkpoint 路径；不设置则从随机初始化开始，设置后读取模型参数并继续训练。
- `--save-every`：每隔多少个 epoch 保存一次 `epoch_XXXX.pt`；设为 `0` 可关闭中间保存。
- `--device`：训练设备，默认自动选择 `cuda` 或 `cpu`。
- `--seed`：训练随机种子。

## 当前实现文件

```text
Data/Mangazero/download_raw.py
Data/Mangazero/build_dataset.py
Solver/env/mangazero_panel_env.py
Solver/agent/mangazero_set2seq_solver.py
Solver/train_mangazero_panel_ordering.py
```

## Pipeline

```text
Hugging Face MangaZero
-> download_raw.py 下载 meta.url1 / meta.url2 和原始标注
-> 保存 raw/pages/* 与 raw/manifest.jsonl
-> build_dataset.py 读取本地 raw 数据
-> 横向拼接为完整页面图
-> 根据 frames[*].bbox 裁剪 panel
-> 根据 dialogs[*].bbox 裁剪 dialog 区域并 PaddleOCR
-> 保存 padded panel
-> 按 page 维护连续 panel 窗口
-> 每凑够一个固定数量同页连续 panel 窗口就打乱输入顺序
-> 写出 output_dir/{index}/sample.json 和对应图片
-> Dataset/Env 读取完整 puzzle 数据
-> train 脚本按 split-ratio 划分 train/val/test
-> Set-to-sequence Transformer solver 训练、验证、测试
```

## 任务定义

任务是固定数量连续 panel 排序。

输入：

```text
同一 manga/chapter 中连续的 K 个 panel，顺序被随机打乱
```

输出：

```text
长度为 K 的 target_order，表示正确顺序应依次选择乱序输入中的哪个 panel
```

例子：

```text
原始顺序:     [p0, p1, p2, p3, p4, p5]
乱序输入:     [p3, p0, p5, p1, p4, p2]
target_order: [1, 3, 5, 0, 4, 2]
```

`target_order[t]` 表示第 `t` 个正确 panel 在乱序输入中的位置。

## MangaZero 数据格式

数据集读取方式：

```python
from datasets import load_dataset

ds = load_dataset("jianzongwu/MangaZero")
```

已适配字段：

```python
{
    "image_path": str,
    "frames": [
        {
            "bbox": [x1, y1, x2, y2],
            "caption": str,
            "characters": [
                {
                    "bbox": [x1, y1, x2, y2],
                    "id": str,
                    "type": int,
                }
            ],
            "dialogs": [
                {
                    "bbox": [x1, y1, x2, y2],
                }
            ],
        }
    ],
    "meta": {
        "url1": str,
        "url2": str,
        "width1": int,
        "width2": int,
    },
}
```

当前假设：

- 一条 sample 对应一张由 `url1` 和 `url2` 横向拼接得到的页面图。
- `frames[*].bbox` 和 `dialogs[*].bbox` 使用拼接后的页面坐标系。
- `frames` 列表顺序作为该页面内 panel 的阅读顺序。
- `image_path` 用于解析 `manga_id` 和 `page_index`。

## 数据集生成脚本

实现文件：

```text
Data/Mangazero/main.py
```

主要参数：

```text
--dataset-name
--split
--output-dir
--cache-dir
--panel-count
--stride
--shuffle-per-window
--target-width
--target-height
--max-pages
--puzzle-num
--disable-ocr
--ocr-lang
--ocr-version
--ocr-device
--ocr-mode
--ocr-timeout
--ocr-max-restarts
--ocr-use-angle-cls
--ocr-enable-pir
--ocr-enable-mkldnn
```

输出结构：

```text
Data/Mangazero/ordering_dataset/
├── 000000/
│   ├── sample.json
│   └── panels_padded/
├── 000001/
│   ├── sample.json
│   └── panels_padded/
├── manifest.jsonl
└── meta.json
```

### Panel 裁剪

脚本会下载 `meta.url1` 和 `meta.url2`，然后按以下方式拼接：

```text
page_width = meta.width1 + meta.width2
page_height = max(url1.height, url2.height)
url1 paste at x = 0
url2 paste at x = meta.width1
```

每个 panel 从拼接页面中按 `frames[*].bbox` 裁剪，只保存 padding 后的图片到 `panels_padded/`；原始裁剪图不再落盘。
生成 puzzle 时只在同一 page 内取连续窗口；若 `start + panel_count > 当前 page 的 panel 数`，则直接跳过。

### 黑边 padding

不同分辨率和长宽比的 panel 会等比例缩放到目标分辨率内部，然后用黑边补齐到统一尺寸。

输出字段：

```text
padded_size
pad = [left, top, right, bottom]
```

### Dialog OCR

MangaZero 原始标注里 dialog 只有 bbox，没有文本。脚本会对每个 `dialogs[*].bbox` 裁剪区域调用 PaddleOCR。

输出字段：

```text
dialog_bboxes
dialog_texts
dialog_text
```

重要规则：

- `dialog_texts` 与 `dialog_bboxes` 一一对应。
- `dialog_text` 是 `dialog_texts` 过滤空字符串后的拼接结果。
- OCR 失败或关闭 OCR 时，`dialog_text` 为空。
- `caption` 只作为原始 metadata 保留。
- 后续 puzzle 和 solver 只读取 `dialog_text`，不读取 `caption`。

## JSONL 样本格式

每个 `output_dir/{index}/sample.json` 是一个 fixed-count puzzle：

```json
{
  "sequence_id": "manga_page_puzzle_000000",
  "manga_id": "manga",
  "chapter_id": "chapter",
  "page_id": "page",
  "page_index": 0,
  "group_key": "manga/chapter",
  "page_key": "manga/chapter/page",
  "panel_count": 6,
  "target_panel_size": [224, 224],
  "panels": [
    {
      "panel_id": "page_panel_000",
      "source_image_path": "20th-century-boys/000.jpg",
      "page_id": "20th-century-boys_000_jpg",
      "page_index": 0,
      "panel_index_in_page": 0,
      "global_order": 0,
      "bbox": [0, 0, 100, 100],
      "page_size": [1465, 1100],
      "padded_path": "000000/panels_padded/input_00.jpg",
      "raw_size": [100, 100],
      "padded_size": [224, 224],
      "pad": [0, 0, 0, 0],
      "caption": "raw caption metadata",
      "dialog_bboxes": [[10, 10, 50, 30]],
      "dialog_texts": ["ocr text"],
      "dialog_text": "ocr text",
      "character_ids": ["1"],
      "character_bboxes": [[0, 0, 20, 40]],
      "character_types": [0]
    }
  ],
  "input_order": [3, 0, 5, 1, 4, 2],
  "target_order": [1, 3, 5, 0, 4, 2],
  "text_source": "dialog_text"
}
```

`panels` 按乱序输入排列保存，`target_order` 指向乱序输入 index。

## 环境与数据读取

实现文件：

```text
Solver/env/mangazero_panel_env.py
```

### `MangaZeroPanelOrderingDataset`

优先读取 `manifest.jsonl` 指向的 `sample.json`；如果没有 manifest，则扫描数字编号目录下的 `sample.json`；最后兼容旧版 `train.jsonl` / `val.jsonl` / `test.jsonl`。返回：

```python
{
    "sequence_id": str,
    "manga_id": str,
    "chapter_id": str,
    "panel_images": FloatTensor,
    "target_order": LongTensor,
    "dialog_texts": list[str],
    "layout_features": FloatTensor,
    "panels": list[dict],
}
```

### `MangaZeroPanelOrderingEnv`

这是一个 sequential one-shot RL 环境：

```text
reset() -> 当前 puzzle observation
step(action) -> 选择下一个 panel index
```

动作：

```text
action = 乱序输入中的 panel index
```

终止条件：

- 已选择 `K` 个 panel。
- 动作越界。
- 重复选择已选 panel。

reward：

```text
完整合法 episode: pairwise_accuracy + exact_match_bonus
非法动作: invalid_action_penalty
中间步骤: 0
```

## Solver

实现文件：

```text
Solver/agent/mangazero_set2seq_solver.py
```

模型结构：

```text
panel feature
-> Transformer set encoder
-> autoregressive Transformer pointer decoder
-> ordered panel index sequence
```

已实现组件：

- `PanelViTEncoder`：轻量 ViT 视觉编码器，`visual` 和 `multimodal` 共用同一套视觉块。
- `TorchvisionPretrainedViTEncoder`：torchvision 预训练 ViT 视觉编码器，`visual` 和 `multimodal` 默认使用。
- `LightweightPanelViTEncoder`：轻量 ViT 兜底版本。
- `HashedDialogTextEncoder`：hash token + `EmbeddingBag` 文本特征提取器。
- `MangaZeroSetToSequenceSolver`：set-to-sequence Transformer。
- `pointer_cross_entropy`：训练用 pointer cross entropy。

支持三种 solver：

```text
visual:     只使用 panel image
text:       只使用 dialog_text
multimodal: 使用 panel image + dialog_text
```

可选：

```text
--use-layout: 额外加入 layout_features
```

## 训练脚本

实现文件：

```text
Solver/train_mangazero_panel_ordering.py
```

训练方式：

- 监督学习。
- teacher forcing。
- pointer cross entropy。
- validation 使用 greedy decode。
- 训练开始时对完整 puzzle 数据按 `--split-ratio` 切分为 train/val/test。
- 每个 epoch 结束后自动跑 validation。
- 每 `--test-per-epoch` 个 epoch 自动跑一次 test，训练结束后额外保存一次最终 test。

训练输出指标：

```text
train_loss
val_loss
val_exact
val_pos
val_pairwise
test_loss
test_exact
test_pos
test_pairwise
```

保存内容：

```text
Solver/checkpoints_mangazero/
├── config.json
├── epoch_XXXX.pt
├── best.pt
├── last.pt
├── test_epoch_XXXX.json
└── test.json
```

## 当前限制

- 数据生成需要网络下载 Hugging Face 数据和页面图片。
- 完整 OCR 版本需要本地 PaddleOCR/PaddlePaddle 环境。
- 当前文本 encoder 是轻量 hash encoder，不是 BERT/CLIP 文本塔。
- 当前训练是监督 set-to-sequence，环境已提供 RL 接口，但还没有单独的 RL 训练脚本接入该 MangaZero 环境。
- 当前 panel 顺序使用 `frames` 列表顺序；如果后续发现 MangaZero 的 `frames` 不是阅读顺序，需要在数据生成脚本中加入同页阅读顺序重排。
