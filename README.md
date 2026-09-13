# Text-jigsaw

面向文本/版面拼图（jigsaw）研究的数据集与求解器项目。项目把文本切割成拼图碎片，构造需要结合图像、OCR 文本和版面信息才能恢复正确顺序的任务，用于探索多模态模型、强化学习（DQN）和排序模型的求解能力。

当前状态：实验性项目，尚未发布正式数据集（Not ready yet）。

## 项目结构

```text
Text-jigsaw/
├── Data/                          # 合成文本拼图数据生成
│   └── DataGenerator.py           # 由 txt + 字体生成 3x3 文本拼图（图片 + JSON 标签）
├── Mangazero/                     # 漫画分镜（panel）排序数据集
│   ├── download_raw.py            # 下载 MangaZero 原始页图与标注
│   ├── build_dataset.py           # 从本地原始数据构建 panel 排序 puzzle
│   ├── main.py                    # 一步式下载 + 构建 pipeline（含 OCR）
│   └── format_report.json         # 数据构建统计/格式报告
├── comics/                        # COMICS 数据集（上游仓库 + 数据创建脚本）
│   ├── create_panel_ordering_dataset.py # 从 raw panels + OCR CSV 构建 panel 排序 puzzle
│   ├── stream_comix.py            # 从 Hugging Face 流式下载 comix 并构建 ordering 数据集
│   └── ...                        # 上游 COMICS 代码与 folds
├── newspaper_navigator/           # 历史报纸版面拼图数据集
│   ├── create_ocr_dataset.py      # 获取报纸页面图像并配对 ALTO OCR XML
│   ├── create_jigsaw_ocr_dataset.py # 按网格切分页面为拼图 piece，生成字符级标签
│   └── generate_visualization.py  # 报纸页面图像 embedding 的 TSNE 可视化
├── Solver/                        # 求解器与训练代码
│   ├── env/                       # RL/数据集环境
│   │   ├── jigsaw_env.py          # 3x3 文本拼图环境（固定中心块 + swap 动作）
│   │   └── mangazero_panel_env.py # MangaZero panel 排序数据集与环境
│   ├── agent/                     # 策略与模型
│   │   ├── q_learning_agent.py    # DQN agent（visual/text/multimodal）
│   │   ├── mangazero_set2seq_solver.py       # panel 排序 set-to-sequence 模型
│   │   └── mangazero_permutation_classifier.py # panel 排序全排列分类模型
│   ├── model_code/fen_model.py    # FEN 系列视觉特征提取器
│   ├── train_rl.py                # DQN 训练入口
│   ├── train_mangazero_panel_ordering.py      # set2seq 训练入口
│   ├── train_mangazero_permutation_classifier.py # 分类器训练入口
│   ├── test_*.py                  # 对应冒烟/单元测试
│   └── visualize_dataset_sample.py # 数据集样本可视化
├── mangazero_panel_ordering_plan.md # MangaZero 排序方案
├── Solver/RL_REQUIREMENTS.md       # 3x3 RL 需求文档
├── Solver/AGENT_DESIGN.md          # Solver agent 设计
└── Solver/RL_PIPELINE.md           # RL 完整 pipeline 文档
```

## 数据集

项目包含三条数据构建路线：

1. `Data/DataGenerator.py`：从文本文件和字体生成合成 3x3 拼图。每个样本输出一张页面图、9 张 piece 图和一个 JSON 标签（包含 piece 文本、segments、行列位置等），并支持侵蚀/泛黄等版面扰动。
2. `Mangazero/`：从 Hugging Face `jianzongwu/MangaZero` 抓取漫画页，裁剪 panel、对对话框文本做 OCR，生成固定 panel 数量（默认 6）的乱序排序样本，输出 `train/val/test.jsonl` 或 manifest 格式。
3. `comics/`：使用 COMICS 官方 raw panel 图片和 `COMICS_ocr_file.csv`（含 textbox 坐标与 OCR 文本），按漫画书/页/panel 顺序生成固定 panel 数量的排序 puzzle，输出与 Mangazero 相同格式。原始数据不在仓库内，需按 `comics/setup.sh` 或官方链接下载。
4. `newspaper_navigator/`：从 Chronicling America / Newspaper Navigator 获取历史报纸页面，结合 ALTO OCR 的字符框，把页面切成网格 piece，产出带字符级标签的拼图数据，用于 3x3 RL 训练。

## Solver

- `TextJigsawEnv`：3x3 文本拼图环境，固定中心块，动作是交换两个非中心位置，奖励由正确相邻对、正确位置和完成奖励组成。
- DQN solver：支持纯视觉、纯文本、多模态三种输入，统一入口为 `python Solver/train_rl.py`。
- MangaZero panel 排序 solver：提供 set-to-sequence pointer 模型和全排列分类模型两个方案，分别用 `train_mangazero_panel_ordering.py` 与 `train_mangazero_permutation_classifier.py` 训练。

## 使用提示

数据生成与训练脚本均带 `argparse` 参数，详细示例见 `Solver/RL_PIPELINE.md`、`Solver/AGENT_DESIGN.md` 和 `mangazero_panel_ordering_plan.md`。主要依赖包括 PyTorch、torchvision、Pillow、NumPy、datasets、PaddleOCR（可选）、OpenCV 等；生成的数据和训练 checkpoint 默认已被 `.gitignore` 排除。

COMICS 构建示例：

```bash
python comics/create_panel_ordering_dataset.py \
  --panels-dir comics/data/raw_panel_images \
  --ocr-csv comics/data/COMICS_ocr_file.csv \
  --ad-pages comics/data/predadpages.txt \
  --output-dir comics/panel_ordering_dataset \
  --panel-count 6 --puzzle-num 500
```

COMIX（Hugging Face）流式构建示例：

```bash
# 完整流程：流式下载 + 构建 ordering 数据集
python comics/stream_comix.py \
  --dataset-name emanuelevivoli/comix-v0_1-pages \
  --split train \
  --max-pages 100 \
  --raw-dir comics/raw_comix \
  --output-dir comics/ordering_dataset \
  --panel-count 6 --puzzle-num 500 \
  --disable-ocr

# 仅流式下载 raw 数据
python comics/stream_comix.py --dataset-name emanuelevivoli/comix-v0_1-pages --stream-only --max-pages 100

# 仅从已有 raw 目录构建 ordering 数据集
python comics/stream_comix.py --build-only --raw-dir comics/raw_comix --disable-ocr
```

`stream_comix.py` 复用 `Mangazero/build_dataset.py` 的 `write_puzzle_directory`、`pad_image`、`clamp_bbox` 等逻辑，输出与 Mangazero 相同格式的 `manifest.jsonl` + `sample.json` + `panels_padded/`，可以直接喂给现有的 `Solver/train_mangazero_panel_ordering.py` 和 `Solver/train_mangazero_permutation_classifier.py`。

官方 pages 数据集字段为 `page["json"]`（元数据，含 `book_id/page_number/page_class/detections`）和 `page["jpg"]`（PIL 页面图）。脚本会保存 `metadata` 与页面图，并从 `metadata["detections"]["fasterrcnn"]["panels"]` 读取 panel 框、从 `textboxes`/`characters` 按重叠归属到 panel，产出可与 Mangazero 训练入口直接对接的 ordering 数据集。
