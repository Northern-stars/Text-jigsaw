# Panel Ordering 字段说明

本文档说明 panel-ordering（漫画分镜排序）数据格式中各个字段的含义、取值范围和消费位置。

ordering 数据由四条生产链路产出：`Mangazero/build_dataset.py`、`Manga109/create_ordering.py`、`comix/create_panel_ordering_dataset.py`、`comix/comix_pipeline.py`。它们统一写出 `manifest.jsonl` + `<puzzle_index>/sample.json` + `<puzzle_index>/panels_padded/*.jpg`，并由
[Solver/env/mangazero_panel_env.py](../Solver/env/mangazero_panel_env.py) 消费。

## 1. 数据集目录布局

```
<dataset_dir>/
  manifest.jsonl                      # 每个 puzzle 一行索引
  000000/
    sample.json                       # 单个 puzzle 的全部字段
    panels_padded/input_00.jpg ...    # 已缩放到统一尺寸的 panel 图
    panels_raw/   (Manga109/部分链路)  # 未 pad 的原始裁剪
```

`manifest.jsonl` 每行字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `puzzle_index` | int | 该 puzzle 在数据集内的连续编号，与目录名一致。 |
| `sequence_id` | str | puzzle 唯一标识（复制自 sample.json）。 |
| `sample_path` | str | 相对 `dataset_dir` 的 `sample.json` 路径，如 `000000/sample.json`。 |

## 2. sample.json 顶层字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `sequence_id` | str | puzzle 唯一标识，格式 `<manga>_<chapter>_<page>_puzzle_<index:06d>`。 |
| `puzzle_index` | int | puzzle 编号。 |
| `manga_id` | str | 作品/书名标识。 |
| `chapter_id` | str | 章节标识，无章节时多为 `default`。 |
| `page_id` | str | 页面标识，通常为 `<manga>_<页码>_jpg`。 |
| `page_index` | int | 该页在章节内的序号，用作 `global_order` 的高位。 |
| `group_key` | str | 分组键 `manga_id/chapter_id`，用于按作品切分 train/val/test。 |
| `page_key` | str | 页键 `manga_id/chapter_id/page_id`，`sequence_id` 由其派生。 |
| `panel_count` | int | 本 puzzle 的 panel 数量（固定窗口长度，默认 6）。 |
| `target_panel_size` | [int, int] | panel 统一目标尺寸 `[w, h]`（默认 224x224）。 |
| `panels` | list[dict] | panel 列表，见第 3 节。 |
| `input_order` | list[int] | 乱序输入：`input_order[i]` = 第 i 个输入槽对应的窗口内原始位置。 |
| `target_order` | list[int] | 正解顺序：`target_order[k]` = 正确阅读顺序中第 k 位对应的输入槽下标。 |
| `text_source` | str | 文本来源标记，`dialog_text` 表示文本取自对白 OCR。 |

### input_order 与 target_order 的关系

设窗口内原始阅读顺序为 `0..panel_count-1`，生产者先做随机置换 `input_order`（`input_order[i]` 是把第 i 个输入槽放成原始的第几个 panel）：

```python
input_order = list(range(panel_count))
rng.shuffle(input_order)
target_order = [input_order.index(i) for i in range(panel_count)]
```

- `input_order`：从"输入槽 → 原始阅读位置"，描述图是怎么被打乱的。
- `target_order`：从"阅读顺序位置 → 输入槽"，描述模型应该按什么顺序输出输入索引。
  即模型每一步应选 `target_order[k]` 作为下一个 panel，这也正是环境里 `selected_order` 与 `target_order` 可以直接比较的原因。
- `input_order` 与 `target_order` 互为逆置换。模型学习目标只依赖 `target_order`，`input_order` 主要用于还原/调试。

## 3. panels 列表中每个 panel 的字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `manga_id` | str | 作品标识。 |
| `chapter_id` | str | 章节标识。 |
| `source_image_path` | str | 原始页面图在 raw 数据中的相对路径。 |
| `page_id` | str | 页面标识。 |
| `page_index` | int | 页面在章节内的序号。 |
| `panel_index_in_page` | int | 该 panel 在页面 frames 里的序号（页内阅读顺序位置）。 |
| `global_order` | int | 全局顺序，`page_index * 10000 + panel_index_in_page`，用于跨页排序。 |
| `bbox` | [int,int,int,int] | panel 在原页面图中的像素框 `[x1, y1, x2, y2]`。 |
| `page_size` | [int, int] | 原页面图尺寸 `[w, h]`，用于把 bbox 归一化。 |
| `padded_path` | str | 已 pad 的 panel 图相对路径，环境据此加载图像。 |
| `raw_size` | [int, int] | 裁剪后、pad 前的原始 panel 尺寸 `[w, h]`。 |
| `padded_size` | [int, int] | pad 之后的尺寸（等于 `target_panel_size`）。 |
| `pad` | [int,int,int,int] | pad 边距 `[left, top, right, bottom]`（黑边像素）。 |
| `caption` | str | panel 内容的自然语言描述（数据集自带，可选）。 |
| `dialog_bboxes` | list[[int,int,int,int]] | 对白气泡在原页面图中的像素框。 |
| `dialog_texts` | list[str] | 每个气泡 OCR 出来的文本（`--disable-ocr` 时为空）。 |
| `dialog_text` | str | `dialog_texts` 拼接后的整段对白文本，供文本/多模态分支使用。 |
| `character_ids` | list[str] | panel 内出现的人物 id。 |
| `character_bboxes` | list[[int,int,int,int]] | 人物框的像素坐标。 |
| `character_types` | list[int] | 人物类型标记（-1 表示未知）。 |
| `panel_id` | str | panel 唯一标识 `<page_id>_panel_<index:03d>`。 |
| `input_index` | int | 该 panel 在乱序输入中的位置（= 模型输出时的索引）。 |
| `original_window_index` | int | 该 panel 在窗口原始阅读顺序中的位置。 |

说明：

- `pad` 由 `pad_image()` 计算，采用等比缩放居中贴到黑底画布，因此 `pad` 之和等于被填充的总边距。
- `padded_path` 由 `padding` 写入，环境 `_resolve_path()` 按 `dataset_dir/` → `dataset_dir/../` → `cwd/` 依次查找，故相对路径无需改绝对路径。

## 4. 环境读取后输出的接口字段

### 4.1 `MangaZeroPanelOrderingDataset.__getitem__`

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `sequence_id` | str | puzzle 标识。 |
| `manga_id` / `chapter_id` | str | 作品 / 章节标识。 |
| `panel_images` | Tensor `[N,3,H,W]` | 按 `input_order` 排列的 RGB 图，float32 归一到 [0,1]（`image_size` 不为空时先缩放）。 |
| `target_order` | Tensor `[N]` long | 正解顺序（输入槽下标序列）。 |
| `dialog_texts` | list[str] | 每个输入槽 panel 的 `dialog_text`。 |
| `panels` | list[dict] | 原始 panel 字典，保留全部字段供调试/可视化。 |
| `layout_features` | Tensor `[N,10]` float32 | `use_layout=True` 时给出，见 4.3。 |

### 4.2 `collate_panel_ordering_batch`

把 batch 组装为：`sequence_id`/`manga_id`/`chapter_id` 为 list，`panel_images`、`target_order`（以及可选的 `layout_features`）按 batch 维 stack，`dialog_texts`、`panels` 保持嵌套 list。

### 4.3 `layout_features` 的 10 个维度

由 `MangaZeroPanelOrderingDataset._layout_features()` 从 panel 字段计算，全部按页面尺寸归一化：

| 位置 | 含义 |
| --- | --- |
| 0-3 | `x1/W, y1/H, x2/W, y2/H`，归一化 bbox。 |
| 4-5 | `center_x/W, center_y/H`，bbox 中心。 |
| 6-7 | `width/W, height/H`，归一化宽高。 |
| 8 | `width*height / (W*H)`，归一化面积占比。 |
| 9 | `page_index / 10000`，缩放后的页码。 |

### 4.4 `MangaZeroPanelOrderingEnv`

把单个 puzzle 作为 episode，动作为"选择下一个 panel 的输入索引"。

- `reset(index=None)` 返回 observation。
- `step(action)` 返回 `(observation, reward, done, info)`。
  - 动作非法（越界或重复）→ `done=True`，reward=`invalid_action_penalty`（默认 -1.0）。
  - 未选满 → reward=0。
  - 选满 → reward=`pairwise_accuracy`，若完全匹配再加 1.0。

observation 字段：

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `sequence_id` | str | puzzle 标识。 |
| `panel_images` | Tensor `[N,3,H,W]` | 当前 puzzle 的 panel 图。 |
| `dialog_texts` | list[str] | 每个 panel 的对白文本。 |
| `selected_order` | Tensor long | 已经按顺序选中的输入槽下标。 |
| `selected_mask` | Tensor bool `[N]` | 已选 panel 的掩码，用于屏蔽重复动作。 |
| `done` | bool | episode 是否结束。 |
| `layout_features` | Tensor `[N,10]` | 可选项，同 4.3。 |

info 字段：

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `target_order` | list[int] | 正解顺序。 |
| `pred_order` | list[int] | 当前预测顺序。 |
| `invalid_action` | int / None | 触发结束的非法动作。 |
| `complete` | bool | 是否完整选满且无非法动作。 |
| `exact_match` | float | 完全匹配为 1.0，否则 0.0。 |
| `position_accuracy` | float | 逐位命中率（命中位 / 总长度）。 |
| `pairwise_accuracy` | float | 成对顺序正确率，见第 5 节。 |

## 5. 评估指标 pairwise_accuracy

`pairwise_accuracy(pred_order, target_order)` 遍历所有 panel 对 `(i, j)`（`i` 在 `j` 之前），统计 `pred` 中 `i` 是否仍排在 `j` 之前，返回正确对数占比：

- 1.0 表示完全有序（与 exact match 一致）。
- 随机排序在 N 个 panel 上的期望为 0.5。
- 只统计两个 panel 都出现在 `pred` 中的对，缺失 panel 的 pair 不计入分母。

## 6. 生产链路的细微差异

- `Mangazero/build_dataset.py`：panel 来自数据集自带 `frames`，`global_order = page_index*10000 + panel_index`，OCR 可关（`--disable-ocr` 时 `dialog_text` 为空）。
- `Manga109/create_ordering.py`：panel 来自 XML `<frame>`，按 `sort_bboxes_reading_order` 计算阅读顺序，用滑动窗口 `stride` 切窗。
- `comix/create_panel_ordering_dataset.py`（COMICS 老数据集）：沿用 OCR CSV，按官方 `panel_no` 排序。
- `comix/comix_pipeline.py`（COMIX Hugging Face）：实现集中在该模块，`stream_comix.py` / `download_raw.py` / `build_ordering_dataset.py` 只是入口转发。默认按 `sort_prepared_panels_reading_order` 几何重排（`--no-sort-panels-reading-order` 可退回检测顺序，`--reading-order-ltr` 可切西式顺序）。
- 以上链路都保证输出可直接喂给 `MangaZeroPanelOrderingDataset`，无需改字段。
