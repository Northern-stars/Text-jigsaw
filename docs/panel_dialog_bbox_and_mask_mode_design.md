# Panel 对话气泡 bbox 与 Solver mask / OCR-only 模式 设计方案

> 目标：
> 1. 在 Manga109 / MangaZero / comix 三个数据集的 ordering puzzle 生成逻辑中，为每个 panel 记录其每个对话框（对话气泡 / textbox）的 bbox 坐标。
> 2. 在 solver 中，除现有 `solve`（正常）、`mirage_inference`（无图、静默）、`guess_inference`（无图、显式告知）之外，新增 `mask` 模式：把图片中所有对话框的文字内容遮掉后再送入 VLM，其余流程与其它模式保持一致。
> 3. 在 solver 中新增纯 OCR 文本模式（OCR-only）：不给图像，只把（人工校验的）画内文字转录作为文本输入供模型排序。
>
> 本文档只做设计，不包含实现。

## 1. 背景与现状

### 1.1 四条数据生产链路

ordering 数据由四条链路产出（见 [panel_ordering_fields.md](panel_ordering_fields.md)），本方案涉及其中三条会写 `sample.json` 的链路：

| 链路 | 入口 | panel 来源 | 对话框来源 |
| --- | --- | --- | --- |
| MangaZero | `Mangazero/build_dataset.py` | raw `frames[].bbox` | `frames[].dialogs[].bbox` |
| Manga109 | `Manga109/create_ordering.py` | XML `<page><frame>` | XML `<page><text>`（按与 frame 的重叠率分配） |
| comix (COMICS 老集) | `comix/create_panel_ordering_dataset.py` | 原始 panel 图 | `COMICS_ocr_file.csv` 的 `x1,y1,x2,y2` |
| comix (COMIX-HF) | `comix/comix_pipeline.py` | fasterrcnn `panels` | fasterrcnn `textboxes`（与 panel 重叠率 >= 0.30） |

`comix/comix_pipeline.py` 复用 `Mangazero/build_dataset.py` 的 `PanelRecord` / `write_puzzle_directory`（见其 `from Mangazero.build_dataset import (...)`），因此改动 `write_puzzle_directory` 会同时惠及 MangaZero 与 comix 两条链路。

### 1.2 已有字段：`dialog_bboxes`

三个生产者的 `PanelRecord` / panel JSON 已经带有对话框坐标字段：

- `dialog_bboxes`: 每个对话框在**原页面坐标系**下的像素框 `[x1, y1, x2, y2]`。
- `dialog_texts`: 与 `dialog_bboxes` 一一对应的 OCR / 标注文本。
- `dialog_text`: 上述文本拼接后的整段文本。

关键点：**`dialog_bboxes` 是页面坐标系，不是 panel 裁剪图坐标系**。

- MangaZero: `build_dataset.py:604-607`，`clamp_bbox(dialog["bbox"], page_width, page_height)`。
- comix (COMIX-HF): `comix_pipeline.py:669-676`，`_extract_bbox(tb, page_width, page_height)`。
- comix (COMICS 老集): `create_panel_ordering_dataset.py:161`，直接来自 CSV（COMICS 每行坐标即该 panel 内坐标，见 1.3）。
- Manga109: **当前写死为空** —— `create_ordering.py:411` `"dialog_bboxes": []`，需要补齐。

### 1.3 坐标语义差异（关键坑）

`dialog_bboxes` 目前在两套坐标系下混用，做遮罩时不能一刀切：

- **页面坐标系（page-space）**：MangaZero、comix(COMIX-HF) 将气泡框记录为整页坐标。要映射到 panel 裁剪图，需先减去 panel bbox 左上角并裁剪，再缩放到 padded 图。
- **panel 内坐标系（panel-space）**：COMICS 老集每行坐标就是该 panel 图内坐标，panel 裁剪本身从 `(0,0)` 起，无页面偏移。
- **Manga109**：XML 的 `<text>` 也是页面坐标，与 frame 重叠分配，可作为 page-space 处理。

无论哪种，都需要一个统一的「把 bbox 映射到 padded panel 图坐标」的换算，见第 2 节。

### 1.4 Solver 现状

`Solver/agent/vlm_ordering_solver.py` 中 `VLMOrderingSolver` 已有三条推理路径，结构完全平行：

| 方法 | prompt generator | 是否给图 | 备注 |
| --- | --- | --- | --- |
| `solve` | `build_ordering_messages` | 是 | 正常多模态 |
| `mirage_inference` | `build_ordering_messages_mirage` | 否（静默） | 只给文字，不提示无图 |
| `guess_inference` | `build_ordering_messages_guess` | 否（显式） | 明说无图 |

三者最后都走 `vlm_infer -> answer_generator -> _validate`，返回 `VLMOrderingResult`。`QwenVLOrderingSolver`（`Solver/agent/qwen_vl_ordering_solver.py`）通过 `_default_*_prompt_generator` 注入对应 generator，并复用基类推理路径。

因此 mask 模式与 OCR-only 模式都应作为**平行的新推理方法 + 新 prompt generator**加入，尽量不改动既有路径。

## 2. 数据集改动：为每个 panel 写入对话框 bbox

### 2.1 字段设计

在每个 panel JSON 中新增一个字段，存放**与 padded panel 图坐标对齐**的对话框框，便于 solver 直接绘制遮罩；同时保留原有的页面坐标系字段以兼容其它消费者。

| 字段 | 坐标系 | 用途 |
| --- | --- | --- |
| `dialog_bboxes`（已存在） | 原页面图（或 panel 内） | 兼容现有消费者 / 调试 |
| `dialog_bboxes_panel`（新增） | panel 裁剪图（pad 之前，原点为裁剪左上） | 中间表示 |
| `dialog_bboxes_padded`（新增） | 最终 padded 图（`target_panel_size`） | solver 直接绘制遮罩 |
| `dialog_bboxes_norm`（可选） | 归一化到 `[0,1]` | 换 target 尺寸后仍可用 |

推荐至少写入 `dialog_bboxes_padded`，其定义：

```jsonc
"dialog_bboxes_padded": [[x1, y1, x2, y2], ...]  // 整数像素，坐标基于 padded_size（如 224x224）
```

### 2.2 映射公式

设 panel 在原页面上的框为 `P = [px1, py1, px2, py2]`，面板裁剪宽高 `pw, ph`，目标尺寸 `W, H`，缩放比 `s = min(W/pw, H/ph)`，pad 偏移为 `[left, top, right, bottom]`（由 `pad_image()` 返回，沿用现有实现）。对每个页面坐标框 `B = [x1, y1, x2, y2]`：

1. 转 panel 裁剪坐标：`bx1 = x1 - px1`，`by1 = y1 - py1`（`px1, py1` 即 `P` 左上角）。
2. 用 `s` 缩放并加 pad 偏移：
   - `X1 = round(bx1 * s) + left`，`Y1 = round(by1 * s) + top`
   - `X2 = round(bx2 * s) + left`，`Y2 = round(by2 * s) + top`
3. 裁剪到 `[0, W] x [0, H]`，并丢弃退化框（`X2 <= X1` 或 `Y2 <= Y1`）。

对 panel-space 数据（COMICS 老集），第 1 步偏移量为 0（等价于 `px1 = py1 = 0`）。

实现上建议新增一个纯函数：

```python
def map_dialog_bboxes_to_padded(
    dialog_bboxes, panel_bbox, raw_size, target_size, pad,
) -> list[list[int]]:
    ...
```

放在 `Mangazero/build_dataset.py`（或 `Manga109/reading_order_utils.py` 一类共享模块）内，供三条链路复用。

### 2.3 各链路改动点

#### MangaZero (`Mangazero/build_dataset.py`)

- `PanelRecord`（`build_dataset.py:22-59`）新增 `dialog_bboxes_padded`（以及可选 `dialog_bboxes_norm`）字段，并在 `to_json()` 中自然带出。
- `process_sample`（`build_dataset.py:573-659`）在算出 `bbox`、`pad`、`dialog_bboxes` 后，用 2.2 的映射函数计算 `dialog_bboxes_padded`，写入 record（`build_dataset.py:628-656`）。
- `write_puzzle_directory`（`build_dataset.py:693-744`）无需改动：它直接 `record.to_json()`。

#### comix (COMIX-HF) (`comix/comix_pipeline.py`)

- `process_comix_sample`（`comix_pipeline.py:583-754`）在 `PanelRecord(...)` 构造处（`comix_pipeline.py:726-752`）补算 `dialog_bboxes_padded`（`panel_dialog_bboxes` 是页坐标，`bbox` 是页坐标，`pad` / `raw_crop.size` 可得）。
- 因为复用 MangaZero 的 `write_puzzle_directory`，写盘自动带上新字段。

#### Manga109 (`Manga109/create_ordering.py`)

- 现有输出 `"dialog_bboxes": []`（`create_ordering.py:411`）需要补齐：
  - `assign_texts_to_frames`（`create_ordering.py:173-196`）目前只返回 `frame_id -> text`；改为同时返回每个 frame 下被分配的 text 框坐标（`frame_id -> list[[x1,y1,x2,y2]]`），或新增一个并行函数返回框列表。
  - 在 panel 拼装处（`create_ordering.py:395-419`）把该 frame 的对话框框写入 `dialog_bboxes`（页坐标）与 `dialog_bboxes_padded`（2.2 换算）。
- 注意 `dialog_texts`（`create_ordering.py:412`）当前是整段拼接后的单元素列表，与逐框 `dialog_bboxes` 不再一一对应；如需一一对应，应改为逐框文本列表。建议保持「一个框一条文本」的语义与其它链路一致。

#### comix (COMICS 老集) (`comix/create_panel_ordering_dataset.py`)

- `panel_to_json`（`create_panel_ordering_dataset.py:206-233`）已写 `dialog_bboxes = [box.bbox for box in panel.text_boxes]`（panel 内坐标）。
- 新增 `dialog_bboxes_padded`：`panel_bbox = [0, 0, width, height]`，用 2.2 映射（此链路坐标已是 panel 内坐标，可直接缩放 + pad）。
- `write_puzzle_directory`（`create_panel_ordering_dataset.py:236-289`）已有 `pad`、`raw_size`、`target_size`，在拼装处传入即可。

### 2.4 兼容性

- 新字段是**追加**，不改名、不删除既有字段，`MangaZeroPanelOrderingDataset`（`Solver/env/mangazero_panel_env.py`）通过 `sample.panels` 原样透传 panel dict，因此旧数据缺字段也能正常加载；新数据多出的字段对旧代码无副作用。
- `dialog_bboxes_padded` 缺失时，mask 模式的回退见 3.3。

## 3. Solver mask 模式设计

### 3.1 API 形态

在 `VLMOrderingSolver` 中新增平行方法，签名与其它三条一致：

```python
def mask_inference(
    self,
    panels: Sequence[PanelImage],
    texts: TextList = None,
    *,
    masks: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    mask_value: int | tuple[int, int, int] = 0,
) -> VLMOrderingResult:
    ...
```

- `masks`：与 `panels` 一一对应的对话框框列表，元素为 padded 坐标 `[x1,y1,x2,y2]`。缺省时从 panel dict 的 `dialog_bboxes_padded` 自动读取（见 3.3）。
- `mask_value`：填充颜色，默认黑 `(0,0,0)`；可配置为白或灰色，或改成半透明 / 马赛克（见 5.2）。
- 返回类型与 `solve` 完全一致，便于评测脚本统一调用。

对应新增一个 prompt generator（文字 prompt 与 `solve` 完全一致，因此直接复用 `build_ordering_messages`，只在图上去掉文字）：

```python
def build_ordering_messages_masked(
    panels, texts=None, *, start_index=0, system_prompt=None,
    masks=None, mask_value=0,
) -> list[dict[str, Any]]:
    # 先对每个 panel 图做遮罩，再走与 build_ordering_messages 相同的组装
```

### 3.2 与其它模式的关系

| 模式 | 图像 | 画内文字 | prompt 文字 | 与 solve 的差异 |
| --- | --- | --- | --- | --- |
| `solve` | 原始 | 保留 | 标准 | 基准 |
| `mirage_inference` | 无 | 无 | 标准（不提缺图） | 去掉图 |
| `guess_inference` | 无 | 无 | 标准 + 提示无图 | 去掉图 + 提示 |
| `mask_inference` | 有（气泡区域被遮） | 被遮 | 标准 | 只在图上遮罩 |

`mask_inference` 与 `solve` 应保持**除视觉输入外的一切因素相同**（同一个 prompt 文本、同样的 `answer_generator`、同样的解析与校验），这样得到的是纯粹的「画内文字消融」对照，符合研究提案四条件里的 Visual-only 条件（见 [mirage_modality_ablation_proposal.md](mirage_modality_ablation_proposal.md)）。

### 3.3 遮罩生成

在 `vlm_ordering_solver.py` 中新增纯函数（只用 PIL，无 torch 依赖）：

```python
def apply_dialog_masks(
    image: Image.Image,              # padded panel 图，RGB
    bboxes: Sequence[Sequence[int]],  # padded 坐标框
    *,
    fill: int | tuple[int, int, int] = 0,
    pad_px: int = 0,                 # 可选：向外扩张若干像素，防止描边残留
    mode: str = "fill",              # fill | blur | white
) -> Image.Image:
    ...
```

读取坐标的优先级：

1. 调用方显式传入 `masks`（便于测试与手工覆盖）。
2. panel dict 的 `dialog_bboxes_padded`（推荐，直接可用）。
3. 回退：panel dict 的 `dialog_bboxes` + `bbox` / `raw_size` / `pad` / `padded_size` 现算（用 2.2 公式），保证旧数据也能跑；若只有 `dialog_bboxes` 且无 `pad`，按 panel-space 处理。

`to_pil`（`vlm_ordering_solver.py:53-85`）已能接受 dict / path / tensor，遮罩应作用在 `to_pil()` 之后、放入 messages 之前，即改 `build_ordering_messages_masked` 内部对每个 panel 先 `to_pil` 再 `apply_dialog_masks`。

### 3.4 `QwenVLOrderingSolver` 接入

- 新增 `_default_mask_prompt_generator`，返回 `build_ordering_messages_masked(..., start_index=self.start_index)`。
- 构造函数增加 `mask_prompt_generator` 参数（对齐现有 `mirage_prompt_generator` / `guess_prompt_generator`）。
- 父类 `VLMOrderingSolver.__init__` 增加 `mask_prompt_generator` 形参并保存默认值。
- 训练路径不受影响（`train` 仍用 `prompt_generator`）；若希望「训练用带字、测试用遮字」，可在后续扩展 `train_masked`，本方案不要求。

### 3.5 结果记录

`VLMOrderingResult` 建议扩展可选字段，便于评测时区分 mode 与记录遮罩框：

```python
mode: str = "solve"          # solve | mirage | guess | mask | ocr_only
masks: list[list[list[int]]] = field(default_factory=list)
```

现有 `solve` 也可顺带填 `mode="solve"`，`mirage_inference` 填 `"mirage"`，`guess_inference` 填 `"guess"`，保持向后兼容（新增字段带默认值）。

## 4. Solver 纯 OCR 文本模式设计（OCR-only）

### 4.1 定位

对应研究提案四条件中的 **OCR-only**：不给任何 panel 图像，只把画内文字的转录作为文本输入，让模型据此排序（见 [mirage_modality_ablation_proposal.md](mirage_modality_ablation_proposal.md)）。它是 `Text-only retention ratio` 与 `Visual gain` 两个指标的输入条件。

与现有无图路径的关系：

| 方法 | 图像 | 文本输入 | 声明无图 | 定位 |
| --- | --- | --- | --- | --- |
| `mirage_inference` | 无 | 可选 | 否 | 静默无图，测语言先验 |
| `guess_inference` | 无 | 可选 | 是 | 明说无图，测缺失模态行为 |
| `ocr_only_inference` | 无 | 必须 | 可配置 | 正式 OCR-only 条件 |

实现上 `ocr_only_inference` 与「传了 `texts` 的 `mirage_inference` / `guess_inference`」非常接近（三者都走同一套 text-only message builder）。之所以单列，是为了：(1) 作为一个有名字、可打标签的正式条件；(2) 固定「使用人工校验转录」这一来源约定；(3) 固定默认的 header 与是否声明无图，避免被调用方的临时参数污染；(4) 供评测聚合时按条件区分。

### 4.2 关键要求：使用人工校验转录

proposal 明确要求 OCR-only 使用 **verified transcription**，避免 OCR 识别错误成为主要混淆因素。现有数据集的文本来源不一：MangaZero / comix-HF 的 `dialog_text` 是 OCR 产物，Manga109 的 `dialog_text` 来自 XML 标注。因此设计两级文本来源：

1. `dialog_text_verified`（新增，可选）：人工校验 / 修正后的转录。
2. 回退 `dialog_text`：无 verified 时使用，并在结果里标记 `text_source`，分析时可按来源分层。

若不想重跑数据集生成，另提供**覆盖文件**方案：一个 JSONL（`panel_id -> verified_text`），在构造 `texts` 时按 `panel_id` 覆盖。

### 4.3 API 形态

在 `VLMOrderingSolver` 中新增平行方法：

```python
def ocr_only_inference(
    self,
    panels: Sequence[PanelImage],
    texts: TextList,
    *,
    source: str = "auto",            # auto | verified | dialog_text
    disclose_text_only: bool = False,
    ocr_header: str = "OCR transcription",
) -> VLMOrderingResult:
    ...
```

- `texts` 必填，每个 panel 一条，顺序与 `panels`（即输入槽顺序）一致；长度不匹配时报错。
- `panels` 仅用于计数 / 读取 verified 文本；其图像不会被送入模型。
- `source`：`auto`（优先 verified，缺失则回退 `dialog_text`）、`verified`（缺则报错或标记缺失）、`dialog_text`。
- `disclose_text_only`：是否在 prompt 里声明「只提供文本转录、没有图像」。默认 `False`（静默，做纯文本条件）；置 `True` 时可作为更强 framing 的敏感性对照。
- 返回 `VLMOrderingResult`，`mode="ocr_only"`。

对应 prompt generator：

```python
def build_ordering_messages_ocr_only(
    panels, texts=None, *, start_index=0, system_prompt=None,
    ocr_header="OCR transcription", text_only_note=None,
) -> list[dict[str, Any]]:
    # include_images=False，其余与 build_ordering_messages 同源
```

内部复用 `build_ordering_text_prompt(..., ocr_header=ocr_header, no_image_note=text_only_note)`：与 `solve` 的文本 prompt 同源，只把 header 换成转录口径，并可选追加 note。调用方可用 `text_only_note` 传入声明无图的句子（对应 `disclose_text_only=True`）。

### 4.4 文本来源解析

runner / 评测脚本构造 `texts` 时：

```python
def panel_text_for_ocr_only(panel, source="auto"):
    if source in ("auto", "verified"):
        value = str(panel.get("dialog_text_verified", "") or "").strip()
        if value:
            return value, "verified"
        if source == "verified":
            return "", "verified_missing"
    return str(panel.get("dialog_text", "") or "").strip(), "dialog_text"
```

`sample["panels"]` 已按 input 顺序排列，逐个 panel 取文本即与输入槽对齐，无需再动 `input_order`。

### 4.5 数据集改动（可选）

- 三个生产者可新增 `dialog_text_verified` 字段，默认等于 `dialog_text`（占位），后续人工修订即可生效。
- Manga109 的 XML 文本本身即人工标注，可令 `dialog_text_verified = dialog_text`，并把 `text_source` 标为 `xml`。
- 若不改数据生成链路，只用 4.2 的覆盖文件。

### 4.6 与 Full + OCR 的区分

proposal 的辅助条件 **Full + OCR** = 图像 + 额外转录文本，直接用 `solve(panels, texts=...)` 即可；**OCR-only** 是只有文本、没有图像。两者共用同一套 `texts` 供给逻辑（4.4），区别只在 `solve` 保留图像。

### 4.7 结果记录与指标对接

- `VLMOrderingResult.mode = "ocr_only"`，并建议记录 `text_source`（`verified` / `dialog_text`）。
- 指标：`Text-only retention ratio = A_ocr_only / A_full`；`Visual gain = A_full - A_ocr_only`。两个指标都以本条件准确率为输入。

## 5. 可选增强

### 5.1 遮罩策略开关（CLI）

建议给 mask 模式提供参数，便于做敏感性实验：

- `--mask-fill {black,white,gray,blur}`
- `--mask-dilate N`：向外扩张像素，避免气泡描边、速度线残留。
- `--mask-source {padded,dialog_bboxes}`：优先用哪个字段。
- `--mask-keep-text`：遮罩但仍在 prompt 里附 OCR 文本（等价于「视觉遮字、文本给字」，用于分离 visual vs text 通道）。

### 5.2 填充方式

- `fill`（纯色）：最接近「抹掉文字」，但大面积色块本身可能成为视觉线索，需在结果解读里说明。
- `blur` / `inpaint`：保留气泡轮廓与位置信息，可能更接近「读不到字但知道有气泡」。建议至少实现 `fill` 与 `blur` 两种对照。
- 若担心纯色块引入伪线索，可做一个 `mask_boxes_only` 变体：只在实验结果中记录框，不真遮，用于评估「模型是否利用框位置」。

### 5.3 无 bbox 时的行为

- 若某 panel 没有可用 bbox（如旧数据、Manga109 未补字段前的产物），`mask_inference` 应记录该情况并**原样保留图片**（不遮），在结果里用 `mask_missing = True` 标记，避免静默产生「假 mask 结果」。

## 6. 测试与验证计划

数据侧：

1. 小样本构建后，随机抽 panel，把 `dialog_bboxes_padded` 画到 padded 图上并目视核对是否贴合气泡（可用 PIL 画红框导出图片）。
2. 断言：`dialog_bboxes` 与 `dialog_texts` 数量一致；`dialog_bboxes_padded` 全部落在 `[0,W] x [0,H]` 内；退化框被丢弃。
3. 跨链路抽查：MangaZero、Manga109、comix 各取一页，人工确认坐标换算正确。

Solver 侧：

4. 用 stub `vlm_infer`（返回固定顺序）+ 假 panel，断言 `mask_inference` 与 `solve` 走的是同一条 `answer_generator` / `_validate`，且传入模型的图确实被遮罩（可比对遮罩前后像素）。
5. 断言 `mask_inference` 在缺 bbox 时原样返回图片并打标。
6. 用 stub `vlm_infer` 断言 `ocr_only_inference`：消息里不含任何 image、prompt 含转录文本、`answer_generator` / `_validate` 与 `solve` 同源。
7. 断言 `ocr_only_inference` 的 `texts` 与输入槽对齐；`source="verified"` 缺字段时被正确标记。
8. 端到端：对同一 puzzle 跑 `solve` / `mask_inference` / `ocr_only_inference` / `mirage_inference` / `guess_inference`，确认准确率口径一致（都转成同一 permutation 后再比较）。

## 7. 与论文实验的映射

本方案直接支撑研究提案里的四条件（Full / Visual-only / OCR-only / No-image）与辅助条件 Full+OCR：

- `solve` ≈ Full
- `mask_inference` ≈ Visual-only（保留图像，遮掉画内文字）
- `ocr_only_inference` ≈ OCR-only（只有转录文本，无图像）
- `mirage_inference` / `guess_inference` ≈ No-image（静默 / 显式两种 framing）
- `solve(panels, texts=...)` ≈ Full + OCR（图像 + 额外转录）

详见 [mirage_modality_ablation_proposal.md](mirage_modality_ablation_proposal.md)。

## 8. 改动清单（实现时按此落地）

| 文件 | 改动 |
| --- | --- |
| `Mangazero/build_dataset.py` | `PanelRecord` 增 `dialog_bboxes_padded`（可选 `dialog_text_verified`）；`process_sample` 计算 padded 框；新增 `map_dialog_bboxes_to_padded` |
| `comix/comix_pipeline.py` | `process_comix_sample` 计算 padded 框 |
| `Manga109/create_ordering.py` | `assign_texts_to_frames` 输出框；panel 拼装写 `dialog_bboxes` / `dialog_bboxes_padded`；令 `dialog_text_verified = dialog_text` |
| `comix/create_panel_ordering_dataset.py` | `panel_to_json` 写入 padded 框 |
| `Solver/agent/vlm_ordering_solver.py` | `apply_dialog_masks`；`build_ordering_messages_masked`；`mask_inference`；`build_ordering_messages_ocr_only`；`ocr_only_inference`；`VLMOrderingResult` 增 `mode` / `masks` / `text_source` 字段 |
| `Solver/agent/qwen_vl_ordering_solver.py` | `_default_mask_prompt_generator`；`_default_ocr_only_prompt_generator`；构造参数 `mask_prompt_generator` / `ocr_only_prompt_generator` |
| `docs/panel_ordering_fields.md` | 补充 `dialog_bboxes_padded`、`dialog_text_verified` 等新字段说明 |
