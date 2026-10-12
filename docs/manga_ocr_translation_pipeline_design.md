# Manga OCR + Sugoi v4 日文翻译 Pipeline 设计方案

## 1. 目标

从日文漫画页面获取文本，经过 manga-ocr 识别日文，再经 Sugoi v4 翻译为英文，最后与现有 ordering puzzle 生成流程整合。

## 2. 现状

- Manga109 提供页面图片和 `<text>` bbox 注解，可直接作为文本区域标注。
- `Manga109/create_ordering.py` 已能从页面提取 frames、按阅读顺序排序、生成 `panel_count` 与 `stride` 采样的 ordering puzzle。
- 现有文本来源是 `assign_texts_to_frames`，直接把 XML `<text>` 内容拼接进 panel。
- `Mangazero/main.py` 已有 PaddleOCR 封装（`OCREngine`、`SubprocessOCREngine`），但只支持 Paddle，没有翻译能力。

## 3. 管线总览

```
Manga109 图片 + XML
        |
        v
[文本检测]  : 复用 Manga109 <text> bbox，或使用文本框检测模型补充
        |
        v
[日文识别]  : manga-ocr  (对 manga 文本训练，识别日文原文)
        |
        v
[文本清洗]  : 合并同一 panel/speech 的段落、去噪、处理竖排/拟声词
        |
        v
[日译英]    : Sugoi v4 按 panel 或整页翻译，输出英文 text
        |
        v
[对齐]      : panel_id / frame_id / text_id 关联，生成英文 dialog_text
        |
        v
[ordering puzzle] : 复用 create_ordering.py 生成 sample.json / manifest.jsonl
```

## 4. 模块设计

### 4.1 文本检测 / 裁剪选择

- manga-ocr 本身不含检测器，需要先提供文本区域。
- 推荐两条路径：
  1. Manga109 有 `<text>` bbox：直接裁剪，最稳定。
  2. 无标注数据：接入页面/panel 文本检测模型，如 `ComicTextDetector`、`DB` 或 `EasyOCR` detector，得到 bbox 后再输入 manga-ocr。
- 策略建议：第一阶段优先使用 XML bbox，后面再补充检测模型。

### 4.2 OCR 识别

- 模型：`kHa-white/manga-ocr`。
- 输入：文本 bbox 裁剪后的 RGB 图像 `PIL.Image`。
- 输出：日文原文字符串，可附带置信度（若模型支持，否则仅文本）。
- 注意：竖排文本在 manga-ocr 中通常能直接处理，无需额外旋转；拟声词、特殊符号要单独清洗或保留标记。
- 接口建议：包装成 `MangaOcrEngine`，兼容现有 `OCREngine.recognize(image)` 接口。

### 4.3 文本清洗 / 规整

- 按 `text_id` / `frame_id` 合并同一段对话。
- 过滤空行、纯符号行、过短机器噪声。
- 竖排与横排统一处理为线性文本。
- 保留日文原文，翻译前不丢失原始信息。

### 4.4 翻译模块

- 模型：Sugoi v4 文本生成模型（如 `mHagos/sugoi-4-japanese-to-english-translator` 或服务端部署实例）。
- 输入：日文文本 + 系统 prompt / translation prompt，可传整体段落保持语境。
- 输出：英文文本，要求：
  - 保留对话气泡顺序；
  - 保留拟声词 / 特效词可选翻译；
  - 避免额外标注或解释文本。
- 缓存：同一日文原句务必缓存翻译结果，避免重复调用。

### 4.5 关联与保存

- 每个翻译输出必须带 `text_id / frame_id / page_id / book_id`。
- 最终 `dialog_text` 采用英文翻译；日文原文作为 `ocr_text` 或 `source_text` 单独保存。
- `sample.json` 中同时保存：
  - `ocr_text` / `ocr_lang`
  - `translated_text` / `translation_lang`
  - 保留 `dialog_text` 作为训练和评测时使用的文本。

## 5. Pipeline 执行流程

```python
for book in books:
    pages = parse_book_pages(xml)
    for page in pages:
        for text in page["texts"]:
            crop = page_image.crop(text["bbox"])
            ocr_text = manga_ocr.recognize(crop)
            en_text = sugoi_translate(ocr_text, target="en")
            save_json({
                "book_id": book,
                "page_id": page["id"],
                "panel_id": page["frame_id"],
                "text_id": text["id"],
                "ocr_text": ocr_text,
                "translation": en_text,
            })
```

### 5.1 中间缓存格式（JSONL）

```json
{
  "book_id": "AisazuNihaIrarenai",
  "page_id": "AisazuNihaIrarenai_p0002",
  "panel_id": "p0002_f1",
  "text_id": "00001d42",
  "bbox": [x1, y1, x2, y2],
  "ocr_text": "……日文原文……",
  "ocr_lang": "ja",
  "translated_text": "……English……",
  "translation_lang": "en"
}
```

### 5.2 生成 puzzle 时合并

- 复用现有 `create_ordering.py` 的读取顺序、sliding window、`input_order` / `target_order`。
- 将 panel 的 `dialog_text` 替换为 `translated_text`。
- 保留 `ocr_text` 作为第二语言字段，供消融和对比实验使用。

## 6. 实现注意

- 依赖新增：`manga-ocr`、`transformers`（Sugoi 需要）、`PIL`（已有）。
- 缓存层：为避免重复 OCR / 翻译，按书/页/文本框 id 缓存结果。
- 加速：OCR 与翻译在 CPU 上较慢，建议 GPU；Sugoi v4 若内存吃紧，可批量调用或走本地推理服务。
- 评测：单独准备 100-300 条日文原文 + 英文翻译测试集，人工验证 OCR 与翻译质量。

## 7. 与现有 pipeline 的兼容

- `create_ordering.py` 可加 `--text-source` 参数：
  - `xml`: 使用原 XML 内容（现有行为）。
  - `manga-ocr`: 使用 OCR + Sugoi 翻译文本。
  - `json`: 从缓存 JSON/JSONL 读取已翻译文本。
- 生成 `sample.json` 时对 `dialog_text` 写翻译文本，其他字段保持原样。

## 8. 执行顺序

1. 安装并验证 manga-ocr 与 Sugoi v4。
2. 写 `manga_ocr_engine.py` 与 `sugoi_translator.py`。
3. 在 Manga109 中选 1-2 本书验证端到端。
4. 写缓存与 JSON 输出。
5. 修改 `create_ordering.py` 加入 `--text-source`。
6. 构建 100-300 条测试集评估 OCR 质量。
7. 在 1 本书上生成完整英文 puzzle，检查可读性。
8. 推广到全部数据集并整合 puzzle 实验。

## 9. 开放问题

- manga-ocr 对竖排与对话框区是否足够可靠？是否需要检测器补充？
- Sugoi v4 的 GPU 要求与批量吞吐？当前机器能否跑？
- 翻译质量评估标准：自动指标是否需要人工 review？
- 同一 dialog 的跨 page 上下文是否需要一起翻译？
