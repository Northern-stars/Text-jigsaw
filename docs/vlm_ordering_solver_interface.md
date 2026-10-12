# VLM Ordering Solver 接口文档

本文档说明通用 VLM 排序接口 [Solver/agent/vlm_ordering_solver.py](../Solver/agent/vlm_ordering_solver.py) 提供的抽象，并在后半部分给出 Qwen2.5-VL 具体实现 [Solver/agent/qwen_vl_ordering_solver.py](../Solver/agent/qwen_vl_ordering_solver.py) 对每个接口的实现逻辑。

## 1. 总体模型

一次 `solve` 调用把三块可替换的组件串起来：

```
panels (+ 可选 OCR texts)
    -> prompt_generator  -> 多模态 prompt（视觉 + 任务文字）
    -> vlm_infer         -> 模型原始输出
    -> answer_generator  -> 最终排序 list[int]
```

- `prompt_generator` 决定"怎么把 panel 列表和文本组装给模型"。
- `vlm_infer` 决定"用哪个模型推理、怎么解码"。
- `answer_generator` 决定"怎么从模型输出里解析出排序"。

三者都是普通可调用对象（打平为函数），所以换模型只需替换 `vlm_infer`，换 prompt 只要替换 `prompt_generator`。

除标准 `solve` 外还有 `mirage_inference`：沿用同一条链路，但**不把图片交给模型**，只送同样的文字 prompt（也不提示图像缺失），输出与解析逻辑完全一致，用于衡量模型在无视觉信息下的表现（见 8.3）。

另有 `guess_inference`：与 `mirage_inference` 流程相同，同样不给图片，但**在文字里明确告知模型没有输入图像、让它基于文本猜测**（见 8.4）。

## 2. 类型约定

| 名称 | 定义 | 说明 |
| --- | --- | --- |
| `PanelImage` | `Any` | 单个 panel，支持 PIL.Image / torch / numpy 张量 / 路径 / dataset panel dict。 |
| `TextList` | `Optional[Sequence[Optional[str]]]` | 每个 panel 的文本；`None` 表示无 OCR，槽位可为 `None`/空串。 |
| `PromptGenerator` | `(panels, texts) -> prompt` | 返回任意 prompt 对象，默认返回 chat messages。 |
| `VLMInfer` | `(prompt) -> Any` | 返回模型原始输出（默认约定为文本）。 |
| `AnswerGenerator` | `(raw_output, panel_count) -> list[int]` | 返回 0-based 排列。 |

排序语义与数据集一致：`order[k]` 表示第 k 个输出位置应放的**输入 panel 下标**（0-based）；`target_order` 也是同一语义，可直接比较。

## 3. 数据结构

### 3.1 `VLMOrderingResult`

`solve` 的返回值。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `order` | `list[int]` | 解析后的排序（0-based 排列）。 |
| `raw_output` | `str` | 模型原始输出文本（`None` 归一为空串）。 |
| `prompt` | `Any` | 实际送入 `vlm_infer` 的 prompt 对象，便于调试。 |
| `panel_count` | `int` | 本次 puzzle 的 panel 数量。 |
| `texts` | `list[Optional[str]]` | 本次使用的文本列表。 |

## 4. 通用辅助函数

| 函数 | 签名 | 作用 |
| --- | --- | --- |
| `to_pil` | `(panel) -> PIL.Image` | 把 panel 归一化为 RGB PIL 图。 |
| `_to_numpy` | `(panel) -> np.ndarray` | 内部：张量转 numpy（含 `.detach().cpu()`）。 |
| `parse_torch_dtype` | `(raw_dtype) -> torch.dtype \| "auto"` | 字符串 dtype 映射到 torch dtype。 |

`to_pil` 接受的形式：

- `PIL.Image`：直接 `convert("RGB")`。
- `dict`：优先取 `padded_path`/`image_path`/`path`，其次 `image`/`padded_image`/`panel_image`；都没有则抛错。这样可以直接传 `dataset[i]["panels"][j]`。
- `str`/`Path`：`Image.open(...).convert("RGB")`。
- 张量/numpy：`CHW` 或 `HWC` 都识别（`shape[0] in (1,3,4)` 且小于最后一维视为 `CHW`）；单通道复制为 3 通道；float 输入按 [0,1] 裁剪后转 uint8；RGBA 丢弃 alpha。

## 5. 默认 prompt 生成

### 5.1 `build_ordering_text_prompt(panel_count, texts=None, *, start_index=0, ocr_header="OCR text", no_image_note=None) -> str`

生成任务引导文字，措辞对齐现有 Qwen-VL 方案（`mangazero_qwen_vl_*.py` 的 `build_ordering_prompt`）：

- 说明这是 manga panel ordering 任务，共 `panel_count` 个编号 panel（`start_index` 起）。
- 指示依据视觉连续性、阅读流、角色、对白气泡、场景转换推断阅读顺序。
- 明确要求"以逗号分隔的 panel 编号从先到后输出，不要其它内容"（纯生成式所需，原分类/回归方案是"不要输出顺序，只做内部表示"）。
- 当 `texts` 非空时，追加 `OCR text:` 块，逐行 `panel <id>: <text>`，跳过空文本。
- 当 `no_image_note` 非空时，在末尾追加该句（guess 路径用它声明"没有图像"）。

### 5.2 `build_ordering_messages(panels, texts=None, *, start_index=0, system_prompt=None, include_images=True, no_image_note=None) -> list[dict]`

默认的 `prompt_generator`，返回 chat 风格 messages：

```
[
  {"role": "user", "content": [
      {"type": "image", "image": <PIL>},   # 每个 panel 一张，按输入顺序
      ...,
      {"type": "text", "text": <build_ordering_text_prompt>}
  ]},
]
```

可选 `system_prompt` 会在最前插入 system 回合。图片在前、指令在后，便于 `apply_chat_template` 与 `process_vision_info` 直接消费。

`include_images=False` 时跳过所有图片，只保留任务文字 prompt，用于 mirage 推理（见 5.3）。`no_image_note` 会透传给 `build_ordering_text_prompt`。

### 5.3 `build_ordering_messages_mirage(panels, texts=None, *, start_index=0, system_prompt=None) -> list[dict]`

mirage 推理专用的 prompt generator：**文字 prompt 与 `build_ordering_messages` 完全一致，但不含任何图片**。它只是把 `include_images=False` 传给 `build_ordering_messages`，因此不会解码 panel，也不提示模型"图像缺失"。注意它仍按 `len(panels)` 生成"共 N 个编号 panel"的表述，调用方可传占位对象代替真实图片。

### 5.4 `build_ordering_messages_guess(panels, texts=None, *, start_index=0, system_prompt=None, no_image_note=GUESS_NO_IMAGE_NOTE) -> list[dict]`

guess 推理专用的 prompt generator：同样不含图片，但在文字 prompt 末尾**明确追加一句"没有提供 panel 图像"**（默认 `GUESS_NO_IMAGE_NOTE`），并要求模型据文本猜测最可能的阅读顺序。它等价于 `build_ordering_messages(include_images=False, no_image_note=...)`，其余任务引导文字与正常推理一致，不解码 panel。

## 6. 默认答案解析

### 6.1 `parse_ordering_answer(raw_text, panel_count, *, one_indexed=False, strict=True) -> list[int]`

把模型文本解析为排列，步骤：

1. 取文本并 `strip`（`None` 归一为空串）。
2. 优先匹配 JSON 数组（正则 `\[[^\[\]]*\]` 后 `json.loads`），失败再退化为提取所有整数。
3. 若要 `one_indexed`，所有数字减 1。
4. 去重（保序）、丢弃越界值，最多取 `panel_count` 个。
5. 若不足 `panel_count`：`strict=True` 抛 `ValueError`；`strict=False` 按升序补齐缺失下标。

兼容 `[2, 0, 1]`、`2 -> 0 -> 1`、`2, 0, 1`、换行分隔等格式。

## 7. 默认 VLM 后端

### 7.1 `QwenVLInfer(model, processor, *, max_new_tokens=64, do_sample=False, generation_kwargs=None)`

默认 `vlm_infer` 实现，`__call__(prompt) -> str`：

1. `_as_conversations` 归一化输入：既接受单会话 `[{role,...}]`（由 `build_ordering_messages` 产出），也接受批量 `[[{...}],[{...}]]`。
2. 对每个会话 `apply_chat_template(tokenize=False, add_generation_prompt=True)`。
3. `_prepare_inputs` 组装输入：有 `qwen_vl_utils` 时走 `process_vision_info` 提取图像，否则退化纯文本；`return_tensors="pt"`, `padding=True`。
4. 把张量搬到模型设备，`model.generate(**inputs, **generation_kwargs)`。
5. 按 `input_ids` 长度裁掉 prompt 部分，`batch_decode(skip_special_tokens=True)`，返回第一条文本。

## 8. 通用包装器

### 8.1 `VLMOrderingSolver`

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `vlm_infer` | 必填 | 推理后端。 |
| `prompt_generator` | `build_ordering_messages` | prompt 组装。 |
| `mirage_prompt_generator` | `build_ordering_messages_mirage` | mirage 推理的 prompt 组装（同样的文字、不含图片）。 |
| `guess_prompt_generator` | `build_ordering_messages_guess` | guess 推理的 prompt 组装（不含图片，并声明没有图像）。 |
| `answer_generator` | `parse_ordering_answer` 包装 | 答案解析。 |
| `panel_count` | `None` | 期望 panel 数；设定后不符会抛错。 |
| `one_indexed` | `False` | 传给默认答案解析。 |
| `strict_answer` | `True` | 传给默认答案解析；解析不出完整排列则抛错。 |

### 8.2 `solve(panels, texts=None) -> VLMOrderingResult`

1. `panels` 转 list，空则抛 `ValueError`。
2. 校验 `panel_count` 一致性。
3. `prompt = prompt_generator(panels, texts)`。
4. `raw = vlm_infer(prompt)`。
5. `order = answer_generator(raw, panel_count)`。
6. `_validate(order, panel_count)`：必须是完整排列，否则抛 `ValueError`。
7. 返回 `VLMOrderingResult`。

### 8.3 `mirage_inference(panels, texts=None) -> VLMOrderingResult`

"海市蜃楼"式推理，用于探测模型在**看不到图片**时能靠文字先验走多远。与 `solve` 完全一致，唯一区别是给模型的 prompt 不携带任何图片：仍然照常组装任务文字 prompt（有 OCR 文本时一并给出），**不提示模型图像缺失**，取模型输出、解析答案、校验排列的逻辑与正常推理没有任何差别。

1. `panels` 转 list，空则抛 `ValueError`。
2. 校验 `panel_count` 一致性。
3. `prompt = mirage_prompt_generator(panels, texts)`（默认即 5.3，同样的文字、无图片；`panels` 只用于计数，可为占位对象）。
4. `raw = vlm_infer(prompt)`。
5. `order = answer_generator(raw, panel_count)`。
6. `_validate(order, panel_count)`。
7. 返回 `VLMOrderingResult`。

当构造时未提供 `mirage_prompt_generator` 时会回退到 `prompt_generator`（含图片），此时该方法退化为普通 `solve`。

### 8.4 `guess_inference(panels, texts=None) -> VLMOrderingResult`

与 `mirage_inference` 流程一致（同样不给模型图片），唯一区别在文字 prompt：**明确告知模型"没有提供图像"并要求其据文本猜测最可能的顺序**。取模型输出、解析答案、校验排列的逻辑与正常推理完全相同。

1. `panels` 转 list，空则抛 `ValueError`。
2. 校验 `panel_count` 一致性。
3. `prompt = guess_prompt_generator(panels, texts)`（默认即 5.4，不含图片且带"无图像"说明；`panels` 只用于计数，可为占位对象）。
4. `raw = vlm_infer(prompt)`。
5. `order = answer_generator(raw, panel_count)`。
6. `_validate(order, panel_count)`。
7. 返回 `VLMOrderingResult`。

当构造时未提供 `guess_prompt_generator` 时会回退到 `prompt_generator`（含图片），此时该方法退化为普通 `solve`。

与 `mirage_inference` 的差异只有 prompt 文字：mirage 不提及图像缺失（考察模型的隐式先验），guess 显式说明没有图像（考察模型被"告知无图"时的表现）。

### 8.5 `train(...)` 抽象方法

```
train(batches, *, epochs=1, optimizer=None, device=None,
      grad_accum_steps=1, grad_clip_norm=0.0, log_every=0) -> dict[str, float]
```

- 基类默认 `raise NotImplementedError`：通用 prompt/infer/answer 路径是**纯推理**的，训练由子类实现。
- 约定 `batches` 为 batch 可迭代对象（或"按 epoch 返回可迭代对象"的可调用对象），batch 含 `panel_images`、`target_order`，可选 `dialog_texts`。返回至少含 `{"loss": ...}` 的指标字典。

### 8.6 `_validate(order, panel_count)`

静态方法，校验 `order` 是 `0..panel_count-1` 的排列。

## 9. Qwen2.5-VL 具体实现

实现类 `QwenVLOrderingSolver(VLMOrderingSolver)`：**不加任何解码头**，直接让 VLM 输出排序文本；训练用 LoRA 微调，标签是正确排序的字符串（如 `2, 0, 1, 3`）。

### 9.1 构造：`__init__`

关键参数：`model_name="Qwen/Qwen2.5-VL-3B-Instruct"`、`panel_count`、`torch_dtype`、`attn_implementation`、`min_pixels`/`max_pixels`、LoRA 组（`use_lora`、`lora_r`、`lora_alpha`、`lora_dropout`、`lora_target_modules`）、`load_in_4bit`、`device_map`、`start_index`、`max_new_tokens`、`generation_kwargs`、`strict_answer`，以及可覆盖的 `prompt_generator`/`answer_generator`。

流程：

1. 加载 `AutoProcessor`（按 `min_pixels`/`max_pixels` 配置）与 `Qwen2_5_VLForConditionalGeneration`（`torch_dtype` 经 `parse_torch_dtype`；可选 4bit `BitsAndBytesConfig`；可选 `device_map`），关闭 `use_cache`。
2. `use_lora=True` 时 `get_peft_model` 套 `LoraConfig(TaskType.CAUSAL_LM)`；4bit 先 `prepare_model_for_kbit_training`。`use_lora=False` 时冻结全部 backbone 参数（此时只适合推理或另行训练）。
3. 调 `super().__init__`，其中：
   - `vlm_infer = QwenVLInfer(self.backbone, self.processor, max_new_tokens, generation_kwargs)`；
   - `prompt_generator = self._default_prompt_generator`（除非外部传入）；
   - `mirage_prompt_generator = self._default_mirage_prompt_generator`（除非外部传入）；
   - `guess_prompt_generator = self._default_guess_prompt_generator`（除非外部传入）；
   - `answer_generator` 交由基类按 `strict_answer` 生成；
   - `one_indexed = bool(start_index)`：用 `start_index` 同时控制 prompt 编号与答案解析的 1-index 偏移。

### 9.2 `_default_prompt_generator(panels, texts)`

直接调用 `build_ordering_messages(panels, texts, start_index=self.start_index)`，即基类的默认 prompt 组装，把每个 panel 转 PIL 后与任务指令拼成 messages。

### 9.3 `_default_mirage_prompt_generator(panels, texts)`

对应 mirage 路径：调用 `build_ordering_messages_mirage(panels, texts, start_index=self.start_index)`，文字与正常推理完全一致（含 OCR 块），只省去所有图片。因此该类的 `mirage_inference` 在传给 processor 时 `images=None`（经 stub 验证），模型的 generate/parse/validate 链路与 `solve` 复用同一条。

### 9.4 `_default_guess_prompt_generator(panels, texts)`

对应 guess 路径：调用 `build_ordering_messages_guess(panels, texts, start_index=self.start_index)`，同样不含图片，并在文字 prompt 末尾追加"没有提供图像"的说明。该类 `guess_inference` 传给 processor 的 `images=None`，其余链路与 `solve` 复用同一条。

### 9.5 `_chat_messages(panels, texts, order=None)`

训练专用：先取 `prompt_generator(panels, texts)` 得到 user 消息；若给了 `order`，追加一条 assistant 消息，内容为 `format_order_label(order)`，即标签字符串 `2, 0, 1, 3`。这是"标签 = 正确排序字符串"的落点。

### 9.6 `_encode_messages(messages)`

把完整会话（含 assistant 标签）编码为模型输入：`apply_chat_template(tokenize=False, add_generation_prompt=False)`，有 `qwen_vl_utils` 时用 `process_vision_info` 取图，`process` 返回 `return_tensors="pt"` 的 `input_ids` 等。

### 9.7 `compute_loss(panel_images, dialog_texts=None, target_order=None) -> Tensor`

每个样本独立编码（因 Qwen-VL 图像 token 变长，逐样本编码最稳）：

1. 把 `panel_images[i]` 逐个 `to_pil`，取该样本的 `dialog_texts[i]` 与 `target_order[i]`。
2. 编码两遍：仅 prompt 的 `prompt_inputs`、含标签的 `full_inputs`。
3. `labels = full_inputs["input_ids"].clone()`，把 `[:prompt_len]` 置 `-100`，使损失只落在答案 token 上。`prompt_len` 由 `common_prefix_length` 比较两次编码的公共 token 前缀得到（而非直接信 prompt 长度），避免 chat template 在尾部处理不一致导致 mask 偏移。
4. `self.backbone(**full_inputs, labels=labels)` 取 `outputs.loss`。
5. 对 batch 内各样本 loss 取均值。

### 9.8 `train(...)`

实现基类抽象方法：

1. 缺省 `device` 取 backbone 参数设备；缺省 `optimizer` 建 `AdamW(可训练参数, lr=1e-4)`。
2. 记录原训练态并 `backbone.train()`。
3. 逐 epoch 取批次（`batches(epoch)` 或直接迭代），搬 `panel_images`/`target_order` 到设备，`dialog_texts` 原样传递。
4. `loss = compute_loss(...) / grad_accum_steps`，`backward()`；每 `grad_accum_steps` 步做 `clip_grad_norm_`（可选）+ `optimizer.step()` + `zero_grad`。
5. 收尾补一次未满累积的 step；恢复原训练/评估态。
6. 返回 `{"train_loss": 平均 loss, "steps": 步数}`。

`_trainable_parameters()` 返回 `backbone` 中 `requires_grad=True` 的参数，即 LoRA 适配器参数。

### 9.9 `solve(panels, texts=None)`

覆写并加 `@torch.no_grad()`：先 `backbone.eval()`，再调 `super().solve(...)`，即复用基类的 prompt -> infer -> answer 全链路。因为后端是 `QwenVLInfer`，这里就是"生成排序文本 -> `parse_ordering_answer` 解析成排列"。

### 9.10 检查点：`trainable_state_dict` / `load_trainable_state_dict`

- `trainable_state_dict()`：优先 `peft.get_peft_model_state_dict`，无 peft 时回退取名字含 `lora_` 的权重，返回 `{"backbone": <LoRA 权重>, "model_name": ...}`。只存 LoRA adapter，不存 3B 全量权重。
- `load_trainable_state_dict(state_dict)`：优先 `set_peft_model_state_dict`，否则 `load_state_dict(strict=False)`。

### 9.11 `format_order_label` 与 `common_prefix_length`

- `format_order_label(order, *, start_index=0)`：把排列渲染为 `", ".join(...)`，例如 `[2,0,1,3] -> "2, 0, 1, 3"`，是训练标签字符串的唯一来源。
- `common_prefix_length(prefix, full)`：逐 token 比较两段序列，返回公共前缀长度，用于 label mask。

## 10. 使用示例

推理（纯生成式）：

```python
from Solver.agent.qwen_vl_ordering_solver import QwenVLOrderingSolver

solver = QwenVLOrderingSolver(model_name="Qwen/Qwen2.5-VL-3B-Instruct", panel_count=6)
result = solver.solve(panels, texts=None)   # texts 传 OCR 结果或 None
print(result.order, result.raw_output)
```

训练（LoRA，标签为正确排序字符串）：

```python
from torch.utils.data import DataLoader
from Solver.env.mangazero_panel_env import MangaZeroPanelOrderingDataset, collate_panel_ordering_batch

dataset = MangaZeroPanelOrderingDataset("Mangazero/ordering_dataset", image_size=(224, 224))
loader = DataLoader(dataset, batch_size=1, collate_fn=collate_panel_ordering_batch)

solver = QwenVLOrderingSolver(model_name="Qwen/Qwen2.5-VL-3B-Instruct", panel_count=6)
metrics = solver.train(loader, epochs=3, grad_accum_steps=8, grad_clip_norm=1.0, log_every=10)
```

对接数据集时，`panels` 直接传 `dataset[i]["panels"]`，`texts` 传 `dataset[i]["dialog_texts"]`（两者字段含义见 panel_ordering_fields.md）。

mirage 推理（只给文字、不给图，用于评估纯文本先验的上限）：

```python
mirage = solver.mirage_inference(panels, texts=None)
print(mirage.order, mirage.raw_output)   # 与 solve 同样返回 VLMOrderingResult
```

guess 推理（不给图，并明确告知模型没有输入图像）：

```python
guess = solver.guess_inference(panels, texts=None)
print(guess.order, guess.raw_output)
```

## 11. 扩展新 VLM

继承 `VLMOrderingSolver`，在 `__init__` 里把新后端包装成 `vlm_infer` 调 `super().__init__`，并按需覆写：

- `prompt_generator`：若该 VLM 的图文顺序/模板不同，改这里。
- `mirage_prompt_generator`：若该 VLM 的无图 prompt 组装方式不同，改这里；缺省用基类的纯文字版。
- `guess_prompt_generator`：若该 VLM 声明"无图"的措辞不同，改这里；缺省用基类的默认提示。
- `answer_generator`：若输出格式不同，改这里。
- `train`：接入该 VLM 的微调方式（LoRA/全参），返回 `{"loss": ...}`。

`QwenVLOrderingSolver` 就是这一模式的参考实现。
