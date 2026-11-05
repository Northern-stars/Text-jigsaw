# LVLM Panel Reordering：四项研究问题的代码实验方案

> RQ1 的通用 VLM 推理接口已接入 `experiment/RQ_1/model_adapters.py`，并提供统一比较入口 `experiment/RQ_1/rq1_model_comparison.py`。本方案中其余尚未实现的 runner/接口仍属于后续计划；已实现状态以各 `experiment/RQ_n/README.md` 为准。

## 1. 实验方法与代码现状

| 方法 | 当前实现 | 适合回答的问题 | 需要的工作 |
| --- | --- | --- | --- |
| Qwen2.5-VL zero-shot | `Solver/agent/qwen_vl_ordering_solver.py`；RQ1 adapter 为 `experiment/RQ_1/model_adapters.py` | Baseline；RQ3 的未微调参照 | `experiment/RQ_1/rq1_model_comparison.py` 已提供统一推理 runner；固定模型 ID、revision、processor 与解码参数；不替代 RQ4 的同 checkpoint 主比较 |
| Qwen2.5-VL 生成式 LoRA | `QwenVLOrderingSolver` 已有 `compute_loss`、`train`、`solve`，标签是排序字符串 | RQ1–RQ4 主方法；同一 checkpoint 支持 prompt 对比 | 新建 `Solver/train_qwen_vl_ordering.py`；接入 split manifest、数据规模采样和可切换 prompt |
| Qwen2.5-VL permutation classifier | `Solver/train_mangazero_qwen_vl_lora.py` + `Solver/agent/mangazero_qwen_vl_lora_classifier.py` | 附加排序架构基线；可补做 RQ1/RQ2 | 已使用统一 manifest split；与主方法分开报告 |
| Qwen2.5-VL autoregressive pointer | `Solver/train_mangazero_qwen_vl_autoregressive.py` + `Solver/agent/mangazero_qwen_vl_autoregressive.py` | 附加排序架构基线；可补做 RQ1/RQ2 | 已使用统一 manifest split；与主方法分开报告 |
| Made2Order | `Solver/train_made2order_panel_ordering.py` + `Solver/agent/made2order_set2order_solver.py` | 纯视觉排序基线；RQ1/RQ2 可作为架构参照 | 已使用统一 manifest split；输入仍为图像，不能参加文本-only 或 prompt 条件比较 |

推荐的主结论来自 **Qwen2.5-VL 生成式 LoRA**，因为它能在同一模型上施加数据、风格、模态和 prompt 干预。其余模型是架构参照，不应把不同输出头的差异解释为同一 RQ 中的 prompt 或模态因果效应。仓库目前训练脚本默认 `Qwen/Qwen2.5-VL-3B-Instruct`；主实验须通过 `--model-name` 指定准备使用的 Qwen2.5-VL checkpoint，并在所有 run 中固定该 ID、revision 和 processor 配置。

## 2. 共享代码结构

### 2.1 统一 work-level split manifest

新增 `scripts/build_panel_ordering_experiment_manifest.py`，输入一个或多个现有 ordering 数据目录，读取每个目录的 `manifest.jsonl` 和对应 `sample.json`，不复制图片，只生成实验索引。每行至少包含：

```json
{
  "dataset": "mangazero",
  "style": "manga",
  "work_id": "作品标识",
  "page_key": "页面标识",
  "window_id": "规范化后的 panel 集合标识",
  "sequence_id": "样本标识",
  "sample_path": "相对源数据目录的 sample.json 路径",
  "split": "train",
  "panel_count": 6,
  "text_source": "dialog_text"
}
```

`work_id` 的提取规则：Manga109 用 `manga_id`（book name）；MangaZero 用 `manga_id`；COMIX Hugging Face 用 `manga_id`（当前来自 `book_id`）；旧版 COMICS 用 `comic_no`。清单另存 `dataset` 与 `canonical_work_id`；默认 canonical ID 使用 source 前缀，跨数据源可能是同一作品时通过 alias 表合并，避免重复作品跨集合泄漏。缺失 work ID 时直接报错，不回退成 page 或 puzzle ID。按 `canonical_work_id` 分配 train/val/test，保存固定 seed 与每个集合的 work ID 清单；同一 work 的所有页、窗口和 shuffle 版本必须在同一 split。

当前 `dataset_split_utils.py` 与若干生成器以页为单位填 split quota，不能保证同一作品不跨集合。训练/评测代码现已统一读取既有 `train` / `val` / `test` split，不再随机按 puzzle 二次划分；但这些 split 仍可能只是 page-level。后续实验 manifest 要按 canonical work ID 重建 work-level split。新 manifest 的 `window_id` 用同一页面内按 `panel_id` 排序后的 panel ID 生成，以便识别同一原始窗口的多种 shuffle 版本；测试/验证每个窗口只保留预先指定的一个 puzzle，训练时由现有在线随机置换增强。

### 2.2 共享 Dataset 与 batch

扩展 `Solver/env/mangazero_panel_env.py`，或新增 `Solver/env/comic_panel_ordering_dataset.py` 提供 `ComicPanelOrderingDataset`。它从统一实验 manifest 读取样本，并返回现有 key 加上实验元数据：

- 已有：`sequence_id`、`panel_images`、`target_order`、`dialog_texts`、`panels`。
- 新增：`dataset`、`style`、`work_id`、`page_key`、`window_id`、`split`、`text_source`。
- 可选：`layout_features`；布局盲实验必须关闭，并且调用模型时不能传 bbox、page index 或原页面图。

兼容保留 `MangaZeroPanelOrderingDataset` 的现有行为，旧 checkpoint 和已有数据目录仍可读取。统一 collate 继续调用 `collate_panel_ordering_batch` 的 tensor/list 约定，并增加元数据 list；所有 solver 从同一 `DataLoader` 和相同样本 ID 集合取数据。图片路径由 manifest 中的数据源配置与 `sample_path` 解析，避免把不同数据集文件复制到一个目录。

### 2.3 统一训练与评估入口

- 新建 `Solver/train_qwen_vl_ordering.py`：复用 `QwenVLOrderingSolver.train()`；每个 epoch 调 `Solver.train_mangazero_panel_ordering.randomize_training_batch()` 的等价共享函数，保证图像、OCR 文本、panel 元数据与 `target_order` 同步置换。
- 已修改 Made2Order、Qwen 分类/指针、set-to-sequence 和 jigsaw 训练入口，通过统一 Dataset 加载已有 split；后续新增 `--experiment-manifest` 以支持跨数据集 work-level split。
- 新建 `Solver/evaluate_panel_ordering_experiment.py`：按 manifest 顺序评估同一 test IDs，写逐样本 JSONL 和汇总 JSON。逐样本记录 `run_id`、方法、模型/checkpoint、RQ 条件、`sequence_id`、`work_id`、`style`、target/prediction、raw output、解析是否成功、各项指标和错误类别。
- 所有训练 checkpoint 保存模型 ID、revision、manifest hash、train work ID、seed、样本数、optimizer steps、训练 token 或可得的计算统计、图像/OCR/prompt 配置。测试集只在选定 checkpoint 后跑一次正式汇报；训练中不据 test 结果选 epoch。

## 3. 四项 RQ 的实验实现

### RQ1：训练数据规模

**目的：**测量 comic-domain 训练 puzzle 增加时的性能曲线和边际增益。

1. 用统一 manifest 先固定 work-level train/val/test，再仅从 train works 构造嵌套子集，例如 5k / 10k / 20k / 40k unique `window_id`。若数据量不够，按可用数量选几何递增档位并报告实际数；验证/测试集合不变。
2. 每个规模在 manga 与 comics 子域内固定比例，按 work 均衡抽样，避免大作品或高窗口页数支配子集。不同规模的子集必须嵌套，至少跑 3 个 seed；val/test puzzle IDs 对所有规模完全一致。
3. 主方法用同一个 Qwen2.5-VL 生成式 LoRA 配置逐档训练；固定 LoRA、优化器、batch、图像分辨率、epoch 数和数据顺序规则。固定 epoch 时 optimizer steps 随样本量增加，应报告实际 steps/tokens/FLOPs（若能测），并把 RQ1 解释为固定每样本训练轮数下的数据扩展效应。
4. 可选的计算敏感性对照：再设一个固定 optimizer-step 预算，所有规模训练相同步数、对小集合循环采样。与主曲线分开报告，不混在一个结论里。
5. Made2Order 可作为可选的纯视觉规模曲线，但另用其自身训练配置和指标表，不与 Qwen 规模曲线合并。

**代码落点：**`scripts/build_panel_ordering_experiment_manifest.py` 产出 work/split/window 索引；新 Qwen runner 增 `--train-window-limit`、`--sampling-seed`、`--experiment-manifest`；Qwen checkpoint 与 `metrics.json` 写出实际窗口 ID 和 optimizer steps。现有 Qwen classifier/pointer 若列为附加结果，也必须读同一窗口子集和 split。

### RQ2：预算匹配的风格平衡训练

**目的：**比较风格均衡混合是否提升风格内留出作品表现及对 held-out comic works 的泛化。

1. 在 train works 构造三组：manga-only、comics-only、manga:comics = 1:1。COMICS 与 COMIX 都可标为 comics 风格，但数据集来源仍保留，结果需分别分层；若其标注/图像处理差异太大，先做 Manga109/MangaZero vs COMIX，再把旧版 COMICS 作为敏感性结果。
2. 三组使用完全匹配的训练 puzzle/window 数、有效训练 token 数（按目标输出 token 及 OCR 输入 token 计）、optimizer steps、batch/gradient accumulation、epoch 或步数定义、图像尺寸、LoRA 配置和随机种子。混合组按 style、再按 work 均衡抽样。避免为了补齐样本而重复单个 puzzle；不足时将三组统一降到共同可用预算。
3. 主测试是相同 style、不同 work 的 within-style held-out 测试。额外的 cross-style 测试单独命名为 unseen-style transfer，避免与 held-out works 混称。test work IDs 在训练前固定且未用于选参。
4. 在同一 Qwen2.5-VL 生成式 LoRA 训练与评测 runner 下跑三组；3 个 seed 起步。Made2Order 可作为纯视觉补充组，但其输入模态和训练 compute 定义不同，应单独呈现，不用于 Qwen 混合训练的直接预算匹配结论。

**代码落点：**manifest 为每个 sample 写 `style`、`work_id`、`split`；新 Dataset 支持 `style_filter` 和按 `work_id` 的均衡采样器（建议新建 `Solver/env/work_balanced_sampler.py`）；新 Qwen runner 参数为 `--train-style {manga,comics,mixed}`、`--train-puzzles`、`--max-train-tokens`、`--seed`。日志写入每风格/每来源有效样本数、work 数、实际 steps 和 token 数。

### RQ3：layout-blind 证据模态消融

**目的：**在同一批 puzzle 上估计视觉证据、画内文字和二者组合的贡献，同时不让原始页面位置/版面泄露顺序。

| 条件 | Qwen 输入 | 数据/solver 路径 |
| --- | --- | --- |
| Visual-only | panel 图像；所有嵌入文字区域遮挡；不附 OCR | `mask_inference` 目标路径 |
| Text-only | 无 panel 图像；逐 panel OCR 转录按输入 panel ID 提供 | `ocr_only_inference` 目标路径 |
| Visual + text | 原始 panel 图像 + 与 panel ID 对齐的 OCR 转录 | `solve(panels, texts=...)` |

1. 在 `Mangazero/build_dataset.py`、`Manga109/create_ordering.py`、`comix/comix_pipeline.py`、`comix/create_panel_ordering_dataset.py` 样本中补齐可投影到 padded panel 图的 `dialog_bboxes_padded`；Manga109 当前无 bbox 输出，需要从 XML `<text>` 框关联到 frame。可依照 [Panel 对话框 bbox 与 Solver mask / OCR-only 设计](panel_dialog_bbox_and_mask_mode_design.md) 实施。
2. 在 `Solver/agent/vlm_ordering_solver.py` 与 `Solver/agent/qwen_vl_ordering_solver.py` 增加 mask/OCR-only message builder 与 inference method，并在 `VLMOrderingResult` 保留 mode、text source、mask missing 等诊断字段。掩码缺失不能静默当成视觉-only；标记后剔除或单独分层报告。
3. OCR-only 主分析优先用人工校验转录；若只具备现有 OCR `dialog_text`，结果标记为 OCR-derived text-only，报告 OCR 覆盖率/错误率，并对人工校验子集做敏感性分析。COMICS OCR CSV、Manga109 XML 文本、MangaZero/COMIX OCR 来源分别统计，不把来源差异藏在合并准确率里。
4. 对所有条件复用同一 test `window_id`、同一输入 panel 顺序、同一个 Qwen checkpoint、同一解码配置。模型只能看到独立 panel；不传页面图、bbox、坐标、`page_index`、reading-order 或 `original_window_index`。OCR 文本按照输入槽对应 panel 呈现，绝不按正解顺序排序。
5. Made2Order 可提供图像排序的外部参照，但它无法切换到 text-only 或 image+OCR 条件，所以不纳入三条件主消融的统计对照。

**代码落点：**数据字段改动如上；Solver 接口按 `panel_dialog_bbox_and_mask_mode_design.md`；统一 evaluator 加 `--modality {visual_only,text_only,visual_text}`，三个条件写成配对 `run_id`。评测前做三项断言：遮罩后目标文字区域像素有变化；Text-only prompt 不含 image block；三种条件的 `sequence_id` 集合和 target 顺序完全一致。

### RQ4：固定 checkpoint 的 prompt 策略

**目的：**分离提示策略本身对排序准确率、格式有效率和错误分布的影响。

1. 只选用一个已训练好的 Qwen2.5-VL 生成式 LoRA checkpoint（主分析建议 RQ1 最大规模、RQ2 混合预算模型之一；在看 test 前固定），三策略都加载完全相同 adapter 和 processor。
2. 在 `Solver/agent/vlm_ordering_solver.py` 增加三种 prompt generator：`direct`（只要求输出排列）、`cot`（先分析线索后给排列）、`structured_evidence`（按 panel ID 输出角色/动作/场景/文本等可观察证据，最后输出排列）。结构化字段不能要求模型访问 panel 坐标或页面布局。
3. 扩展 `Solver/agent/qwen_vl_ordering_solver.py` 的构造/runner，让 prompt 可按 `prompt_profile` 切换而不改变 checkpoint。新增评估脚本按完全相同的 puzzle 顺序依次运行三个 profile；固定 `max_new_tokens` 上限、`do_sample=False`、温度/采样参数、重试次数和解析规则。记录每个 profile 的实际生成 token 数，确保相同预算上限；CoT 不能因重试或二次调用获得额外预算。
4. evaluator 必须保留原始输出；解析失败单独记为 invalid，不用宽松补齐后的排列计算主准确率。主指标报 exact sequence、pairwise、合法排列率；错误画像在一个预先抽样并由人工标注的 test 子集上分类：局部相邻倒置、首尾/转场、角色/动作连续性、文本误用、遗漏/重复 panel、格式/解析错误。标注人员看不到 prompt 条件，至少双人标注一部分并报告一致性。
5. Made2Order、classifier head 和 pointer decoder 不进入该 RQ；它们没有等价的自然语言推理 prompt 接口。

**代码落点：**在 `vlm_ordering_solver.py` 实现纯 prompt builder 和严格 parser；在 `qwen_vl_ordering_solver.py` 注入 profile；新增 `Solver/evaluate_qwen_vl_prompt_profiles.py`，输出同一 checkpoint 的逐样本三行记录及配对统计。

## 4. 数据生产链路与字段对接

| 数据集 | 当前 ordering 代码 | 作品 ID | 需处理的问题 |
| --- | --- | --- | --- |
| MangaZero | `Mangazero/build_dataset.py` | `manga_id` | 现有 manifest split 是 page/quota 级；实验 manifest 重按作品分组。OCR 来源是模型识别；补 padded dialog bbox。 |
| Manga109 | `Manga109/create_ordering.py` | `manga_id` / book name | 现有 split 只避免同页跨集合，不保证整部书隔离；当前 `dialog_bboxes` 为空，需解析 XML text bbox。 |
| COMIX (HF) | `comix/comix_pipeline.py`；入口 `download_raw.py`、`build_ordering_dataset.py` | `manga_id`（由 `book_id` 得到） | 当前 split 避免同页泄漏，不保证 held-out works；OCR/textbox 来自检测或 OCR 流程，补 bbox 并记录来源。 |
| COMICS (legacy) | `comix/create_panel_ordering_dataset.py` | `comic_no` | 当前只写合并 manifest、样本顶层有 `comic_no`；生成统一 work manifest 时按 comic_no 划分并补齐 style/work 元数据。 |

数据生成时 `target_order[k]` 表示排序第 k 位应选择哪个输入槽。数据读取、在线 shuffle 和 evaluator 必须沿用该语义；当前 `randomize_training_batch()` 已负责重建规范序列后同步置换图像、标签、文本和 panel metadata，可抽到共享模块避免不同训练脚本各自实现。输入 shuffle 不应泄露 canonical panel index、原始 window index 或原阅读坐标给模型。

## 5. 推荐实施顺序

1. 写 split-manifest 生成器并对四条数据生产链路做 work ID 覆盖率检查；输出 work、page、window 计数和 train/val/test 交集断言。
2. 改造 Dataset/Collate，使 `split=train|val|test` 真正过滤 manifest；验证各方法读到完全相同的 sample IDs。
3. 接入/新增 Qwen 生成式 LoRA runner 与通用 evaluator；先用极小数据跑一次 train/eval/checkpoint round-trip。
4. 完成 zero-shot、Made2Order、classifier、pointer 基线；保存基线配置，正式主表使用 Qwen2.5-VL 生成式 LoRA。
5. 跑 RQ1 scale sweep，再跑 RQ2 三种训练构成；根据资源预算冻结最佳 checkpoint 选择规则。
6. 完成 bbox/mask/OCR-only 数据和 Solver 接口，跑 RQ3 配对消融。
7. 冻结 RQ4 checkpoint 后跑三 prompt；最后执行盲标注错误画像与配对统计。

## 6. 建议的实验产物

```text
experiments/panel_reordering/
  manifests/work_split_seed_42.jsonl
  configs/model_and_data.yaml
  checkpoints/<run_id>/
  predictions/<run_id>.jsonl
  metrics/<run_id>.json
  analysis/rq1_scaling.csv
  analysis/rq2_style_balance.csv
  analysis/rq3_modalities.csv
  analysis/rq4_prompt_profiles.csv
```

正式汇总应同时给出样本级配对差异和 work-level bootstrap 置信区间，避免大量同作品 puzzle 被当作完全独立观测。保存数据 manifest、split seed、模型 revision、运行配置和逐样本预测后，应能从原始结果重算表格与图。

## 7. 与相关文档的关系

- [LVLM 干预实验提案](lvlm_panel_reordering_intervention_proposal.md)：定义 RQ 和科学问题。
- [Panel Ordering 字段说明](panel_ordering_fields.md)：定义现有 ordering sample 与 `target_order` 语义。
- [VLM Ordering Solver 接口](vlm_ordering_solver_interface.md)：定义通用 VLM prompt/infer/parse API。
- [Panel 对话框 bbox 与 Solver mask / OCR-only 设计](panel_dialog_bbox_and_mask_mode_design.md)：定义 RQ3 所需文本区域遮罩和 OCR-only 扩展。
- [README](../README.md)：列出现有训练、评测命令与 Made2Order 用法。
