# Manga Ordering 研究提案与项目 TODO

相关论文：EACL *CHROMIC: Chronological Reasoning Across Multi-Panel Comics*。

## 0. 当前进度

- 数据集收集基本完成：日漫（Manga109 / MangaZero）、美漫（COMICS / COMIX）均有本地 raw 与 ordering 构建脚本。
- 当前 solver：QwenVL + LoRA（classifier / autoregressive 两种 head），另有独立 Made2Order 视觉 transformer solver。
- 尚未落地：专项 SFT 数据规模实验、风格混合训练、推理策略对比（CoT / 结构化 CoT）。
- Human baseline 与数据集难度评估尚未设计。

## 1. 研究提案 statement

研究方向从「提升漫画理解能力」收窄为：**通过受控干预，研究训练数据规模、风格多样性和推理策略如何影响 LVLM 的漫画面板排序能力**。核心不是提出更好的排序模型，而是系统研究哪些训练与推理干预能够改善排序、这些改善在什么条件下成立、哪些缺陷仍然存在。

完整提案见 [lvlm_panel_reordering_intervention_proposal.md](lvlm_panel_reordering_intervention_proposal.md)。

## 2. Research Questions

- **RQ1 — Data Scaling**：增加漫画训练数据规模，能否持续提升 Panel Reordering 表现？提升是否存在边际收益递减？
- **RQ2 — Style Diversity**：在相同训练预算下，日漫与美漫混合训练能否改善跨风格泛化？是否存在对某一风格的性能权衡？
- **RQ3 — Inference Strategies**：CoT 与结构化推理提示能否进一步改善排序？它们的收益是否会随数据规模、漫画风格或文本线索可用性而变化？

## 3. 实验计划

### 3.1 Baseline（零样本）

- [ ] 用开源 LVLM（Qwen2.5-VL 等）在 Panel Reordering 上跑零样本排序。
- [ ] 建立统一评测协议：Exact sequence accuracy、Pairwise ordering accuracy，按日漫 / 美漫 / 跨风格切分报告。

### 3.2 Data Scaling

- [ ] 从日漫 ordering 数据中按作品级划分抽取 5k / 10k / 20k / 40k 训练子集。
- [ ] 用相同超参数分别做 SFT（QwenVL + LoRA），记录各规模 checkpoint 的排序指标。
- [ ] 绘制数据规模-准确率曲线，判断边际收益递减是否存在。

### 3.3 Style mixing

- [ ] 固定总样本量和训练步数，构造「纯日漫」「纯美漫」「日漫:美漫 = 1:1（按作品数均衡）」三种训练集。
- [ ] 在日漫、美漫、跨风格测试集上分别评估，报告风格权衡。

### 3.4 Inference strategies

- [ ] 实现三种 prompt：直接输出排序、普通 CoT（先解释再排序）、结构化 CoT（逐格列出关键线索后排序）。
- [ ] 对同一组 checkpoint 做三种推理策略对比，解析输出为合法排序后计算指标。
- [ ] 结合 mask / OCR-only 辅助条件，分析文本线索可用性是否调节推理策略收益。

## 4. 数据与评测

### 4.1 数据构成

- 日漫：Manga109、MangaZero（`Mangazero/build_dataset.py` 产出 ordering puzzle）。
- 美漫：COMICS（comics 旧集）、COMIX（Hugging Face pages，`comix/` 产出 ordering puzzle）。

### 4.2 作品级划分

- [ ] 训练 / 验证 / 测试按作品划分，避免同一作品跨集合泄露。
- [ ] 分别报告日漫、美漫、跨风格三类测试结果。

### 4.3 辅助实验

- [ ] mask / OCR-only 作为辅助条件，用于解释推理策略收益来源（设计见 [panel_dialog_bbox_and_mask_mode_design.md](panel_dialog_bbox_and_mask_mode_design.md)）。
- [ ] 文本线索可用性（有无 OCR）作为调节变量纳入 Inference 分析。

## 5. 预期贡献

- 提供漫画专项 SFT 数据规模效应的实证结果。
- 分离训练样本数量与风格多样性对排序能力及跨风格泛化的影响。
- 评估推理提示策略在漫画排序中的有效性及其局限。

即使 CoT 无效、性能饱和、或混合风格只对部分测试集有效，仍可形成有解释力的研究结论。

## 6. 优先级

1. 建立零样本 Baseline 与统一评测协议。
2. 完成 Data Scaling 实验谱系。
3. 完成 Style mixing 实验。
4. 实现并比较三种推理策略。
5. 补充 mask / OCR-only 辅助分析。
6. 设计 human baseline 与数据集难度评估（可选增强）。

## 7. 开放问题

- Scaling 实验的最小/最大数据规模如何确定，是否受 OCR 质量影响。
- 日漫与美漫的样本难度差异如何控制，避免风格权衡被难度混淆。
- CoT 输出解析失败率是否成为第二种混淆因素。
- Human baseline 是否纳入正式实验设计。