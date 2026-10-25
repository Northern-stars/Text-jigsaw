# MIRAGE 适用性与漫画排序模态消融研究提案

> 定位：研究 MIRAGE 所揭示的评估问题是否也存在于原生多模态叙事任务中，并通过漫画排序中的模态消融，区分**语言捷径**、**视觉依赖**与**跨模态互补性**。
>
> 核心原则：重点不是预先证明 MIRAGE 正确或错误，而是建立一个能检验其适用范围（boundary conditions）的实验。

## 暂定标题

- 英文：Beyond Image Ablation: Evaluating Visual-Linguistic Integration in Natively Multimodal Comic Sequencing
- 中文：超越图像消融：评估原生多模态漫画排序中的视觉—语言整合能力

相关文档：[Manga Ordering 研究提案与项目 TODO](manga_ordering_research_todo.md)、[VLM Ordering Solver 接口](vlm_ordering_solver_interface.md)。

## 1. Research motivation

MIRAGE 指出，视觉语言模型（VLM）在常规视觉问答基准上取得较高准确率，不一定意味着它真正使用了图像。即使图像被移除，模型也可能通过语言先验、问题线索或数据集偏差完成任务。

然而，漫画与常规图像问答存在一个值得专门研究的区别：**漫画中的图像和嵌入式文字共同构成叙事内容**。人物动作、空间关系、对话、拟声词等信息可能相互补充。只测试移除图像后的表现，无法充分分辨这些不同的信息来源分别发挥了什么作用。

因此，我们计划使用漫画排序（comic-panel ordering）任务，研究以下问题：

- 模型能否只凭漫画中的文字恢复故事顺序？
- 视觉信息是否提供了超出文本信息的独立贡献？
- 视觉和文字结合后，是否能产生超出单一模态的收益？
- 当模型没有获得漫画图像时，它会承认信息缺失，还是仍然表现得像看到了图像？

## 2. Research objectives

研究的总体目标是检验 MIRAGE 式现象在原生多模态叙事任务中的适用范围，而不是简单地以另一项任务的高分或低分来推翻原论文。

- **Objective 1 — Modality dependence**：测量漫画排序在不同模态输入下的表现，判断模型在多大程度上依赖视觉和文字信息。
- **Objective 2 — Cross-modal complementarity**：判断完整漫画是否显著优于视觉单模态和文本单模态输入，并检验两种信息的联合收益。
- **Objective 3 — Mirage behavior**：检查模型在漫画图像被静默移除时，是否仍生成未经输入支持的具体视觉描述，以及明确告知图像缺失是否会改变其行为。

## 3. 实验设计

采用同一批漫画、同一批排序问题和相同的面板随机化方式，建立以下四种核心条件。

| 条件 | 图像 | 画内文字 | 输入内容及目的 |
| --- | --- | --- | --- |
| Full | 保留 | 保留 | 原始漫画，测量自然输入下的综合排序能力 |
| Visual-only | 保留 | 遮挡 | 测量移除可读文字后，视觉叙事线索能够支持多少排序 |
| OCR-only | 移除 | 以文本提供 | 测量画内文字能否独立支持排序 |
| No-image / No-OCR | 移除 | 移除 | 测量无漫画内容时的任务表现与非视觉线索影响 |

设计要求：

- OCR-only 应优先使用**人工校验的 OCR 转录**，确保文字识别错误不会成为主要混淆因素。
- 面板标签必须随机化，并与其文字内容正确对应，不能通过原始顺序、文件名或 prompt 泄露答案。
- No-image / No-OCR 条件用于建立严格的无图像对照。对于没有任何可用面板内容的排序任务，准确率理论上应接近随机基线；若模型仍声称看到了具体场景，还需要单独分析其生成内容是否缺乏证据。这里需要分别评估「具体视觉描述是否缺乏输入依据」以及「排序准确率如何变化」，不能只因为模型在无图像时答错，就判定它出现了 Mirage Effect。
- 可设置一个**辅助条件 Full + OCR**，判断显式提供 OCR 转录能否进一步提升性能。它不是主要四条件实验的一部分，因为重复提供文字会改变自然输入条件。

## 4. Evaluation metrics

主要指标：

- **Exact sequence accuracy**：完整排序是否正确。
- **Pairwise ordering accuracy**：两两面板的先后关系是否正确，补充反映排序质量。
- **Text-only retention ratio**：OCR-only 准确率相对于 Full 准确率的比例。
- **Visual gain**：完整输入相对于 OCR-only 的准确率增益。
- **Text gain**：完整输入相对于 Visual-only 的准确率增益。
- **Cross-modal interaction**：视觉和文字同时提供时，联合效应是否超出两种单模态效应的加性预期。
- **Unsupported visual description rate**：无图像条件下，模型生成具体但没有输入证据支持的视觉描述的比例。

需要特别注意：这里的 Text-only retention ratio **不应直接称为 MIRAGE 原论文的 Mirage Score**，因为 OCR-only 仍然提供了从漫画图像中提取出来的文字，与原论文的严格无图像条件并不相同。

为直接研究 Mirage behavior，可再进行一个辅助的 prompt 对照：

1. **Silent omission**：沿用原本要求模型排序漫画的指令，但静默移除全部面板。
2. **Explicit absence**：在 prompt 中明确告知模型没有提供图像，要求其据文本线索作答。

对比两种条件下的排序行为、拒答行为和具体视觉描述的生成比例。`mirage_inference`（静默无图）与 `guess_inference`（显式告知无图）已在 solver 中实现，对应这两种设置，见 [VLM Ordering Solver 接口](vlm_ordering_solver_interface.md)。

## 5. Research hypotheses

设置三个主要假设和一个辅助假设，分别测试文本捷径、视觉依赖与跨模态整合，而不是把所有结果压缩成一个「模型有没有视觉理解能力」的判断。

### H1 — Textual shortcut hypothesis

- **假设**：即使在原生多模态漫画排序任务中，嵌入式文本仍可能提供足够的线索，使模型在没有原始图像时取得高于随机水平的准确率；在部分文字线索较强的样本中，OCR-only 表现可能接近 Full。
- **对应比较**：`A_OCR-only > A_chance`，并考察 `Δ_text_retention = A_Full - A_OCR-only`。
- **预期解释**：若 OCR-only 表现较高，尤其在文字密集、对话本身包含时间或因果线索的故事中接近 Full，就说明漫画排序仍存在通过文字完成任务的可能性。这支持 MIRAGE 所强调的评估风险在此类任务中仍然存在。
- **注意**：应避免将 OCR-only 高准确率直接说成模型「没有理解视觉信息」。它只说明视觉信息并非解决这些样本的唯一途径。

### H2 — Visual evidence dependence hypothesis

- **假设**：对于依赖人物动作、空间关系和事件状态变化的漫画排序，完整漫画应显著优于 OCR-only 条件。
- **对应比较**：`Δ_V = A_Full - A_OCR-only`。
- **预期结果**：在视觉叙事线索重要的样本中，`Δ_V` 显著大于零，而且这种差异在未见过的漫画系列或故事上仍然成立。
- **预期解释**：如果完整漫画表现显著优于 OCR-only，就说明文本线索无法完全解释模型的排序成绩，视觉输入为任务提供了额外的信息。
- **注意**：这一比较单独不能证明模型进行了跨模态整合，因为视觉信息可能单独就足以完成排序。

### H3 — Cross-modal complementarity hypothesis

- **假设**：对于确实需要结合画内文字与视觉事件的样本，Full 条件应显著优于 Visual-only 和 OCR-only 条件。
- **对应比较**：`A_Full > max(A_Visual-only, A_OCR-only)`；同时计算准确率尺度下的交互项 `I = A_Full - A_Visual-only - A_OCR-only + A_No-info`。
- **预期解释**：如果在预先定义的跨模态依赖子集上，Full 显著优于两种单模态条件，并出现正向交互效应，就可以认为视觉和文字之间存在互补性的行为证据。
- **注意**：正交互效应并不等于直接证明了模型内部的跨模态推理机制，还需要结合具体错误案例和逐题结果解释。

### H4 — Mirage prompt sensitivity（辅助假设）

- **假设**：当漫画图像被静默移除时，模型更容易生成仿佛源自实际视觉观察的具体描述；明确告知图像缺失后，模型的回答方式会发生变化。
- **主要比较**：静默移除图像时模型生成未经输入支持的具体视觉描述的比例；明确告知图像缺失时同类描述的比例；两种条件下模型的排序行为、拒答行为和置信表达。
- **定位**：该假设最直接地连接 MIRAGE 原文的 Mirage mode 与 Guess mode 实验，主要研究缺失模态下的行为差异。它与 H1–H3（主要研究排序任务的可解性与信息来源）应分开报告。

## 6. 数据设计：不要让四个条件只产生四个总体分数

四个条件是实验的基础，但如果最终只报告四个总体准确率，研究贡献可能仍然有限。建议在测试之前，依据漫画的叙事特点，将样本划分为三个预先定义的类别。

| 子集 | 主要信息来源 | 用途 |
| --- | --- | --- |
| Text-dominant | 对话包含明显的时间、因果或事件顺序线索 | 检验 H1 |
| Visual-dominant | 人物动作、空间关系或状态变化是关键线索 | 检验 H2 |
| Jointly dependent | 文字和画面分别提供不充分但互补的信息 | 检验 H3 |

- 这些类别应通过**独立人工标注或预先制定的规则**确定，而不应根据被测模型的实验结果事后挑选，否则容易造成选择偏差。
- 例如：漫画里一个角色说「我终于找到了」，但画面必须显示他找到的究竟是什么，才能把该格与之前的寻找过程正确连接，那么这类故事可能需要文字与视觉线索共同参与理解。
- Jointly dependent 是一个**待验证的数据属性**，而不是只要 Full 的准确率最高就自动成立的标签。人工标注者可以分别评估视觉遮挡文字版和文本转录版是否足以恢复顺序，以验证预先划分是否合理。

## 7. 不同实验结果能支持什么结论

- **结果 A：OCR-only 接近 Full。** 说明许多漫画排序题可通过文本线索完成，原生多模态形式本身并不能消除 MIRAGE 所关注的评估风险。
- **结果 B：Full 显著优于 OCR-only，但与 Visual-only 接近。** 说明视觉信息确实重要，但现有结果更支持视觉主导的排序，而不是视觉与文字之间存在明显的互补作用。
- **结果 C：Full 同时显著优于 OCR-only 和 Visual-only。** 如果在独立验证的 Jointly dependent 子集中也观察到这一现象，就能提供更强的行为证据，说明两个模态都为排序任务贡献了信息。
- **结果 D：无图像时仍有较高的非视觉条件准确率，或出现大量无依据描述。** 这需要进一步区分文本捷径与虚构视觉证据。OCR-only 的高准确率反映文本足以支持任务，而无图像、无 OCR 时生成具体视觉描述则属于另一种缺失模态行为。

## 8. 科学边界

- 即使**结果 C** 成立，也不能直接推出 MIRAGE 的主要观点错误。它表明，在经过适当构造的原生多模态任务中，模型能够从两种模态的联合输入中获得额外收益；MIRAGE 对常规 benchmark 的批评仍可能成立。
- 同样，如果**结果 A** 成立，也不能证明所有 VLM 的视觉能力都只是幻觉。结论应限于所测试的模型、漫画数据和排序任务。

## 9. 正式 proposal 摘要（可直接引用）

**Research Proposal: Evaluating Visual-Linguistic Integration in Natively Multimodal Comic Sequencing**

**Motivation.** MIRAGE raises concerns that strong performance on conventional vision-language benchmarks may not reliably demonstrate genuine visual understanding, as models can sometimes retain substantial task performance when visual inputs are removed. However, the generalizability of this phenomenon to natively multimodal narrative tasks, in which visual events and embedded text jointly convey meaning, remains an empirical question.

**Research Question.** To what extent do vision-language models rely on visual evidence, embedded linguistic information, and their interaction when solving comic-panel ordering tasks?

**Methodology.** We will evaluate vision-language models on shuffled comic-panel sequences under four controlled input conditions: (1) full comic images, (2) images with embedded text masked, (3) verified OCR transcriptions without images, and (4) neither images nor OCR text. We will measure exact sequence accuracy, pairwise ordering accuracy, modality-specific performance gains, and cross-modal interaction effects. We will additionally compare silent image omission with explicit disclosure of missing visual inputs to investigate whether prompt framing affects unsupported visual descriptions and model behavior. Where feasible, samples will be stratified into text-dominant, visual-dominant, and jointly dependent narratives through independent annotation.

**Hypotheses.**

- **H1 — Textual shortcuts:** OCR-only input will support above-chance ordering performance, with potentially small performance gaps from full input on text-dominant examples.
- **H2 — Visual dependence:** Full comic input will outperform OCR-only input on examples in which visual actions and spatial relationships provide essential ordering information.
- **H3 — Cross-modal complementarity:** On independently identified jointly dependent examples, full input will outperform both visual-only and OCR-only conditions, providing behavioral evidence of complementary information use.
- **H4 — Prompt sensitivity:** Silently omitted images will produce a higher rate of unsupported, visually grounded descriptions than explicitly disclosed image absence.

**Expected Contribution.** Rather than seeking to confirm or reject MIRAGE through a single alternative benchmark, this study will examine the boundary conditions of MIRAGE-style effects in native multimodal narrative reasoning. The proposed modality-controlled evaluation will help distinguish textual shortcuts, visual evidence dependence, and cross-modal complementarity in comic-panel ordering.

**建议**：把 H1–H3 作为主要研究假设，H4 作为辅助实验。H1 检验文字捷径是否依然存在，H2 检验视觉信息能否增加任务所需证据，H3 检验两个模态是否具有互补性。这样的安排既能让研究目标清楚，也能允许不同实验结果对 MIRAGE 提出更细致、而不是非黑即白的结论。
