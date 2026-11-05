# From Diagnosis to Intervention: LVLM 漫画 Panel Reordering 的训练与推理干预研究提案

> 定位：不再以「提出一种更好的漫画排序模型」为目标，而是通过受控干预，研究**训练数据规模、风格平衡、可用模态证据和提示策略**如何影响 LVLM 的漫画面板排序能力，以及这些改善在什么条件下成立、哪些缺陷仍然存在。
>
> 核心原则：不预设模型一定能被改善；实验旨在检验哪些干预有效、哪些无效、在哪些边界条件下有效。

## 暂定标题

- 英文：From Diagnosis to Intervention: Data Scaling, Style-Balanced Training, Evidence Modalities, and Prompting for LVLM Panel Reordering
- 中文：从诊断到干预：LVLM 漫画面板排序中的数据规模、风格平衡、证据模态与提示策略

相关文档：[Manga Ordering 研究提案与项目 TODO](manga_ordering_research_todo.md)、[VLM Ordering Solver 接口](vlm_ordering_solver_interface.md)、[四项研究问题的代码实验方案](lvlm_panel_reordering_experiment_code_plan.md)。

## 1. Research motivation

### 1.1 现有问题

CHROMIC 等 benchmark 已经揭示了 LVLM 在漫画 Panel Reordering 上与人类表现之间存在明显差距。但这类 benchmark 的作用主要是**评估和诊断**模型的失败：它们告诉我们模型在哪类题目上表现差，却没有系统检验这些缺陷能否通过专项训练得到改善。

### 1.2 研究空白

漫画面板排序同时涉及角色状态、动作连续性、视觉布局和跨格叙事关系。目前尚不清楚性能不足究竟在多大程度上能通过以下干预缓解：

- 增加训练数据规模；
- 在匹配训练数据量与计算预算的前提下平衡训练风格；
- 控制视觉证据、嵌入文本及其组合的可用性，并排除原始版面布局线索；
- 调整推理提示策略（直接预测 / CoT / 结构化证据提示）。

### 1.3 本研究动机

通过控制训练数据规模、风格构成、输入证据和推理提示，识别哪些干预能够改善排序表现，以及这些改善能否迁移到未见过的作品。研究的定位是系统研究「哪些训练与推理干预能够改善漫画排序，以及这些改善在什么条件下成立」，而不是追求一个绝对更高的排序准确率。

## 2. Research questions

四个核心研究问题：

- **RQ1 — Data Scaling**: How does the amount of comic-domain training data affect panel-reordering performance, and do returns diminish as the training set grows?
- **RQ2 — Style-Balanced Training**: Under matched training-data and compute budgets, does style-balanced mixed-style training improve within-style performance and generalisation to held-out comic works?
- **RQ3 — Evidence Modalities**: To what extent do visual evidence, embedded text, and their combination support panel reordering under a controlled, layout-blind evaluation?
- **RQ4 — Prompting Strategies**: With the same trained model and decoding budget, do direct prediction, chain-of-thought prompting, and structured evidence prompts improve ordering performance or alter its error profile?

## 3. 实验设计

评测以 Panel Reordering 为唯一任务，采用**作品级数据划分**，并分别报告日漫、美漫及未见作品结果。四个问题对应彼此区分的受控实验：数据规模曲线、等预算风格构成比较、布局盲模态消融、以及固定模型的提示策略比较。测试 puzzle 在各输入条件和提示条件间配对复用。

### 3.1 核心实验

| 实验 | 设计 | 目标 |
| --- | --- | --- |
| Baseline | 选定 LVLM 的未微调基线及统一的微调基线 | 建立参照性能 |
| RQ1: Scaling | 逐步增加 comic-domain 训练集规模 | 衡量规模效应及边际收益递减 |
| RQ2: Style balance | 匹配训练 puzzle 数、训练 token/step 与计算预算；比较单风格和风格均衡混合训练 | 衡量各风格内性能及留出作品泛化 |
| RQ3: Evidence modality | 在 layout-blind 输入协议下比较视觉-only、文本-only、视觉+文本 | 隔离不同证据来源的作用 |
| RQ4: Prompting | 固定同一 checkpoint、输入和解码预算，比较直接预测、CoT、结构化证据提示 | 衡量排序变化及错误类型变化 |

### 3.1.1 对比方法

| 方法 | 当前代码/模型 | 输入与训练方式 | 在本研究中的角色 |
| --- | --- | --- | --- |
| **Qwen2.5-VL 生成式 LoRA（主方法）** | `Solver/agent/qwen_vl_ordering_solver.py`；基座通过 `--model-name` 指定 | panel 图像 + 可选 OCR；LoRA 微调生成排序字符串 | RQ1–RQ4 主实验统一使用该方法；RQ4 在同一 checkpoint 上只替换 prompt |
| **Made2Order** | `Solver/agent/made2order_set2order_solver.py` | 纯视觉、监督式排序；共享 ordering puzzle 和主要准确率指标 | 已有视觉排序基线；用于规模/风格结果的外部参照及 RQ3 视觉条件参照，不适用于文本-only 或提示策略比较 |
| **Qwen2.5-VL permutation classifier** | `Solver/agent/mangazero_qwen_vl_lora_classifier.py` | LoRA backbone + 全排列分类头 | 现有任务专用架构对照；建议作为补充基线，不替代 RQ4 可提示的生成式主方法 |
| **Qwen2.5-VL autoregressive pointer** | `Solver/agent/mangazero_qwen_vl_autoregressive.py` | LoRA backbone + 自回归指针解码头 | 现有任务专用架构对照；可用于总体方法比较，不用于 CoT/结构化提示消融 |
| **Qwen2.5-VL zero-shot** | `Solver/agent/qwen_vl_ordering_solver.py` 的无微调推理 | 与主方法相同的视觉输入和输出解析，不加载排序微调 adapter | 所有训练干预的参照基线；RQ4 主比较仍限定在同一个已训练 checkpoint 上 |

本研究的四条 RQ 主矩阵以 Qwen2.5-VL 生成式 LoRA 为核心，确保数据规模、风格、模态和提示策略可以在同一求解器接口内比较。Made2Order、Qwen2.5-VL permutation classifier 和 pointer decoder 是架构对照；其中只有适用的实验格才纳入，避免把不支持文本输入或自然语言提示的方法强行放进不公平的对比。仓库现有 Qwen 训练脚本默认基座为 `Qwen/Qwen2.5-VL-3B-Instruct`；实验应通过配置固定实际可用的 Qwen2.5-VL checkpoint（包括 7B 若已准备），并在所有 RQ 记录相同的模型 ID、revision 和 processor 配置。

#### 开源 VLM 候选与 32GB 显存配置

Qwen2.5-VL 生成式 LoRA 仍是四项 RQ 的主方法。为检验结果是否依赖单一基座，并适配单卡 32GB 显存，增加以下候选；参数量和量化格式仅用于初始筛选，正式实验前须在目标硬件、目标图像尺寸和实际 batch 下运行显存探测，并记录峰值显存与吞吐量。

| 候选模型 | 初始运行配置 | 计划用途 | 注意事项 |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B-Instruct | 4-bit NF4 / bitsandbytes QLoRA；推理可用 4-bit AWQ/GPTQ | 首选 7B 扩展基座；通过同一生成式 ordering runner 覆盖 RQ1–RQ4 | 与 3B 使用同一数据、输入分辨率和训练预算定义；验证视觉编码器量化兼容性。若显存或训练稳定性不满足要求，保留 3B 主实验，不以缩小测试集代替预算匹配 |
| InternVL2.5-8B | 4-bit 权重量化；必要时采用 QLoRA | 独立开源 VLM 推理/微调复现，优先作为补充模型 | 需实现或验证 InternVL 专用 processor、图像裁剪与消息格式；只有在统一数据和评估协议可复现时纳入主表 |
| Llama 3.2 11B Vision Instruct | 4-bit 权重量化推理；训练仅在 QLoRA 显存探测通过后进行 | 较大参数量开源视觉指令模型的补充比较 | 视觉输入尺寸和 Transformers/量化支持需实测；不默认其能以 32GB 完成训练 |
| Qwen2.5-VL-3B-Instruct | BF16/FP16 或 4-bit，按实测选择 | 保持可复现的低显存主基线与故障回退方案 | 仓库现有训练脚本默认基座；主结果须固定模型 revision、processor 和量化配置 |

32GB 约束按可用显存而非显卡标称容量判断。训练配置统一记录量化方法、compute dtype、LoRA target modules、图像最小/最大像素、gradient checkpointing、batch/gradient accumulation、优化器状态和峰值显存；先做单 batch 前向、反向及短程训练 smoke test。只有在 32GB 内稳定完成且无 OOM 时才列为可训练模型。若某候选仅能量化推理，则将其标为 inference-only，不与 LoRA 微调结果混列。模型许可、权重可获得性和具体软件版本在实验开始时核验并记录。

#### 闭源大模型测试方案

闭源模型作为零样本/提示推理的外部参照，不参与 RQ1 的训练数据规模曲线或 RQ2 的训练风格平衡因果比较，也不与开源微调模型直接作“同预算训练”结论。候选使用实验启动时可用且支持图像输入的固定 API 版本，例如 OpenAI GPT-4.1 / GPT-4o、Google Gemini 2.5 Pro / Flash、Anthropic Claude Sonnet 系列；最终只选能稳定处理多张 panel 图、满足数据使用要求且预算可控的模型，并记录供应商、精确 model ID、API/模型版本、调用日期和参数。模型名称是候选而非对长期可用性的承诺。

评测流程：

1. 通过统一 evaluator 将同一批测试 puzzle 转换为独立 panel 图像及随机 panel ID；不传原页面、坐标、原始顺序、split 以外的答案元数据。每题一次请求包含全部 panels，prompt 只要求输出 panel ID 的完整排列；保存原始响应、规范化解析结果和 API 请求元数据。
2. 做配对的闭源推理基线：direct prediction 与 structured evidence prompt 使用相同 puzzle、图像顺序、最大输出 token 数和解析规则。只有当 API 确实支持可观察的逐步推理输出时才单列 CoT；隐藏 reasoning token 不可见时，不将其等同于显式 CoT，改为报告 direct 与可见结构化证据两种条件。
3. RQ3 可在支持图像与文本输入的 API 上执行 visual-only、text-only、visual+text 三种 layout-blind 配对条件；纯文本条件不发送图像，文本按随机 panel ID 对齐。无法满足遮字、OCR 文本注入或输入约束的模型标记为不适用，不以不同输入协议的分数横向合并。
4. RQ1/RQ2 仅将闭源 zero-shot 结果作为固定外部参考线；闭源模型不可训练，不能据此推断训练数据规模或风格平衡效果。闭源模型之间也单独报告，不与开源模型做同模型提示干预的因果比较。
5. 预先固定模型版本、system/user prompt、图像压缩尺寸、温度（优先 0）、输出 token 上限、超时和重试规则。网络/API 错误可按预注册次数重试；格式无效的成功响应保留为 invalid，不反复提示直到得到正确格式。记录每题延迟、输入/输出 token、费用、错误码、解析成功率与 Exact sequence / Pairwise accuracy；报告实际有效样本数和估算总费用。
6. 仅发送获准用于第三方 API 的测试图像和文本；对每个供应商核实数据保留/训练政策。若数据许可不允许外传，闭源评测只使用获准子集或取消该数据源，并明确报告覆盖范围。

闭源 API 结果与开源本地结果分开成表，统一列出任务协议、样本集、指标及错误率，并标注模型版本和测试日期。至少保存逐样本 JSONL（匿名样本 ID、模型 ID、prompt profile、请求状态、原始输出、解析排序、token/费用、指标），使结果可审计；API 行为可能随服务更新变化，因此闭源结果的可复现性限定于记录的版本和日期。

实验控制：

- **RQ1 / 数据规模：**从同一训练池构造递增且嵌套的训练子集（例如 5k / 10k / 20k / 40k puzzles）；保持风格构成、作品采样规则、训练配置和每个样本的训练处理一致。对每个规模使用多个随机种子或子集抽样重复，报告性能曲线及相邻规模的增益，避免把单次采样差异解释成规模效应。
- **RQ2 / 风格平衡：**在匹配训练 puzzle 数、总训练 token、优化步数和计算预算的条件下，比较日漫-only、美漫-only 与日漫:美漫均衡混合训练。混合组按风格均衡采样，并尽量按作品均衡，避免少数长篇作品支配训练。训练作品与测试作品严格隔离；分别报告见过风格中的留出作品（within-style）以及跨风格/留出作品表现。多随机种子下比较各风格变化，检验总体增益是否以某一风格退化为代价。
- **RQ3 / 证据模态：**对完全相同的测试 puzzle 构造三种配对条件：视觉-only（嵌入文字被遮蔽）、文本-only（只提供逐 panel 的 OCR 转录，不提供 panel 图像）、视觉+文本（原始 panel 图像及其中嵌入文字）。三个条件使用相同排序目标、panel 标识和模型/解码设置。实施 layout-blind 协议：只向模型提供独立 panel，不提供原页面、原始坐标、阅读顺序标注或能还原这些信息的布局元数据；panel 展示次序随机化并在各条件保持一致。文本-only 的 OCR 转录按随机化后的 panel ID 对齐，避免其顺序泄露答案。报告 OCR 覆盖率/错误率，便于区分文本证据缺失与识别噪声。
- **RQ4 / 提示策略：**在完全相同的已训练 checkpoint 和测试输入上比较直接预测、chain-of-thought prompting 与 structured evidence prompt。结构化提示要求按 panel ID 给出可观察证据（如角色、动作、场景/转场、文本线索）后再输出唯一排序；各策略使用相同最大生成 token 数、解码参数及重试次数/解析规则。避免把额外生成预算误当成提示策略收益；记录合法排序率，并对格式错误采用预先固定的解析/计分规则。
- 所有实验报告 Exact sequence accuracy、Pairwise ordering accuracy，并按漫画风格和作品是否在训练中出现切分汇总。RQ1 报告规模曲线；RQ2 报告风格内增益与留出作品泛化；RQ3 报告三种模态条件的配对差异；RQ4 报告提示差异及错误构成。

### 3.2 数据说明

- 作品级划分：同一部作品的所有 puzzle 只能出现在训练、验证或测试中的某一个集合，避免跨作品泄露。报告留出作品与留出风格的结果时，清楚区分这两种泛化设置。
- 日漫数据以 Manga109 / MangaZero 为主；美漫数据以 COMICS / COMIX 为主。
- 对模态实验，预先固定 OCR、文字遮蔽方法、panel 预处理和随机化协议；保留 OCR 质量统计。布局盲评测中不提供原页面布局或原始 panel 坐标。

## 4. Evaluation

对每个模型和每个实验条件输出：

- **Exact sequence accuracy**：完整排序是否正确。
- **Pairwise ordering accuracy**：两两面板的先后关系是否正确。
- **按风格切分的准确率**：日漫测试集、美漫测试集、跨风格测试集分别报告。
- **边际收益曲线**：Scaling 实验下各数据规模的准确率，判断是否存在收益递减。
- **风格内与留出作品表现**：RQ2 分别报告每种训练风格上的留出作品表现，以及跨风格测试表现；同时列出相对单风格训练的变化。
- **模态差异**：RQ3 对同一 puzzle 的视觉-only、文本-only、视觉+文本得分进行配对比较，并报告 OCR 覆盖率和识别错误。
- **风格权衡指标**：Style-balanced training 实验中日漫 / 美漫各自的增益或损失，判断总体改善是否伴随特定风格退化。
- **错误画像**：RQ4 按预先定义的类别统计顺序错误，例如局部相邻次序错误、首尾/转场错误、角色或动作连续性错误、文本线索误用、无效输出/解析失败。比较各提示策略的错误比例与合法排序率。

辅助指标（用于解释）：

- 模态实验中的 OCR 覆盖率/错误率，用于说明文本-only 条件的测量限制。
- 各提示策略成功输出并解析为合法排序的比例，用于区分排序能力变化与格式/解析失败。

## 5. 预期贡献

- 提供关于漫画专项 SFT 数据规模效应的实证结果。
- 分离训练样本规模与风格均衡训练对风格内排序能力及留出作品泛化的影响。
- 在控制版面布局线索的条件下，估计视觉证据、嵌入文本及二者组合对排序的贡献。
- 评估直接预测、CoT 和结构化证据提示对准确率及错误画像的影响。

定位说明：即使 CoT 无效、性能出现饱和、或混合风格训练只对部分测试集有效，也仍然可以形成有解释力的研究结论。本研究不承诺提出新的 SOTA 排序模型，而是提供关于「哪些干预有效、在什么条件下有效、哪些缺陷仍然存在」的实证证据。

## 6. 科学边界与可能结果

- 如果 Scaling 出现收益递减：报告曲线并讨论数据规模的上限效应，不强行归因于模型容量或数据质量。
- 如果风格均衡混合训练只改善某一风格或留出作品：报告分组结果和风格权衡，不将总体均值解释为普遍改善。
- 如果文本-only 表现较低：结合 OCR 覆盖率/错误率解释，并避免将 OCR 噪声直接归因于文本线索无用。
- 如果 CoT 或结构化证据提示无效：结合错误画像和合法排序率判断准确率是否稳定但错误类型改变，或收益被额外输出/格式错误抵消。
- 结论严格限定在本实验覆盖的模型、漫画数据、排序任务和训练协议内，不外推到一般漫画理解能力。
