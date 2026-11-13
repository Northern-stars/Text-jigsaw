# Panel Ordering Ground Truth 与 Human Benchmark 审计报告

审计日期：2026-11-13。本文基于当前工作区代码、本地 Manga109 原始压缩包，以及用户确认的阅读方向约定。初始检查只记录发现和建议；后续按报告完成了代码修复，实施状态见本文末尾。既有生成数据没有被覆盖或重建。

## 1. 结论与阅读方向约定

当前 human benchmark 低分至少受到两类独立错误影响：**评分器误解 `target_order`，以及生成器将双页作为一张单页排序**。此外，COMIX 默认方向与项目要求不符，几何分行算法对标注输入顺序敏感，first-panel 实验也存在标签语义错误。

| 数据集 | 本项目要求 | 当前默认值 | 判断 |
| --- | --- | --- | --- |
| Manga109 | RTL，页内同层从右到左；普通双页先右页后左页 | `rtl=true` | 方向默认值正确，双页与页内布局处理有问题 |
| MangaZero | RTL，页内同层从右到左；普通双页先右页后左页 | `reading_order_rtl=true` | 方向默认值正确，双页与页内布局处理有问题 |
| COMIX（Hugging Face pages 链路） | LTR，页内同层从左到右；普通双页先左页后右页 | `reading_order_rtl=true` | 方向默认值错误，页内几何算法也有问题 |

这里的左右方向是**恢复原始版面阅读顺序**的规则，不是“把乱序展示的 panel 从右往左/从左往右依次选择”。它也不意味着翻转 panel 图片。气泡内部阅读方向应单独定义，不能仅凭 panel 方向推断。

方向开关并非完全失效：已独立执行实际参数解析函数并加载对应 JSON，确认 `--reading-order-ltr` 得到 `False`，`--reading-order-rtl` 得到 `True`。问题在于 COMIX 的默认值，以及该开关目前只控制“行内 x 排序”，没有控制物理页先后、分区或复杂布局。

旧 COMICS 链路 `comix/create_panel_ordering_dataset.py` 使用官方 `panel_no` 排序，和 COMIX pages 的几何排序链路不同，不能把 COMIX 默认方向错误直接推广到旧 COMICS 链路。

## 2. 项目相关结构与 GT 生成流程

| 模块 | 职责与关键行为 |
| --- | --- |
| `Manga109/create_ordering.py` | 解析 XML frame/text，分配对白，几何排序，取连续窗口，打乱后写 `sample.json` |
| `Manga109/reading_order_utils.py` | Manga109、MangaZero、COMIX 共用的几何排序函数 |
| `Mangazero/download_raw.py` | 保存原始标注与两张页图 |
| `Mangazero/build_dataset.py` | 拼接本地两张页图，裁剪 panel、OCR，几何排序与切窗 |
| `Mangazero/main.py` | 下载与构建一体化入口，重复实现了部分 MangaZero 构建逻辑 |
| `comix/comix_pipeline.py` | COMIX pages 下载/构建共用实现，复用 MangaZero 的记录格式和 puzzle 写入函数 |
| `Solver/data/panel_ordering_dataset.py` | 统一读取清单、panel 图片和 `target_order` |
| `Solver/human_benchmark.py` | 再次随机排列展示顺序，将人类输入与答案比较 |
| `experiment/first_panel_classifier/` | 从同一格式标签中提取第一格分类目标 |
| `config/` | 各 CLI 入口的 JSON 默认参数，显式 CLI 参数优先 |

`Manga109/create_jigsaw.py` 属于另一条规则网格拼图链路，不是本次 panel 阅读顺序生成器。

当前两种漫画数据的 `target_order` 都由同一种公式产生：

```python
# window 已经被生成器认定为阅读顺序。
input_order = list(range(panel_count))
rng.shuffle(input_order)
panels = [window[i] for i in input_order]
target_order = [input_order.index(i) for i in range(panel_count)]
```

因此 `input_order[i]` 表示输入槽 i 的阅读排名，`target_order[k]` 表示阅读位置 k 应选择的输入槽。二者互为逆排列，不能混用。生成器计算逆排列的公式本身正确，但它依赖的 `window` 顺序可能错误。

## 3. 已确认的问题与修改建议

### F1：human benchmark 对 `target_order` 多求了一次逆排列

**优先级：P0。所有使用标准 ordering 格式的人工评测均受影响，与 RTL/LTR 无关。**

位置：[human_benchmark.py](../Solver/human_benchmark.py)，`correct_display_order()`。

当前代码先按 `target_order[index]` 排序输入索引，相当于 `argsort(target_order)`，把“选择序列”误当成“每个输入 panel 的排名”。之后的展示编号转换无法修正这个语义错误。

例如：

```text
input_order       = [3, 0, 5, 1, 4, 2]
target_order      = [1, 3, 5, 0, 4, 2]
displayed_indices = [0, 1, 2, 3, 4, 5]

正确的 1-based 展示答案：2 4 6 1 5 3
当前评分器使用的答案：  4 1 6 2 5 3
```

独立执行实际评分函数，对全部 720 种六格输入排列进行枚举，并为每种排列再随机调整展示顺序：一个完全按照生成器 GT 作答的参与者仅有 **76/720 = 10.56%** 会被判对。原因是只有排列等于自身逆排列时，这个错误才被掩盖。

这是均匀遍历输入排列时的理论/程序验证结果，不是任意有限题集的固定上限，也不是实际人类能力估计。真实 GT 还可能包含其他错误。

建议改成直接进行输入槽到展示编号的映射：

```python
def correct_display_order(target_order, displayed_indices):
    count = len(target_order)
    expected = set(range(count))
    if len(displayed_indices) != count:
        raise ValueError("target_order and displayed_indices must have equal length")
    if set(target_order) != expected or set(displayed_indices) != expected:
        raise ValueError("orders must be permutations of input panel indices")
    input_to_display = {
        input_index: display_position + 1
        for display_position, input_index in enumerate(displayed_indices)
    }
    return [input_to_display[input_index] for input_index in target_order]
```

`Solver/test_human_benchmark.py` 中已有测试也误解了标签。例如 `target_order` 与 `displayed_indices` 同为 `[2,0,3,1]` 时，正确答案应是 `[1,2,3,4]`，现有测试却期待 `[4,3,2,1]`，需要一并纠正。

### F2：双页被当成单页，导致两页交叉读取和错误切窗

**优先级：P0。Manga109、MangaZero 的构建逻辑均有此缺陷。COMIX 若输入双页图，也需要同样的物理页处理。**

位置：[reading_order_utils.py](../Manga109/reading_order_utils.py)、[create_ordering.py](../Manga109/create_ordering.py)、[build_dataset.py](../Mangazero/build_dataset.py)、[main.py](../Mangazero/main.py)。

Manga109 的原始图包含左右双页；MangaZero 明确将两张图拼接成一张图，其中第二张图从 `x=width1` 开始。当前算法在整个拼接图上按纵向中心分行，没有利用物理页边界。

```text
普通 RTL 双页期望：右页上 → 右页下 → 左页上 → 左页下
当前 RTL 分行输出：右页上 → 左页上 → 右页下 → 左页下

普通 LTR 双页期望：左页上 → 左页下 → 右页上 → 右页下
当前 LTR 分行输出：左页上 → 右页上 → 左页下 → 右页下
```

将 `rtl` 改成 `False` 只能改变同一行的左右位置，不能修复双页交叉读取。错误排序还会污染后续滑窗，使声称“连续”的窗口实际跳过中间剧情。

已查看 Manga109 `ARMS/007.jpg` 原图，并与六个 frame 对照：当前 GT 依次输出 frame ID `0000008e, 00000086, 0000008c, 0000007e, 00000088, 0000008d`；遵循双页顺序应为 `0000008e, 00000086, 00000088, 0000008c, 0000007e, 0000008d`。

修改建议：

1. 显式区分 `source_image_id` / `spread_id` 与 `physical_page_id`，不要把一张图片恒等于一个物理页。
2. MangaZero 优先使用已知拼接边界 `width1`；Manga109 对普通双页使用数据约定或经过验证的页缝检测。不能把所有宽图都直接从中间切开。
3. 普通双页先对每个物理页独立排序，再按 RTL 的右→左、LTR 的左→右合并；有可靠原始页序元数据时优先使用元数据。
4. 明确选择窗口策略：只在单个物理页内切窗，或允许跨页取真正连续的窗口。两者都合理，但必须记录策略并重新核算六格样本数量。
5. 跨页大格、嵌套格、覆盖页缝的 bbox 单独处理；不能强行将一个 panel 切成两个独立标签。无法确定时标记待复核或排除。

### F3：COMIX 默认值仍为 RTL，与要求的 LTR 不符

**优先级：P0。影响没有显式覆盖方向的 COMIX pages 构建。**

代码及配置均为 RTL：

- [comix_pipeline.py](../comix/comix_pipeline.py)：`--reading-order-rtl` 的默认值为 `True`，排序调用中的缺省回退也为 `True`。
- [comix_pipeline.json](../config/comix/comix_pipeline.json)：`reading_order_rtl=true`。
- [stream_comix.json](../config/comix/stream_comix.json)：`reading_order_rtl=true`。
- [build_ordering_dataset.json](../config/comix/build_ordering_dataset.json)：`reading_order_rtl=true`。

已验证这些入口在不传方向参数时解析为 RTL，显式传 `--reading-order-ltr` 时可以切换到 LTR。没有历史运行参数，不能据此断定所有已生成的 COMIX 样本都使用了错误方向。

建议同步修改 COMIX 的 parser 默认值、JSON 配置和排序调用缺省值为 LTR，保留显式覆盖。MangaZero、Manga109 继续默认 RTL。

短期可显式使用 `--reading-order-ltr` 构建 COMIX，但这不能解决 F2/F4。长期建议统一参数为 `--reading-direction {rtl,ltr}`，避免 Manga109 的 `rtl` 与其余入口的 `reading_order_rtl` 命名不一致。两种方向参数应互斥。

### F4：纵向中心分行既不稳定，也不足以恢复复杂布局

**优先级：P1。三条几何排序链路都受影响，LTR/RTL 均可复现。**

当前阈值为所有 panel 平均高度的 0.5 倍，逐个将 panel 加入第一个符合阈值的行，并更新行中心。输入顺序改变会改变分组；一个大格还可能通过改变平均高度影响其他区域。

反例为“一侧一个跨两行高格，另一侧上下两个小格”。仅交换 frame 列表顺序，同一组 bbox 就可能输出 `高格→上小格→下小格` 或 `上小格→高格→下小格`。镜像后在 LTR 下也能复现。

建议：

- 将“输入 bbox 列表重新排列后，输出的 source panel ID 序列不变”作为基础不变量；几何排序必须具有确定性。
- 按物理页分区后，使用页内沟槽/投影、递归 XY-cut 或布局约束图识别阅读块，再在块内排序。对普通横向行切分与纵向列切分使用明确规则。
- 不要只做 bbox 预排序或调整阈值：这可改善确定性，但不等于布局语义正确。
- 对跨行、斜切、重叠、嵌套及四格纵列等布局建立独立人工标注样例。几何规则无法可靠决定的顺序应显式标记歧义。
- 将算法名称与版本写入数据，避免将几何推导顺序描述为官方人工阅读顺序。

### F5：first-panel classifier 提取了错误的监督目标

**优先级：P0（该实验）。不改变完整排序生成器的标签，但会使第一格分类实验训练/评测错误任务。**

位置：[model.py](../experiment/first_panel_classifier/model.py)，`first_panel_targets()` 当前返回 `target_order.argmin(dim=1)`。

标准标签是选择序列，因此第一格应是 `target_order[:, 0]`。例如 `[2,0,1]` 的第一格是输入槽 2，而不是输入槽 1。

需同步修正代码、测试、README、训练脚本说明与 `label_definition` 日志字段。旧 checkpoint 无法通过只修改评测代码变成正确第一格任务的模型，需要重新训练或明确标记为旧任务。

### F6：panel 身份与顺序元数据不一致，影响追溯

**优先级：P1。不能单独解释所有 human benchmark 低分，但会妨碍去重、泄露检查和纠错。**

Manga109 在切窗后用窗口内的 `original_idx` 写入 `panel_id`、`panel_index_in_page`、`global_order`，且不保留 XML frame ID。同一 frame 在不同窗口的 ID/索引会变化，同一 ID 也可能指向不同 frame。

MangaZero/COMIX 的 `panel_index_in_page`、`global_order` 则保留排序前的来源列表序号，几何重排后不再保证表示阅读排名。保留来源序号本身合理，但文档不应再称其为阅读顺序。

建议拆分字段：

| 字段 | 建议语义 |
| --- | --- |
| `source_panel_id` | 稳定来源身份，Manga109 使用 book + spread/page + XML frame ID |
| `source_panel_index` | 来源检测/标注列表序号，不充当阅读排名 |
| `reading_rank_in_page` | 经当前算法或人工确认的物理页内阅读排名 |
| `original_window_index` | 此 panel 在当前窗口内的阅读排名 |
| `input_index` | 当前乱序输入槽 |

若保留 `global_order`，需重新定义适用范围并保证和物理页序一致；否则避免让下游将它误认为权威阅读顺序。

## 4. Manga109 原始标注量化结果

输入为本地 `Manga109_released_2026_05_21.zip` 中最新的 `annotations/*.xml`，未使用旧版本标注目录。直接调用当前 `sort_bboxes_reading_order(..., rtl=True)`。

| 检查项 | 数量 / 比例 |
| --- | --- |
| 作品数 | 109 |
| XML 页条目数（包含双页图条目，不等于物理页数） | 10,619 |
| frame 数 | 103,850 |
| 至少六格的页条目数 | 9,201 |
| 至少六格、所有框均不跨图像中线、左右两侧均有框 | 8,712 |
| 上述 8,712 个条目中，当前排序出现左页后返回右页 | 8,626 / 8,712 = 99.01% |
| 上述条目按六格、stride=1 形成的全部窗口 | 52,775 |
| 其中有左页 panel 出现在后续右页 panel 之前的窗口 | 48,219 / 52,775 = 91.37% |
| 至少六格条目中，改变输入列表顺序后至少一次输出不同 | 5,836 / 9,201 = 63.43% |

方法说明：左右归属使用 `xmin >= width/2` 为右，`xmax <= width/2` 为左；跨中线框所在条目不纳入该双页冲突统计。输入顺序敏感性检查按 XML 路径排序遍历、随机种子 109，每个合格条目比较原列表、反转列表和三次随机重排的结果。

这些结果衡量**普通双页约束冲突和算法不稳定性**，不是人工确认的完整 GT 错误率。窗口之间有重叠，不能当作相互独立的统计样本；也没有模拟生成器的最终样本预算与 split 截断。MangaZero/COMIX 本地缺少 raw 与实际 benchmark 答题记录，尚未量化它们的实际受影响比例。

## 5. 对现有结果的解释与修复顺序

1. **先修评分与标签语义。** 修 F1/F5，纠正已有错误测试，并用生成器实际输出构造端到端验证。现有 human accuracy 不宜作为能力结论；若曾记录完整答案、样本 ID 和展示排列，可重算，否则需要重测。
2. **修 COMIX 方向配置并记录有效参数。** 不仅修改 JSON，还检查所有公开入口、parser 与缺省回退；核对旧构建是否显式传过 LTR。
3. **修双页和页内排序后重建数据。** 使用新输出目录与 GT 版本。双页排序变化会改变窗口成员，不能保证只替换旧 `target_order` 就能得到新定义下的连续窗口。
4. **按作品和布局分层人工复核。** 先建立独立于旧算法的小型 gold set，对方向、双页、跨行、跨页大格等分别验证；对歧义页保留标记或暂不纳入正式测试集。
5. **重跑 human benchmark 与受影响实验。** human 与 model 必须使用同一 GT 版本；报告 exact match、position accuracy、pairwise accuracy 和有效题数。修评分不代表旧训练模型已经学到了正确阅读顺序。

默认 224×224 的缩放、文本截断也可能增加人工作答难度；应提供可放大的高分辨率 panel，并记录实际展示条件。`panel_text_mode=auto` 可能使用 description/caption，而模型通常使用 `dialog_text`，需统一或单独标记文本证据条件。这些属于评测协议因素，不能替代对明确代码错误的修复。

对白目前通常按来源 text/dialog 列表直接拼接，没有独立恢复气泡顺序。这是额外的文本证据质量风险，尚未量化，建议在页内 GT 修复后单独审计，不将其混同于 panel 排序。

## 6. 建议的可追溯字段与验收标准

建议每个构建保存有效 CLI/JSON 参数、代码版本、随机种子、数据来源版本，并为每个 sample 保存 `reading_direction`、`source_image_id`、`physical_page_id`、页缝位置/识别方法、`reading_order_method`、`reading_order_version`、复核状态和窗口策略。

Human benchmark 每题至少保存 `sequence_id`、GT 版本、展示排列、用户答案、参考答案和各项指标，以便评分逻辑修正后重算。不能只保留最终总分。

| 验收项 | 通过条件 |
| --- | --- |
| 方向默认值 | MangaZero/Manga109 为 RTL，COMIX 为 LTR；JSON、CLI 和入口一致 |
| 标签契约 | `input_order[target_order[k]] == k`，`target_order` 为完整合法排列 |
| 人工展示映射 | 任意合法展示排列下，按 `target_order` 作答均判正确；遍历六格全部排列验证 |
| 第一格目标 | 对所有合法标签均等于 `target_order[0]` |
| 普通双页 | RTL 右页完成后才进入左页；LTR 左页完成后才进入右页 |
| 输入顺序不变性 | bbox 与稳定 ID 不变，仅排列输入列表，输出 source panel ID 序列不变 |
| 镜像一致性 | 对无歧义测试布局，水平镜像且 RTL/LTR 互换后保留对应逻辑序列 |
| 复杂布局 | 与独立人工 gold set 一致；无法确定的样本显式记录歧义 |
| 稳定身份 | 同一来源 panel 在不同窗口保持同一 source ID，窗口位置独立记录 |
| 划分隔离 | 同一 spread 的两个物理页以及所有关联窗口不得跨 train/val/test；若目标是跨作品泛化，再要求作品隔离 |
| 人机可比性 | GT 版本、panel 集合、文本证据与展示条件均有记录 |

当前已抽查的完整排序训练/评测路径主要把 `target_order` 当作选择序列，在线重排也先按它恢复顺序；没有证据表明所有模型路径都存在 F1 的逆排列错误。因此应分别处理“生成 GT 错误”“人工评分错误”和“特定实验标签错误”，不能将所有低分归为同一个原因。

## 8. 修复实施状态

后续代码修复采用 `reading_order_version=2`：

- human benchmark 直接按 `target_order` 映射展示编号；新增结果 JSONL 选项，记录样本、GT 版本、展示顺序、标准答案、回答和展示尺寸。
- first-panel classifier 改为用 `target_order[0]` 监督，并拒绝缺少新版标签定义的旧 checkpoint。
- 共用排序器改为确定性的递归留白切分；支持明确提供物理页边界，按 RTL/LTR 先后排序物理页。
- Manga109/MangaZero 默认 RTL + spread；COMIX pages 默认 LTR + single。所有方向/布局都写入配置与生成 metadata。
- 排序无法判定或 panel 横跨物理页边界时跳过整页并在 `meta.json` 计数；不会把歧义排序伪装成有效 GT。
- Manga109 保留 XML frame ID 和源列表序号，分别记录物理页内阅读排名、source-image 阅读排名、窗口内排名与乱序槽位。
- 新 puzzle / panel 记录来源图、spread、物理页、页缝、方向、排序方法/版本、状态和窗口策略。

自动几何排序仍是启发式；`reading_order_status=heuristic` 不代表官方人工阅读顺序。用当前排序器扫描 Manga109 最新标注的 9,201 个至少六格的页条目，3,286 个通过保守布局检查，5,915 个被跳过（5,554 个无可分离布局，361 个 panel 跨中缝）。正式使用前应人工审阅跳过页面、审计自动接受的页面，并决定是否要针对高覆盖率复杂布局引入人工标签或专用规则。

修复后的定向测试为 20 项通过，其中包含六格答案映射全排列检查、RTL/LTR 双页顺序、输入检测顺序不变性、歧义 bbox 拒绝、first-panel 标签和旧 checkpoint 标签拒绝，以及 split/config 回归。另以合成 Manga109 双页端到端运行生成器，确认 `R0 → R1 → L0 → L1` 和来源元数据正确。没有重建已有数据；使用旧 GT 的训练、human benchmark 和实验结果应重新生成或重测。

## 7. 本次验证边界

完成了代码追踪、方向参数解析、评分函数六格排列枚举、LTR/RTL 几何反例验证、Manga109 最新 XML 全量扫描及一个双页原图的视觉核对。参数解析与评分验证通过 AST 提取原始函数并独立执行，避免导入当前环境缺失的模型和图像依赖。

当前 Python 环境缺少 Pillow、PyTorch，未运行完整图像生成/训练/交互测试，也未重建数据。本文给出的修复代码与验收标准是修改建议，尚未应用到生产脚本。
