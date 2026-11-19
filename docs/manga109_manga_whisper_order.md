# Manga109 的 MangaWhisper 阅读顺序生成

Manga109 现在使用 **Magi v1 / The Manga Whisperer（CVPR 2024）** 的 panel 排序算法，替代此前严格的页内 XY-cut。MangaZero 和 COMIX 继续使用各自现有排序链路。

GitHub 项目 [ragavsachdeva/magi](https://github.com/ragavsachdeva/magi) 的 README 指向 Hugging Face 的实际实现。本项目移植的是 [`utils.py` 的 `sort_panels` 及其几何辅助函数](https://huggingface.co/ragavsachdeva/magi/blob/15ad498ad45093ad3b833d625728220c6309ad99/utils.py)，固定 revision 为 `15ad498ad45093ad3b833d625728220c6309ad99`。未接入 v2/v3 的检测模型或下载模型权重。

## 算法与适配

1. 按已有 `page_layout` 拆分物理页；Manga109 默认 spread，使用图像中线作为左右页边界，RTL 先右页后左页。
2. 每个物理页内，按上游比例先收缩 bbox 5%，建立所有 panel 对的阅读先后关系。重叠时继续收缩；斜向关系通过局部水平/垂直切分判断。
3. 建立带几何距离权重的有向图；按上游方式枚举环，移除选中环中距离最大的边，直至无环，再做拓扑排序。
4. 合并完整物理页顺序后，使用原有连续窗口、输入打乱和逆排列标签公式生成 puzzle。

适配代码位于 [manga_whisper_order.py](../Manga109/manga_whisper_order.py)。几何交集和距离由等价的轴对齐矩形运算计算，不依赖 Shapely；图操作保留 NetworkX。几何输入先按坐标与来源 ID 规范化排序，使重合框、环消解等模糊情形的结果不依赖 XML 列表顺序。该规范化是项目适配层，可能改变上游对原始任意输入序号的 tie-break 结果。

`--no-rtl` 仍可用：通过镜像 x 坐标复用 RTL 算法，这是本项目扩展。跨物理页缝的大格仍跳过并计入 `skipped_reading_order_pages`，不会强行切开。侵蚀循环限制为 2,048 次，若数值收敛失败则跳过该页，避免上游无界循环卡住构建。几何顺序仍为启发式，`reading_order_status=heuristic`，不代表人工确认。

## 安装与生成

```bash
python -m pip install -r Manga109/requirements-ordering.txt
python Manga109/create_ordering.py \
  --manga109-dir Data/Manga109 \
  --output-dir Data/Manga109/ordering_manga_whisper_v3 \
  --panel-count 6 --puzzle-num 200
```

已拆成单物理页的输入使用 `--page-layout single`。翻译仍由原有翻译配置控制。运行时无需访问 Magi 仓库或下载检测模型。

`sample.json`、panel 与 `meta.json` 会写入：

- `reading_order_method=manga_whisper_v1`
- `reading_order_version=3`
- `reading_order_source` 与固定的 `reading_order_source_revision`

来源 frame ID、物理页、阅读排名、窗口位置、乱序槽位及 `target_order` 语义保留。应生成到新目录；旧 GT 不会自动更新，窗口成员也可能改变。

## 验证

测试使用固定 revision 上游代码实际计算的六组参考布局，覆盖普通网格、跨行长格、重叠格、嵌套格、同中心框和不规则布局。另直接执行固定的上游函数，与移植实现对照了随机抽取的 300 个 Manga109 物理页，输出全部一致（均使用相同的规范化输入顺序）。端到端测试覆盖页内重叠、完整双页顺序、连续窗口、稳定来源身份及 GT/provenance 写入。

对本地 `Manga109_released_2026_05_21.zip` 最新 `annotations/*.xml` 的全量扫描中，109 本书共 9,201 个至少六格的双页条目，8,840 个完成排序且保留全部 frame，361 个因 panel 跨页缝而跳过。该检查只执行标注排序，未重建图片数据，也未人工确认阅读顺序。相比旧 XY-cut 的 3,286 个通过条目，新算法能处理更多重叠布局；覆盖率提高不等于 GT 准确率已得到验证。

```bash
python -m pytest Manga109/test_manga_whisper_order.py Manga109/test_reading_order_utils.py Mangazero/test_build_dataset_reading_order.py config/test_cli.py
```

## 来源与使用条件

作者为 Ragav Sachdeva 和 Andrew Zisserman。上游 GitHub README 声明其提供的模型和数据集仅供学术研究，且没有单独列出本段 Python 源码的开源许可证；本项目保留来源与该声明，不将其描述为 MIT/Apache 代码。

```bibtex
@InProceedings{magiv1,
  author = {Sachdeva, Ragav and Zisserman, Andrew},
  title = {The Manga Whisperer: Automatically Generating Transcriptions for Comics},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024},
  pages = {12967--12976}
}
```
