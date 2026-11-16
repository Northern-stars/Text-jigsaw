# NAR checkpoint 检查记录（2026-11-16）

检查对象：`Solver/checkpoints/qwen_vl_nonautoregressive/last.pt`，以及同目录的 `config.json`、`startup_config.json`、`metrics.jsonl` 和 `run.log`。

结论：文件和优化器状态没有发现非有限值；训练确实执行了更新。保存的 decoder 在合成输入下表现出明显的位置表示趋同和加性打分头 `tanh` 饱和，与日志中的随机水平性能一致。尚不能仅凭 checkpoint 确定最初触发退化的原因。

## 1. 实际训练配置与日志

- 保存于 epoch 17；backbone 和 head 的优化器状态均为 17,000 步。
- 6 个 panel，首尾已知，只排序中间 4 个。
- train/val/test：8,000 / 1,000 / 1,000。
- batch size 1，gradient accumulation 8，每轮 1,000 次 optimizer 更新。
- backbone 学习率 `1e-4`，decoder 学习率 `1e-3`，clip norm 1，weight decay 0。
- 默认 backbone 为 `Qwen/Qwen2.5-VL-3B-Instruct`，训练 LoRA；decoder 维度 256。

以下配置来自本次运行保存的配置，不能用当前仓库的默认训练配置替代。

| Epoch | Train loss | Val exact match |
| --- | ---: | ---: |
| 1 | 2.77265810 | 0.047 |
| 2 | 2.77258979 | 0.042 |
| 4 | 2.77261616 | 0.050 |
| 17 | 2.77258873 | 0.044 |

最后四轮验证结果完全相同：exact match `0.044`，position accuracy `0.24875`，pairwise accuracy `0.49466668`。停滞从最初几轮就存在，并非第 14 轮才开始。

当前 loss 为行、列两个交叉熵之和，列权重为 1。中间 4 个 panel 均匀预测的基线为：

`loss = 2 × ln(4) = 2.772588722`；exact match `1/24 ≈ 0.04167`；position accuracy `0.25`；pairwise accuracy `0.5`。

## 2. 权重与优化器完整性

- 751 个保存的权重张量全部有限，没有 NaN/Inf。
- 两组优化器的一阶、二阶矩全部有限，每个参数状态都记录了 17,000 步。
- 348 个 `lora_B` 张量全部非零，与标准零初始化相比发生过更新。
- backbone/head 的一阶矩合并 L2 范数分别约为 `8.75e-7` / `3.74e-7`，说明保存时更新信号已很小。这是 Adam 一阶矩，不能等同于最后一个 batch 的实际梯度。
- 原始中间位置 query 参数仍有差异，余弦相似度均值约 `0.129`。趋同主要出现在后续 decoder 和 query 投影输出中，不能描述为位置参数本身全部相同。

优化器步数和非零 LoRA 排除了“完全没有执行更新”这一解释，但不能证明训练数据、标签和真实特征均正确。

## 3. 不依赖数据的 decoder 对照

使用 CPU，仅重建保存的 projection、LayerNorm 和 decoder，不加载 Qwen。对新初始化 decoder 和 checkpoint decoder 输入同一组 16 个随机 memory，每个样本 6 个 256 维向量；两个 decoder 都处于 eval 模式。输入使用同一份保存的 panel LayerNorm。中间 panel 向量余弦相似度均值为 `0.0105`，输入本身有明显差异。

| 指标 | 新初始化 decoder | Checkpoint decoder |
| --- | ---: | ---: |
| 中间位置 decoder 输出余弦相似度均值 | 0.930925 | 0.997573 |
| 中间位置投影后 query 余弦相似度均值 | 0.926562 | 0.999765 |
| Query 向量平均范数 | 9.13 | 72.73 |
| Key 向量平均范数 | 9.24 | 15.78 |
| 全部位置-panel 打分的 `abs(query + key) > 3` 比例 | 0.034% | 85.14% |
| 全部位置-panel 打分的平均 tanh 导数 | 0.681 | 0.024 |
| 中间位置-panel 分数交互 RMS | 0.01732 | 0.00291 |
| 中间 panel 行分布平均熵 | 1.37071 | 1.38567 |

4 类均匀分布的最大熵为 `ln(4) = 1.386294`。分数交互 RMS 是分数矩阵去掉行均值、列均值并加回全局均值后的 RMS，用于衡量真正依赖“哪个位置匹配哪个 panel”的部分。

Checkpoint 的一个中间分数矩阵为：

```text
[-3.960634, -4.029891, -4.029983, -3.981936]
[-3.960613, -4.029849, -4.029932, -3.981961]
[-3.960613, -4.029849, -4.029932, -3.981961]
[-3.953552, -4.027818, -4.026631, -3.976425]
```

不同位置的分数几乎相同，位置相关的区分能力很弱。高幅度且相近的 query 使加性头进入 tanh 饱和区，进一步削弱梯度和 panel 差异。这里展示的是合成输入上的现象；不能断言真实数据或任意输入都产生同样矩阵。打分层共同 bias 约为 `-3.757`，共同平移不影响 softmax/交叉熵，其负值本身不是异常原因。

额外将随机 2048 维 hidden 经保存的 projection/norm 输入 decoder，也出现高 query 相似度（约 `0.99922`）和饱和，但行熵约为 `1.23958`。因此不能把“所有输入都严格均匀输出”当作已确认事实。

## 4. 固定合成小样本拟合

对同一组 16 个 memory 和有效的随机中间排序标签训练 100 步。只训练 decoder，两组各自新建 AdamW（学习率 `1e-3`，clip norm 1），不复用 checkpoint 优化器；训练使用 dropout。下表 loss 为 train 模式记录，最终 exact match 为 eval 模式。

| 指标 | 新初始化 decoder | Checkpoint decoder |
| --- | ---: | ---: |
| Step 0 loss | 2.79627 | 2.77309 |
| Step 0 位置 query 梯度范数 | 0.56403 | 0.00269 |
| Step 50 loss | 0.01200 | 2.00566 |
| Step 100 loss | 0.00313 | 1.88285 |
| 最终 exact match | 100% | 12.5% |

新 decoder 可以拟合，说明当前 decoder/loss 在这些人工特征和标签上具有学习能力。保存的 decoder 初始 query 梯度小约 210 倍、拟合明显慢，但也能部分恢复，不能认定它已经不可恢复。此实验没有验证真实 Qwen 特征、真实标签或数据加载。

## 5. 后续验证建议

1. 优先冻结 backbone，重置 decoder 和优化器，用 16–32 个固定真实 puzzle 做过拟合检查，同时确认 panel/target 对齐和不同 panel 的 pooled feature 是否有差异。
2. 记录真实输入上的 query/key 范数、位置输出相似度、tanh 饱和比例、行列熵，以及 backbone/head 梯度。不要只记录总 loss。
3. 从新初始化开始对照较小 decoder 学习率（如 `1e-4`）与 warmup；当前 `1e-3` 可能放大表示趋同与饱和，但仅凭最终 checkpoint 尚不能确定学习率是唯一原因。
4. 如真实数据上复现退化，再对照 query/key 归一化、缩放点积打分、加强位置条件等方案，每次只改一个因素。

不建议直接沿用同一 checkpoint 和优化器继续长时间训练。此次检查没有修改模型架构、训练流程或原 checkpoint。

## 6. 复查方式

```powershell
python Solver/diagnose_nar_checkpoint.py --checkpoint Solver/checkpoints/qwen_vl_nonautoregressive/last.pt
```

默认在 checkpoint 同目录保存 `diagnosis.json`，包含完整参数统计、优化器统计、合成探针及训练历史；可以通过 `--output` 指定其他 JSON 路径，通过 `--steps` 调整合成拟合步数。脚本仅在内存中更新实验 decoder，不写回 checkpoint。

限制：本地没有实际数据集及 Qwen 基础权重，本次无法复现真实验证集推理，也无法从最终 checkpoint 恢复早期训练的因果过程。
