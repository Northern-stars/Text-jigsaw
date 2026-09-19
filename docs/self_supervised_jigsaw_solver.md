# JigsawPuzzlePytorch 技术文档

## 1. 项目概述

本仓库是论文 **Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles** 的 PyTorch 实现。项目使用 ImageNet 图像进行自监督预训练，不依赖图像类别标签来训练主任务，而是人为构造一个“拼图还原”任务：

1. 将一张图像裁剪成 3×3 的九个图块。
2. 使用预先选出的一个九图块排列对图块进行打乱。
3. 将打乱后的九个图块输入网络。
4. 让网络预测使用的是哪一个排列。

模型为了完成排列分类，需要学习图块之间的语义、结构和空间关系。因此，排列分类器之前的卷积特征可以作为通用视觉表征，后续可迁移到分类、检测等下游任务。

仓库的主要特点：

- 自监督学习，不使用 ImageNet 类别作为训练目标。
- 默认使用 1000 个九图块排列作为 1000 分类任务。
- 九个图块使用共享的卷积网络分别提取特征。
- 采用类似 AlexNet 的卷积结构，并保留 LRN 层。
- 使用 SGD、动量、权重衰减和分阶段学习率衰减。
- 代码以 Python 2.7、PyTorch 0.3 时代的 API 为主要目标，属于较早期实现。

## 2. 整体流程

```text
ImageNet 图像
      |
      v
缩放到约 256，中心裁剪为 255×255
      |
      v
划分为 3×3 九宫格
      |
      v
每个图块随机裁剪 64×64，再缩放到 75×75
      |
      v
颜色扰动、随机灰度化、每个图块独立归一化
      |
      v
按照 1000 个候选排列中的一个随机打乱
      |
      v
九路共享卷积网络
      |
      v
拼接九个图块特征
      |
      v
全连接层 + 1000 分类器
      |
      v
交叉熵损失：预测排列编号
```

训练标签不是 ImageNet 原始类别，而是排列在 `permutations_1000.npy` 中的行号。对于某张图像，如果随机抽到第 `k` 个排列，则训练标签为 `k`。

## 3. 自监督任务设计

### 3.1 九宫格切分

`Dataset/JigsawImageLoader.py` 对输入图像进行如下处理：

- `Resize(256)`：将图像缩放，使短边为 256，同时保持纵横比。
- `CenterCrop(255)`：中心裁剪为 255×255。
- 将图像按 3×3 划分成 9 个约 85×85 的区域。
- 对每个区域单独进行随机裁剪和缩放。

255 可以被 3 整除，因此可以比较稳定地划分出九个图块。每个图块后续会经过：

- `RandomCrop(64)`；
- `Resize((75, 75))`；
- 轻微 RGB 颜色扰动；
- `ToTensor()`；
- 按图块、按通道独立计算均值和标准差并归一化。

这里将图块输入尺寸设为 75×75，而不是论文描述中的 64×64。原因是仓库作者在 README 中说明：按照当前网络结构，如果直接使用 64×64，卷积和池化后的空间尺寸会变成 2×2，而全连接层按照 3×3 特征图构建，因此代码选择 75×75 以得到 3×3 输出。

### 3.2 排列标签

`permutations_1000.npy` 是一个形状为 `(1000, 9)` 的数组，保存 1000 个不同排列。当前文件中的编号范围为 0 到 8，每一行都是 0 到 8 的一个排列。

例如：

```text
[5, 8, 6, 1, 4, 2, 7, 3, 0]
```

表示打乱后的第 0 个位置取原始图块 5，第 1 个位置取原始图块 8，以此类推。

数据加载器中的逻辑等价于：

```python
order = random_integer(0, 1000)
shuffled_tiles = [tiles[p[order][i]] for i in range(9)]
label = order
```

因此，分类器的每个类别对应一个固定的空间排列，而不是对应一个自然图像类别。

### 3.3 为什么要选择相互差异较大的排列

理论上九个图块共有 `9! = 362880` 个排列，但全部使用会导致分类规模过大，也会产生许多过于相近的类别。`select_permutations.py` 尝试通过排列之间的平均 Hamming 距离选择一组差异较大的排列：

1. 枚举全部 9! 个排列。
2. 随机选择第一个排列。
3. 每轮计算候选排列与已选排列之间的平均 Hamming 距离。
4. 默认选择平均距离最大的候选排列。
5. 重复直到选择指定数量。

默认配置是选择 1000 个排列，生成的目标是让不同分类类别尽可能容易区分。

## 4. 网络结构

实现位于 `JigsawNetwork.py`，类名为 `Network`。

### 4.1 九路共享卷积编码器

网络接收形状为：

```text
(B, 9, 3, 75, 75)
```

其中：

- `B`：batch size；
- `9`：图块数量；
- `3`：RGB 通道；
- `75×75`：图块空间尺寸。

前向传播首先将输入转为 `(9, B, 3, 75, 75)`，然后循环处理 9 个图块。九个分支调用的是同一个 `self.conv` 和同一个 `self.fc6`，因此它们共享参数，属于典型的 Siamese/共享权重结构。

### 4.2 卷积部分

| 阶段 | 结构 | 输出通道/特征 |
|---|---|---:|
| conv1 | 11×11 卷积，stride=2，ReLU，3×3 最大池化，LRN | 96 |
| conv2 | 5×5 分组卷积，groups=2，ReLU，3×3 最大池化，LRN | 256 |
| conv3 | 3×3 卷积，ReLU | 384 |
| conv4 | 3×3 分组卷积，groups=2，ReLU | 384 |
| conv5 | 3×3 分组卷积，groups=2，ReLU，3×3 最大池化 | 256 |

卷积模块使用了较早期 AlexNet 风格的设计：

- 大卷积核；
- 分组卷积；
- ReLU；
- LRN（Local Response Normalization）。

`Utils/Layers.py` 中的 `LRN` 使用 `AvgPool3d` 在通道维度上计算局部响应归一化因子。

### 4.3 全连接与分类器

对每一个图块：

```text
卷积输出 256×3×3
        |
        v
fc6: 256×3×3 -> 1024
ReLU + Dropout(0.5)
```

九个图块各自产生一个 1024 维特征，拼接后得到：

```text
9×1024 = 9216
        |
        v
fc7: 9216 -> 4096
ReLU + Dropout(0.5)
        |
        v
fc8: 4096 -> classes
```

默认 `classes=1000`，所以最终输出形状为：

```text
(B, 1000)
```

输出经过交叉熵损失训练，目标是预测排列编号。

## 5. 数据加载与增强

### 5.1 默认数据加载器

训练脚本默认导入：

```python
from JigsawImageLoader import DataLoader
```

该加载器继承 `torch.utils.data.Dataset`，由 PyTorch 的 `DataLoader` 负责 batch 组装、随机打乱和多进程加载。

每次调用 `__getitem__` 时，都会重新随机生成：

- 是否转为灰度图；
- 每个图块的随机裁剪位置；
- RGB 颜色扰动；
- 图块排列编号。

这意味着同一张原始图片在不同 epoch 或不同访问时，通常会产生不同的训练样本。

### 5.2 颜色与灰度增强

当前默认实现包含：

- 30% 概率将图像转为灰度后再转回 RGB；
- 每个 RGB 通道增加一个很小的随机偏移；
- 每个图块单独标准化。

图块独立归一化的目的，是减少网络通过低级颜色、亮度或边缘统计直接猜测图块位置的机会，迫使模型更多利用语义和结构信息。

### 5.3 备用加载器

`Dataset/ImageDataLoader.py` 是一个早期的自定义迭代器版本。README 声明它在单 CPU 核心场景下可能更快，但当前训练脚本没有使用它。

它与默认加载器存在行为差异：

- 中心裁剪尺寸写为 225，而不是 255；
- 使用 ImageNet 均值和标准差，而不是每个图块独立归一化；
- 返回接口和现代 PyTorch `DataLoader` 不完全一致；
- 主要面向 Python 2 的迭代器接口。

因此它更像是历史备选实现，不建议直接与默认数据管线混用。

## 6. 训练流程

训练入口是 `JigsawTrain.py`。

### 6.1 数据目录约定

脚本期望传入 ImageNet 根目录，并在其中查找：

```text
<data>/
├── ILSVRC2012_img_train/
├── ILSVRC2012_img_val/
├── ILSVRC2012_img_train_255x255/  # 可选预处理目录
├── ILSVRC2012_img_val_255x255/    # 可选预处理目录
├── ilsvrc12_train.txt
└── ilsvrc12_val.txt
```

如果存在带 `_255x255` 后缀的目录，训练脚本优先使用该目录。文本文件每行包含：

```text
相对图片路径 ImageNet类别编号
```

原始 ImageNet 类别编号会被读取，但本项目的自监督训练目标不会使用它。

### 6.2 优化配置

默认参数：

| 参数 | 默认值 |
|---|---:|
| 训练 epoch | 70 |
| batch size | 256 |
| 类别数 | 1000 |
| 初始学习率 | 0.001 |
| 优化器 | SGD |
| momentum | 0.9 |
| weight decay | 5e-4 |
| 学习率衰减 | 每 20 个 epoch 乘以 0.1 |

损失函数为：

```text
CrossEntropyLoss(logits, permutation_index)
```

训练过程中每 20 个 step 打印并记录一次 loss 和 top-1 accuracy。每 10 个 epoch 在验证集上评估一次，每 1000 个 step 保存一个 checkpoint。

### 6.3 验证方式

验证集仍然执行同样的自监督任务：随机生成图块排列，然后判断模型能否预测排列编号。因此验证准确率衡量的是排列识别能力，不是 ImageNet 物体分类准确率。

`compute_accuracy` 同时计算 top-1 和 top-5，但训练日志和验证输出当前只使用 top-1 结果。

### 6.4 Checkpoint 与恢复

checkpoint 文件名格式为：

```text
jps_<epoch>_<step>.pth.tar
```

如果 `--checkpoint` 目录中已经存在包含 `pth` 的文件，脚本会按文件名排序并加载最后一个文件，同时从文件名中推断起始 step。

当前保存的是模型 `state_dict`，没有保存：

- 优化器状态；
- 当前 epoch 的完整状态；
- 随机数状态；
- 学习率调度器状态。

因此恢复训练属于“加载模型权重后继续训练”，不等价于完整断点续训。

## 7. 文件职责

| 文件 | 作用 |
|---|---|
| `JigsawTrain.py` | 训练、验证、日志和 checkpoint 管理 |
| `JigsawNetwork.py` | 九路共享卷积网络和分类器 |
| `Dataset/JigsawImageLoader.py` | 默认 PyTorch Dataset，生成拼图自监督样本 |
| `Dataset/ImageDataLoader.py` | 历史自定义批量加载器 |
| `Dataset/produce_small_data.py` | 将原始图像预处理为 255×255 并保存 |
| `Utils/Layers.py` | LRN 层实现 |
| `Utils/TrainingUtils.py` | 学习率调整和 top-k 准确率计算 |
| `Utils/logger.py` | 使用 TensorFlow summary 写入标量和图片日志 |
| `select_permutations.py` | 从 9! 个排列中选择差异较大的排列 |
| `permutations_1000.npy` | 默认使用的 1000 个排列 |
| `Utils/convert2h5.py` | 尝试将模型权重导出为 HDF5 |
| `run_jigsaw_training.sh` | 训练命令模板 |

## 8. 运行方式

### 8.1 直接训练

```bash
python JigsawTrain.py <path_to_imagenet> \
  --checkpoint <path_to_checkpoint> \
  --gpu 0 \
  --batch 256 \
  --classes 1000 \
  --epochs 70 \
  --cores 4
```

仓库提供的 shell 脚本需要先修改：

```bash
IMAGENET_FOLD=path_to_ILSVRC2012_img
```

然后执行：

```bash
./run_jigsaw_training.sh <GPU_ID>
```

### 8.2 仅评估

```bash
python JigsawTrain.py <path_to_imagenet> \
  --checkpoint <path_to_checkpoint> \
  --gpu 0 \
  --evaluate
```

评估输出是排列分类准确率，不是 ImageNet 监督分类准确率。

### 8.3 生成预处理数据

编辑 `Dataset/produce_small_data.py` 中的 `datapath` 和 `trainval`，然后运行：

```bash
python Dataset/produce_small_data.py
```

脚本会创建类似以下目录：

```text
ILSVRC2012_img_train_255x255/
ILSVRC2012_img_val_255x255/
```

## 9. 当前实现的注意事项与风险

### 9.1 版本较旧

README 明确以 Python 2.7、PyTorch 0.3 为测试环境。当前代码使用了多处旧式 API 或旧式写法，例如：

- `Variable`；
- 旧版 `torchvision.transforms` 行为；
- `scipy.misc.toimage`；
- TensorFlow 旧版 summary API；
- 依赖 Python 2 整数除法语义的坐标计算；
- 旧版权重初始化接口。

在现代 Python、PyTorch、torchvision、TensorFlow 或 SciPy 环境中，不能直接假设可以运行。

### 9.2 CPU 模式参数不完整

`--gpu` 默认值为 0，代码判断 `args.gpu is not None` 后就会调用 CUDA。因此不传参数时默认尝试使用 GPU。若要支持现代 CPU-only 运行，需要显式改造设备选择逻辑。

### 9.3 排列生成脚本和加载器路径不一致

`select_permutations.py` 的输出路径形如：

```text
permutations/permutations_hamming_max_1000.npy
```

而数据加载器读取的是：

```text
permutations_1000.npy
```

生成排列后，需要手动移动或重命名文件，或者修改其中一方的路径约定。

### 9.4 `convert2h5.py` 当前不可直接使用

该脚本存在与主网络不一致的历史代码痕迹，包括：

- 导入路径 `JpsTraininig` 与仓库目录不一致；
- 实例化 `Network` 时传入了当前构造函数不支持的 `groups` 参数；
- 依赖旧版模型结构和旧版权重格式。

它应被视为未维护的辅助脚本，而不是当前稳定导出流程。

### 9.5 恢复训练不完整

checkpoint 只保存模型权重，不保存优化器和随机状态。恢复后学习率、动量缓存和数据随机序列不会与中断前完全一致。

### 9.6 日志依赖额外组件

训练脚本会导入 TensorFlow，日志实现还依赖 SciPy 的旧图片接口。README 提到如果没有 TensorFlow，需要移除 Logger 相关逻辑；在现代环境中还需要替换图片写入方式。

### 9.7 训练脚本中的若干历史遗留

以下变量或逻辑目前没有发挥完整作用：

- `N` 被赋值但未使用；
- `criterion` 在验证函数中传入但未使用；
- `tqdm` 被导入但训练循环未使用；
- `prec5` 被计算但没有记录；
- `weights_init` 定义了但没有启用。

## 10. 方案总结

这个仓库的核心方案可以概括为：

> 用图像自身的空间一致性构造监督信号。网络不需要知道图像是什么类别，只需要判断九个图块被怎样打乱；为了完成该任务，网络被迫学习局部图块之间的语义和空间关系，从而获得可迁移的视觉特征。

从工程结构上看，项目分为四层：

1. **样本构造层**：裁剪九宫格、增强、归一化、排列打乱。
2. **特征提取层**：九个图块共享一套卷积编码器。
3. **关系建模层**：拼接九个局部特征，用全连接层建模整体排列关系。
4. **训练管理层**：交叉熵优化、验证、日志和 checkpoint。

从当前仓库状态看，它更适合作为论文方法和早期 PyTorch 自监督学习实现的参考代码。若要在现代环境中复现，应优先处理依赖升级、设备管理、Python 3 坐标计算、日志系统、checkpoint 完整保存，以及排列文件路径统一等问题。
