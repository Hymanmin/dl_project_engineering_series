# 从 CNN 分类任务出发，搭建一个可扩展的深度学习工程框架

> 深度学习项目工程化实战系列 01

很多深度学习项目都是从一个 `train.py` 开始的。刚开始它很高效：读取数据、定义网络、计算 loss、反向传播、保存模型，全都写在一个文件里。但只要项目稍微复杂一点，问题就会很快出现：

- 换一个数据集，需要改数据读取逻辑。
- 换一个网络结构，需要改模型构建逻辑。
- 换一种 loss，需要在训练脚本里继续加分支。
- 想做推理、导出、多卡训练时，`train.py` 越来越臃肿。
- 实验参数散落在代码里，结果很难复现。

本文从一个最小 CNN 分类任务出发，搭建一个通用深度学习工程框架。重点不是追求复杂网络，而是把一个完整训练和推理闭环拆成稳定的工程模块：`config`、`dataset`、`network`、`model`、`loss`、`launcher` 和 `main`。

本文对应代码目录：

[GitHub：article_01_basic_framework/project](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_01_basic_framework/project)

## 1. 本篇最终工程目录

本篇最终得到的工程结构如下。注意这里不是只列空目录，而是把每个目录下的典型文件也列出来，因为深度学习工程真正需要维护的是“文件职责”和“调用关系”。

```text
project/
  main.py                         # 统一入口：解析 mode/config/opts，调度 train 或 predict
  requirements.txt                # 最小依赖
  run_main.sh                     # 训练启动脚本示例

  config/
    __init__.py
    loader.py                     # 配置加载器：支持 _base_ 继承和命令行覆盖
    base.yaml                     # 公共基础配置
    cnn_cls_train.yaml            # CNN 分类训练配置
    cnn_cls_predict.yaml          # CNN 分类推理配置

  datatxt/
    train.txt                     # 训练集样本列表
    val.txt                       # 验证集样本列表
    test.txt                      # 测试/推理样本列表

  launcher/
    __init__.py
    train.py                      # 训练流程：dataset/model/loss/optimizer/backward/checkpoint
    predict.py                    # 推理流程：加载 checkpoint，执行前向，保存预测结果

  core/
    __init__.py
    registry.py                   # DATASETS/NETWORKS/LOSSES/MODELS 注册表
    hooks.py                      # Hook 扩展点，后续可接入日志、EMA、评估等

    dataset/
      __init__.py
      base_dataset.py             # BaseDataset，定义数据集接口约定
      cnn_cls/
        __init__.py
        cnn_cls_dataset.py        # CNN 分类任务数据集

    networks/
      __init__.py
      factory.py                  # 网络构建入口
      cnn_cls/
        __init__.py
        backbone.py               # CNN backbone
        neck.py                   # 特征变换模块
        head.py                   # 分类头
        framework.py              # 组合 backbone/neck/head，注册完整网络

    losses/
      __init__.py
      cnn_cls_loss.py             # 分类损失函数

    models/
      __init__.py
      base_model.py               # BaseModel，定义模型接口
      cnn_cls/
        __init__.py
        cnn_cls_model.py          # 完整任务模型，组织前向逻辑

  utils/
    __init__.py
    logger.py                     # 日志工具
```

这个结构的核心思想是：稳定流程放在 `launcher/`，可替换组件放在 `core/`，实验参数放在 `config/`，入口统一放在 `main.py`。

## 2. 完整运行链路

一个训练任务的完整调用链如下：

```text
main.py
  -> load_config(config/cnn_cls_train.yaml)
  -> run_train(cfg)
      -> CnnClsDataset(cfg["dataset"])
      -> build_model(cfg["model"], cfg)
          -> CnnClsModel(cfg)
              -> build_network(cfg["network"])
                  -> CnnBackbone
                  -> CnnNeck
                  -> CnnHead
      -> build_loss(cfg["loss"])
      -> optimizer
      -> forward
      -> loss
      -> backward
      -> optimizer.step
      -> save checkpoint
```

推理任务的调用链如下：

```text
main.py
  -> load_config(config/cnn_cls_predict.yaml)
  -> run_predict(cfg)
      -> CnnClsDataset(cfg["dataset"], split="test")
      -> build_model(cfg["model"], cfg)
      -> load checkpoint
      -> forward
      -> argmax
      -> save predict_results.txt
```

训练和推理共用同一套配置加载、注册表、dataset、network 和 model。不同的只是 `launcher/train.py` 和 `launcher/predict.py` 定义的流程不同。

## 3. Config：把实验参数从代码里拿出来

配置文件负责描述“这次实验要使用哪些组件、组件参数是什么、运行参数是什么”。

基础配置 `config/base.yaml`：

```yaml
task: cnn_cls
seed: 42

runtime:
  device: cpu
  work_dir: runs/cnn_cls
  log_dir: runs/logs
  checkpoint_dir: runs/checkpoints

dataset:
  name: cnn_cls_dataset
  train_list: datatxt/train.txt
  val_list: datatxt/val.txt
  test_list: datatxt/test.txt
  batch_size: 8
  num_workers: 0
  input_dim: 16
  num_classes: 2

network:
  name: cnn_cls_network
  input_dim: 16
  hidden_dim: 32
  num_classes: 2

model:
  name: cnn_cls_model
  num_classes: 2

loss:
  name: cnn_cls_loss

optimizer:
  name: adam
  lr: 0.001
  weight_decay: 0.0
```

训练配置 `config/cnn_cls_train.yaml` 只覆盖训练相关参数：

```yaml
_base_: base.yaml

train:
  epochs: 3
  print_freq: 10
  save_name: cnn_cls.pt
```

配置加载器 `config/loader.py` 做了两件事：

1. 读取 `_base_`，把基础配置和当前配置做递归合并。
2. 解析命令行 `--opts`，支持临时覆盖配置。

例如：

```bash
python main.py --mode train --config config/cnn_cls_train.yaml --opts train.epochs=10 optimizer.lr=0.0005
```

这条命令不会修改 YAML 文件，但运行时会把 `train.epochs` 改成 `10`，把 `optimizer.lr` 改成 `0.0005`。

这就是配置驱动工程的第一步：常见实验变化不再通过改代码完成，而是通过配置文件和命令行完成。

## 4. Registry：让配置中的 name 找到对应代码

如果没有注册表，训练脚本里通常会出现大量分支：

```python
if cfg["network"]["name"] == "resnet18":
    model = ResNet18(...)
elif cfg["network"]["name"] == "mobilenet":
    model = MobileNet(...)
```

这样的代码很快会变得难维护。模板工程使用注册表来统一管理可插拔组件。

`core/registry.py` 中定义了四类注册表：

```python
DATASETS = Registry("dataset")
NETWORKS = Registry("network")
LOSSES = Registry("loss")
MODELS = Registry("model")
```

组件通过装饰器注册：

```python
@NETWORKS.register("cnn_cls_network")
class CnnClsNetwork(nn.Module):
    ...
```

配置中通过 `name` 指向注册名：

```yaml
network:
  name: cnn_cls_network
```

构建时统一调用：

```python
model = build_model(cfg["model"], cfg)
criterion = build_loss(cfg["loss"])
```

这样新增网络、数据集、loss 时，主训练流程不用新增大量 `if/else`。只要新组件完成注册，并在配置里写对 `name`，工程就能找到它。

## 5. Dataset：统一管理输入、标签和 batch 字典

数据集基类放在 `core/dataset/base_dataset.py`：

```python
class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
```

CNN 分类任务数据集放在：

```text
core/dataset/cnn_cls/cnn_cls_dataset.py
```

它负责读取 `datatxt/train.txt`、`datatxt/val.txt`、`datatxt/test.txt` 中的样本列表，并返回标准 batch 字典：

```python
{
    "input": input_tensor,
    "target": label_tensor,
}
```

为什么推荐返回字典，而不是直接返回 `(x, y)`？

因为字典更适合扩展。分类任务只需要 `input` 和 `target`，但检测任务可能需要 `boxes`，分割任务可能需要 `mask`，Transformer 任务可能需要 `input_ids` 和 `attention_mask`。使用字典后，训练流程可以通过字段名明确知道每个张量的语义。

当前示例为了让工程无数据也能跑通，如果列表文件不存在或为空，会自动生成合成数据。这适合模板验证。真实项目中，可以把这里替换为图片读取、数据增强和标签解析逻辑。

## 6. Network：用 backbone、neck、head 组合 CNN 网络

网络结构放在：

```text
core/networks/cnn_cls/
  backbone.py
  neck.py
  head.py
  framework.py
```

`backbone.py` 负责主干特征提取：

```python
class CnnBackbone(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
```

`neck.py` 负责中间特征变换：

```python
class CnnNeck(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
```

`head.py` 负责输出分类 logits：

```python
class CnnHead(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
```

`framework.py` 组合完整网络，并注册到 `NETWORKS`：

```python
@NETWORKS.register("cnn_cls_network")
class CnnClsNetwork(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, num_classes=2):
        super().__init__()
        self.backbone = CnnBackbone(input_dim=input_dim, hidden_dim=hidden_dim)
        self.neck = CnnNeck(hidden_dim=hidden_dim)
        self.head = CnnHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)
```

这个拆分方式适合 CNN 分类、检测、分割等任务。后续如果换成真实 CNN，可以把当前 `Linear` 示例替换成卷积模块、ResNet、MobileNet 或 ViT backbone。

## 7. Model：组织完整前向逻辑

`network` 只关心结构，`model` 负责把网络放进一个任务模型里。

模型基类：

```text
core/models/base_model.py
```

任务模型：

```text
core/models/cnn_cls/cnn_cls_model.py
```

核心代码：

```python
@MODELS.register("cnn_cls_model")
class CnnClsModel(BaseModel):
    def __init__(self, cfg: dict, num_classes: int = 2) -> None:
        super().__init__()
        network_cfg = dict(cfg.get("network", {}))
        network_cfg["num_classes"] = num_classes
        self.network = build_network(network_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

在简单分类任务里，`model.forward()` 只是调用 network。但保留 `model` 这一层很重要，因为复杂任务通常不只是一个网络前向：

- GAN 需要 generator 和 discriminator。
- Diffusion 需要加噪、采样 timestep、调用 denoiser。
- Transformer 可能需要处理 attention mask、position ids、labels。

所以工程上建议保持这条边界：

```text
network：描述网络结构
model：组织任务前向
launcher/trainer：处理 loss、backward、optimizer.step
```

## 8. Loss：把损失函数也作为可插拔组件

分类损失放在：

```text
core/losses/cnn_cls_loss.py
```

代码很简单：

```python
@LOSSES.register("cnn_cls_loss")
class CnnClsLoss(nn.CrossEntropyLoss):
    pass
```

配置中引用：

```yaml
loss:
  name: cnn_cls_loss
```

训练流程中构建：

```python
criterion = build_loss(cfg.get("loss", {}))
```

简单任务可以直接使用 `CrossEntropyLoss`。复杂任务建议让 loss 返回字典，例如：

```python
{
    "loss": total_loss,
    "loss_cls": cls_loss,
    "loss_reg": reg_loss,
}
```

这样日志、可视化和多任务训练会更清晰。

## 9. Launcher：训练和推理流程

`launcher/train.py` 负责完整训练流程：

```text
读取 dataset
构建 dataloader
构建 model
构建 loss
构建 optimizer
循环 epoch/step
前向
计算 loss
反向传播
参数更新
保存 checkpoint
```

关键代码：

```python
outputs = model(inputs)
loss = criterion(outputs, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`launcher/predict.py` 负责推理流程：

```text
构建 test dataset
构建 model
加载 checkpoint
model.eval()
前向推理
保存结果
```

训练和推理拆开的好处是：它们共享组件，但流程职责不同。后续还可以继续增加：

```text
launcher/eval.py        # 验证/测试集评估
launcher/export.py      # ONNX/TorchScript 导出
launcher/benchmark.py   # 吞吐、延迟、显存统计
```

这会在后续文章中继续展开。

## 10. Main：统一入口

`main.py` 是整个工程的入口。它不应该写具体训练细节，只负责三件事：

1. 解析命令行参数。
2. 加载配置。
3. 根据 `mode` 调度对应流程。

核心逻辑：

```python
args = parse_args()
cfg = load_config(Path(args.config), args.opts)

if args.mode == "train":
    run_train(cfg, logger)
else:
    run_predict(cfg, logger)
```

这种设计可以保证入口稳定。后续增加 `eval/export/benchmark` 时，也是在入口增加模式分发，而不是把所有逻辑写进 `main.py`。

## 11. 配置与代码的映射关系

工程中最重要的一条关系是：

```text
YAML 中的 name -> Registry 中的注册名 -> Python 类
```

以 network 为例：

配置：

```yaml
network:
  name: cnn_cls_network
```

代码注册：

```python
@NETWORKS.register("cnn_cls_network")
class CnnClsNetwork(nn.Module):
    ...
```

构建：

```python
network = build_network(cfg["network"])
```

dataset、model、loss 也是同样的模式：

```yaml
dataset:
  name: cnn_cls_dataset

model:
  name: cnn_cls_model

loss:
  name: cnn_cls_loss
```

对应：

```python
@DATASETS.register("cnn_cls_dataset")
class CnnClsDataset(BaseDataset):
    ...

@MODELS.register("cnn_cls_model")
class CnnClsModel(BaseModel):
    ...

@LOSSES.register("cnn_cls_loss")
class CnnClsLoss(nn.CrossEntropyLoss):
    ...
```

这就是后续扩展工程的基础。

## 12. 扩展一个 cat_dog_cls 任务后的目录结构

如果要在当前框架下新增一个猫狗分类任务，不建议直接改 `cnn_cls` 文件，而是新增一个独立任务目录。

扩展后的目录可以这样组织：

```text
project/
  config/
    base.yaml
    cat_dog_train.yaml            # 新增：猫狗分类训练配置
    cat_dog_predict.yaml          # 新增：猫狗分类推理配置

  datatxt/
    cat_dog_train.txt             # 新增：训练图片路径和标签
    cat_dog_val.txt               # 新增：验证图片路径和标签
    cat_dog_test.txt              # 新增：测试图片路径

  core/
    dataset/
      base_dataset.py
      cat_dog_cls/
        __init__.py               # 新增：导出 CatDogDataset
        cat_dog_dataset.py        # 新增：读取猫狗图片和标签

    networks/
      factory.py
      cat_dog_cls/
        __init__.py               # 新增：导出 CatDogNetwork
        backbone.py               # 新增：ResNet/MobileNet 主干
        neck.py                   # 新增：可选特征变换层
        head.py                   # 新增：二分类输出头
        framework.py              # 新增：组合完整 CNN 分类网络

    losses/
      cat_dog_loss.py             # 新增：猫狗分类损失，也可以复用通用 CE loss

    models/
      base_model.py
      cat_dog_cls/
        __init__.py               # 新增：导出 CatDogModel
        cat_dog_model.py          # 新增：组织完整前向逻辑
```

新增部分和任务的对应关系如下：

- `cat_dog_train.yaml`：指定猫狗任务使用哪个 dataset、network、model、loss。
- `cat_dog_dataset.py`：把图片路径和标签转成 batch 字典。
- `framework.py`：把 backbone、neck、head 组合成完整分类网络。
- `cat_dog_model.py`：调用网络完成前向，输出 logits。
- `cat_dog_loss.py`：定义或复用分类损失。
- `launcher/train.py`：不需要针对猫狗任务大改，只通过配置构建组件。

这就是框架扩展性的核心：新增任务主要新增组件文件和配置文件，而不是重写训练主流程。

## 13. 运行命令

进入本文配套工程：

```bash
cd dl_project_engineering_series/article_01_basic_framework/project
```

安装依赖：

```bash
pip install -r requirements.txt
```

训练：

```bash
python main.py --mode train --config config/cnn_cls_train.yaml
```

推理：

```bash
python main.py --mode predict --config config/cnn_cls_predict.yaml
```

如果 Windows 默认 `python` 不是安装 PyTorch 的环境，也可以指定 Python 版本：

```bash
py -3.11 main.py --mode train --config config/cnn_cls_train.yaml
py -3.11 main.py --mode predict --config config/cnn_cls_predict.yaml
```

运行后会生成：

```text
runs/
  checkpoints/
    cnn_cls.pt
  logs/
    run.log
  predict_results.txt
```

## 14. 本篇工程小结

到这里，我们已经把一个 CNN 分类训练脚本拆成了几个稳定模块：

- `config`：控制实验参数和组件选择。
- `registry`：把配置中的 `name` 映射到 Python 类。
- `dataset`：统一管理输入和标签。
- `network`：负责网络结构搭建。
- `model`：负责完整任务前向。
- `loss`：负责损失计算。
- `launcher`：负责训练和推理流程。
- `main`：负责统一入口和模式分发。

这个工程现在仍然很小，但它已经具备后续扩展的基础。下一篇会继续围绕配置驱动和注册表展开，重点解决一个问题：如何做到新增数据集、网络或损失函数时，尽量少改主流程代码。

## 15. GitHub 对应代码目录

本文代码位于：

[GitHub：article_01_basic_framework/project](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_01_basic_framework/project)

建议读者阅读文章时同步打开 GitHub 工程目录，对照每个文件理解调用关系。

## 系列规划

1. [从 CNN 分类任务出发，搭建一个可扩展的深度学习工程框架](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_01_basic_framework)
2. [配置驱动与注册表：让深度学习工程从 hard code 变成可组合](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_02_config_registry)
3. [训练流程插件化：把 train/eval/predict/export/benchmark 拆成独立模式](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_03_trainer_plugins)
4. [多卡训练实战：DataParallel、DDP 与 Pipeline 在工程中放在哪里](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_04_parallel_training)
5. [从 CNN 扩展到 GAN、Diffusion、Transformer：同一工程框架下的不同任务范式](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_05_task_paradigms)
6. [训练部署与模型导出：从 checkpoint 到 TorchScript、ONNX 和 benchmark](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_06_export_benchmark)
7. [并行测试框架搭建：让深度学习工程扩展后仍然可验证](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_07_testing)
