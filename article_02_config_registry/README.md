# 配置驱动与注册表：让深度学习工程从 hard code 变成可组合

> 深度学习项目工程化实战系列 02

上一篇我们搭建了一个最小 CNN 分类工程，把训练闭环拆成了 `config`、`dataset`、`network`、`model`、`loss` 和 `launcher`。这一篇继续解决一个更实际的问题：

> 当我要新增一个数据集、一个网络结构或一个损失函数时，能不能尽量不改训练主流程？

答案是可以。核心方法就是两件事：

1. 用配置文件描述本次实验要使用的组件。
2. 用注册表把配置中的 `name` 映射到具体 Python 类。

本文对应工程目录：

```text
dl_project_engineering_series/
  article_02_config_registry/
    README.md
    project/
```

## 1. 本篇最终代码目录

第 2 篇在第 1 篇基础上增加了 `core/imports.py` 和一个可切换的 tiny 网络配置。

```text
project/
  main.py                         # 新增 import_modules 调用

  config/
    loader.py                     # 配置加载器：_base_ 继承和 --opts 覆盖
    base.yaml                     # 新增 imports 字段，集中声明注册模块
    cnn_cls_train.yaml            # 默认 CNN 分类训练配置
    cnn_cls_tiny_train.yaml       # 新增：切换到 tiny 网络的训练配置
    cnn_cls_predict.yaml          # 推理配置

  core/
    imports.py                    # 新增：按配置导入模块，触发注册表注册
    registry.py                   # DATASETS/NETWORKS/LOSSES/MODELS

    dataset/
      base_dataset.py
      cnn_cls/
        cnn_cls_dataset.py        # @DATASETS.register("cnn_cls_dataset")

    networks/
      cnn_cls/
        backbone.py
        neck.py
        head.py
        framework.py              # 注册 cnn_cls_network 和 cnn_cls_tiny_network

    losses/
      cnn_cls_loss.py             # @LOSSES.register("cnn_cls_loss")

    models/
      cnn_cls/
        cnn_cls_model.py          # @MODELS.register("cnn_cls_model")

  launcher/
    train.py
    predict.py
```

## 2. 本篇要解决的工程问题

没有注册表时，训练代码经常会变成这样：

```python
if cfg["network"]["name"] == "resnet18":
    network = ResNet18(...)
elif cfg["network"]["name"] == "mobilenet":
    network = MobileNet(...)
elif cfg["network"]["name"] == "vit":
    network = VisionTransformer(...)
else:
    raise ValueError(...)
```

这类代码的问题很明显：

- 新增组件必须修改主流程。
- 分支越来越多，训练代码越来越难读。
- 组件参数和构建逻辑混在一起。
- 不利于复现实验，因为“用了哪个组件”没有完全写在配置里。

注册表的目标是让主流程只做一件事：

```python
network = build_network(cfg["network"])
```

具体构建哪个网络，由配置中的 `name` 决定。

## 3. 配置继承：让实验配置只写变化部分

基础配置 `config/base.yaml` 保存公共参数：

```yaml
task: cnn_cls
seed: 42

imports:
  - core.dataset.cnn_cls.cnn_cls_dataset
  - core.networks.cnn_cls.framework
  - core.losses.cnn_cls_loss
  - core.models.cnn_cls.cnn_cls_model

network:
  name: cnn_cls_network
  input_dim: 16
  hidden_dim: 32
  num_classes: 2
```

训练配置 `config/cnn_cls_train.yaml` 只写训练相关内容：

```yaml
_base_: base.yaml

train:
  epochs: 3
  print_freq: 10
  save_name: cnn_cls.pt
```

如果要切换成 tiny 网络，新建 `config/cnn_cls_tiny_train.yaml`：

```yaml
_base_: cnn_cls_train.yaml

network:
  name: cnn_cls_tiny_network
  input_dim: 16
  hidden_dim: 16
  num_classes: 2

train:
  epochs: 2
  print_freq: 10
  save_name: cnn_cls_tiny.pt
```

这里没有改 `launcher/train.py`，只是把 `network.name` 从 `cnn_cls_network` 换成了 `cnn_cls_tiny_network`。

## 4. imports：解决注册表必须先 import 的问题

注册表模式有一个容易被忽略的点：只有模块被 import 后，装饰器才会执行。

例如：

```python
@NETWORKS.register("cnn_cls_network")
class CnnClsNetwork(nn.Module):
    ...
```

如果 `core.networks.cnn_cls.framework` 从来没有被 import，那么 `cnn_cls_network` 就不会注册到 `NETWORKS` 中。

因此第 2 篇新增了：

```text
core/imports.py
```

代码很简单：

```python
import importlib


def import_modules(modules: list[str]) -> None:
    for module in modules:
        importlib.import_module(module)
```

然后在 `main.py` 中加载配置后调用：

```python
cfg = load_config(Path(args.config), args.opts)
import_modules(cfg.get("imports", []))
```

这样，配置文件中的 `imports` 字段就变成了“注册模块清单”。新增任务时，只需要把新模块路径加进配置。

## 5. 注册表：从 name 到 Python 类

注册表文件：

```text
core/registry.py
```

核心结构：

```python
DATASETS = Registry("dataset")
NETWORKS = Registry("network")
LOSSES = Registry("loss")
MODELS = Registry("model")
```

每类组件都有自己的注册表。以网络为例：

```python
@NETWORKS.register("cnn_cls_network")
class CnnClsNetwork(nn.Module):
    ...


@NETWORKS.register("cnn_cls_tiny_network")
class CnnClsTinyNetwork(nn.Module):
    ...
```

配置中选择哪个：

```yaml
network:
  name: cnn_cls_tiny_network
```

构建时：

```python
network = build_network(network_cfg)
```

这条映射关系非常重要：

```text
config.network.name
  -> "cnn_cls_tiny_network"
  -> NETWORKS.get("cnn_cls_tiny_network")
  -> CnnClsTinyNetwork(...)
```

dataset、model、loss 也是同样逻辑。

## 6. 新增网络组件后的目录结构

如果继续新增一个 `cnn_cls_wide_network`，目录可以保持不变，只需要在 `framework.py` 里新增一个注册类：

```text
project/
  core/
    networks/
      cnn_cls/
        backbone.py
        neck.py
        head.py
        framework.py              # 新增：CnnClsWideNetwork 注册类

  config/
    cnn_cls_wide_train.yaml       # 新增：选择 cnn_cls_wide_network 的训练配置
```

新增配置：

```yaml
_base_: cnn_cls_train.yaml

network:
  name: cnn_cls_wide_network
  input_dim: 16
  hidden_dim: 128
  num_classes: 2
```

新增类：

```python
@NETWORKS.register("cnn_cls_wide_network")
class CnnClsWideNetwork(nn.Module):
    ...
```

训练命令：

```bash
python main.py --mode train --config config/cnn_cls_wide_train.yaml
```

主训练流程不需要知道 `wide` 网络存在，只需要通过注册表构建它。

## 7. 运行命令

默认网络训练：

```bash
cd dl_project_engineering_series/article_02_config_registry/project
python main.py --mode train --config config/cnn_cls_train.yaml
```

切换 tiny 网络训练：

```bash
python main.py --mode train --config config/cnn_cls_tiny_train.yaml
```

命令行临时覆盖：

```bash
python main.py --mode train --config config/cnn_cls_tiny_train.yaml --opts train.epochs=1 optimizer.lr=0.01
```

## 8. 本篇工程小结

这一篇解决了深度学习工程扩展中最核心的问题之一：如何让新增组件不污染主流程。

现在工程具备了三层解耦：

```text
YAML 配置决定使用哪个组件
Registry 负责从 name 找到 Python 类
launcher/train.py 只负责训练流程，不关心具体组件类型
```

后续新增 dataset、network、loss、model 时，主要工作是：

1. 新增组件文件。
2. 用装饰器注册组件。
3. 在配置中写入注册名。
4. 在 `imports` 中加入模块路径。

下一篇会继续把训练流程本身插件化，把 `train/eval/predict/export/benchmark` 拆成更清晰的模式。

## 9. GitHub 对应代码目录

[GitHub：article_02_config_registry/project](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_02_config_registry/project)
