# 训练流程插件化：把 train/eval/predict/export/benchmark 拆成独立模式

> 深度学习项目工程化实战系列 03

前两篇解决了工程骨架、配置驱动和注册表问题。但如果继续把所有训练细节都写在 `launcher/train.py` 里，工程仍然会膨胀。

普通 CNN 分类任务可能只需要：

```text
one model + one optimizer + one loss
```

但 GAN 需要：

```text
generator + discriminator + two optimizers + D step + G step
```

Diffusion 又需要：

```text
timestep sampling + add noise + denoise + EMA + sampling
```

如果这些逻辑都塞进同一个 `train.py`，训练流程会变成新的大杂烩。本文的目标是把训练流程插件化，引入 `TRAINERS` 注册表，并把 `train/eval/predict/export/benchmark` 拆成独立模式。

本文对应工程目录：

```text
dl_project_engineering_series/
  article_03_trainer_plugins/
    README.md
    project/
```

## 1. 本篇最终代码目录

```text
project/
  main.py                         # mode 扩展为 train/eval/predict/export/benchmark

  config/
    base.yaml                     # 新增 trainer 配置
    cnn_cls_train.yaml
    cnn_cls_predict.yaml

  core/
    registry.py                   # 新增 TRAINERS 和 build_trainer
    trainers/
      __init__.py
      base_trainer.py             # BaseTrainer，定义 train/evaluate 接口
      classification_trainer.py   # CNN 分类 Trainer

    dataset/
      cnn_cls/
        cnn_cls_dataset.py
    networks/
      cnn_cls/
        framework.py
    losses/
      cnn_cls_loss.py
    models/
      cnn_cls/
        cnn_cls_model.py

  launcher/
    train.py                      # 只负责构建 trainer 并调用 train()
    eval.py                       # 构建 trainer 并调用 evaluate()
    predict.py                    # 推理流程
    export.py                     # 导出流程占位
    benchmark.py                  # 性能测试流程
```

## 2. Trainer 为什么需要注册表

第 2 篇中我们已经把 dataset、network、loss、model 变成了可注册组件。训练流程也应该同样处理。

在 `core/registry.py` 中新增：

```python
TRAINERS = Registry("trainer")


def build_trainer(cfg: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
    return TRAINERS.build(cfg, *args, **kwargs)
```

配置中新增：

```yaml
trainer:
  name: classification_trainer
  amp: false
  grad_accum_steps: 1
```

Trainer 注册：

```python
@TRAINERS.register("classification_trainer")
class ClassificationTrainer(BaseTrainer):
    ...
```

构建：

```python
trainer = build_trainer(cfg["trainer"], cfg, logger)
```

这样后续新增 `gan_trainer`、`diffusion_trainer`、`transformer_trainer` 时，不需要把所有逻辑都写进一个通用训练函数。

## 3. BaseTrainer：定义训练流程接口

`core/trainers/base_trainer.py`：

```python
class BaseTrainer(ABC):
    def __init__(self, cfg: dict, logger) -> None:
        self.cfg = cfg
        self.logger = logger

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError
```

这里没有规定每个 Trainer 必须怎么训练，只规定它至少应该有 `train()`，并且可以实现 `evaluate()`。

这个设计给不同任务留下空间：

- CNN 分类：一个优化器，一个 loss。
- GAN：两个优化器，两个 step。
- Diffusion：噪声调度、EMA、采样。
- Transformer：AMP、梯度累积、warmup scheduler。

## 4. ClassificationTrainer：把 CNN 分类训练逻辑收进去

`core/trainers/classification_trainer.py` 负责 CNN 分类任务完整训练：

```text
构建 model
构建 loss
构建 optimizer
构建 dataloader
执行 forward/loss/backward/step
保存 checkpoint
```

核心代码：

```python
@TRAINERS.register("classification_trainer")
class ClassificationTrainer(BaseTrainer):
    def __init__(self, cfg: dict, logger) -> None:
        super().__init__(cfg, logger)
        self.device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
        self.model = build_model(cfg.get("model", {}), cfg).to(self.device)
        self.criterion = build_loss(cfg.get("loss", {}))
        self.optimizer = _build_optimizer(self.model.parameters(), cfg)
```

训练 step：

```python
outputs = self.model(inputs)
loss = self.criterion(outputs, targets)

self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

评估：

```python
pred = logits.argmax(dim=1)
acc = correct / total
```

这样 `launcher/train.py` 就不用关心分类训练细节。

## 5. Launcher：流程入口变薄

改造后 `launcher/train.py` 只剩下：

```python
from core.registry import build_trainer


def run_train(cfg: dict, logger) -> None:
    trainer = build_trainer(cfg.get("trainer", {}), cfg, logger)
    trainer.train()
```

`launcher/eval.py`：

```python
def run_eval(cfg: dict, logger) -> None:
    trainer = build_trainer(cfg.get("trainer", {}), cfg, logger)
    trainer.evaluate()
```

这就是插件化后的好处：launcher 只负责调度，具体训练逻辑由 Trainer 决定。

## 6. 五种模式的职责边界

本文把入口扩展为：

```text
train | eval | predict | export | benchmark
```

它们的职责应该明确区分：

```text
train
  训练模型，保存 checkpoint

eval
  在验证集或测试集上计算指标

predict
  加载 checkpoint，对样本输出预测结果

export
  把模型导出成 TorchScript、ONNX、TensorRT 等部署格式

benchmark
  测试吞吐、延迟、显存占用
```

不要把这些逻辑都混进 `train.py`。否则后续部署和测试时，训练脚本会越来越难维护。

## 7. 新增 GAN Trainer 后的目录结构

假设后续新增 GAN 任务，不应该改 `ClassificationTrainer`，而应该新增：

```text
project/
  core/
    trainers/
      base_trainer.py
      classification_trainer.py
      gan_trainer.py              # 新增：GAN 双优化器训练流程

    networks/
      gan_toy/
        generator.py              # 新增：生成器
        discriminator.py          # 新增：判别器
        framework.py              # 新增：组合完整 GAN 网络

    models/
      gan_toy/
        gan_model.py              # 新增：暴露 generator/discriminator

    losses/
      gan_loss.py                 # 新增：D loss 和 G loss

  config/
    gan_toy_train.yaml            # 新增：选择 gan_trainer
```

配置：

```yaml
trainer:
  name: gan_trainer

network:
  name: gan_network

model:
  name: gan_model

loss:
  name: gan_loss
```

主入口 `main.py` 不需要为 GAN 增加专门分支，因为它仍然只是：

```python
run_train(cfg, logger)
```

真正不同的是 Trainer。

## 8. 运行命令

训练：

```bash
cd dl_project_engineering_series/article_03_trainer_plugins/project
python main.py --mode train --config config/cnn_cls_train.yaml --opts train.epochs=1
```

评估：

```bash
python main.py --mode eval --config config/cnn_cls_train.yaml
```

推理：

```bash
python main.py --mode predict --config config/cnn_cls_predict.yaml
```

导出占位：

```bash
python main.py --mode export --config config/cnn_cls_train.yaml
```

性能测试：

```bash
python main.py --mode benchmark --config config/cnn_cls_train.yaml
```

## 9. 本篇工程小结

这一篇完成了训练流程插件化：

- `TRAINERS` 注册表负责管理训练流程。
- `ClassificationTrainer` 封装 CNN 分类训练和评估。
- `launcher/train.py` 和 `launcher/eval.py` 变成薄调度层。
- `main.py` 支持 `train/eval/predict/export/benchmark` 五种模式。

这一步非常关键，因为后续 GAN、Diffusion、Transformer 的差异主要不在入口，而在 Trainer。

## 10. GitHub 对应代码目录

[GitHub：article_03_trainer_plugins/project](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_03_trainer_plugins/project)
