# 模块化深度学习工程模板设计文档

本工程是一个面向教学、实验和中小型研发项目的模块化深度学习模板。它的核心目标不是绑定某一种网络，而是提供一套稳定的工程组织方式，让 CNN、GAN、Diffusion、Transformer 等任务都能复用同一套入口、配置、注册表、数据集、网络、损失函数和训练/推理流程。

默认示例任务名为 `cnn_cls`。实际使用时，可以将它替换为 `cat_dog_cls`、`mnist_gan`、`ddpm_cifar10`、`vit_cls` 等具体任务名。

## 1. 总体架构建议

当前模板采用了比较合理的工程分层：

- `main.py` 作为统一入口，负责选择 `train` 或 `predict` 模式。
- `config/` 负责配置文件、配置继承和命令行覆盖。
- `datatxt/` 管理数据列表文件。
- `launcher/` 定义训练和推理流程。
- `core/` 保存核心组件，包括注册表、模型、网络、损失函数和数据集。
- `utils/` 保存日志等通用工具。

从工程灵活性、扩展性和使用效率看，建议后续重点增强以下部分：

1. **把训练流程进一步插件化**

   当前 `launcher/train.py` 是一个单任务训练模板。对 CNN 分类任务足够，但 GAN、Diffusion、Transformer 预训练通常需要不同的 step 逻辑，例如多优化器、EMA、梯度累积、混合精度、采样评估等。建议新增 `TRAINERS` 注册表，将训练逻辑抽象为 `BaseTrainer`，不同任务注册不同 Trainer。

2. **把推理/评估/导出拆成独立模式**

   现在 `mode` 只有 `train` 和 `predict`。实际工程中建议扩展为：

   - `train`：训练。
   - `eval`：验证集/测试集评估。
   - `predict`：单样本或批量推理。
   - `export`：导出 TorchScript、ONNX、TensorRT 等部署格式。
   - `benchmark`：测速和显存统计。

3. **配置建议按职责拆分**

   当前 `base.yaml` 已经支持 `_base_` 继承。后续可以把配置拆成：

   - `config/dataset/*.yaml`
   - `config/network/*.yaml`
   - `config/train/*.yaml`
   - `config/runtime/*.yaml`

   对教学项目，当前单文件配置更直观；对复杂项目，分层配置更利于复用。

4. **注册表应支持自动导入或显式模块清单**

   注册表模式依赖模块被 import 后才能完成注册。当前模板通过在使用处 import 示例模块触发注册。更完整的方案是提供 `core/imports.py` 或 `config.imports` 字段，启动时统一导入任务模块，避免新增模块后忘记 import。

5. **数据集建议返回标准 batch 字典**

   当前数据集返回：

   ```python
   {
       "input": tensor,
       "target": tensor,
   }
   ```

   这是一个好习惯。建议长期保持 batch 字典格式，并按任务扩展字段，例如：

   - 分类：`input`, `target`
   - 检测：`image`, `boxes`, `labels`, `meta`
   - GAN：`real`, `label`, `noise`
   - Diffusion：`image`, `condition`, `timestep`
   - NLP/Transformer：`input_ids`, `attention_mask`, `labels`

6. **模型层负责完整前向，网络层负责结构组合**

   建议保持当前职责边界：

   - `network`：只描述网络结构，如 backbone、neck、head、generator、discriminator、unet、encoder、decoder。
   - `model`：组织完整训练前向，包括调用网络、处理输出、必要时组合多个网络。
   - `loss`：只计算损失，尽量不持有训练状态。
   - `trainer` 或 `launcher`：处理优化器、反向传播、分布式、日志、保存、评估。

   这样做可以避免“所有逻辑都堆在模型里”。

## 2. 工程目录

```text
template_project/
  main.py                         # 统一入口，支持 train/predict 模式
  config/                         # 配置管理
    __init__.py
    loader.py                     # 配置加载器，支持 _base_ 继承和命令行覆盖
    base.yaml                     # 基础配置
    cnn_cls_train.yaml       # 训练配置
    cnn_cls_predict.yaml     # 推理配置
  datatxt/                        # 数据列表文件
    train.txt
    val.txt
    test.txt
  launcher/                       # 流程定义
    train.py
    predict.py
  core/                           # 核心组件
    registry.py                   # Registry 类与 build_* 构建函数
    hooks.py                      # Hook 扩展点
    dataset/
      base_dataset.py
      cnn_cls/
        cnn_cls_dataset.py
    networks/
      factory.py
      cnn_cls/
        backbone.py
        neck.py
        head.py
        framework.py
    losses/
      cnn_cls_loss.py
    models/
      base_model.py
      cnn_cls/
        cnn_cls_model.py
  utils/
    logger.py
  requirements.txt
  run_main.sh
  ARCHITECTURE.md
  README.md
```

## 3. 快速使用

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

命令行覆盖配置：

```bash
python main.py --mode train --config config/cnn_cls_train.yaml --opts train.epochs=10 optimizer.lr=0.0005 runtime.device=cpu
```

## 4. 配置系统设计

配置文件是工程的“实验说明书”。所有可变参数都应该优先放进 YAML，而不是硬编码在 Python 文件里。

当前配置加载器支持两类能力：

1. `_base_` 继承：

   ```yaml
   _base_: base.yaml

   train:
     epochs: 20
   ```

2. 命令行覆盖：

   ```bash
   python main.py --mode train --config config/cnn_cls_train.yaml --opts train.epochs=50 optimizer.lr=0.0001
   ```

一个常规配置建议包含以下字段：

```yaml
task: cat_dog_cls
seed: 42

runtime:
  device: cuda
  work_dir: runs/cat_dog_cls
  log_dir: runs/logs
  checkpoint_dir: runs/checkpoints
  amp: true
  distributed: false

dataset:
  name: image_classification_dataset
  train_list: datatxt/train.txt
  val_list: datatxt/val.txt
  batch_size: 64
  num_workers: 8
  image_size: 224
  num_classes: 2

network:
  name: resnet_classifier
  backbone: resnet18
  pretrained: true
  num_classes: 2

model:
  name: classification_model

loss:
  name: cross_entropy_loss

optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.01

train:
  epochs: 100
  print_freq: 20
  save_name: cat_dog_cls.pt
```

配置层面的建议：

- 每个实验保留一份独立 YAML，便于复现实验。
- 基础配置只放稳定参数，实验配置只覆盖变化参数。
- 不要在代码里写死 batch size、学习率、类别数、路径。
- 对 DDP、AMP、梯度累积、EMA、checkpoint 等训练行为，也应该通过配置控制。

## 5. Registry 组件注册机制

工程使用统一注册表管理可插拔组件：

```python
DATASETS = Registry("dataset")
NETWORKS = Registry("network")
LOSSES = Registry("loss")
MODELS = Registry("model")
```

新增组件时，使用装饰器注册：

```python
from core.registry import NETWORKS


@NETWORKS.register("resnet_classifier")
class ResNetClassifier(nn.Module):
    ...
```

配置中写入注册名：

```yaml
network:
  name: resnet_classifier
```

构建时统一调用：

```python
network = build_network(cfg["network"])
```

这种模式的优点是：

- 新增组件不需要修改大量 if/else。
- 配置文件决定使用哪个组件。
- 同一训练流程可以复用不同数据集、网络和损失函数。

使用时需要注意：注册只有在模块被 import 后才生效。新增任务后，应在对应 `__init__.py`、`launcher` 或统一导入文件中导入新模块。

## 6. Dataset 扩展方式

数据集组件建议只负责三件事：

1. 读取数据列表或索引。
2. 执行数据解析和变换。
3. 返回标准 batch 字典。

分类数据集示例：

```python
from core.dataset.base_dataset import BaseDataset
from core.registry import DATASETS


@DATASETS.register("image_classification_dataset")
class ImageClassificationDataset(BaseDataset):
    def __init__(self, train_list, image_size=224, num_classes=2, **kwargs):
        self.samples = load_samples(train_list)
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = load_image_tensor(image_path, self.image_size)
        return {
            "input": image,
            "target": label,
            "meta": {"path": image_path},
        }
```

GAN 数据集示例：

```python
@DATASETS.register("gan_image_dataset")
class GANImageDataset(BaseDataset):
    def __getitem__(self, index):
        image = load_real_image(index)
        return {
            "real": image,
        }
```

Diffusion 数据集示例：

```python
@DATASETS.register("diffusion_image_dataset")
class DiffusionImageDataset(BaseDataset):
    def __getitem__(self, index):
        image = load_image(index)
        condition = load_condition(index)
        return {
            "image": image,
            "condition": condition,
        }
```

Transformer 文本数据集示例：

```python
@DATASETS.register("text_classification_dataset")
class TextClassificationDataset(BaseDataset):
    def __getitem__(self, index):
        item = tokenize_text(index)
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }
```

## 7. Network 扩展方式

`network` 层用于表达网络结构。当前模板把网络拆成：

- `backbone.py`：主干特征提取。
- `neck.py`：中间特征融合或变换。
- `head.py`：任务输出头。
- `framework.py`：组合 backbone、neck、head。

### 7.1 常规 CNN

CNN 分类可以沿用 backbone/neck/head：

```text
input image -> backbone -> neck -> classifier head -> logits
```

示例配置：

```yaml
network:
  name: cnn_classifier
  backbone: resnet18
  hidden_dim: 512
  num_classes: 10
```

示例结构：

```python
@NETWORKS.register("cnn_classifier")
class CNNClassifier(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=10, **kwargs):
        super().__init__()
        self.backbone = build_cnn_backbone(backbone)
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)
```

### 7.2 GAN

GAN 不适合强行套 backbone/neck/head。建议在 `core/networks/<task>/` 下拆成：

```text
core/networks/mnist_gan/
  generator.py
  discriminator.py
  framework.py
```

`framework.py` 负责注册完整网络容器：

```python
@NETWORKS.register("dcgan_network")
class DCGANNetwork(nn.Module):
    def __init__(self, noise_dim=128, **kwargs):
        super().__init__()
        self.generator = Generator(noise_dim)
        self.discriminator = Discriminator()
```

GAN 的训练通常需要两个优化器，因此建议新增 `GANTrainer`，而不是把所有逻辑塞进普通 `run_train`。

```yaml
trainer:
  name: gan_trainer

network:
  name: dcgan_network
  noise_dim: 128

loss:
  name: gan_loss
```

### 7.3 Diffusion

Diffusion 常见结构是 UNet + Scheduler + Noise Predictor。建议目录：

```text
core/networks/ddpm_cifar10/
  unet.py
  time_embedding.py
  condition_encoder.py
  framework.py
core/models/ddpm_cifar10/
  ddpm_model.py
core/losses/
  diffusion_losses.py
```

配置示例：

```yaml
network:
  name: ddpm_unet
  in_channels: 3
  model_channels: 128
  num_res_blocks: 2
  num_timesteps: 1000

model:
  name: ddpm_model
  prediction_type: epsilon

loss:
  name: diffusion_mse_loss
```

Diffusion 的 `model` 层可以负责加噪和预测：

```python
@MODELS.register("ddpm_model")
class DDPMModel(BaseModel):
    def __init__(self, cfg, prediction_type="epsilon"):
        super().__init__()
        self.unet = build_network(cfg["network"])
        self.prediction_type = prediction_type

    def forward(self, batch):
        x0 = batch["image"]
        t = sample_timesteps(x0)
        xt, noise = add_noise(x0, t)
        pred = self.unet(xt, t, batch.get("condition"))
        return {
            "pred": pred,
            "target": noise,
            "timestep": t,
        }
```

### 7.4 Transformer

Transformer 可以用于 NLP、视觉 ViT、多模态任务。建议按照 encoder/decoder/head 拆分：

```text
core/networks/text_cls/
  embedding.py
  encoder.py
  head.py
  framework.py
```

配置示例：

```yaml
network:
  name: transformer_classifier
  vocab_size: 30522
  hidden_dim: 768
  num_layers: 12
  num_heads: 12
  max_length: 512
  num_classes: 2
```

示例 batch：

```python
{
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels,
}
```

Transformer 的关键工程点：

- 数据集需要处理 tokenizer、padding、truncation。
- 训练流程通常需要 warmup scheduler。
- 大模型训练可能需要 gradient accumulation、AMP、DDP、ZeRO 或 pipeline parallel。

## 8. Loss 扩展方式

Loss 层建议保持纯函数或轻量 `nn.Module`，输入模型输出和目标，返回可反向传播的 loss。

分类损失：

```python
@LOSSES.register("cross_entropy_loss")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    pass
```

GAN 损失：

```python
@LOSSES.register("gan_loss")
class GANLoss(nn.Module):
    def d_loss(self, real_logits, fake_logits):
        ...

    def g_loss(self, fake_logits):
        ...
```

Diffusion 损失：

```python
@LOSSES.register("diffusion_mse_loss")
class DiffusionMSELoss(nn.Module):
    def forward(self, outputs):
        return F.mse_loss(outputs["pred"], outputs["target"])
```

Transformer 损失：

```python
@LOSSES.register("masked_lm_loss")
class MaskedLMLoss(nn.Module):
    def forward(self, logits, labels):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
```

建议：

- 普通分类可以让 loss 接收 `(logits, targets)`。
- 复杂任务建议让 loss 接收 `outputs` 字典，避免参数越来越多。
- 多损失任务建议返回字典：`{"loss": total_loss, "loss_cls": ..., "loss_reg": ...}`。

## 9. Model 扩展方式

`model` 层表示一个完整任务模型，不只是网络结构。它可以组合多个 network，也可以封装训练所需的特殊前向。

CNN 分类：

```python
@MODELS.register("classification_model")
class ClassificationModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.network = build_network(cfg["network"])

    def forward(self, batch):
        return self.network(batch["input"])
```

GAN：

```python
@MODELS.register("gan_model")
class GANModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.network = build_network(cfg["network"])

    @property
    def generator(self):
        return self.network.generator

    @property
    def discriminator(self):
        return self.network.discriminator
```

Diffusion：

```python
@MODELS.register("diffusion_model")
class DiffusionModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.denoiser = build_network(cfg["network"])
        self.scheduler = build_noise_scheduler(cfg["diffusion"])
```

Transformer：

```python
@MODELS.register("language_model")
class LanguageModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.transformer = build_network(cfg["network"])

    def forward(self, batch):
        return self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
```

## 10. 训练流程扩展

当前模板的 `launcher/train.py` 是一个基础训练循环：

```text
dataset -> dataloader -> model -> loss -> optimizer -> backward -> save checkpoint
```

适合 CNN 分类等单模型、单优化器任务。复杂任务建议引入 Trainer 注册表：

```text
core/trainers/
  base_trainer.py
  classification_trainer.py
  gan_trainer.py
  diffusion_trainer.py
  transformer_trainer.py
```

建议新增注册表：

```python
TRAINERS = Registry("trainer")


def build_trainer(cfg, *args, **kwargs):
    return TRAINERS.build(cfg, *args, **kwargs)
```

配置示例：

```yaml
trainer:
  name: classification_trainer
  amp: true
  grad_accum_steps: 1
  clip_grad_norm: 1.0
```

### 10.1 CNN Trainer

CNN 分类训练通常包含：

- 单模型。
- 单优化器。
- 交叉熵损失。
- train/eval 指标。
- checkpoint 保存。

### 10.2 GAN Trainer

GAN 训练通常包含：

- generator 和 discriminator。
- 两个优化器。
- D step 和 G step。
- 固定噪声采样可视化。
- EMA 可选。

伪代码：

```python
for batch in loader:
    real = batch["real"]
    z = sample_noise(batch_size)

    fake = model.generator(z).detach()
    d_loss = loss.d_loss(model.discriminator(real), model.discriminator(fake))
    optimize_d(d_loss)

    fake = model.generator(z)
    g_loss = loss.g_loss(model.discriminator(fake))
    optimize_g(g_loss)
```

### 10.3 Diffusion Trainer

Diffusion 训练通常包含：

- 随机采样 timestep。
- 对图像加噪。
- UNet 预测噪声或 x0。
- MSE 或 v-prediction loss。
- EMA。
- 周期性采样保存图片。

### 10.4 Transformer Trainer

Transformer 训练通常包含：

- AMP。
- 梯度累积。
- warmup + cosine scheduler。
- 梯度裁剪。
- DDP 或 ZeRO。
- 长序列任务的显存优化。

## 11. DDP 训练扩展

DDP 适合多卡数据并行训练。建议通过配置控制：

```yaml
runtime:
  distributed: true
  backend: nccl
  device: cuda

train:
  amp: true
  grad_accum_steps: 1
```

启动方式：

```bash
torchrun --nproc_per_node=4 main.py --mode train --config config/cat_dog_train.yaml
```

工程层面需要增加：

1. `utils/distributed.py`

   负责：

   - 初始化进程组。
   - 获取 `rank`、`local_rank`、`world_size`。
   - 设置当前 GPU。
   - 判断是否主进程。
   - 包装 `DistributedSampler`。

2. `launcher/train.py` 或 `BaseTrainer`

   负责：

   - 将模型包装为 `DistributedDataParallel`。
   - 每个 epoch 调用 `sampler.set_epoch(epoch)`。
   - 只在主进程保存 checkpoint 和打印日志。
   - 指标做跨进程聚合。

DDP 注意事项：

- batch size 通常指单卡 batch size，总 batch size = 单卡 batch size * GPU 数 * 梯度累积步数。
- 保存 checkpoint 时应保存原始模型权重，避免直接保存 DDP wrapper。
- 验证集指标需要跨进程汇总，否则多卡评估结果可能不完整。

## 12. Pipeline 并行扩展

Pipeline 并行适合模型本身很大、单卡放不下的场景。它和 DDP 的目标不同：

- DDP：每张卡一份完整模型，切分 batch。
- Pipeline：不同 GPU 放模型的不同部分，切分模型。

Transformer 大模型、超大 Diffusion UNet 或多阶段多模态模型可能需要 Pipeline。

配置示例：

```yaml
runtime:
  parallel:
    type: pipeline
    stages: 4
    micro_batch_size: 2

network:
  name: large_transformer
  num_layers: 48
  hidden_dim: 4096
```

工程层面建议：

1. 在 `network` 中提供可切分的 stage：

   ```python
   class LargeTransformer(nn.Module):
       def build_stages(self):
           return [
               nn.Sequential(self.embedding, *self.layers[:12]),
               nn.Sequential(*self.layers[12:24]),
               nn.Sequential(*self.layers[24:36]),
               nn.Sequential(*self.layers[36:], self.head),
           ]
   ```

2. 在 `trainer` 中选择并行策略：

   ```python
   if cfg["runtime"]["parallel"]["type"] == "pipeline":
       model = build_pipeline_model(model, cfg)
   ```

3. 对 batch 做 micro-batch 切分，减少 pipeline bubble。

实际项目中，Pipeline 建议直接基于成熟库实现，例如 PyTorch Pipeline、DeepSpeed Pipeline 或 Megatron-LM 风格的并行训练框架。模板工程只保留配置和接口，不建议从零手写完整 pipeline 调度器。

## 13. 部署与导出扩展

建议新增 `launcher/export.py`，并把 `main.py` 的 `mode` 扩展为：

```text
train | eval | predict | export | benchmark
```

导出配置示例：

```yaml
export:
  format: onnx
  checkpoint: runs/checkpoints/cat_dog_cls.pt
  output_path: exports/cat_dog_cls.onnx
  input_shape: [1, 3, 224, 224]
  dynamic_axes: true
```

常见导出目标：

- TorchScript：适合 PyTorch 内部部署。
- ONNX：适合跨框架部署。
- TensorRT：适合 NVIDIA GPU 推理加速。

部署相关逻辑不建议写进 `model` 或 `network`，应单独放在 `launcher/export.py` 或 `deploy/` 目录下。

## 14. 新任务扩展示例

以新增 `cat_dog_cls` 分类任务为例：

1. 新建数据集：

   ```text
   core/dataset/cat_dog_cls/cat_dog_dataset.py
   ```

2. 新建网络：

   ```text
   core/networks/cat_dog_cls/backbone.py
   core/networks/cat_dog_cls/head.py
   core/networks/cat_dog_cls/framework.py
   ```

3. 新建模型：

   ```text
   core/models/cat_dog_cls/cat_dog_model.py
   ```

4. 新建配置：

   ```text
   config/cat_dog_train.yaml
   config/cat_dog_predict.yaml
   ```

5. 在模块中注册组件：

   ```python
   @DATASETS.register("cat_dog_dataset")
   class CatDogDataset(BaseDataset):
       ...

   @NETWORKS.register("cat_dog_network")
   class CatDogNetwork(nn.Module):
       ...

   @MODELS.register("cat_dog_model")
   class CatDogModel(BaseModel):
       ...
   ```

6. 配置中引用注册名：

   ```yaml
   dataset:
     name: cat_dog_dataset

   network:
     name: cat_dog_network

   model:
     name: cat_dog_model

   loss:
     name: cross_entropy_loss
   ```

## 15. 推荐演进路线

如果把该模板继续打造成更完整的深度学习工程框架，建议按以下顺序演进：

1. 新增 `core/trainers/` 和 `TRAINERS` 注册表。
2. 新增 `utils/distributed.py`，支持 DDP。
3. 新增 `utils/checkpoint.py`，统一保存和恢复模型、优化器、scheduler、epoch、best metric。
4. 新增 `core/metrics/` 和 `METRICS` 注册表。
5. 新增 `core/schedulers/` 和 `SCHEDULERS` 注册表。
6. 新增 `launcher/eval.py`、`launcher/export.py`、`launcher/benchmark.py`。
7. 新增 `config/imports` 或 `core/imports.py`，集中导入注册模块。
8. 新增单元测试，至少覆盖配置继承、注册表构建、dataset 输出格式和一次最小训练 step。

## 16. 最小测试建议

建议补充以下测试：

```text
tests/
  test_config_loader.py
  test_registry.py
  test_dataset.py
  test_model_forward.py
  test_train_step.py
```

重点测试：

- `_base_` 是否正确继承。
- `--opts` 是否正确覆盖。
- 未注册组件是否抛出清晰错误。
- 数据集 batch 字段是否符合约定。
- 模型 forward 输出 shape 是否正确。
- 单个训练 step 是否能完成反向传播。

这样可以保证后续扩展 CNN、GAN、Diffusion、Transformer 时，不会破坏基础框架。


