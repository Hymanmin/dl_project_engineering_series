# 从 CNN 扩展到 GAN、Diffusion、Transformer：同一工程框架下的不同任务范式

> 深度学习项目工程化实战系列 05

前几篇建立了一个配置驱动、注册表管理、Trainer 插件化的工程框架。到这里，工程已经不再只能支持一个 CNN 分类任务。

但不同网络任务的训练范式差异很大：

```text
CNN 分类：单模型、单优化器、单 loss
GAN：generator/discriminator、双优化器、D step/G step
Diffusion：加噪、timestep、denoiser、noise prediction、EMA
Transformer：tokenizer、attention mask、warmup、梯度累积
```

本文重点说明：这些任务如何在同一基础框架下扩展，以及哪些模块可以复用，哪些模块必须新增。

本文对应工程目录：

```text
dl_project_engineering_series/
  article_05_task_paradigms/
    README.md
    project/
```

## 1. 本篇最终代码目录

```text
project/
  config/
    toy_gan_train.yaml            # GAN 任务配置
    toy_diffusion_train.yaml      # Diffusion 任务配置
    toy_transformer_train.yaml    # Transformer 任务配置

  core/
    networks/
      gan_toy/
        generator.py              # GAN 生成器
        discriminator.py          # GAN 判别器
        framework.py              # 组合 generator/discriminator
      diffusion_toy/
        unet.py                   # 简化 denoiser
      transformer_toy/
        framework.py              # embedding/encoder/head

    models/
      gan_toy/
        gan_model.py              # 暴露 generator/discriminator
      diffusion_toy/
        diffusion_model.py        # 加噪、采样 timestep、调用 denoiser

    losses/
      gan_loss.py                 # d_loss/g_loss
      diffusion_loss.py           # noise prediction MSE

    trainers/
      classification_trainer.py   # 当前已有分类 Trainer
      base_trainer.py             # 后续 GAN/Diffusion/Transformer Trainer 的基类
```

## 2. CNN 分类范式

CNN 分类是最简单的训练范式：

```text
input -> network -> logits -> cross entropy -> backward -> optimizer.step
```

它适合当前已有的目录结构：

```text
networks/cnn_cls/
  backbone.py
  neck.py
  head.py
  framework.py

models/cnn_cls/
  cnn_cls_model.py

losses/
  cnn_cls_loss.py

trainers/
  classification_trainer.py
```

这种任务通常只需要一个模型、一个优化器、一个 loss。

## 3. GAN：不要强行套 backbone/neck/head

GAN 的核心是两个网络：

```text
z -> generator -> fake
real/fake -> discriminator -> logits
```

因此目录应该按 GAN 的自然结构拆分：

```text
core/
  networks/
    gan_toy/
      generator.py                # 生成器
      discriminator.py            # 判别器
      framework.py                # GAN 网络容器

  models/
    gan_toy/
      gan_model.py                # 暴露 generator/discriminator

  losses/
    gan_loss.py                   # d_loss 和 g_loss
```

`framework.py`：

```python
@NETWORKS.register("toy_gan_network")
class ToyGANNetwork(nn.Module):
    def __init__(self, noise_dim=16, data_dim=16):
        super().__init__()
        self.generator = ToyGenerator(noise_dim=noise_dim, output_dim=data_dim)
        self.discriminator = ToyDiscriminator(input_dim=data_dim)
```

`gan_model.py`：

```python
@MODELS.register("toy_gan_model")
class ToyGANModel(BaseModel):
    @property
    def generator(self):
        return self.network.generator

    @property
    def discriminator(self):
        return self.network.discriminator
```

GAN 的 loss 也不是一个普通 `forward(logits, target)`：

```python
class ToyGANLoss(nn.Module):
    def d_loss(self, real_logits, fake_logits):
        ...

    def g_loss(self, fake_logits):
        ...
```

因此 GAN 后续应该新增 `GANTrainer`：

```text
core/trainers/gan_trainer.py
```

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

这就是 Trainer 插件化的价值。

## 4. Diffusion：模型层负责加噪和 denoise 组织

Diffusion 的训练流程不是简单分类，而是：

```text
x0 -> sample timestep t
x0 + noise -> xt
denoiser(xt, t) -> pred_noise
MSE(pred_noise, noise)
```

对应目录：

```text
core/
  networks/
    diffusion_toy/
      unet.py                     # denoiser

  models/
    diffusion_toy/
      diffusion_model.py          # 加噪、timestep、调用 denoiser

  losses/
    diffusion_loss.py             # MSE(pred, target)
```

`diffusion_model.py` 的职责比 CNN model 更重：

```python
def forward(self, batch):
    x0 = batch["input"]
    noise = torch.randn_like(x0)
    t = torch.randint(0, 1000, (x0.shape[0],), device=x0.device)
    xt = x0 + 0.1 * noise
    pred = self.denoiser(xt, t)
    return {"pred": pred, "target": noise, "timestep": t}
```

Diffusion 的 loss 接收字典：

```python
@LOSSES.register("toy_diffusion_loss")
class ToyDiffusionLoss(nn.Module):
    def forward(self, outputs):
        return F.mse_loss(outputs["pred"], outputs["target"])
```

真实项目中还需要：

- beta/noise scheduler
- EMA
- classifier-free guidance
- 周期性采样保存图片
- 推理采样流程

这些不应该塞进分类 Trainer，而应该新增 `DiffusionTrainer`。

## 5. Transformer：输入不再是 image tensor

Transformer 任务常见输入是：

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...
}
```

因此 dataset、model、trainer 都会发生变化。

本文工程提供一个简化 Transformer classifier：

```text
core/networks/transformer_toy/
  framework.py
```

结构：

```python
embedding -> TransformerEncoder -> CLS head
```

代码：

```python
@NETWORKS.register("toy_transformer_classifier")
class ToyTransformerClassifier(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return self.head(x[:, 0])
```

真实 Transformer 工程还需要：

- tokenizer
- padding/truncation
- attention mask
- warmup scheduler
- gradient accumulation
- AMP
- DDP/ZeRO

这些也应该通过 Trainer 和 config 扩展。

## 6. 三类任务扩展后的配置示例

GAN：

```yaml
network:
  name: toy_gan_network

model:
  name: toy_gan_model

loss:
  name: toy_gan_loss
```

Diffusion：

```yaml
network:
  name: toy_denoiser

model:
  name: toy_diffusion_model

loss:
  name: toy_diffusion_loss
```

Transformer：

```yaml
network:
  name: toy_transformer_classifier
```

配置仍然通过 `name` 连接注册表，但不同任务需要不同的 model 和 trainer。

## 7. 本篇工程小结

同一工程框架可以支持不同网络范式，但不能要求所有任务都长得像 CNN 分类。

建议保持这条原则：

```text
相同的部分复用：config、registry、launcher、logger、checkpoint
不同的部分插件化：dataset、network、model、loss、trainer
```

GAN、Diffusion、Transformer 的主要差异不只是网络结构，而是训练 step 不同。因此 Trainer 插件化是扩展复杂任务的关键。

## 8. GitHub 对应代码目录

[GitHub：article_05_task_paradigms/project](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_05_task_paradigms/project)
