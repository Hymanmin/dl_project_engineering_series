# 多卡训练实战：DataParallel、DDP 与 Pipeline 在工程中放在哪里

> 深度学习项目工程化实战系列 04

当模型和数据规模变大后，单卡训练通常会遇到两个问题：

- 训练太慢，需要多卡提升吞吐。
- 模型太大，单卡显存放不下。

这时就会遇到三类并行方案：

```text
DataParallel：单进程多卡，使用简单，但效率和可控性一般
DDP：多进程多卡，工程实践中最常用的数据并行
Pipeline：把模型切到多张卡上，适合单卡放不下的大模型
```

本文不追求一次性实现完整大模型并行框架，而是回答一个更工程化的问题：这些能力应该放在项目的哪些文件里，如何和现有 Trainer、config、launcher 结合。

本文对应工程目录：

```text
dl_project_engineering_series/
  article_04_parallel_training/
    README.md
    project/
```

## 1. 本篇最终代码目录

```text
project/
  config/
    base.yaml                     # 新增 runtime.parallel 配置
    cnn_cls_train.yaml
    cnn_cls_dp_train.yaml         # DataParallel 配置
    cnn_cls_ddp_train.yaml        # DDP 配置
    cnn_cls_pipeline_train.yaml   # Pipeline 配置占位

  utils/
    distributed.py                # DDP 初始化、rank/local_rank/world_size、sampler、cleanup
    parallel.py                   # DataParallel/DDP 包装和 unwrap_model
    logger.py

  core/
    trainers/
      classification_trainer.py   # 接入 init_distributed、DistributedSampler、主进程保存

  launcher/
    train.py
    eval.py
    predict.py
    export.py
    benchmark.py
```

## 2. 三种并行方式的区别

### DataParallel

`DataParallel` 是单进程多卡。它会把一个 batch 切到多张 GPU 上，把模型复制到多卡，然后汇总梯度。

优点：

- 使用简单。
- 单进程，调试相对容易。

缺点：

- 主卡压力大。
- 多卡效率通常不如 DDP。
- 大项目中可控性较弱。

配置示例：

```yaml
runtime:
  device: cuda
  parallel:
    type: dp
```

### DDP

DDP 是多进程多卡。每张卡一个进程，每个进程持有一份模型副本，通过梯度同步实现数据并行。

优点：

- PyTorch 官方推荐的多卡训练方式。
- 性能和可控性更好。
- 适合绝大多数多卡训练项目。

配置示例：

```yaml
runtime:
  device: cuda
  parallel:
    type: ddp
    backend: nccl
```

CPU 或教学环境可以使用：

```yaml
runtime:
  device: cpu
  parallel:
    type: ddp
    backend: gloo
```

启动方式：

```bash
torchrun --nproc_per_node=4 main.py --mode train --config config/cnn_cls_ddp_train.yaml
```

### Pipeline

Pipeline 并行不是切 batch，而是切模型。比如一个 48 层 Transformer，可以把前 12 层放在 GPU0，接下来 12 层放在 GPU1，以此类推。

适合：

- 模型太大，单卡放不下。
- Transformer 大模型。
- 超大 Diffusion UNet。
- 多阶段多模态模型。

本文工程只保留 Pipeline 配置和接口占位，不从零手写调度器。实际项目建议使用 PyTorch Pipeline、DeepSpeed 或 Megatron-LM 风格的成熟实现。

## 3. distributed.py：DDP 相关逻辑集中管理

多卡训练最容易把代码写乱。建议不要把 `rank`、`local_rank`、`world_size`、`DistributedSampler` 全部散落在 Trainer 里，而是集中放在：

```text
utils/distributed.py
```

核心函数：

```python
def init_distributed(cfg: dict) -> dict[str, int]:
    parallel_cfg = cfg.get("runtime", {}).get("parallel", {})
    if parallel_cfg.get("type") != "ddp":
        return {"rank": 0, "local_rank": 0, "world_size": 1}

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = parallel_cfg.get("backend", "gloo")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
```

这个函数只在配置声明 `type: ddp` 时初始化进程组。

另一个重要函数：

```python
def is_main_process() -> bool:
    return not is_dist_available_and_initialized() or dist.get_rank() == 0
```

它用于控制日志、checkpoint 保存、评估结果输出。多进程训练时，如果每个进程都保存模型，很容易产生冲突。

## 4. parallel.py：模型包装单独管理

模型包装放在：

```text
utils/parallel.py
```

核心逻辑：

```python
def wrap_model(model, cfg, device):
    parallel_type = cfg["runtime"]["parallel"]["type"]

    if parallel_type == "dp":
        return DataParallel(model)

    if parallel_type == "ddp":
        return DistributedDataParallel(model)

    if parallel_type == "pipeline":
        raise NotImplementedError(...)

    return model
```

保存 checkpoint 时要注意：DDP 或 DP 包装后的模型外层有 `.module`。因此工程里提供：

```python
def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
```

保存时：

```python
torch.save({"model": unwrap_model(self.model).state_dict()}, ckpt_path)
```

这样 checkpoint 不会绑定并行 wrapper。

## 5. Trainer 中如何接入 DDP

`ClassificationTrainer` 中新增三处逻辑。

第一处：初始化分布式状态。

```python
self.dist_state = init_distributed(cfg)
```

第二处：包装模型。

```python
self.model = build_model(cfg.get("model", {}), cfg).to(self.device)
self.model = wrap_model(self.model, cfg, self.device)
```

第三处：构建 sampler。

```python
sampler = build_sampler(dataset, shuffle=shuffle)

DataLoader(
    dataset,
    shuffle=shuffle if sampler is None else False,
    sampler=sampler,
)
```

DDP 下必须使用 `DistributedSampler`，否则每个进程可能读到相同数据，导致有效 batch 和统计结果错误。

## 6. 多卡扩展后的目录结构

```text
project/
  utils/
    distributed.py                # 新增：DDP 初始化和进程信息
    parallel.py                   # 新增：DP/DDP/Pipeline 包装接口

  config/
    cnn_cls_dp_train.yaml         # 新增：DataParallel 配置
    cnn_cls_ddp_train.yaml        # 新增：DDP 配置
    cnn_cls_pipeline_train.yaml   # 新增：Pipeline 配置占位

  core/
    trainers/
      classification_trainer.py   # 修改：接入 sampler、wrap_model、主进程保存
```

这几个文件的职责边界应该保持稳定：

- `utils/distributed.py`：进程和通信。
- `utils/parallel.py`：模型包装。
- `trainer`：训练流程接入并行能力。
- `config`：选择使用哪种并行策略。

## 7. 运行命令

单卡或 CPU：

```bash
python main.py --mode train --config config/cnn_cls_train.yaml
```

DataParallel：

```bash
python main.py --mode train --config config/cnn_cls_dp_train.yaml
```

DDP：

```bash
torchrun --nproc_per_node=2 main.py --mode train --config config/cnn_cls_ddp_train.yaml
```

Pipeline 配置占位：

```bash
python main.py --mode train --config config/cnn_cls_pipeline_train.yaml
```

当前工程会对 Pipeline 抛出 `NotImplementedError`，这是有意设计：Pipeline 调度不建议在教学模板里从零手写，应该接入成熟库。

## 8. 本篇工程小结

这一篇把多卡训练相关逻辑放进了明确位置：

- 配置决定并行策略。
- `utils/distributed.py` 管进程。
- `utils/parallel.py` 管模型包装。
- Trainer 管训练流程接入。
- 只有主进程保存 checkpoint。

这套组织方式可以避免 DDP 代码污染整个工程。

## 9. GitHub 对应代码目录

[GitHub：article_04_parallel_training/project](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_04_parallel_training/project)
