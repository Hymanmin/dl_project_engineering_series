# 并行测试框架搭建：让深度学习工程扩展后仍然可验证

> 深度学习项目工程化实战系列 07

工程框架越灵活，越需要测试。否则新增一个 dataset、network、trainer 或 export 模式时，很容易在不知不觉中破坏已有流程。

深度学习项目不一定一开始就写复杂测试，但至少应该有一套快速 smoke test，覆盖最关键的工程链路：

```text
config loader
registry
dataset batch contract
model forward
train step
export
benchmark
```

本文使用 Python 标准库 `unittest` 搭建最小测试框架，避免引入额外依赖。

本文对应工程目录：

```text
dl_project_engineering_series/
  article_07_testing/
    README.md
    project/
```

## 1. 本篇最终代码目录

```text
project/
  run_tests.py                    # 统一测试入口

  tests/
    __init__.py
    test_config_loader.py         # 测试 _base_ 继承和命令行覆盖
    test_registry.py              # 测试注册表构建 model/loss
    test_dataset.py               # 测试 dataset 返回 batch 字典
    test_model_forward.py         # 测试模型 forward 输出 shape
    test_train_step.py            # 测试一轮最小训练
    test_export_benchmark.py      # 测试导出和 benchmark smoke

  config/
  core/
  launcher/
  utils/
```

## 2. 为什么测试重点不是精度

工程模板阶段，测试重点不是模型精度，而是工程链路是否稳定。

例如：

- 配置能不能正确继承？
- 注册表能不能找到组件？
- dataset 输出字段是否符合约定？
- model forward 是否能跑通？
- train step 是否能完成反向传播和保存 checkpoint？
- export/benchmark 是否能生成产物？

这些测试不需要大数据集，也不需要长时间训练。它们的目标是快速发现工程破坏。

## 3. Config 测试

`tests/test_config_loader.py`：

```python
cfg = load_config(Path("config/cnn_cls_train.yaml"), ["train.epochs=1", "optimizer.lr=0.01"])
self.assertEqual(cfg["task"], "cnn_cls")
self.assertEqual(cfg["train"]["epochs"], 1)
self.assertAlmostEqual(cfg["optimizer"]["lr"], 0.01)
```

这个测试保证：

- `_base_` 正常继承。
- `--opts` 风格覆盖能正确解析。
- 数字类型能正确转换。

## 4. Registry 测试

`tests/test_registry.py`：

```python
import_modules(cfg["imports"])
model = build_model({"name": "cnn_cls_model", "num_classes": 2}, cfg)
loss = build_loss({"name": "cnn_cls_loss"})
```

这个测试保证：

- imports 能触发注册。
- `build_model` 能根据 `name` 构建模型。
- `build_loss` 能根据 `name` 构建损失函数。

## 5. Dataset batch contract 测试

`tests/test_dataset.py`：

```python
sample = dataset[0]
self.assertIn("input", sample)
self.assertIn("target", sample)
self.assertEqual(tuple(sample["input"].shape), (16,))
```

这个测试看起来简单，但很重要。后续 trainer 依赖 batch 字段名，如果 dataset 改坏了，训练会直接失败。

## 6. Model forward 测试

`tests/test_model_forward.py`：

```python
logits = model(torch.randn(4, 16))
self.assertEqual(tuple(logits.shape), (4, 2))
```

这个测试保证模型至少能完成一次前向，并且输出 shape 符合分类任务约定。

## 7. Train step 测试

`tests/test_train_step.py` 执行一轮最小训练：

```python
cfg = load_config(Path("config/cnn_cls_train.yaml"), ["train.epochs=1"])
trainer = build_trainer(cfg["trainer"], cfg, logger)
trainer.train()
self.assertTrue(Path("runs/test_checkpoints/cnn_cls.pt").exists())
```

这个测试覆盖：

- dataset
- dataloader
- model
- loss
- optimizer
- backward
- checkpoint

它是整个工程里最有价值的 smoke test。

## 8. Export 和 benchmark 测试

`tests/test_export_benchmark.py`：

```python
run_export(cfg, logger)
self.assertTrue(Path("exports/test_model.pt").exists())

run_benchmark(bench_cfg, logger)
self.assertTrue(Path("runs/test_benchmark.json").exists())
```

这个测试保证部署相关入口没有被训练代码变更破坏。

## 9. DDP 测试怎么做

DDP 测试通常不建议放进普通单进程单元测试里。可以单独做 smoke test：

```bash
torchrun --nproc_per_node=2 main.py --mode train --config config/cnn_cls_ddp_train.yaml --opts train.epochs=1
```

DDP smoke test 重点看：

- 进程能否正常启动。
- `DistributedSampler` 是否可用。
- 只有主进程保存 checkpoint。
- 训练是否能正常退出。

## 10. 运行测试

进入工程目录：

```bash
cd dl_project_engineering_series/article_07_testing/project
```

运行全部测试：

```bash
python run_tests.py
```

或：

```bash
python -m unittest discover tests
```

Windows 上如果 torch 安装在 Python 3.11：

```bash
py -3.11 run_tests.py
```

## 11. 本篇工程小结

这一篇为深度学习工程补上了质量闭环：

- 测 config，保证实验参数可复现。
- 测 registry，保证组件可构建。
- 测 dataset，保证 batch 字段稳定。
- 测 model forward，保证网络输出正确。
- 测 train step，保证训练闭环能跑通。
- 测 export/benchmark，保证部署入口可用。

后续每增加一个任务或组件，都应该至少补一条 smoke test。

## 12. GitHub 对应代码目录

[GitHub：article_07_testing/project](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_07_testing/project)
