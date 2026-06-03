# 训练部署与模型导出：从 checkpoint 到 TorchScript、ONNX 和 benchmark

> 深度学习项目工程化实战系列 06

训练得到的 checkpoint 不等于部署模型。checkpoint 通常包含训练态信息，适合恢复训练或继续实验；部署模型则要求输入输出稳定、格式可加载、推理链路清晰，并且最好有性能数据。

本文在前面工程基础上增加两个模式：

```text
export：导出模型
benchmark：测试推理延迟和吞吐
```

本文对应工程目录：

```text
dl_project_engineering_series/
  article_06_export_benchmark/
    README.md
    project/
```

## 1. 本篇最终代码目录

```text
project/
  config/
    cnn_cls_export_torchscript.yaml   # TorchScript 导出配置
    cnn_cls_export_onnx.yaml          # ONNX 导出配置
    cnn_cls_benchmark.yaml            # benchmark 配置

  launcher/
    export.py                         # 构建模型、加载 checkpoint、导出
    benchmark.py                      # 构建模型、预热、计时、保存 JSON

  core/
    models/
    networks/
    registry.py
```

## 2. checkpoint 和部署模型的区别

训练 checkpoint 通常用于实验恢复：

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "cfg": cfg,
}
```

部署模型关注的是：

- 模型结构是否可重建。
- 权重是否正确加载。
- 输入 shape 是否明确。
- 输出名称和语义是否稳定。
- 推理框架是否支持该格式。

所以部署逻辑不应该塞进 `train.py`，而应该单独放在：

```text
launcher/export.py
```

## 3. export.py：导出流程

导出流程：

```text
load config
import registered modules
build model
load checkpoint
create dummy input
export torchscript or onnx
```

核心代码：

```python
model = build_model(cfg.get("model", {}), cfg).to(device).eval()
_load_checkpoint_if_exists(model, checkpoint, device, logger)
dummy = torch.randn(1, input_dim, device=device)
```

TorchScript：

```python
traced = torch.jit.trace(model, dummy)
traced.save(str(output_path))
```

ONNX：

```python
torch.onnx.export(
    model,
    dummy,
    str(output_path),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
)
```

## 4. 导出配置

TorchScript：

```yaml
_base_: cnn_cls_predict.yaml

export:
  format: torchscript
  checkpoint: runs/checkpoints/cnn_cls.pt
  output_path: exports/cnn_cls_torchscript.pt
```

ONNX：

```yaml
_base_: cnn_cls_predict.yaml

export:
  format: onnx
  checkpoint: runs/checkpoints/cnn_cls.pt
  output_path: exports/cnn_cls.onnx
  opset_version: 17
```

运行：

```bash
python main.py --mode export --config config/cnn_cls_export_torchscript.yaml
python main.py --mode export --config config/cnn_cls_export_onnx.yaml
```

## 5. benchmark.py：推理性能测试

benchmark 的目标不是训练，而是回答：

- 单次推理平均耗时多少？
- 每秒能处理多少样本？
- 使用哪个 device？
- 结果是否可以保存，便于对比？

配置：

```yaml
benchmark:
  warmup: 5
  iters: 50
  output_path: runs/benchmark.json
```

核心流程：

```python
for _ in range(warmup):
    model(x)

start = time.perf_counter()
for _ in range(iters):
    model(x)
elapsed = time.perf_counter() - start
```

输出：

```json
{
  "device": "cpu",
  "iters": 50,
  "latency_ms": 0.12,
  "throughput_samples_per_s": 8000.0
}
```

运行：

```bash
python main.py --mode benchmark --config config/cnn_cls_benchmark.yaml
```

## 6. 部署扩展后的目录结构

```text
project/
  launcher/
    train.py
    eval.py
    predict.py
    export.py                 # 新增：部署导出逻辑
    benchmark.py              # 新增：性能测试逻辑

  config/
    cnn_cls_export_torchscript.yaml
    cnn_cls_export_onnx.yaml
    cnn_cls_benchmark.yaml

  exports/                    # 运行后生成，不提交源码时可忽略
  runs/
    benchmark.json            # 运行后生成
```

部署相关逻辑保持在 launcher 中，而不是写进 model 或 network。原因是导出是运行流程，不是模型结构本身。

## 7. 运行顺序

先训练：

```bash
python main.py --mode train --config config/cnn_cls_train.yaml --opts train.epochs=1
```

再导出：

```bash
python main.py --mode export --config config/cnn_cls_export_torchscript.yaml
```

再测速：

```bash
python main.py --mode benchmark --config config/cnn_cls_benchmark.yaml
```

## 8. 本篇工程小结

这一篇完成了训练到部署之间的工程衔接：

- `export.py` 负责模型导出。
- `benchmark.py` 负责推理性能测试。
- 配置决定导出格式、checkpoint、输出路径和测试次数。
- 部署流程和训练流程解耦。

后续如果接入 TensorRT、OpenVINO 或服务化部署，也应该继续沿用这个边界。

## 9. GitHub 对应代码目录

[GitHub：article_06_export_benchmark/project](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_06_export_benchmark/project)
