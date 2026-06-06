# 面向大规模推理测试集的并行测试框架搭建

> 深度学习项目工程化实战系列 07

模型进入业务测试阶段后，真正耗时的通常不是代码链路检查，而是大规模测试集上的推理执行：

> 有几十万、几百万条测试样本，单进程推理太慢；某些样本失败时不能影响整轮测试；最终还要统计准确率、吞吐、耗时、失败样本和报告。

本文所说的“业务测试”，专指**模型推理过程的测试**，不是训练流程测试，也不是 `config`、`registry`、dataset contract 这类工程代码测试。配置文件在这里只是推理测试的参数来源，用来说明测试哪个模型、哪份测试集、使用什么设备和输出到哪里。

本文对应工程目录：

```text
dl_project_engineering_series/
  article_07_testing/
    README.md
    project/
```

## 1. 本篇最终代码目录

本篇新增的是一套大规模推理测试执行框架：

```text
project/
  run_business_tests.py              # 推理业务测试入口：并行跑测试集

  testing/
    __init__.py
    config.py                        # TestConfig / TestResult 数据类
    base.py                          # BaseDataLoader / BaseTaskProcessor / BaseResultCollector
    framework.py                     # ParallelTestFramework，并行编排引擎
    network_business.py              # 网络推理测试：切分样本、执行推理、汇总 JSON 报告

  config/
    cnn_cls_predict.yaml             # 示例推理测试配置
  datatxt/
    test.txt                         # 示例测试样本列表

  core/
    dataset/                         # 示例测试集读取
    networks/                        # 示例网络结构
    models/                          # 示例推理模型
```

这套框架解决的是：

```text
大规模测试集
  -> 切分成多个推理任务
  -> 多进程并行执行
  -> 单任务失败隔离
  -> 汇总指标与失败信息
  -> 生成业务测试报告
```

## 2. 为什么推理业务测试需要单独设计

小规模验证时，可以写一个脚本顺序遍历测试集：

```python
for sample in test_dataset:
    pred = model(sample)
    metric.update(pred, sample["label"])
```

但业务测试集变大后，这种写法会遇到几个问题：

- 单进程推理耗时太长。
- 某个样本读取失败、预处理失败或推理异常时，整轮测试容易中断。
- 无法清晰知道每个任务的耗时、吞吐和失败原因。
- 难以把 bad case、错误信息、分批指标统一落盘。
- 后续想切换分类、检测、OCR、推荐等任务时，主流程容易堆满业务分支。

因此本文把测试框架拆成“任务加载、任务处理、结果汇总、并行调度”四层。框架本身不关心具体模型类型，只负责高效、稳定地执行推理测试。

## 3. 核心架构

并行推理测试框架由三类组件和一个编排引擎组成：

```text
ParallelTestFramework
  |
  |-- BaseDataLoader
  |     |-- load_data              # 加载测试集或测试清单
  |     |-- filter_data            # 过滤不需要测试的样本
  |     |-- prepare_task_args      # 把样本切成子进程任务参数
  |
  |-- BaseTaskProcessor
  |     |-- initialize_worker      # 子进程初始化模型或资源
  |     |-- process                # 执行一批样本的推理测试
  |     |-- cleanup_worker         # 子进程释放资源
  |
  |-- BaseResultCollector
        |-- collect                # 收集一个任务的测试结果
        |-- finalize               # 生成最终测试报告
```

完整数据流：

```text
测试配置
  -> DataLoader 加载测试集
  -> 按 chunk 切分样本
  -> ProcessPoolExecutor 并行执行
  -> 子进程独立加载模型并推理
  -> TestResult 返回主进程
  -> ResultCollector 汇总 summary.json
```

这类架构的关键点是：**主进程只做调度和汇总，子进程只做推理任务；单个任务失败会变成结构化错误结果，不影响其他任务继续执行。**

## 4. 测试配置与结果对象

`testing/config.py` 定义推理测试配置：

```python
@dataclass
class TestConfig:
    max_workers: int | None = None
    progress_interval: int = 10
    timeout: int = 300
    output_dir: str = "runs/business_tests"
```

字段含义：

```text
max_workers         并行进程数
progress_interval   每完成多少个任务打印一次进度
timeout             整轮任务等待超时时间
output_dir          测试报告输出目录
```

单个任务返回 `TestResult`：

```python
@dataclass
class TestResult:
    task_id: str
    status: str
    data: dict[str, Any]
    error_message: str = ""
    execution_time: float = 0.0
```

推理测试结果必须结构化。只打印日志很难支撑后续分析，结构化结果则可以继续生成：

- 总体指标报告。
- 失败任务列表。
- bad case 明细。
- 每类指标统计。
- 延迟和吞吐统计。

## 5. 并行编排引擎

`testing/framework.py` 中的 `ParallelTestFramework` 只负责通用执行流程：

```python
raw_data = self.data_loader.load_data(data_source)
filtered_data = self.data_loader.filter_data(raw_data)
task_args = [self.data_loader.prepare_task_args(item, context) for item in filtered_data]

with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
    futures = [
        executor.submit(_worker_wrapper, self.task_processor_class, args)
        for args in task_args
    ]
    for future in as_completed(futures, timeout=self.config.timeout):
        result = future.result()
        self.result_collector.collect(result)
```

框架没有写死分类、检测或 OCR 的逻辑。它只知道：

1. 从哪里加载测试任务。
2. 怎样把任务并行提交到子进程。
3. 怎样收集每个任务返回的 `TestResult`。

具体模型怎么加载、输入怎么预处理、指标怎么计算，由业务自己的 `TaskProcessor` 决定。

## 6. 子进程隔离策略

`_worker_wrapper` 是推理任务隔离的核心：

```python
processor = processor_class()
try:
    processor.initialize_worker()
    result = processor.process(*task_args)
    return result
except Exception as exc:
    return TestResult(
        task_id=str(task_args[0]),
        status="error",
        data={},
        error_message=f"{type(exc).__name__}: {exc}",
    )
finally:
    processor.cleanup_worker()
```

这样做的好处是：

- 单个任务异常不会中断整轮测试。
- 每个子进程拥有独立的模型实例和中间状态。
- 模型、文件句柄、缓存等资源可以在子进程内初始化和释放。
- 主进程收到的始终是统一格式的结果。

业务测试里，局部失败是常态。测试框架要做的不是假设所有样本都会成功，而是把失败样本、失败原因和失败位置完整记录下来。

## 7. 网络推理测试实现

`testing/network_business.py` 给出一个最小的网络推理测试示例。

`NetworkCaseLoader` 负责把测试集切成多个 chunk：

```python
for start in range(0, total, self.chunk_size):
    indices = list(range(start, min(start + self.chunk_size, total)))
    chunks.append(
        {
            "task_id": f"{self.split}_{start:06d}_{indices[-1]:06d}",
            "config_path": data_source,
            "indices": indices,
            "split": self.split,
        }
    )
```

这里每个任务包含一批样本，而不是一个样本。原因是进程调度本身有开销，真实业务中通常会按 batch 或 shard 来提交任务。

`NetworkBatchProcessor` 在子进程中执行推理：

```python
cfg = load_config(Path(config_path), list(global_context.get("opts", [])))
import_modules(cfg.get("imports", []))

dataset = CnnClsDataset(cfg.get("dataset", {}), split=split)
model = build_model(cfg.get("model", {}), cfg).to(device).eval()
```

然后遍历当前 chunk 中的样本：

```python
with torch.no_grad():
    for index in indices:
        sample = dataset[index % len(dataset)]
        x = sample["input"].unsqueeze(0).to(device)
        target = sample["target"].view(1)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu()
```

每个任务返回：

```python
{
    "samples": total,
    "correct": correct,
    "accuracy": correct / max(total, 1),
    "inference_time": inference_time,
    "samples_per_second": total / max(inference_time, 1e-12),
}
```

示例工程没有加载真实业务权重，因此 accuracy 只用于演示统计流程。真实项目中通常会在 `TaskProcessor` 中增加：

- 模型权重加载。
- 图像、文本或特征预处理。
- 预测结果保存。
- bad case 记录。
- 混淆矩阵或分类型指标。
- 置信度、阈值、Top-K 等业务字段。

## 8. 结果汇总报告

`JsonResultCollector` 最终生成：

```text
runs/business_tests/summary.json
```

报告示例：

```json
{
  "total_tasks": 4,
  "success_count": 4,
  "error_count": 0,
  "success_rate": 1.0,
  "total_samples": 32,
  "total_correct": 18,
  "accuracy": 0.5625,
  "tasks_per_second": 3.4,
  "inference_samples_per_second": 1200.5
}
```

真实业务中，结果收集器可以继续扩展输出：

- `summary.json`：总体指标。
- `failed_tasks.jsonl`：失败任务列表。
- `bad_cases.jsonl`：预测错误样本。
- `metrics_by_class.csv`：按类别统计指标。
- `latency_percentile.json`：P50 / P90 / P99 延迟。

## 9. 运行命令

进入工程目录：

```bash
cd dl_project_engineering_series/article_07_testing/project
```

运行推理业务测试：

```bash
python run_business_tests.py --config config/cnn_cls_predict.yaml --split test --chunk-size 8 --max-workers 2
```

限制样本数，便于快速调试并观察任务切分：

```bash
python run_business_tests.py --config config/cnn_cls_predict.yaml --max-samples 16 --chunk-size 4 --max-workers 2
```

指定设备和测试输出目录：

```bash
python run_business_tests.py --config config/cnn_cls_predict.yaml --device cpu --output-dir runs/business_tests
```

通过 `--opts` 临时覆盖推理测试参数：

```bash
python run_business_tests.py --config config/cnn_cls_predict.yaml --opts runtime.device=cpu dataset.input_dim=16
```

## 10. 如何扩展到真实业务

如果要测试一个新的业务模型，例如猫狗分类、缺陷检测、OCR 或推荐模型，通常只需要实现三类组件：

```text
testing/
  cat_dog_business.py
    CatDogCaseLoader             # 读取测试图片列表
    CatDogProcessor              # 加载推理模型，执行预测
    CatDogResultCollector        # 汇总准确率、吞吐和 bad case
```

扩展示例：

```python
class CatDogProcessor(BaseTaskProcessor):
    def initialize_worker(self):
        self.model = load_inference_model(self.model_path)

    def process(self, task_id, image_paths, global_context):
        results = []
        for image_path in image_paths:
            pred = model_predict(self.model, image_path)
            results.append({"image": image_path, "pred": pred})
        return TestResult(task_id=task_id, status="success", data={"results": results})
```

框架本身不需要知道图片如何读取、模型如何预测、指标如何计算。它只负责并行调度、失败隔离、进度管理和结果收集。

## 11. 小结

本文的测试框架只面向一件事：**让大规模模型推理测试更高效、更稳定、更容易分析结果**。

核心设计是：

```text
DataLoader 负责把大规模测试集切成任务
TaskProcessor 负责在子进程中执行推理
ResultCollector 负责汇总结构化结果
ParallelTestFramework 负责并行调度、失败隔离和进度管理
```

训练流程和工程代码链路测试不属于本文范围。业务测试的主角是推理样本、推理耗时、推理结果、失败样本和最终报告。

## 12. GitHub 对应代码目录

[GitHub：article_07_testing/project](https://github.com/Hymanmin/dl_project_engineering_series/tree/main/article_07_testing/project)
