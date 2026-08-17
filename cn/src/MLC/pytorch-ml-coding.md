# PyTorch 机器学习编程题：面试问题与参考回答

[返回第 2 章：ML/数据编码](./ml-coding.md)

本指南通过限时模拟面试、练习题、编程挑战和简明参考回答，覆盖 Python 与 PyTorch 中常见的机器学习工具用法。这里给出的答案具有代表性，但并不是唯一正确解。在真实面试中，优秀候选人应主动说明假设、讨论权衡、处理边界情况，并解释会如何测试代码。

难度标签遵循[本章的统一标准](./ml-coding.md#难度与公司标签)。只有存在题目级精确来源时才会添加公司标签；其余题目不做推断。

## 目录

- [模拟面试：60 分钟](#mock-interview-60-minute-session)
- [练习题库与参考回答](#practice-question-bank-with-reference-responses)
- [补充：PyTorch 基础与进阶问题](#supplemental-pytorch-core-and-advanced-questions)
- [快速练习：通用 Python 与 PyTorch 机器学习工具](#quick-drill-general-python-and-pytorch-ml-utilities)
- [更广泛的讨论题与参考回答](#broader-discussion-questions-with-reference-responses)

## 模拟面试：60 分钟

### 1. 张量基础 —— 10 分钟

#### 问题

**难度：** 中等（Medium）

不用 Python 循环，实现如下函数：

```python
def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize each row to zero mean and unit variance."""
    ...
```

要求：

- `x` 的形状为 `[batch_size, num_features]`。
- 保持 `x` 的 device 和 dtype 不变。
- 避免除零。
- 不要原地修改 `x`。

#### 参考回答

```python
import torch


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, received shape {tuple(x.shape)}")
    if not (x.is_floating_point() or x.is_complex()):
        raise TypeError("x must have a floating-point or complex dtype")

    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std
```

这里的归约使用了 `keepdim=True`，因此形状为 `[B, 1]` 的均值和标准差可以正确广播到 `[B, D]`。`clamp_min` 可以保证常数行也是安全的；在做完中心化之后，这类行会全部变成 0。整个过程都不是原地操作，并且会自然保留输入的 device 和 dtype。

#### 追问参考回答

1. **如果某一行方差为 0，应当怎样处理？** 减去均值后，常数行只包含 0。此时把标准差截断到 `eps`，输出就会是有限的全 0 向量。
2. **如果输入是 `[B, C, H, W]`，该如何修改？** 如果要做按样本归一化，就在 `(1, 2, 3)` 上归约；如果要做按通道的数据集归一化，就在 `(0, 2, 3)` 上归约。具体取决于你希望表达的语义。
3. **`view`、`reshape`、`squeeze` 和 `unsqueeze` 有什么区别？** `view` 要求 stride 兼容；`reshape` 可能返回视图，也可能分配拷贝。`squeeze` 删除大小为 1 的维度，`unsqueeze` 插入一个大小为 1 的维度。
4. **广播什么时候会出错？** 形状为 `[B]` 的张量可能会和 `[B, D]` 的最后一个维度对齐，而不是和 batch 维对齐；当维度“刚好能对上”时，甚至可能不会报错。应显式写出想要广播的维度，例如 `[B, 1]`。

### 2. 批量预处理 —— 10 分钟

#### 问题

**难度：** 中等（Medium）

编写一个可复用的预处理组件，要求能够替换非有限值、使用训练集统计量做标准化、同时支持 CPU 和 GPU、可保存状态，并且避免验证集泄漏。

#### 参考回答

```python
import torch
from torch import nn


class FeatureStandardizer(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("scale", torch.ones(num_features))
        self.register_buffer("is_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "FeatureStandardizer":
        if x.ndim != 2 or x.shape[1] != self.mean.numel():
            raise ValueError("Expected [batch, num_features] with the configured width")

        # Accumulate statistics in float32 for low-precision input.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        work = torch.nan_to_num(work)
        mean = work.mean(dim=0)
        scale = work.std(dim=0, unbiased=False).clamp_min(self.eps)
        self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
        self.scale.copy_(scale.to(device=self.scale.device, dtype=self.scale.dtype))
        self.is_fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fitted):
            raise RuntimeError("FeatureStandardizer must be fitted before use")
        clean = torch.nan_to_num(x)
        return (clean - self.mean.to(dtype=clean.dtype)) / self.scale.to(dtype=clean.dtype)
```

这个模块应只在训练数据上执行 `fit`，之后对验证、测试和推理阶段统一调用不再变化的 `transform`/`forward`。这里使用 buffer 很合适，因为这些统计量属于状态，但不是需要优化的参数；它们会随着 `.to(device)` 一起迁移，也会被包含在 `state_dict()` 中。

追问：

- 对于无状态预处理，函数就足够；如果需要保存已拟合状态并跨设备迁移，则更适合用类或 `nn.Module`。
- 统计量应保存为 buffer，而不是 parameter，因为优化器不应更新它们。
- 低精度输入在统计量计算时应使用更高精度累积，之后在变换阶段再有意识地做类型转换。
- 泄漏测试可以这样做：先在一个已知训练子集上拟合，再大幅修改验证集取值，确认已保存的统计量保持不变。

### 3. 分类指标 —— 10 分钟

#### 问题

**难度：** 中等（Medium）

从 logits 实现二分类精确率、召回率和 F1。输入可能位于 GPU 上，targets 可能只包含一个类别，不应创建 autograd 计算图，返回结果应为 Python float。

#### 参考回答

```python
import torch


@torch.inference_mode()
def binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    predictions = torch.sigmoid(logits) >= threshold
    actual = targets.bool()

    tp = (predictions & actual).sum()
    fp = (predictions & ~actual).sum()
    fn = (~predictions & actual).sum()

    def divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.where(denominator > 0, numerator.float() / denominator, 0.0)

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = divide(2 * precision * recall, precision + recall)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }
```

追问：

- Logits 是无界分数。应先经过 `sigmoid` 再与概率阈值比较；或者把概率阈值换算成等价的 logit 阈值。
- 当正类很稀少时，准确率可能掩盖模型失败。精确率、召回率、F1、PR-AUC 以及代价敏感指标通常更有信息量。
- 对于多分类指标，先用 `argmax` 得到类别预测，再按类别计算 TP/FP/FN，并提供 micro、macro 和 weighted 聚合。
- 应全局聚合充分统计量。像 F1 这样的非线性指标，按 batch 求平均通常不等于数据集级别的 F1。

### 4. 训练与评估循环 —— 20 分钟

#### 问题

**难度：** 困难（Hard）

实现一个训练 epoch 和一个完整的评估流程，要求正确处理模式切换、梯度、device 放置、按样本加权的 loss、空 dataloader，以及最终预测结果。

#### 参考回答

```python
import torch


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Cannot train on an empty dataloader")
    return {"loss": total_loss / total_examples}


@torch.inference_mode()
def evaluate(model, dataloader, loss_fn, device):
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions = []
    targets_out = []

    try:
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            predictions.append(outputs.detach().cpu())
            targets_out.append(targets.detach().cpu())
    finally:
        model.train(previous_mode)

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataloader")

    return {
        "loss": total_loss / total_examples,
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets_out),
    }
```

这里假设 `loss_fn` 返回的是 batch 平均值。先乘以 batch size，再除以总样本数，可以避免最后一个较小 batch 被赋予和其他 batch 相同权重。

追问：

- `model.eval()` 会改变模块行为，例如 Dropout 和 BatchNorm；`inference_mode()` 则会关闭 autograd 相关开销。两者解决的问题不同。
- 梯度应在下一次优化更新的前向/反向传播前清空。
- 梯度裁剪应在 `backward()` 之后执行；若使用 AMP，则应在 unscale 之后、`optimizer.step()` 之前执行。
- AMP 通常是在前向与 loss 计算周围使用 `autocast`，并在 CUDA 上用 `GradScaler` 处理缩放后的反向传播和优化器更新。
- 若做梯度累积，应先把 loss 除以累积步数，再对每个 microbatch 调用 `backward()`，并只在累积边界处执行 step/清梯度。
- 想做到精确恢复训练，通常需要保存 model、optimizer、scheduler、scaler、epoch/step、sampler 位置、配置，以及 Python/NumPy/PyTorch 的 RNG 状态。

### 5. 代码审查与调试 —— 10 分钟

#### 问题

**难度：** 中等（Medium）

审查下面这段代码：

```python
def evaluate(model, loader):
    model.train()
    losses = []

    for x, y in loader:
        prediction = model(x)
        loss = torch.nn.CrossEntropyLoss()(prediction, y)
        losses.append(loss)

    return torch.tensor(losses).mean().item()
```

#### 参考回答

问题包括：

- `model.train()` 会在评估时开启训练行为。
- 输入没有移动到模型所在 device。
- autograd 仍然开启，而把 loss 存起来会保留计算图。
- `torch.tensor(losses)` 试图从一组 tensor 构造新 tensor，可能导致 device 或类型转换错误。
- 即使 batch 大小不同，每个 batch 也被等权处理。
- loss 对象在每次迭代中都被重复创建。
- 没有处理空 loader。
- 没有恢复模型原先的模式。

修正版本如下：

```python
@torch.inference_mode()
def evaluate(model, loader, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0

    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
    finally:
        model.train(previous_mode)

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return total_loss / total_examples
```

## 练习题库与参考回答

### Python 与干净的工具设计

1. **写一个函数，把序列切分为多个 batch，并可选择是否保留最后一个不完整 batch。** — **难度：** 简单（Easy）
   先校验 `batch_size > 0`，然后遍历 `range(0, len(items), batch_size)`，依次产出切片。只有在 `drop_last=True` 且最后一个切片长度小于 `batch_size` 时才跳过它。

2. **不借助 scikit-learn，实现可复现的训练/验证集划分。** — **难度：** 简单（Easy）
   先创建索引，用带固定 seed 的本地随机生成器打乱，再在文档中明确的边界处分割。要校验比例，并避免使用全局 RNG 状态，以免无关随机调用改变结果。

3. **写一个生成器，惰性读取并转换大数据集中的记录。** — **难度：** 中等（Medium）
   在 context manager 中打开数据源，一次解析并转换一条记录，然后 `yield`。需要明确：格式错误的行是抛异常、记录日志后跳过，还是发送到错误流。

4. **在保持可逆映射的前提下，将嵌套字典展平。** — **难度：** 中等（Medium）
   递归遍历，并把 key 保存为元组，例如 `("model", "hidden_size")`。元组路径可以避免分隔符转义问题，也便于之后重建原始嵌套字典。

5. **合并多个指标字典并计算加权平均。** — **难度：** 中等（Medium）
   要求每条记录都包含权重，例如样本数。对每个 key 计算 `metric * weight` 的总和，再除以该 key 的总权重。对缺失 key 和总权重为 0 的情况要显式定义行为。

6. **实现一个统计执行时间的装饰器。** — **难度：** 简单（Easy）
   使用 `functools.wraps`，在调用前记录 `time.perf_counter()`，并在 `finally` 中再次记录。返回原函数结果，不做修改。使用 `finally` 还能保证函数抛异常时也能记录耗时。

7. **为 loss、优化器或预处理函数设计一个注册表。** — **难度：** 简单（Easy）
   使用显式映射，把经过校验的名称映射到构造器，例如 `REGISTRY = {"adamw": torch.optim.AdamW}`。避免使用 `eval`；报错时列出支持的名称，并把“查找”和“实例化”分开。

8. **把一个庞大的训练脚本重构为可测试模块。** — **难度：** 中等（Medium）
   拆分出配置、数据加载、预处理、模型构建、训练、评估、指标、checkpoint 和日志模块。显式传入依赖，这样每个组件都可以用简单 fake 对象或合成 tensor 做测试。

9. **定义一个带类型的配置对象。** — **难度：** 简单（Easy）
   可使用嵌套 dataclass，或带校验的 settings model，并提供明确的默认值和类型。对跨字段约束做校验，把解析后的完整配置与 checkpoint 一起序列化；如果运行时无需修改，优先考虑不可变设计。

10. **在不依赖可变全局状态的前提下，让一个工具具备确定性。** — **难度：** 中等（Medium）
    将 seed 或 generator 作为参数传入，并使用本地对象，例如 `random.Random(seed)` 或 `torch.Generator().manual_seed(seed)`。确定性应由调用方控制，而不是隐藏在模块全局状态中。

11. **什么时候该用 dataclass、函数、类或抽象基类？** — **难度：** 简单（Easy）
    无状态行为适合函数；结构化数据适合 dataclass；有状态行为或生命周期管理适合类；只有当存在多种实现且需要稳定、强制执行的接口契约时，才适合抽象基类。

12. **可变默认参数有什么问题？** — **难度：** 简单（Easy）
    这个对象只会创建一次，并在多次调用之间共享。应把默认值设为 `None`，再在函数内部创建列表或字典。

13. **结合机器学习配置，解释浅拷贝与深拷贝。** — **难度：** 简单（Easy）
    浅拷贝会创建新的外层字典，但内部嵌套对象仍然共享，因此修改 `copy["optimizer"]["lr"]` 可能影响原对象。深拷贝会递归复制嵌套的可变值，但对于不可变对象和外部资源仍需特别处理。

14. **如何为无效的预处理 shape 或 dtype 设计错误信息？** — **难度：** 简单（Easy）
    应指出参数名、预期契约、实际收到的 shape/dtype，并给出可能的修复建议。例如：`Expected features with shape [B, 32] and floating dtype; received [32] torch.int64`。

15. **如何在行为清晰的前提下，同时接受 NumPy array 和 PyTorch tensor？** — **难度：** 中等（Medium）
    文档中应说明一种规范的内部表示以及明确的返回策略。例如，使用 `torch.as_tensor` 转换输入；只有 tensor 输入才保留其 device；返回值要么始终是 tensor，要么显式恢复到原始后端。

### 张量操作

16. **使用 `[B, T]` mask 计算 `[B, T, D]` 的 masked mean。** — **难度：** 中等（Medium）
    先把 mask 转为 `x.dtype`，再用 `mask.unsqueeze(-1)` 扩展维度，对时间维上的 `x * mask` 求和，并除以 `mask.sum(dim=1, keepdim=True).clamp_min(1)`。还要定义：当一个序列全是 padding 时，是返回 0 还是抛错。

17. **计算 `[N, D]` 与 `[M, D]` 的两两余弦相似度。** — **难度：** 简单（Easy）
    先沿特征维用 `F.normalize` 对两个张量归一化，再计算 `a_normalized @ b_normalized.T`，结果形状为 `[N, M]`。

18. **判断真实标签是否出现在 top-k 预测中。** — **难度：** 简单（Easy）
    先计算 `topk_indices = logits.topk(k, dim=1).indices`，再与 `targets[:, None]` 比较，并调用 `.any(dim=1)`。同时校验 `1 <= k <= num_classes`。

19. **不使用 `F.one_hot` 实现 one-hot 编码。** — **难度：** 简单（Easy）
    在标签所在 device 上创建形状为 `[B, C]` 的全 0 张量，然后调用 `scatter_(1, labels[:, None], 1)`。需要校验标签范围；如果结果后续用于 loss，可能还要使用合适的浮点 dtype。

20. **构造带 label smoothing 的目标。** — **难度：** 简单（Easy）
    先创建 `[B, C]` 张量，并填入 `smoothing / (C - 1)`，再在真实类别列上用 `scatter` 写入 `1 - smoothing`。要说明平滑质量是否包含真实类别，因为这两种约定都存在。

21. **不按样本循环，计算 batched confusion matrix。** — **难度：** 中等（Medium）
    把每对 `(target, prediction)` 编码为 `targets * C + predictions`，然后调用 `torch.bincount(..., minlength=C*C)`，最后 reshape 成 `[C, C]`。多个 batch 之间可继续累加这些矩阵。

22. **对变长 tensor 做 padding 并创建 mask。** — **难度：** 中等（Medium）
    使用 `pad_sequence(sequences, batch_first=True)`，计算长度，再比较 `torch.arange(max_len)[None, :] < lengths[:, None]`。应把长度或 mask 和 padding 后的 batch 一起保留。

23. **从 `[B, C]` 中为每个样本取出某个类别分数。** — **难度：** 简单（Easy）
    使用 `logits.gather(1, labels[:, None]).squeeze(1)`。这种写法能明确表达“按 batch 逐个索引”的意图。

24. **高级索引、`gather` 和 `index_select` 有什么区别？** — **难度：** 简单（Easy）
    高级索引更灵活，可以同时使用多个索引张量。`gather` 是沿某一维按位置提供索引；`index_select` 则是对整个维度使用一份共享的一维索引列表。

25. **如何诊断 GPU predictions 搭配 CPU targets 的问题？** — **难度：** 简单（Easy）
    PyTorch 运算通常要求参与计算的 tensor 位于同一 device。应在前向/loss 计算前，把完整 batch 递归移动到模型所在 device，并避免把 device 迁移隐藏在深层工具函数里。

26. **解释 contiguous tensor 与 `.contiguous()`。** — **难度：** 中等（Medium）
    转置和某些切片会产生 stride 不标准的视图。`view` 可能拒绝这类 tensor；`.contiguous()` 会创建标准连续存储；`reshape` 则会在需要时通过拷贝处理这种情况。

27. **找出 `squeeze()` 在 batch size 为 1 时的 bug。** — **难度：** 简单（Easy）
    不带维度参数调用 `squeeze()` 会删除所有大小为 1 的维度，可能把 batch 维也删掉。应改用 `squeeze(-1)` 或其他明确指定的目标维度。

28. **原地操作会如何影响 autograd？** — **难度：** 中等（Medium）
    原地操作会修改反向传播公式可能依赖的 tensor 存储，可能触发 version counter 错误，或者让推理过程更难理解。除非非常清楚其安全性，否则应避免对需要梯度的叶子 tensor，以及反向传播所需的中间结果做原地修改。

29. **比较 `detach()`、`clone()` 和 `torch.no_grad()`。** — **难度：** 简单（Easy）
    `detach()` 返回一个与原 tensor 共享存储、但与当前计算图断开的 tensor。`clone()` 会复制数据，同时保留梯度历史；`no_grad()` 则是在其上下文中关闭后续操作的计算图记录。

30. **为什么 `.item()` 会拖慢 GPU 循环？** — **难度：** 中等（Medium）
    把 GPU 上的标量转成 Python 值会触发同步，CPU 需要等待队列中的 GPU 工作完成。应尽量在 device 端聚合，并减少不必要的数据回传频率。

### 预处理与数据集

31. **实现标准化的 `fit`、`transform` 和 `inverse_transform`。** — **难度：** 中等（Medium）
    `fit` 只保存训练集上的均值和安全的缩放系数，`transform` 应用 `(x - mean) / scale`，`inverse_transform` 应用 `z * scale + mean`。要校验特征宽度，并序列化已拟合状态。

32. **使用按通道统计量对图像做归一化。** — **难度：** 简单（Easy）
    对 `[B, C, H, W]`，先把 mean 和 std reshape 成 `[1, C, 1, 1]`，再计算 `(x - mean) / std.clamp_min(eps)`。要确认预期输入范围、颜色通道顺序和 dtype。

33. **为数值特征和类别特征实现自定义 `Dataset`。** — **难度：** 中等（Medium）
    保存不可变引用或基于索引的数组，实现 `__len__`，并让 `__getitem__` 返回文档化的结构，例如 `{"numeric": ..., "categorical": ..., "target": ...}`。编码器应只在训练数据上拟合，并放在 dataset 外部处理。

34. **为变长序列编写 `collate_fn`。** — **难度：** 中等（Medium）
    提取序列、记录长度、用 `pad_sequence` 做 padding、创建有效位置 mask，并堆叠标签与 ID。输出结构应与训练循环保持一致。

35. **如何在不泄漏的前提下处理缺失值？** — **难度：** 中等（Medium）
    数值特征的填充值和类别词表都只能在训练数据上拟合。必要时可保留缺失指示器或专门的缺失类别，并在验证和推理阶段复用冻结后的预处理状态。

36. **推理时遇到未见过的类别怎么办？** — **难度：** 中等（Medium）
    可以映射到保留的 unknown token，使用 hashing，或根据明确契约直接拒绝。除非模型和部署流程显式支持，否则绝不能悄悄扩展已学习的嵌入词表。

37. **预处理应放在哪一层？** — **难度：** 中等（Medium）
    按样本解析通常属于 dataset；padding 和 batch 组装属于 collator；可微或可导出的变换适合放在模型内；请求校验和生产一致性通常属于服务层。每一种变换都应有唯一责任方，避免重复实现。

38. **如何让带 DataLoader worker 的随机增强可复现？** — **难度：** 中等（Medium）
    为 DataLoader 的 generator 设置 seed，并在 `worker_init_fn` 中派生不同的 worker seed；还要给 worker 内部使用到的其他库设置 seed。如果希望每个 epoch 的增强不同，应有意识地把 epoch 纳入 seed 设计。

39. **为什么增加 DataLoader worker 数量反而可能更慢？** — **难度：** 中等（Medium）
    worker 启动、序列化、进程间通信、重复内存占用、小数据集场景或存储争用，可能比节省下来的工作还多。应实际 profile worker 数、batch size、persistent workers、prefetch 配置以及真实变换成本。

40. **Pinned memory 和 non-blocking 传输有什么作用？** — **难度：** 中等（Medium）
    page-locked 主机内存可以加速异步的 host-to-GPU 拷贝。`non_blocking=True` 只有在源数据已 pinned，且计算与拷贝的重叠组织合理时才有价值；它并不会在所有 pipeline 中自动变快。

### 指标

41. **从 logits 实现二分类准确率。** — **难度：** 简单（Easy）
    可以使用 `logits >= 0`，也可以用 `sigmoid(logits) >= 0.5` 得到预测，再与布尔 targets 比较，并对结果张量按浮点数求平均。

42. **实现多分类 top-k accuracy。** — **难度：** 简单（Easy）
    调用 `logits.topk(k, dim=1).indices`，与 `targets[:, None]` 比较，使用 `any(dim=1)` 归约，再求平均。分布式场景下还应同时上报分子和分母以便全局聚合。

43. **什么是 micro、macro 和 weighted F1？** — **难度：** 中等（Medium）
    Micro 是先聚合 TP/FP/FN 再计算 F1；macro 是对各类别 F1 等权平均；weighted F1 是按每个类别的 support 加权。还要说明对 support 为 0 的类别如何处理。

44. **如何实现 masked mean squared error？** — **难度：** 简单（Easy）
    先计算平方误差，用布尔 mask 做选择或乘法，累加有效误差，再除以有效样本数。若没有任何有效目标，应根据约定抛错或返回文档中定义的哨兵值。

45. **MAPE 及其失效场景是什么？** — **难度：** 简单（Easy）
    MAPE 计算 `abs(prediction - target) / abs(target)` 的平均值，因此当 target 为 0 或接近 0 时会无定义或极不稳定。可选替代包括 MAE、WAPE、sMAPE，或结合业务场景定义专用分母规则。

46. **如何在不保存全部预测结果的情况下计算 confusion matrix？** — **难度：** 中等（Medium）
    维护一个 `[C, C]` 的整数张量，每个 batch 累加一次基于 `bincount` 的矩阵。分布式评估时，先对所有 worker 的矩阵求和，再从中导出指标。

47. **如何设计流式指标（streaming metric）？** — **难度：** 中等（Medium）
    `update` 负责累积充分统计量，`compute` 在不修改内部状态的前提下计算最终指标，`reset` 负责清空状态。device、dtype、分布式归约以及空状态时的行为都应明确。

48. **为什么对 batch F1 求平均是错的？** — **难度：** 中等（Medium）
    F1 是基于聚合计数构成的非线性比值。应先在整个数据集级别汇总 TP/FP/FN，再计算一次 F1，而不是平均多个 batch 上、且样本量分布不同的局部比值。

49. **类别极不平衡的欺诈检测该看哪些指标？** — **难度：** 中等（Medium）
    应重点看精确率、召回率、PR-AUC、F-beta、固定精确率下的召回率，以及预期业务成本。还应评估校准与阈值行为；单纯准确率，甚至某些场景下的 ROC-AUC，都可能掩盖稀有正类表现。

50. **分类阈值应如何选择？** — **难度：** 中等（Medium）
    应在验证集上，根据预先声明的代价函数或指标目标来选择阈值，并在最终测试评估前固定下来。生产环境中若类别比例、错误代价或校准发生漂移，应重新评估该阈值。

### 训练与评估

51. **实现 early stopping。** — **难度：** 简单（Easy）
    记录最佳验证指标及对应 checkpoint，统计“未达到 `min_delta` 改善”的评估次数，达到 `patience` 次后停止。需要明确指标方向、并列情况、warm-up，以及 patience 统计的是 epoch 还是评估次数。

52. **加入梯度裁剪。** — **难度：** 简单（Easy）
    先调用 `loss.backward()`；若使用 AMP，应先 unscale 梯度，再在优化器更新前调用 `clip_grad_norm_` 或 `clip_grad_value_`。排查不稳定训练时，最好记录裁剪前的梯度范数。

53. **加入梯度累积。** — **难度：** 中等（Medium）
    把每个 microbatch 的 loss 除以 `accumulation_steps`，对每个 microbatch 都调用 backward，并只在累积边界处 step/清梯度。要有意识地处理最后一个不完整累积组，以及 DDP 同步问题。

54. **加入自动混合精度。** — **难度：** 中等（Medium）
    在与 device 匹配的 `autocast` 上下文中执行前向和 loss 计算，然后在 CUDA 上使用 `GradScaler.scale(loss).backward()`、`step` 和 `update`。必要时，应让数值敏感的操作保持更高精度。

55. **精确保存并恢复 checkpoint。** — **难度：** 困难（Hard）
    需要保存 model、optimizer、scheduler、AMP scaler、epoch/global step、最佳指标、配置、在相关场景下的数据/采样器位置，以及 Python/NumPy/PyTorch 的 RNG 状态。恢复时应先把这些对象完整还原，再继续下一个 batch。

56. **学习率调度器应该什么时候 step？** — **难度：** 中等（Medium）
    应遵循各自语义：OneCycle 和很多 warm-up 调度器按优化器更新步进；StepLR 常见做法是按 epoch 步进；ReduceLROnPlateau 则在拿到监控的验证指标后步进。还应在文档中说明它与 `optimizer.step()` 的先后顺序。

57. **如何先冻结 backbone、后续再解冻？** — **难度：** 中等（Medium）
    对冻结参数设置 `requires_grad=False`，并只用可训练参数构建优化器。解冻时，把目标参数改回 true，并重建或扩展优化器参数组，通常还会给预训练层使用更小的学习率。

58. **为什么 `train()` 和 `eval()` 下的验证结果会不同？** — **难度：** 简单（Easy）
    Dropout 在训练模式下是随机的；BatchNorm 在训练模式下使用 batch 统计量并更新运行统计。评估模式会关闭 Dropout，并使用保存下来的 BatchNorm 统计量。

59. **如何在验证时避免梯度产生？** — **难度：** 简单（Easy）
    同时使用 `model.eval()` 与 `torch.inference_mode()` 或 `torch.no_grad()`。不要调用 backward 或 optimizer 相关方法，并且对循环后仍需保留的 tensor 做 detach。

60. **最后一个 batch 更小时，epoch loss 应如何计算？** — **难度：** 简单（Easy）
    如果 criterion 返回的是 batch 平均值，就累积 `loss.item() * batch_size`，最后再除以总样本数。若是 token 级 loss，则应按有效 token 数加权。

61. **如何处理非有限 loss？** — **难度：** 中等（Medium）
    用 `torch.isfinite` 检测，记录当前 batch 和相关诊断信息，并遵循明确策略：停止、跳过、降低 scale，或从 checkpoint 恢复。分布式训练中要确保所有 rank 做出相同决策。

62. **如何用一套循环同时支持分类和回归？** — **难度：** 困难（Hard）
    让主循环保持任务无关，把目标准备、loss 计算、输出转预测、指标更新这些逻辑封装进 task adapter。不要把大量 `if task == ...` 分支散落在循环各处。

63. **如何干净地加入 callback hook？** — **难度：** 中等（Medium）
    定义一个小型生命周期接口，例如 `on_train_start`、`on_batch_end`、`on_validation_end` 和 `on_exception`。传入只读上下文或受控状态，避免让日志与 checkpoint callback 掌握核心优化逻辑。

64. **如何把嵌套 batch 移动到指定 device？** — **难度：** 中等（Medium）
    使用递归 tree-map，对 tensor 调用 `.to(device)`，保留映射和序列结构，字符串或 ID 保持不变。应把这部分逻辑集中到一处，确保所有循环行为一致。

65. **预测阶段如何保留输入顺序和样本 ID？** — **难度：** 中等（Medium）
    在 dataset 输出中包含稳定 ID，并与预测结果一起返回。推理时关闭 shuffle；或者按记录下来的源索引对收集结果重新排序。分布式 sampler 的 padding 也要考虑进去。

### 测试与调试

66. **如何测试标准化工具，包括常数特征？** — **难度：** 中等（Medium）
    验证拟合后的均值/缩放系数、变换后均值接近 0、非常数特征方差接近 1、常数特征输出为有限的全 0、逆变换可回到原值、shape 错误能正确报错、状态可序列化，以及 CPU/GPU 行为一致。

67. **如何验证评估过程不会更新模型状态？** — **难度：** 中等（Medium）
    在评估前保存参数和 BatchNorm buffer 快照，执行评估后再比较完整状态。同时断言 `model.training` 已按预期恢复，并且输出不需要梯度。

68. **如何比较 CPU 与 GPU 上的指标实现？** — **难度：** 中等（Medium）
    在两个 device 上运行完全相同的固定输入，并用与 dtype 相匹配的容忍误差比较结果。避免使用非确定性操作；若做不到完全一致，则在文档中说明预期误差范围。

69. **如何测试空输入、batch size 为 1 和意外 shape？** — **难度：** 简单（Easy）
    对这些场景做参数化测试，并断言返回明确定义的结果或清晰异常。必要时还要覆盖标量/单元素维度、长度为 0 的轴、非连续 tensor、错误 dtype 和 device 不匹配。

70. **如何用一个极小数据集验证模型具备过拟合能力？** — **难度：** 中等（Medium）
    在关闭数据增强和正则化的前提下，用一个很小的 batch 训练。若模型表达能力足够，训练 loss 应能降得很低；否则很可能是数据、标签、loss、梯度或优化器更新逻辑存在 bug。

71. **如何诊断 loss 一直不变化？** — **难度：** 中等（Medium）
    检查标签是否正确、学习率、模型模式、`requires_grad`、梯度是否非零、优化器参数组、`zero_grad`/`backward`/`step` 的顺序、输出是否被 detach、loss 与任务是否匹配，以及参数是否真的在变化。

72. **如何诊断完全相同运行下验证结果仍然变化？** — **难度：** 困难（Hard）
    检查 seed、评估模式、随机变换、DataLoader worker seed、样本顺序、GPU 非确定性 kernel、未初始化状态、数据竞争，以及浮点归约顺序。

73. **如何发现由计算图保留造成的内存泄漏？** — **难度：** 困难（Hard）
    检查是否把 `loss`、输出或隐藏状态直接存入列表或日志而没有 detach。标量应存 `.item()`，tensor 应存 `.detach().cpu()`；除非确有必要，否则不要使用 `retain_graph=True`。

74. **如何做梯度数值检查？** — **难度：** 困难（Hard）
    使用 `torch.autograd.gradcheck`，输入应是需要梯度的双精度张量，数值规模要小且条件良好，函数输出必须是可微的。测试点应避开不可导边界。

75. **如何验证 checkpoint 恢复后预测完全一致？** — **难度：** 中等（Medium）
    保存模型后，在一个全新实例中加载，令两个模型都处于评估模式，然后在同一固定输入上比较输出。同时测试 device 映射，并使用合适的浮点容忍误差。

## 补充：PyTorch 基础与进阶问题

下面这组补充题共 50 题，基于本指南附带的 GitHub 格式片段，以及 [Devinterview 发布的完整题目索引](https://devinterview.io/questions/machine-learning-and-data-science/pytorch-interview-questions/)。以下回答均为重新撰写、压缩整理并更新到当前 PyTorch 实践的版本，并非原文复制。

### PyTorch 基础

1. **什么是 PyTorch？它与 TensorFlow 等框架有什么不同？** — **难度：** 简单（Easy）
   PyTorch 是一个开源的张量与深度学习框架，支持 eager execution、自动求导、加速器、分布式训练、编译以及部署工具链。PyTorch 与 Python 生态和研究工作流结合得非常紧密；TensorFlow 则有不同的生态与部署历史。现代版本的两者都支持 eager execution 和图优化，因此框架选择应更多基于团队经验、目标运行时、依赖库和生产约束，而不是只看过时的“动态图 vs 静态图”区别。

2. **PyTorch 中的 tensor 是什么？** — **难度：** 简单（Easy）
   Tensor 是带有 shape、stride、dtype、layout 和 device 的多维类型化数组。与普通 NumPy array 不同，它可以参与 autograd、运行在加速器上，并支持稠密、稀疏、量化等专门表示形式。

3. **Tensor 和 Variable 有什么区别？** — **难度：** 简单（Easy）
   在现代 PyTorch 中基本没有区别。`Variable` 已在 PyTorch 0.4 合并进 `torch.Tensor`；现在 tensor 直接携带 `requires_grad`、`grad` 和 `grad_fn` 相关行为。新代码不应再把 tensor 包一层 `Variable`。

4. **如何把 NumPy array 转成 tensor？** — **难度：** 简单（Easy）
   `torch.from_numpy(array)` 以及通常情况下的 `torch.as_tensor(array)` 会与 NumPy array 共享兼容的 CPU 内存；而 `torch.tensor(array)` 会拷贝数据，并推断或接受显式 dtype。若需要独立存储，应再调用 `.clone()`；转换后若要放到加速器上，应显式迁移。

5. **`.grad` 的作用是什么？** — **难度：** 简单（Easy）
   调用 `backward()` 后，`.grad` 会保存那些 `requires_grad=True` 的叶子 tensor 的累计梯度，其中包括模型参数。非叶子 tensor 默认不会保留 `.grad`；若调试时需要，可调用 `retain_grad()`。梯度会持续累积，直到被清空或设为 `None`。

6. **什么是 CUDA？它和 PyTorch 有什么关系？** — **难度：** 简单（Easy）
   CUDA 是 NVIDIA 的 GPU 计算平台。启用了 CUDA 的 PyTorch 构建可以在 NVIDIA GPU 上分配 tensor 并执行支持的运算。模型和参与计算的 tensor 必须显式放到兼容的 device 上；性能取决于 batch 设计、内存传输、kernel 效率和同步开销，而不是简单调用一下 `.cuda()` 就够了。

7. **autograd 是如何工作的？** — **难度：** 简单（Easy）
   当梯度模式开启时，PyTorch 会把涉及需要梯度 tensor 的操作记录成动态图。调用 `backward()` 时，会应用反向模式自动微分和链式法则，把导数累计到叶子 tensor 上。除非保留计算图，否则反向传播后保存的中间结果通常会被释放。

### 神经网络设计

8. **创建一个神经网络模型的主要步骤有哪些？** — **难度：** 简单（Easy）
   先定义数据与指标契约，再创建 `nn.Module`，有意识地初始化参数，选择与任务兼容的 loss 和优化器，编写训练/验证循环，确认模型能在极小数据集上过拟合，随后结合监控进行训练，在隔离好的数据上评估，最后将模型与预处理和配置一起打包用于推理。

9. **`nn.Sequential` 和继承 `nn.Module` 有什么区别？** — **难度：** 简单（Easy）
   `nn.Sequential` 适合把多个模块串成一条线性链路。自定义 `nn.Module` 则支持分支、残差连接、多输入多输出、条件计算、共享层等非顺序结构。`Sequential` 本身也是 `nn.Module`。

10. **如何实现自定义层？** — **难度：** 中等（Medium）
    继承 `nn.Module`，在 `__init__` 中创建子模块、`nn.Parameter` 或注册 buffer，在 `forward` 中实现张量计算。如果默认初始化不合适，应显式初始化参数，并测试 shape、梯度、device、dtype、状态序列化以及编译行为。

11. **`forward` 的作用是什么？** — **难度：** 简单（Easy）
    `forward` 定义了模块的张量计算逻辑。应通过 `module(inputs)` 调用，而不是直接调用 `forward`，这样 PyTorch 才能在计算前后正确执行 hook 和 `nn.Module.__call__` 中的其他行为。

### 训练与优化

12. **什么是优化器？如何使用？** — **难度：** 简单（Easy）
    优化器根据参数梯度以及动量、自适应矩等内部状态来更新指定参数。标准更新流程是 `zero_grad`、前向、loss、反向传播，再 `step`。如果训练需要可靠恢复，checkpoint 中必须保存优化器状态。

13. **为什么、什么时候调用 `zero_grad()`？** — **难度：** 简单（Easy）
    PyTorch 默认会累积梯度，因此在开始一次逻辑上新的优化器更新前，必须清空梯度，除非你就是在有意做梯度累积。`optimizer.zero_grad(set_to_none=True)` 通常更省开销，也能区分“没有梯度”和“梯度恰好为 0”；它本身并不会执行反向传播。

14. **如何实现学习率调度？** — **难度：** 中等（Medium）
    创建绑定在优化器上的调度器，并按其契约调用 step：有的按每次优化器更新，有的按 epoch，有的在验证指标之后调用。例如，OneCycle 类调度通常按 update 步进，StepLR 常见按 epoch 步进，ReduceLROnPlateau 则需要消费一个监控指标。

15. **描述一下 PyTorch 中的反向传播。** — **难度：** 中等（Medium）
    前向传播产生输出以及标量 loss（或显式种子的 loss）。`loss.backward()` 会逆序遍历记录下来的计算图，并累计参数梯度。随后优化器读取这些梯度并更新参数；反向传播负责算梯度，优化器负责真正更新参数。

16. **梯度裁剪是如何工作的？为什么有用？** — **难度：** 中等（Medium）
    梯度裁剪会限制梯度值或全局范数，从而减少更新不稳定，尤其是在循环网络、深层网络或混合精度训练中。应在 backward 之后执行；若使用 AMP，则要先 unscale，再在优化器更新前裁剪。它只能缓解症状，不能替代对错误目标函数、数据问题或学习率问题的根因诊断。

### 调试与模型改进

17. **如何检查模型是否真的在使用 GPU？** — **难度：** 简单（Easy）
    可查看 `next(model.parameters()).device`，以及代表性输入输出的 device；还可结合 PyTorch profiler 或系统工具验证加速器利用率和显存占用。仅仅把东西分配到 GPU 上，并不代表得到了有效利用；还要观察 kernel、输入阻塞、数据传输和同步。

18. **如何监控并减少过拟合？** — **难度：** 中等（Medium）
    在具有代表性的划分上比较训练与验证曲线，并先排查数据泄漏。常见控制手段包括更多或更好的数据、数据增强、权重衰减、dropout、early stopping、更小的模型容量、迁移学习、label smoothing，以及更合理的验证设计。

19. **什么是 batch normalization？它如何影响训练？** — **难度：** 中等（Medium）
    BatchNorm 在训练时用 batch 统计量归一化激活，并学习缩放和偏置，同时更新运行统计量。它常常有助于优化并允许使用更大的学习率，但当 batch 太小或不具代表性时，统计量会很噪。因此 `train()` 和 `eval()` 下的行为是不同的。

20. **PyTorch 如何初始化神经网络权重？** — **难度：** 中等（Medium）
    标准模块通常定义了默认的 `reset_parameters` 方法。也可以在 `torch.no_grad()` 下或通过 `model.apply` 执行自定义初始化，例如根据激活函数和结构选择 Xavier 或 Kaiming 初始化。初始化还应考虑 bias、归一化层参数、嵌入参数以及可复现性。

21. **训练中常见问题有哪些？如何调试？** — **难度：** 中等（Medium）
    常见问题包括 device 或 shape 不匹配、target 或 loss 错误、梯度缺失或爆炸、非有限值、数据泄漏、模型模式错误、过度同步、输入瓶颈，以及参数没有真正更新。应先通过断言和极小数据集过拟合测试验证正确性，再检查梯度与参数变化，最后才进入性能分析。

### 数据处理与预处理

22. **如何为自定义数据集构建 DataLoader？** — **难度：** 中等（Medium）
    可以实现带 `__len__` 和 `__getitem__` 的 map-style `Dataset`；如果是流式数据，则使用 `IterableDataset`。然后把它交给 `DataLoader`，并有意识地设置 batch、shuffle 或 sampler、worker、collation 和内存参数。对于变长或嵌套样本，应使用自定义 `collate_fn`。

23. **`torchvision` transforms 用来做什么？** — **难度：** 简单（Easy）
    它们用于解码、格式转换、调整尺寸、归一化以及图像类输入的数据增强。训练阶段的随机变换应与验证/推理阶段的确定性预处理分开。输入范围、dtype、通道顺序以及 target 相关变换都必须和模型保持一致。

24. **如何为 RNN 预处理时序数据？** — **难度：** 中等（Medium）
    应按时间顺序划分，避免未来信息泄漏；只在训练数据上拟合归一化；构造输入/目标窗口；保留长度和 mask；并对变长序列做 padding 或 packing。还要明确隐藏状态是否在序列之间重置，并避免把不相关实体错误地串到同一连续状态中。

25. **什么是数据增强？通常如何实现？** — **难度：** 简单（Easy）
    数据增强通过采样“保持标签语义不变”的变化来提升泛化和鲁棒性。它可以放在 dataset 或 transform pipeline 中，包括几何、光照、时间、混合式或领域特定操作。需要验证每种变换是否真的保持任务语义；若要求可复现，还应为 worker 设置 seed。

### 进阶主题

26. **如何用 GPU 做分布式 PyTorch 训练？** — **难度：** 困难（Hard）
    `DistributedDataParallel` 通常采用每块 GPU 一个进程，借助分布式 sampler 切分输入，并通过集体通信同步梯度。要选择合适后端，在构建模型前设置好 device，并谨慎处理 checkpoint，同时保证所有 rank 遵循兼容的控制流。当单卡放不下模型或优化器状态时，可采用 FSDP 这类分片方案。

27. **如何实现迁移学习？** — **难度：** 中等（Medium）
    加载预训练模型，替换任务相关 head，使用其要求的预处理，并在合适时先冻结 backbone、只训练新 head。随后逐步解冻部分层，对预训练权重使用更小学习率，并监控灾难性遗忘与领域不匹配问题。

28. **比较 RNN、LSTM 和 GRU。** — **难度：** 中等（Medium）
    基础 RNN 的循环状态结构简单，但更容易出现梯度消失或爆炸。LSTM 通过输入门、遗忘门、输出门和 cell state 改善这一点；GRU 则使用更简化的门控状态，参数更少。具体选择取决于序列长度、数据特点、延迟要求，以及是否已有更合适的卷积或注意力结构。

29. **什么是 TorchScript？它如何帮助部署？** — **难度：** 中等（Medium）
    TorchScript 过去用于把 PyTorch 程序转成一种可序列化表示，通过 scripting 或 tracing 在无 Python 环境下运行。当前 PyTorch 文档已将 TorchScript 标记为 deprecated，并建议新的导出工作流转向 `torch.export`；但已有的 TorchScript 部署仍可能需要维护和迁移规划。

### 编程挑战

30. **为 CSV 数据集实现一个 DataLoader。** — **难度：** 中等（Medium）

    ```python
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset


    class CsvDataset(Dataset):
        def __init__(self, path: str, feature_cols: list[str], target_col: str):
            frame = pd.read_csv(path)
            self.x = torch.tensor(frame[feature_cols].to_numpy(), dtype=torch.float32)
            self.y = torch.tensor(frame[target_col].to_numpy(), dtype=torch.long)

        def __len__(self) -> int:
            return self.y.shape[0]

        def __getitem__(self, index: int):
            return self.x[index], self.y[index]


    loader = DataLoader(
        CsvDataset("train.csv", ["f1", "f2"], "label"),
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )
    ```

    对于大文件，不要让每个 worker 都把整个 CSV 读进内存；应预处理成列式或分片格式，或者改用可迭代的数据 pipeline。

31. **演示切片、索引、拼接与转置。** — **难度：** 简单（Easy）

    ```python
    x = torch.arange(12).reshape(3, 4)
    first_two_columns = x[:, :2]
    selected_rows = x[torch.tensor([0, 2])]
    stacked_rows = torch.cat([x, x], dim=0)
    transposed = x.transpose(0, 1)
    ```

    需要解释各结果的 shape，并注意转置后的 tensor 可能不是 contiguous 的。

32. **创建一个前馈式 MNIST 模型。** — **难度：** 中等（Medium）

    ```python
    from torch import nn


    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10),
    )

    loss_fn = nn.CrossEntropyLoss()
    ```

    对于 `CrossEntropyLoss`，模型应直接输出原始 logits，不要在模型里额外加 softmax。

33. **手动计算线性回归梯度。** — **难度：** 中等（Medium）

    ```python
    def linear_regression_gradients(x, y, weight, bias):
        prediction = x @ weight + bias
        residual = prediction - y
        count = y.numel()
        grad_weight = (2.0 / count) * x.transpose(0, 1) @ residual
        grad_bias = (2.0 / count) * residual.sum(dim=0)
        return grad_weight, grad_bias
    ```

    在单元测试中，应使用相同的均方误差定义，把结果与 autograd 进行对比。

34. **实现一个用于图像分类的 CNN。** — **难度：** 中等（Medium）

    ```python
    class SmallCnn(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.features(x).flatten(1)
            return self.classifier(x)
    ```

    训练时仍然需要处理输入归一化、兼容的 loss、优化器、验证流程以及 device 放置。

35. **保存并加载一个训练好的模型。** — **难度：** 简单（Easy）

    ```python
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        "checkpoint.pt",
    )

    checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    ```

    加载前应先重建一致的模型结构与配置，校验产物来源；用于推理时还应调用 `eval()`。

### 案例与场景题

36. **如何处理类别不平衡？** — **难度：** 中等（Medium）
    可采用分层评估、类别加权 loss 或 focal loss、均衡采样、定向数据增强、阈值调优，以及按类召回率、macro F1、PR-AUC 等指标。具体方法应根据错误代价选择，不能只看准确率。

37. **PyTorch 如何支持实时推理？** — **难度：** 中等（Medium）
    可使用 inference/no-grad 上下文、评估模式、受控 batching、编译或导出、更低精度、量化以及结构优化。应在目标硬件上测量端到端 tail latency、warm-up、内存、输入处理、并发、device 传输、数值漂移，以及降级/回退行为。

38. **什么时候会把模型转换为 ONNX？** — **难度：** 中等（Medium）
    当部署环境不是 Python、需要跨框架互操作，或目标推理引擎要求 ONNX 时，它会很有用。应验证算子支持、动态 shape、数值一致性、预处理和目标运行时性能；转换本身并不自动意味着优化。

39. **如何把模型部署成 REST API？** — **难度：** 困难（Hard）
    可以用 FastAPI 之类的服务把带版本的模型和预处理 pipeline 封装起来，在启动时一次性加载模型，校验请求，在合适时做 batch 推理，在无梯度环境中执行，并返回稳定的响应 schema。还应补充认证、超时、并发控制、可观测性、健康检查、金丝雀发布和回滚机制。

40. **如何微调一个预训练模型？** — **难度：** 中等（Medium）
    要匹配预训练阶段的预处理与 tokenizer 或特征契约，替换任务 head，先建立“冻结 backbone”的基线，再有意识地解冻，并给预训练层使用更小学习率。还要监控领域偏移、过拟合、灾难性遗忘和校准问题。

### 进阶与研究话题

41. **什么是 GNN？如何在 PyTorch 中实现？** — **难度：** 困难（Hard）
    图神经网络通过沿图连接关系做 message passing，更新节点、边或整图表示。可以用张量索引和稀疏运算自己实现，也可以使用 PyTorch Geometric 等库。需要仔细定义 batching、邻居采样、聚合方式以及置换不变性。

42. **NAS 的重要方向有哪些？PyTorch 如何支持它们？** — **难度：** 困难（Hard）
    神经网络结构搜索可能采用强化学习、进化搜索、贝叶斯优化、可微松弛，或权重共享的 supernet。当前更实用的方向强调硬件感知、多目标搜索、降低搜索成本、稳定排序和可复现性。PyTorch 可用于表达候选模块和训练循环，而试验编排层负责管理实验与实测延迟。

43. **GAN 是如何实现的？为什么难？** — **难度：** 困难（Hard）
    训练时让生成器去欺骗判别器，让判别器区分真实样本与生成样本，分别使用各自目标函数和优化步骤。常见问题包括 mode collapse、振荡、梯度不稳定、对结构和超参数敏感，以及评估困难。常见改进包括 Wasserstein 目标、梯度惩罚、归一化，以及更仔细的更新频率设计。

44. **什么是模型量化？什么时候有用？** — **难度：** 中等（Medium）
    量化会把权重、有时还包括激活，表示为更低精度格式，以降低内存、带宽、功耗或延迟。训练后量化成本更低；量化感知训练则通过在训练中模拟量化来尽量恢复精度。当前 PyTorch 中很多量化工作流正在集中到 `torchao`；无论哪种方式，都必须实测精度与目标硬件上的真实性能。

45. **PyTorch 在强化学习中扮演什么角色？** — **难度：** 困难（Hard）
    PyTorch 提供可微策略、价值函数、分布、优化器、向量化张量计算以及加速器支持。比如在 actor-critic 实现中，可从策略分布采样动作、估计 return 或 advantage，并更新策略损失和价值损失；环境交互通常由专门的环境库负责。

### 实践实现与开源贡献

46. **如何创建自定义 C++ 或 CUDA 算子？** — **难度：** 困难（Hard）
    需要定义并注册算子 schema 及其各 device 实现，使用官方支持的扩展工具进行构建；如果需要梯度或编译支持，还要注册 autograd 实现或 decomposition。在决定写自定义 kernel 之前，应先测试 CPU/CUDA 一致性、shape、dtype、非连续 tensor、数值梯度、错误处理、stream 行为和性能。

47. **如何在面试中讨论开源贡献或社区工具？** — **难度：** 中等（Medium）
    最好用一个具体例子来讲：问题是什么、为什么选择该工具或贡献方向、你做了哪些技术工作、评审或兼容性上遇到什么挑战、补了哪些测试和文档，以及最终带来了什么可量化影响。如果没有向上游提交过代码，也可以诚实地讲如何负责任地评估工具、报告 issue、维护内部扩展或社区依赖。

48. **如何讲一个以 PyTorch 为核心的项目？** — **难度：** 中等（Medium）
    可以按产品问题、数据与约束、模型结构与训练选择、评估设计、调试过程、部署方案以及量化结果来组织回答。要分清你个人做出的决策与团队工作，并说明一个改变了最终系统设计的权衡或失败经验。

49. **如何提升实验可复现性？** — **难度：** 中等（Medium）
    记录代码、环境、配置、数据与产物版本；为 Python、NumPy、PyTorch 及 DataLoader worker 设置 seed；控制 sampler；在需要时启用确定性算法。需要注意：跨 PyTorch 版本、平台以及 CPU/GPU 执行环境，并不能保证绝对一致；确定性 kernel 也可能更慢，因此应先定义清楚你要保证的可复现边界。

50. **PyTorch Lightning 如何简化 PyTorch 工作流？** — **难度：** 简单（Easy）
    Lightning 可以把训练 hook、device 与分布式设置、精度、日志、checkpoint 以及常见循环机制标准化，同时底层仍然是 PyTorch。它能减少样板代码并统一团队规范，但候选人仍应理解原生 PyTorch 语义，并能权衡框架抽象、调试、定制、依赖和迁移成本。

## 快速练习：通用 Python 与 PyTorch 机器学习工具

1. **如何在不使用 Python 循环的情况下，对 tensor 的每一行做归一化？** — **难度：** 简单（Easy）

   沿特征维计算均值和标准差，并保留该维以便广播：

   ```python
   def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
       mean = x.mean(dim=1, keepdim=True)
       std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
       return (x - mean) / std
   ```

   `keepdim=True` 会产生形状为 `[B, 1]` 的统计量，可广播到 `[B, D]`。对标准差做截断可以防止常数行除零，且这些操作会保留 tensor 的 device 和浮点 dtype。

2. **`view`、`reshape` 和 `squeeze` 有什么区别？** — **难度：** 简单（Easy）

   `view` 基于同一块存储返回不同 shape，要求内存 stride 兼容，所以在转置后经常会失败。`reshape` 具有相同的形状语义，但当无法返回视图时会创建一个连续拷贝。`squeeze(dim)` 只会在指定维度大小为 1 时删除该维。若 batch size 可能为 1，应避免直接调用不带参数的 `squeeze()`，否则可能误删 batch 维。

3. **如何设计一个可复用的预处理组件？** — **难度：** 中等（Medium）

   给它定义一个小而清晰的契约，例如 `fit`、`transform` 和 `state_dict`。统计量只在训练数据上拟合；像均值和缩放系数这类不可训练状态应保存为已注册的 buffer；还要校验输入 shape 和 dtype，并明确 device 行为。训练阶段的随机增强应与验证和推理阶段的确定性预处理分开。

4. **什么是数据泄漏？预处理阶段如何避免？** — **难度：** 简单（Easy）

   数据泄漏指训练时本不应获得的信息影响了模型或评估结果，例如在划分数据之前就用全量数据拟合标准化器。正确做法是先划分，再只在训练集上拟合预处理，随后把冻结后的变换用于验证、测试和在线服务。对于时序数据，应按时间切分，并确保特征中不含未来信息。

5. **解释精确率、召回率和 F1。分别适用于什么场景？** — **难度：** 简单（Easy）

   精确率是 `TP / (TP + FP)`，回答的是“预测为正的样本中，有多少是真的正类”；召回率是 `TP / (TP + FN)`，回答的是“所有真实正类中，有多少被找到了”；F1 是两者的调和平均。假阳性代价高时优先看精确率，假阴性代价高时优先看召回率；当两者都重要且类别不平衡时，F1 通常更合适。最终仍应以产品侧的错误代价为准。

6. **为什么对各个 batch 的 F1 求平均通常不正确？** — **难度：** 中等（Medium）

   F1 是 TP、FP 和 FN 的非线性函数。按 batch 计算再平均，通常与在整个数据集上一次性计算出来的 F1 不同，尤其是在 batch 大小或类别分布差异较大时。正确做法是在所有 batch 上累积 TP、FP 和 FN——分布式评估时还要跨 worker 归约——最后只计算一次 F1。

7. **PyTorch 训练循环的基本步骤是什么？** — **难度：** 中等（Medium）

   把模型切到训练模式，把 batch 移到正确的 device，清空旧梯度，执行前向传播，计算 loss，反向传播，再更新参数：

   ```python
   model.train()
   for x, y in loader:
       x, y = x.to(device), y.to(device)
       optimizer.zero_grad(set_to_none=True)
       logits = model(x)
       loss = loss_fn(logits, y)
       loss.backward()
       optimizer.step()
   ```

   生产级训练循环通常还需要处理按样本加权的指标、非有限 loss 检查、日志、checkpoint，以及在适合时引入混合精度。

8. **`model.eval()` 与 `torch.no_grad()` 或 `torch.inference_mode()` 有什么区别？** — **难度：** 简单（Easy）

   `model.eval()` 会改变层行为，例如关闭 Dropout，并让 BatchNorm 使用保存好的运行统计量。`no_grad()` 会关闭梯度记录；`inference_mode()` 还会进一步减少 autograd 开销，因此在其更严格语义可接受时，更适合纯推理/评估场景。评估通常需要同时使用 `model.eval()` 和 inference/no-grad 上下文。

9. **如果忘了调用 `optimizer.zero_grad()` 会发生什么？** — **难度：** 简单（Easy）

   PyTorch 会把梯度累积到每个参数的 `.grad` 中。如果不清空，下一次更新使用的就是多次 backward 的梯度和。只有在你有意识地实现了梯度累积，并正确处理 loss 缩放和 step 时序时，这才是合理的；否则会悄悄改变优化行为。

10. **如何把一个庞大的机器学习脚本重构成干净模块？** — **难度：** 中等（Medium）

    把配置、数据集与预处理、模型构建、训练、评估、指标、checkpoint 和日志拆开。尽量把张量变换与指标写成纯函数，让主流程循环尽量薄，并通过显式依赖注入替代全局状态。组件之间应通过稳定的 tensor 接口和小型结果字典协作，而不是对某个框架抽象产生过度耦合。

11. **如何编写一个同时支持 CPU 和 GPU 的工具函数？** — **难度：** 简单（Easy）

    新 tensor 应从已有 tensor 派生，或显式使用输入的 `device` 和 `dtype` 创建；不要在函数内部偷偷调用 `.cuda()`，也不要构造隐藏的 CPU 常量。应在清晰边界处迁移完整 batch，让参与计算的 tensor 保持在兼容 device 上；除非契约另有说明，返回结果也应保留在输入所在 device。最后还要做 CPU/GPU 一致性测试，并设置合理数值容忍度。

12. **你会如何测试一个张量预处理或指标工具？** — **难度：** 中等（Medium）

    应覆盖常规情况，以及单样本 batch、常数值、缺失值或非有限值、支持时的空输入、错误 rank、不兼容 dtype，以及非连续 tensor。要验证预期 shape、device 和 dtype 是否保持、数值结果是否正确、是否存在意外原地修改、已拟合状态是否可序列化，以及 CPU/加速器行为是否一致。预处理工具要额外加泄漏测试；指标工具则应覆盖全正、全负和分母为 0 的情况。

## 更广泛的讨论题与参考回答

1. **什么样的机器学习工具才算“可复用而不过度抽象”？** — **难度：** 中等（Medium）
   它应具备小而明确的契约、实用的默认值、清晰的 shape/device/dtype 行为、可组合的输入输出，以及聚焦的测试。只有在多个真实用例暴露出稳定共性接口之后，才值得继续抽象。

2. **模块边界上应做哪些校验？** — **难度：** 中等（Medium）
   要校验那些如果不提前检查，就会在后续更晚处失败或产生静默错误的假设：rank、特征宽度、dtype 类型族、值域、已拟合状态、必要 key，以及不兼容的配置组合。若上游契约已保证正确，则不要在高频内层循环中重复做昂贵检查。

3. **工具什么时候该返回 tensor，什么时候该返回 Python 数值？** — **难度：** 简单（Easy）
   当调用方可能还需要在 device 端继续组合、做 batch 处理、分布式归约或保留梯度时，应返回 tensor。到了报表、日志或序列化边界，且同步本就是有意行为时，再返回 Python 数值。

4. **如何保证训练、评估与推理阶段的预处理一致？** — **难度：** 中等（Medium）
   只实现一套带版本的预处理组件，只在训练数据上拟合，并把它的状态与模型产物一起序列化，在评估和服务端复用同一套逻辑。最好再用 golden example 做一致性测试。

5. **机器学习代码库如何组织，才能支持独立测试？** — **难度：** 中等（Medium）
   应把纯变换与指标和有状态的流程编排分开。通过依赖注入，把模型、loss、优化器、数据源和 callback 传给薄薄的主循环，这样每个边界都能用合成数据和简单 fake 进行独立测试。

6. **PyTorch 能对可复现性控制到什么程度？** — **难度：** 中等（Medium）
   它可以设置自身 RNG seed、请求确定性算法、控制 generator 的使用，但硬件、库版本、分布式调度、数据 pipeline 和浮点归约顺序仍然会影响结果。可复现性本质上是端到端系统属性，而不是单个框架属性。

7. **如何平衡 tensor 代码的可读性与向量化？** — **难度：** 中等（Medium）
   先写出最清晰且正确的张量表达式，再对真实瓶颈做基准测试，只在收益明显时优化。可通过显式命名中间 shape、加入断言、写明语义注释以及补充等价性测试来保持可读性。

8. **训练循环应该记录哪些日志？** — **难度：** 中等（Medium）
   应记录解析后的配置、代码/数据/模型版本、epoch 与 global step、loss、关键指标、学习率、吞吐、在有价值时记录梯度范数、验证结果、checkpoint 标识、耗时，以及诸如“跳过了非有限 batch”这类重要告警。

9. **如何审查跨 device、dtype 和 shape 的正确性？** — **难度：** 中等（Medium）
   先定义支持矩阵，再做参数化测试，覆盖 singleton、空输入和非连续场景，并在容忍误差范围内比较 CPU 与加速器输出，同时测试混合精度和序列化。对不支持的组合，应做到明确失败而不是静默行为异常。

10. **把 notebook 代码迁移到生产环境前，需要改哪些地方？** — **难度：** 中等（Medium）
    需要抽出配置与可复用模块，移除隐藏状态，补齐校验与测试，让预处理可复现，加入日志、checkpoint 和错误处理，固定依赖版本，分析资源使用，定义产物/版本契约，并记录部署与回滚策略。
