# <a name="ml-coding"></a> 2. 机器学习/数据编程 :robot:

ML 编码面试因公司而异。有些更关注从零实现经典算法，另一些则会考察实用的 Python 和 PyTorch 能力，例如张量操作、预处理、指标计算和训练循环。无论是哪种形式，面试官通常都会评估：正确性、数值稳定性、代码质量、边界情况处理、复杂度，以及你解释设计取舍的能力。

## 如何使用本章

- 将 [`solutions/ml_algorithms.py`](./solutions/ml_algorithms.py) 作为核心面试题的权威、可执行 NumPy 参考实现。
- 运行 [`solutions/test_ml_algorithms.py`](./solutions/test_ml_algorithms.py) 来验证实现，并学习一些有价值的边界情况。
- 将较早的 [notebooks](./notebooks/) 用作补充性的探索材料。其中一些早于权威解法，可能不够完整。
- 练习在不看参考实现的情况下写出每个重点题目，然后对比正确性、复杂度和边界情况处理。

在仓库根目录运行参考测试：

```bash
uv run --with numpy python src/MLC/solutions/test_ml_algorithms.py
```

## PyTorch ML 编码

现代 ML 编码面试可能会考察实用的 PyTorch 能力，而不只是要求候选人从零实现算法。[PyTorch ML Coding Problems](./pytorch-ml-coding.md) 指南包括：

- 一场 60 分钟的模拟面试，覆盖张量、预处理、指标、训练/评估循环和调试
- 面向 ML 工作流的 Python 工具题与高质量代码题
- 张量操作、数据集、batch 处理、设备管理和 autograd
- 训练、优化、混合精度、checkpoint 保存与恢复，以及可复现性
- 测试、调试、部署和高级 PyTorch 问题
- 编码挑战与简明参考答案

## 优先级最高的 ML 编码题

下面这组题目结合了仍然常见的经典问题，以及在 AI/ML 面试中越来越常被要求掌握的现代基础能力。

| 题目 | 权威解法 | 补充 notebook | 一个优秀解法应覆盖的内容 |
| --- | --- | --- | --- |
| 数值稳定的 softmax 和交叉熵 | `softmax`, `cross_entropy_from_logits` | — | 最大值平移、log-sum-exp、shape、类别索引校验 |
| 使用梯度下降的线性回归 | `linear_regression_gradient_descent` | [Linear regression](./notebooks/linear_regression_md.ipynb) | 向量化梯度、偏置项、MSE 缩放、收敛性 |
| 使用梯度下降的逻辑回归 | `logistic_regression_gradient_descent` | [Logistic regression](./notebooks/logistic_regression_md.ipynb) | 稳定的 sigmoid、二元交叉熵梯度、阈值 |
| k 最近邻 | `knn_predict` | [k-NN](./notebooks/k_nearest_neighbors.ipynb) | 成对距离、top-k 选择、平票处理、复杂度 |
| k-means 聚类 | `kmeans` | [k-means](./notebooks/k_means_2.ipynb) | 初始化、向量化分配、收敛、空簇 |
| 决策树划分 | `gini_impurity`, `best_gini_split` | [Decision tree](./notebooks/decision_tree.ipynb) | 候选阈值、加权不纯度、停止条件 |
| 主成分分析 | `principal_component_analysis` | — | 中心化、SVD/特征分解、主成分排序、方差 |
| 二维卷积 | `conv2d_valid` | [Convolution](./notebooks/convolution.ipynb) | 输出 shape、步幅、互相关与卷积的区别 |
| 缩放点积注意力 | `scaled_dot_product_attention` | — | Q/K/V 的 shape、`1/sqrt(d_k)`、在稳定 softmax 前做 mask |
| 二分类指标与 ROC-AUC | `binary_classification_metrics`, `roc_auc` | — | 分母为零、类别不平衡、平票、排序解释 |
| 蓄水池抽样 | `reservoir_sample` | — | 流长度未知、均匀概率、O(k) 内存 |
| TF-IDF | `tfidf` | — | token 计数、文档频率、平滑、稀疏缩放 |

所有权威函数都在 [`solutions/ml_algorithms.py`](./solutions/ml_algorithms.py) 中。

## 其他经典算法

这些题目也很适合作为后续练习，尤其是在它们与目标团队的业务领域匹配时：

- 线性 SVM 和 hinge loss（[notebook](./notebooks/svm.ipynb)）
- 感知机学习规则（[notebook](./notebooks/perceptron.ipynb)）
- 前馈神经网络与反向传播（[notebook](./notebooks/feedforward.ipynb)）
- 指标和损失的多分类或多标签扩展
- 用于文本分类的朴素贝叶斯
- 用于推荐系统的矩阵分解
- 梯度提升：解释训练循环，并实现一个简单的残差拟合步骤

## 数据与采样问题

- 实现无泄漏的训练集/验证集/测试集划分
- 仅使用训练集统计量对特征做标准化
- 一致地处理缺失值和未见类别
- 实现均匀采样、分层采样、加权采样和蓄水池抽样
- 构建 mini-batch，并对变长序列进行 padding
- 正确聚合按样本加权的损失和流式指标

## 面试中应该说明什么

1. 在开始编码前，先说明输入 shape、dtype、前提假设和期望输出。
2. 先写出正确的基线版本，再对瓶颈部分做向量化或优化。
3. 讨论时间与空间复杂度，包括成对矩阵的计算开销。
4. 处理数值稳定性、空输入、平票、常量特征和非法标签。
5. 为正常情况编写小测试，并至少覆盖一个失败或边界情况。
6. 说明如果面对大规模数据集、GPU、分布式训练或生产级库，实现会如何变化。
