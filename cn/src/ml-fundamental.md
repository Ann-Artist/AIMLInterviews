# <a name="breadth"></a> 3. 机器学习基础（广度）
顾名思义，这类面试旨在考察你对机器学习概念的整体理解，既包括理论层面，也包括实践层面。与机器学习深度面试不同，广度面试在不同面试官和候选人之间，整体结构和覆盖范围通常都比较相似。

准备这类面试的最佳方式，是复习你在机器学习课程中的笔记，同时参考一些高质量的在线课程和资料。尤其是下面这些资源，我觉得非常有帮助。

# 1. 课程与复习资料：
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)（你也可以在[Youtube](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)上找到课程视频）
- [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects)
- [Udacity's deep learning nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) 或 [Coursera's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)（适合深度学习）

如果你已经掌握了这些概念，那么下面这些资源非常适合快速复习不同知识点：
- [StatQuest Machine Learning videos](https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
- [StatQuest Statistics](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)（用于统计学复习——对 Data Science 岗位尤其有用）
- [Machine Learning cheatsheets](https://ml-cheatsheet.readthedocs.io/en/latest/)
- [Chris Albon's ML falshcards](https://machinelearningflashcards.com/)

# 2. 机器学习基础主题

下面是最重要、需要覆盖的主题：

## 1. 经典机器学习概念
### 机器学习算法的分类
  - 监督学习、无监督学习和半监督学习（附示例）
    - 分类 vs 回归 vs 聚类
  - 参数化算法 vs 非参数化算法
  - 线性算法 vs 非线性算法

### 监督学习
  - 线性算法
    - 线性回归
      - 最小二乘、残差、单变量回归 vs 多元回归
    - 逻辑回归
      - 代价函数（公式、代码）、sigmoid 函数、交叉熵
    - 支持向量机
    - 线性判别分析

  - 决策树
    - Logits
    - 叶节点
    - 训练算法
      - 停止条件
    - 推理
    - 剪枝

  - 集成方法
    - Bagging 和 Boosting 方法（附示例）
    - 随机森林
    - Boosting
      - Adaboost
      - GBM
      - XGBoost
  - 不同算法的对比
    - [TBD: LinkedIn lecture]

  - 优化
    - 梯度下降（概念、公式、代码）
    - 梯度下降的其他变体
      - SGD
      - Momentum
      - RMSprop
      - ADAM
  - 损失函数
    - Logistic Loss function
    - Cross Entropy（公式也要记住）
    - Hinge loss（SVM）

- 特征选择
  - 特征重要性
- 模型评估与选择
  - 评估指标
    - TP、FP、TN、FN
    - 混淆矩阵
    - 准确率、精确率、召回率/敏感性、特异性、F-score
      - 应该如何在这些指标之间做选择？（类别不平衡数据集）
      - precision vs TPR（为什么看 precision）
    - ROC 曲线（TPR vs FPR、阈值选择）
    - AUC（模型对比）
    - 上述指标在多分类（n 元）场景下的扩展
    - 特定算法对应的指标 [TBD]
  - 模型选择
    - 交叉验证
      - k 折交叉验证（k 取多少比较合适？）

### 无监督学习
  - 聚类
    - 基于质心的模型：k-means 聚类
    - 基于连通性的模型：层次聚类
    - 基于密度的模型：DBSCAN
  - 高斯混合模型
  - 潜在语义分析
  - 隐马尔可夫模型（HMM）
    - 马尔可夫过程
    - 转移概率和发射概率
    - Viterbi 算法 [Advanced]
  - 降维技术
    - 主成分分析（PCA）
    - 独立成分分析（ICA）
    - T-sne

### 偏差 / 方差（欠拟合/过拟合）
- 正则化技术
  - L1/L2（Lasso/Ridge）

### 采样
- 采样技术
  - 均匀采样
  - 水塘抽样
  - 分层采样

### 数据处理
 - 缺失数据
 - 不平衡数据
 - 数据分布漂移

### 机器学习算法的计算复杂度
- [TBD]

## 2. 深度学习
- 前馈神经网络
  - 深入理解其工作原理
  - [EX] 对于类别并非互斥的任务，应使用什么激活函数
- RNN
  - 时间反向传播（BPTT）
  - 梯度消失/梯度爆炸问题
- LSTM
  - 梯度消失/梯度爆炸问题
  - gradient?
- Dropout
  - 如何在 LSTM 中应用 dropout？
- Seq2seq 模型
- Attention
  - self-attention
- * Transformer 架构（要深入，绝不是开玩笑！）
  - [Illustrated transformer](http://jalammar.github.io/illustrated-transformer/)
- 嵌入向量（词嵌入）

## 3. 统计机器学习
### 贝叶斯算法
  - 朴素贝叶斯
  - 最大后验（MAP）估计
  - 最大似然（ML）估计

### 统计显著性
- R-squared
- P-values

## 4. 其他主题：
  - 离群点
  - 相似度/差异度量
    - 欧氏距离、曼哈顿距离、余弦、马氏距离（高级）

## 5. 基础模型与大语言模型（LLM）

> 2026 年机器学习面试中最大的变化：LLM / 基础模型现在已经成为**广度**要求，而不再是某种专项能力。下面列出的是面试官重点考察的核心概念。关于端到端 GenAI **系统设计**（RAG、智能体、服务部署），请参见[第 4 章](./MLSD/ml-system-design.md)以及[Agentic AI Systems repo](https://github.com/alirezadir/Agentic-AI-Systems.git)。

### Transformer 与 LLM 内部机制
- Self-attention 回顾：scaled dot-product attention、**为什么要除以 √dₖ**（避免 softmax 进入饱和区 → 梯度更稳定）、multi-head attention
- Attention 的变体（内存 / 吞吐权衡）：**MHA → MQA (multi-query) → GQA (grouped-query) → MLA (multi-head latent, DeepSeek)**
- 位置编码：绝对位置 / 可学习位置、**RoPE**（rotary）、ALiBi；长上下文扩展（position interpolation、YaRN）
- **KV cache**：缓存每层的 key/value，使每个新 token 的计算变为 O(n)，而不是每次都重算 O(n²)；在长上下文 × 大 batch 下，它往往是内存占用的主导因素
- **FlashAttention**（IO-aware、在 SRAM 中分块计算 attention）——为什么长上下文训练终于变得可行
- Block 内部结构：pre-norm vs post-norm、**RMSNorm**、**SwiGLU / GeGLU** 激活
- **Mixture-of-Experts (MoE)**：稀疏专家路由、负载均衡、活跃参数 vs 总参数（Mixtral、DeepSeek-V3）
- 分词：**BPE / byte-level BPE / SentencePiece**、词表大小、上下文窗口
- 缩放定律（Chinchilla compute-optimal）、涌现能力

### 训练流水线（pretraining → post-training）
1. **Pretraining** —— 在 web 规模语料上做自监督的 next-token prediction
2. **SFT / instruction tuning** —— 在（prompt → response）监督示例上训练
3. **Preference alignment (RLHF & alternatives)** —— 对齐人类偏好与安全性
4. **Reasoning RL** —— 面向链式推理模型的可验证奖励强化学习（o-series、DeepSeek-R1）

### 后训练：SFT 与 RL / 对齐算法（2026）
2026 年面试中的一个核心主题，是**如何根据你的数据和算力选择合适的后训练方法**。现代训练栈通常会把这些方法*分层叠加*（SFT → 偏好优化 → RL），而不是只使用某一种单体方法。

| 算法 | 所需数据 | 所需额外模型 | 相对成本 | 适用场景 |
|---|---|---|---|---|
| **SFT** | 精选的指令示例（prompt→response） | 无 | 低 | 学习输出格式和指令遵循能力（通常应首先进行） |
| **RLHF (PPO)** | 人类偏好标签 | 奖励模型 + critic/value 模型 + 参考模型 | 高 | 经典方法；目前多数场景已改用更简单的方法 |
| **DPO** | 偏好**样本对**（chosen 与 rejected） | 参考模型 | 中 | 默认的离线对齐方法；训练方式类似 SFT，无需 RL 循环 |
| **SimPO** | 偏好样本对 | 无（无需参考模型） | 中低 | 不需要参考模型的 DPO（以平均对数概率作为隐式奖励） |
| **KTO** | **二元**赞成/反对反馈（无样本对） | 无 | 低 | 适用于低成本、噪声较大的反馈，或只有非配对信号时 |
| **ORPO** | 仅需指令示例 | 无 | 低 | 将 SFT 与偏好调优合并为一个阶段 |
| **GRPO** | 仅需 prompt（采样一**组**回答并计算组内相对优势） | 无（不需要 critic） | 中 | 不依赖价值网络的 RL；适合推理/数学任务（DeepSeek） |
| **RLVR** | 具有**可验证**奖励的任务（单元测试、数学答案、合法 JSON） | 自动验证器 | 中 | 适用于结果可验证的代码、数学和工具调用任务 |

关键讨论点：DPO 将 reward model + RL loop 折叠为一个基于偏好对的监督损失；**GRPO 去掉了 critic**（基于组归一化奖励估计 advantage → 比 PPO 更省内存）；**GRPO / RLVR 是推理模型的重要驱动方法**；不同算法的优劣**依赖模型规模**（例如在线 RL 在约 1.5B 规模下可能更强，而 SimPO 在约 7B 规模下可能更好）。另外还要了解 DAPO（用于稳定长链式推理 RL）和 RLAIF（用 AI 反馈替代人工标注）。

### 参数高效微调（PEFT）
- 全量微调 vs PEFT 的权衡（算力、存储、灾难性遗忘）
- **LoRA / QLoRA**（低秩适配器；QLoRA 在量化后的基座模型上微调）、adapters、prefix / prompt tuning
- 关键超参数：LoRA rank / α、learning rate、epochs、batch size

### 推理与服务优化
- **量化**：PTQ vs QAT；INT8 / INT4（GPTQ、AWQ）、**FP8**、**KV-cache quantization**（长上下文下，KV 可能比权重更占内存）
- **Paged attention / continuous batching**（vLLM）、prefix caching
- **Speculative decoding**（draft + verify）、蒸馏、用于稀疏计算的 **MoE**
- 并行化：tensor / pipeline / sequence；prefill vs decode 阶段

### 解码与上下文学习
- 解码：greedy、beam、**temperature、top-k、top-p (nucleus)**、repetition penalty；结构化 / 受约束解码（JSON / grammar）
- 上下文学习：zero / few-shot、**chain-of-thought**、self-consistency；**test-time compute / inference-time scaling**（推理模型）
- 增加知识：**RAG vs long-context vs fine-tuning**（权衡取舍）

### 生成模型评估（2026：*"eval is the new system design"*）
- 为什么经典指标不适用于开放式生成；**hallucination** 与校准
- **RAG triad**（RAGAS）：faithfulness、answer relevance、context relevance；检索指标（recall@k、MRR、nDCG）
- **LLM-as-judge**、pairwise win-rate / Arena（Elo）、golden sets 与回归测试
- 智能体指标：工具选择质量、任务 / 步骤成功率、轨迹遵循度
- 基准测试（MMLU、GPQA、SWE-bench 等）；安全性 / red-teaming、jailbreak 鲁棒性

## 6. 多模态与生成式 AI

### 多模态基础模型（FMs）
- 核心思想：在多种模态（文本、图像、音频、视频、动作）之间学习共享表征
- **融合方式**：对比式双编码器（**CLIP / SigLIP**）；投影 / adapter 到 LLM token 空间（**LLaVA**）；cross-attention（Flamingo）；早期融合 vs 后期融合；原生 / “omni” any-to-any（GPT-4o、Gemini）
- 统一做理解**和**生成；用于自回归生成的图像 / 音频分词器（VQ-VAE）

### 视觉语言模型（VLMs）
- 架构：**vision encoder (ViT / SigLIP / DINOv2) → projector → LLM**
- 任务：VQA、图像描述、**OCR / 文档理解**、grounding / detection、图表 / UI 理解
- 示例：GPT-4o、Gemini、Claude、Qwen-VL、**LLaVA**、**PaliGemma**、InternVL
- 训练：图文预训练 + 视觉指令微调

### 视觉-语言-动作模型（VLA）
- 将 VLM 扩展到**具身 / 机器人控制**——感知 → 理解指令 → 输出动作，通常可在一次前向计算中完成
- 动作表示：**离散动作 token**（RT-2、OpenVLA）vs **通过 diffusion / flow-matching action head 表示连续动作**（π0）
- 示例：
  - **RT-2**（Google DeepMind）——基于 PaLI-X / PaLM-E VLM 构建；将网页知识和 chain-of-thought 迁移到机器人控制
  - **OpenVLA** —— 7B、开源；DINOv2 + SigLIP 视觉 + Llama-2；基于 97 万条真实演示训练；参数量约少 7× 的情况下超过 RT-2-X（55B）
  - **π0 (Pi-Zero)** —— PaliGemma VLM + **flow-matching** 动作专家；约 50 Hz 高频灵巧控制
- 应用场景：机器人操作、类人机器人、通用机器人策略；数据来自遥操作演示 + Open X-Embodiment

### Diffusion 与自回归生成对比
| | **自回归（AR）** | **扩散模型** |
|---|---|---|
| 生成方式 | 逐步预测下一个 token | 从噪声开始迭代去噪 |
| 似然 | 可精确计算 | 变分 / 基于 score |
| 优势 | 离散序列、变长、推理能力 | 连续高维模态（图像 / 视频 / 音频）、高保真 |
| 速度 | 每个 token 需 1 次前向计算（KV cache 可加速） | 需要多步去噪（可通过 distillation / consistency / **flow matching** 减少） |
| 示例 | 文本（GPT）、图像 token（Parti、VAR）、音频（AudioLM） | 图像（Stable Diffusion、Imagen）、**视频（Sora、Veo — DiT）**、音频 / 音乐（Stable Audio）、机器人动作（π0） |

- **Diffusion 深入**：前向 / 反向过程（DDPM）、score / noise prediction、**latent diffusion**（Stable Diffusion）、**DiT**（diffusion transformers，用于视频）、**classifier-free guidance**、快速采样器（DDIM）、**flow matching / rectified flow**（SD3、π0）
- **AR 深入**：先将某个模态离散化为 token，再做 next-token prediction；这使统一的 token-based 多模态模型成为可能；图像 AR（next-scale prediction / VAR）、音频（AudioLM / MusicGen）
- 新趋势：**text diffusion / masked diffusion LMs**；consistency 与 few-step 模型；统一的 AR + diffusion 栈

# 3. 机器学习基础示例问题
- 什么是机器学习？它与传统编程有什么区别？
- 机器学习技术有哪些不同类型？
- 监督学习和无监督学习有什么区别？
- 什么是半监督学习？
- 构建机器学习模型的各个阶段是什么？
- 你能解释一下机器学习中的偏差-方差权衡吗？
- 什么是过拟合？如何防止过拟合？
- 为什么要把数据划分为训练集、测试集和验证集？具体应如何划分？
- 什么是交叉验证？为什么它很重要？
- 你能解释一下正则化的概念及其类型（L1、L2 等）吗？
- 你会如何处理数据集中的缺失或损坏数据？
- 什么是决策树？它是如何工作的？
- 你能解释一下逻辑回归吗？
- 你能解释一下 K 最近邻（KNN）算法吗？
- 比较 K-means 和 KNN 算法。
- 解释基于决策树的算法（随机森林、GBDT）
- 什么是梯度下降？它是如何工作的？
- 你能解释一下支持向量机（SVM）算法吗？什么是 Kernel SVM？
- 你能解释一下神经网络及其工作原理吗？
- 什么是深度学习？它与传统机器学习有什么不同？
- 你能解释一下反向传播算法及其在训练神经网络中的作用吗？
- 什么是卷积神经网络（CNN）？它是如何工作的？
- 什么是迁移学习？它在实践中是如何使用的？
* [45 ML interview questions](https://www.simplilearn.com/tutorials/machine-learning-tutorial/machine-learning-interview-questions)

### LLM / GenAI / 多模态示例问题（2026）
- 请完整讲一下一个 transformer block。为什么 attention score 要除以 √dₖ？
- 什么是 KV cache？它为什么对推理很重要？它的内存规模如何增长？
- 比较 MHA、MQA 和 GQA。为什么 Llama 采用了 GQA？
- 什么是 RoPE？为什么旋转位置嵌入比可学习位置嵌入更常用？
- 什么情况下你会选择 RAG、微调或 long-context？
- 比较 SFT、DPO、GRPO 和 RLVR：各自需要什么数据、需要哪些额外模型、分别适用于什么场景。
- 为什么 GRPO 去掉了 critic 网络？它如何估计 advantage？
- 什么是 LoRA / QLoRA？什么情况下你会选择 PEFT 而不是全量微调？
- 解释一下量化（INT8/INT4、FP8）和 KV-cache quantization。它们在精度/延迟上的权衡是什么？
- Speculative decoding 是如何加速生成的？
- 你会如何评估一个 LLM / RAG 系统？什么是 RAG triad 和 LLM-as-judge？
- VLM 是如何把 vision encoder 与 LLM 结合起来的？（例如 LLaVA / PaliGemma）
- 什么是 VLA 模型？对比离散动作 token（RT-2、OpenVLA）与 flow-matching action head（π0）。
- Diffusion 与自回归生成有什么区别？它们分别适用于什么场景（图像/视频/音频/文本）？
- 什么是 classifier-free guidance？什么是 flow matching / rectified flow？
- 什么是 Mixture-of-Experts 模型？请区分活跃参数和总参数。
