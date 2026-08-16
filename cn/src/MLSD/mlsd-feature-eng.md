# 特征预处理

## 文本预处理
归一化 -> 分词 -> token 转 ID
* 归一化
* 分词
  * 词级分词
  * 子词分词
  * 字符级分词
* token 转 ID
  * 查找表
  * 哈希


## 文本编码器：
文本 -> 向量（嵌入向量）
两种方法：
  - 统计方法
    - BoW：将文档转换为词频向量，忽略词序和语法
    - TF-IDF：衡量一个词（term）在某篇文档中相对于整个文档集合的重要性。它由两个部分相乘得到：

      - 词频（TF）：该部分衡量某个词在特定文档中出现的频率，计算方式是该词在文档中出现的次数（记为`term_count`）除以该文档中的总词数（记为`total_terms`）。TF 的公式为：

        TF(t, d) = \frac{\text{term_count}}{\text{total_terms}}

      - 逆文档频率（IDF）：该部分衡量某个词在整个文档集合中的稀有程度，计算方式是文档集合中的文档总数（记为`total_documents`）与包含该词的文档数（记为`document_frequency`）之比的对数。IDF 的公式为：

        IDF(t) = \log\left(\frac{\text{total_documents}}{\text{document_frequency}}\right)

      某个词“t”在文档“d”中的最终 TF-IDF 分数由 TF 和 IDF 相乘得到：
      TF-IDF(t,d)=TF(t,d)×IDF(t)

  - 机器学习编码器
    - Embedding（查找）层：一种可训练层，用于将词或 ID 等类别型输入转换为连续值向量，使网络能够在训练过程中学习这些输入的有效表示。
    - Word2Vec：基于浅层神经网络，主要包括两种方法：Continuous Bag of Words（CBOW）和 Skip-gram。

      - CBOW（Continuous Bag of Words）：

        在 CBOW 中，模型基于固定窗口内的上下文词（即目标词周围的词）来预测目标词。
        它通过对上下文词的嵌入向量取平均来生成目标词。
        CBOW 计算效率较高，适合较小的数据集。
      - Skip-gram：

        在 Skip-gram 中，模型给定目标词来预测上下文词（周围的词）。
        它能够学习目标词与上下文词之间的关系。
        Skip-gram 尤其擅长捕捉细粒度的语义关系，并且在大规模数据集上表现良好。

      CBOW 和 Skip-gram 都使用浅层神经网络来学习词嵌入。得到的词向量是稠密且连续的，因此适用于多种 NLP 任务，例如情感分析、语言建模和文本分类。

    - 基于 Transformer 的模型，例如 BERT：会考虑上下文，同一个词在不同上下文中会有不同的嵌入向量


## 视频预处理
帧级：
解码帧 -> 采样帧 -> 调整尺寸 -> 缩放、归一化、颜色校正
### 视频编码器：
  - 视频级
    - 处理整个视频以生成一个嵌入向量
    - 使用 3D 卷积或 Transformer
    - 成本更高，但能够捕捉时序理解
    - 示例：ViViT（Video Vision Transformer）
  - 帧级（从采样帧中提取并聚合帧嵌入向量）
    - 成本更低（训练和服务速度、算力）
    - 示例：ViT（Vision Transformer）
      - 它将图像划分为互不重叠的 patch，并通过自注意力机制进行处理，从而分析图像内容；这与最初为序列数据（如文本）设计的原始 Transformer 不同，后者依赖 1D 位置编码。
