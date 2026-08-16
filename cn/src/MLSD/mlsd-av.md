# 自动驾驶汽车
- 在几乎没有人工干预，甚至完全没有人工干预的情况下自行驾驶
- 存在不同级别的自动驾驶能力

## 硬件支持

### 传感器

* Camera
  * 用于分类、分割和定位。
  * 在夜间以及雾、暴雨等极端条件下效果较差。
* LiDAR (Light Detection And Ranging,)
  * 使用激光或光来测量附近物体的距离。
  * 增加深度信息（3D感知），输出点云。
  * 在夜间或黑暗环境中也能工作，但在雨雾噪声较大时仍会失效。
* RADAR (Radio detection and ranging)
  * 使用无线电波（而不是激光），因此几乎可在各种条件下工作。
  * 通过反射感知距离。
  * 噪声很大（需要清理，如阈值处理、FFT），空间分辨率较低，还会与其他无线电系统产生干扰。
  * 点云
* Audio

## 技术栈

![stack](https://miro.medium.com/v2/resize%3Afit%3A720/0%2AV3LATCuich2XSNKz)

* **感知**

- Perception 
objects, 
原始传感器（lidar、camera等）数据（图像、点云）->  世界理解
  * 目标检测（交通灯、行人、路标、人行道、停车位、车道线等）、交通灯状态检测等
  * 定位
    * 在车辆行驶过程中计算其位置和姿态（Visual Odometry，VO）。
    * 深度学习用于提升VO性能，以及进行目标分类。
      * 示例：PoseNet 和 VLocNet++，使用点数据估计3D位置和姿态。
  * ....
* **行为预测**
  * 预测智能体未来轨迹
* **规划**：决策并生成轨迹
* **控制器**：生成控制指令：加速、刹车、向左转向或向右转向

* 注：延迟要求方面，某些任务需要毫秒级，另一些任务需要几十毫秒级

## 感知

* 2D目标检测：
  * 两阶段检测器：使用 Region Proposal Network (RPN) 学习潜在目标的 RoI，再进行边界框预测（使用 RoI pooling）：（R-CNN、Fast R-CNN、Faster R-CNN、Mask-RCNN，后者还做分割）
    * 在 focal loss 出现之前通常表现更好
  * 单阶段：跳过 proposal 生成；直接输出目标 BB：YOLO、SSD、RetinaNet
    * 计算上更有吸引力（适合实时）
  * 基于 Transformer：
    * Detection Transformer（[DETR](https://github.com/facebookresearch/detr)）：End-to-End Object Detection with Transformers
      * 使用 transformer encoder-decoder 架构，backbone CNN 作为编码器，基于 transformer 的模块作为解码器。
      * 输入图像 -> CNN -> feature map -> decoder -> 最终的 object queries、对应的类别标签和边界框。
      * 不依赖固定的一组候选框，因此可以处理图像中数量可变的目标。
      * [更多](https://towardsdatascience.com/detr-end-to-end-object-detection-with-transformers-and-implementation-of-python-8f195015c94d)
    * TrackFormer: Multi-Object Tracking with Transformers
      * 构建在 DETR 之上
  * NMS：

* 3D目标检测：
  * 基于点云数据，很多思路从2D检测迁移而来
  * 示例：
    * 在体素化点云上做3D卷积
    * 在 BEV 上做2D卷积
  * 计算开销大

* 目标跟踪：
  * 使用 EKF 等概率方法
  * 使用基于机器学习的模型
    * 使用/微调预训练 CNN 做特征提取 -> 再通过相关或回归方式进行跟踪。
    * 使用基于深度学习的跟踪算法，如 SORT（Simple Online and Realtime Tracking）或 DeepSORT

* 语义分割
  * 对图像做像素级分类（每个像素分配一个类别）
* 实例分割
  * 结合目标检测 + 语义分割 -> 对每个目标实例的像素进行分类

## 行为预测

  * 主要任务：运动预测 / 轨迹预测（未来）
  * 给定多个历史帧，预测每个目标未来会出现在哪里
  * 示例：
    * 使用 RNN/LSTM 做预测

* 输入来自感知系统 + HDMap
* 方案：
  * 俯视图表示：input -> CNN -> ..
  * 向量化表示：context map
  * 图表示：GNN

* 在单张 RGB 图像上渲染 bird eye view 图像
  * 一种处理历史信息的方式：也渲染到单张图像中
  * 另一种方式：对每一帧使用特征提取器（CNN），再用 LSTM 获取时序信息
  * 输入：BEV image + (v, a, a_v)
  * 输出：(x, y, std)
  ![Alt text](https://miro.medium.com/v2/resize%3Afit%3A1400/format%3Awebp/1%2AWiAw3Rl5kP0cuX2A4Vg0uw.png)
* 也可以使用 LSTM 网络按顺序生成轨迹中的 waypoints。

* 挑战：多模态性（不同模式的分布）——未来具有不确定性

  <!-- - 生成式方法：从噪声出发并施加条件
  - 基于锚点：加入人工先验（可能的目标位置）后进行选择
  - 多头方法：建模多模态分布
  - 基于意图的方法： -->

## 规划

- 决策并生成轨迹
- 输入：route（从 A 到 B）、context map、附近智能体的预测结果

- proposal：规划有哪些可能选项（数学方法 vs 模仿学习）——预测哪个最优

* 可以使用分层强化学习
  * 高层规划器：让行、停车、左转/右转、车道跟随等
  * 低层规划器：执行指令

- motion validation：检查例如碰撞、红灯等 -> 拒绝 + 排序

## 多任务方法

* ### 感知 + 行为预测
  * Fast& Furious（Uber）：
    * 任务：检测、跟踪、短时（例如1秒）运动预测
    * 从点云数据创建 BEV：
      * 量化 3D → 3D voxel grid（占用为二值）→ 将高度>channel（第3维）映射到 RGB，时间作为第4维 → 类似 SSD 的单阶段检测器
    * 以两种方式处理时间维度：
      * early fusion（在最开始的第一层聚合时序信息）
      * late fusion（逐步融合时序信息：允许模型捕获高层运动特征。）
    * 为 feature map 的每个位置使用多个预定义框（类似 SSD）
    * feature map 后接两个分支：
      * 二分类（每个预分配框对应的 P（是否为车辆））
      * 预测（回归）当前帧以及未来 n − 1 帧的 BB → 尺寸和朝向
      ![](https://miro.medium.com/v2/resize:fit:720/0*0jeh53rYLHXJrzwO)
  * IntentNet：从原始传感器数据学习预测意图（Uber）
    * 融合由点云生成的 BEV 和 HDMap 信息，用于检测、意图预测和轨迹预测。
    * I：BEV 中体素化的 LiDAR，栅格化的 HDMap
    * O：检测到的目标、轨迹、8类意图（保持车道、左转等）
  ![]()
      ![stack](https://miro.medium.com/v2/resize:fit:720/0*u8FqnNniHgRr7bF2)

* ### 行为预测 + 规划（Mid-to-Mid Model）

  * ChauffeurNet（Waymo）
    * 使用单个神经网络，通过模仿学习（IL）同时做预测和规划
    * 更多信息见[这里](https://medium.com/aiguys/behavior-prediction-and-decision-making-in-self-driving-cars-using-deep-learning-784761ed34af)

* ### 端到端

  * Learning to drive in a day（wayve.ai）
    * 使用强化学习从零训练一个驾驶策略，在不到20分钟内学会沿车道行驶！
    * 不需要任何 HDMap，也不需要手写规则！
  * Learning to Drive Like a Human
    * 模仿学习 + 强化学习
    * 使用了一些辅助任务，如分割、深度估计和光流估计，以学习更好的场景表征，并用其训练策略。

---

# 示例
设计一个机器学习系统，用于检测行人是否将要闯红灯/横穿马路（jaywalking）。

### 1. 问题定义

- Jaywalking：行人在没有人行横道或路口的地方穿越道路。
- 目标：开发一个机器学习系统，能够在实时场景下准确预测行人是否会在短时间范围内（例如1秒）发生 jaywalking。

- 行人行为预测比车辆更难：未来行为还依赖于身体姿态、当前活动等其他因素。

* 机器学习目标
  * 二分类（预测行人在未来 T 秒内是否会发生 jaywalking。）

* 讨论数据来源及其可获取性。

### 2. 指标
#### 组件级指标
* 目标检测
  * 精确率
    * 基于 IOU 阈值计算
  * AP：在不同 IOU 阈值上的平均
  * mAP：C 个类别上的 AP 均值
* jaywalking 检测：
  * Precision、Recall、F1
#### 端到端指标
* 人工干预
* 仿真误差
  * 带有专家司机的历史日志（场景记录）
  * 输入到我们的系统中，并将系统决策与专家司机进行对比

### 3. 架构组件
* 视觉理解系统
  * Camera：目标检测（行人、可行驶区域？）+ 跟踪
  * [可选] Camera + 目标检测：活动识别
  * Radar：3D目标检测（跳过）
* 行为预测系统
   * 轨迹估计
      * 需要运动历史
   * 基于机器学习的方法（分类）
      * 输入：
         * 视觉：局部上下文：行人裁剪图像序列（最近 k 帧）+ 全局上下文（最近 k 帧的语义分割图像）
         * 非视觉：行人轨迹（以 BB 表示，最近 k 帧）+ context map + 上下文（位置、年龄段等）

### 4. 数据收集与准备

* 数据收集与标注：
  * 收集行人行为数据集，包含 jaywalking 和非 jaywalking 行为。这些数据可以来自公开视频，也可以通过我们自己录制视频获得。
  * 收集多样化的视频片段或图像序列，覆盖不同地点，包括城市和郊区区域，以及不同的行人行为、交通状况和光照条件。
  * 对数据进行标注，标出行人、其位置，以及是否发生 jaywalking。可以通过给行人画边界框并打标签来完成（初期由人工标注，最终可演进为自动标注系统）
  * 定向数据收集：
    * 在后续迭代中，我们检查司机因行人 jaywalking 而必须接管的案例，检查最后20帧上的表现，并让标注人员标这些数据，再加入数据集（需要看到这些样本）

* 标注：
  * 对视频的每一帧标注 BB + 行人姿态信息 + 活动标签（walking、standing、crossing、looking 等）+ 行人属性（年龄、性别、位置等），
  * 对每个视频标注天气情况和时间段。

* 数据预处理：
  * 将数据集划分为训练集、验证集和测试集。
  * 对图像做归一化和调整尺寸，以保持输入数据一致性。
  * 应用数据增强技术（例如旋转、翻转、亮度调整）来扩大数据集规模并提升模型泛化能力。
     * 也可以用 GAN 增强数据

* 数据增强

### 5. 特征工程

* 从视频中提取相关特征，例如行人的位置、速度和运动方向。
* 还可以使用计算机视觉技术提取其他特征，例如是否存在人行横道、交通灯或其他相关环境线索。

* 来自帧的特征：对每个 BB，使用 Faster R-CNN 目标检测器的 fc6 特征（4096T vector）
  * 假设：我们可以从内置的目标检测和跟踪系统中，查询检测到的行人在最近 T（例如5）帧中的裁剪图像
* 来自裁剪帧的特征：活动识别
* context map：交通标志、道路宽度等
* 行人历史（BB 信息序列）+ 当前信息（BB + 姿态信息（openPose）+ 活动 + 局部上下文）+ 全局上下文（context map）+ 上下文（位置、年龄段等）-> JW/NJW 分类器
  * 其他可融合特征：行人姿态、BB、语义分割图（相关目标的语义 mask）、道路几何、周围人群、与其他智能体的交互

### 6. 模型开发与离线评估

模型选择与架构：

假设已有内置目标检测器和跟踪器。否则，
  * 目标检测：使用预训练目标检测模型，如 Faster R-CNN、YOLO 或 SSD，识别并定位视频帧中的行人。
  * 目标跟踪：
    * 使用基于 EKF 的方法，或基于机器学习的方法（SORT 或 DeepSORT）
* 活动识别：
  * 3D CNN，或 CNN + RNN(GRU)（这里选择后者，以适配整体架构）

（目标检测和跟踪的输出可以转换为每个 actor 的栅格化图像 -> Base CNN）

* 编码器：
  * 视觉编码器：视觉内容（最近 k 帧）-> CNN 基础编码器 + 用于时序信息的 RNN（GRU）[另一个选择是 3D CNN]
    * CNN 基础编码器 -> 再接一个 RNN 用于活动识别
  * 非视觉编码器：时序内容使用 GRU

* 融合策略：
  * early fusion
  * late fusion
  * hierarchical fusion

* Jaywalking clf：设计一个自定义 clf 层，将检测到的行人分类为 jaywalking 或非 jaywalking。
  * 示例：RF，或 FC layer

* 可以通过消融实验来选择融合架构，以及视觉/非视觉编码器
另一个例子：
![sd](https://github.com/OSU-Haolin/Pedestrian_Crossing_Intention_Prediction/blob/main/model_architecture.png?raw=true)

模型训练与评估：
a. 使用标注数据集训练模型，
+ 目标检测的损失函数（MSE、BCE、IoU）
+ jaywalking 分类任务（BCE）。

b. 定期在验证集上评估模型，以监控性能并避免过拟合。如有必要，调整学习率、batch size 等超参数。

c. 模型收敛后，在测试集上评估其性能，使用精确率、召回率、F1 score 和 Intersection over Union（IoU）等相关指标。

目标检测的迁移学习（使用预训练模型中强大的特征检测器）
* 例如做微调时，可使用 500 段视频，每段 5-10 秒，30fps

### 7. 预测服务
* 路上的 SDV：将接收实时图像 -> ...

* 模型优化：通过模型剪枝、量化和 TensorRT 优化等技术，将模型优化到适合实时部署。

### 8. 在线测试与部署

部署：将训练好的模型部署到配备摄像头的边缘设备或服务器上，用于监控实时视频流（例如交通摄像头系统）并检测 jaywalking 事件。将该系统与现有交通基础设施集成，例如交通信号灯和监控系统。

### 9. 扩展、监控与更新

持续改进：定期用新数据更新并重新训练模型，以提升性能并适应不断变化的行人行为和环境条件。

* 其他点：
  * 遮挡检测
    * hallucinated agent
  * 当视觉信号不精确时
    * 光照条件差

<!-- 行人行为预测：行人比车辆更难预测，因为其未来行为取决于身体姿态
日志 -> 筛选 100 米内出现行人的场景 -> 感知：目标检测 -> 截取过去 5 秒的图像 -> 前期需要人工标注，后期使用自动标注器 -> 按时间排列的目标序列 -> 身体姿态 + 活动识别 + 历史信息 + 上下文 -> 神经网络 -> 输出是否会横穿马路

目标检测 [A] -> 活动 + 上下文地图 + 历史信息 [B] -> 分类
其他特征：年龄、位置等 -->
