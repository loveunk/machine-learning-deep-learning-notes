# 机器学习

## 绪论
### 机器学习的应用领域和案例
人工智能主要包括**感知智能**（比如图像识别、语言识别和手势识别等）和**认知智能**（主要是语言理解知识和推理）。它的核心是数据驱动来提升生产力、提升生产效率。

在现实生活中，机器学习技术主要体现在以下几个部分：
* 数据挖掘（Data Minning)：发现数据间的关系
* 计算机视觉（CV - Computer Vision）：让机器看懂世界
* 自然语言处理（NLP）：让机器读懂文字
* 语音识别（Speech Recognition）：让机器听懂
* 决策（Decision Making）：让机器做决定，比如无人驾驶中的汽车控制决策

### 学习类问题的分类
* **监督学习**：训练数据中有我们想要预测的属性，也就是说对每一组 _输入_ 数据，都有对应的 _输出_。问题可以分为两类：
  * **分类问题**：数据属于有限多个类别，希望从已标记数据中学习如何预测未标记数据的类别。
    * 例子：手写数字的识别（0-9共10个类别）。
    * 例子：血糖值预测，根据性别、年龄、血液各种参数（血小板、白蛋白等等）预测血糖值
  * **回归问题**：每组数据对应的输出是一个或多个连续变量。
    * 例子：是根据鲑鱼长度作为其年龄和体重。
    * 例子：有无糖料病预测，根据性别、年龄、血液各种参数预测有无糖尿病
* **无监督学习**：训练数据无对应的输出值。
  * 例子：数据聚类、降维。

### 定义
科学家们的定义：
* 机器学习是不显示编程地赋予计算机能力的研究领域。—— Arthur Samuel
* 机器学习研究的是从数据中产生模型（model）的算法。—— 周志华《西瓜书》

更通俗的理解：
* 根据已知的数据，学习一个数学函数（决策函数），使其可以对未知的数据做出响应（预测或判断）。

#### 专有名词
* 数据集（data set）：为机器学习准备一组记录集合
* 样本（sample）或示例（instance）：数据集中记录的关于一个事件或对象的记录
* 模型（model）
* 特征（feature）或属性（attribute）
* 样本空间（sample space）或属性空间（attribute space）：属性张成的空间
* 特征向量（feature vector）
* 维数（dimensionality）
* 训练数据（training data）
* 训练集（training set）
* 训练样本（training sample）
* 假设（hypothesis）
* 真相（ground-truth）
* 标记（label）
* 分类（classification）
* 回归（regression）
* 正类（positive class）
* 反类（negative class）
* 多分类（multi-class classification）
* 测试（testing）
* 测试样本（testing sample）
* 监督学习（supervised learning）：训练数据有标记信息
* 无监督学习（unsupervised learning）：训练数据无标记信息
* 泛化（generalization）：学得模型适用于新样本的能力称为泛化能力
* 分布（distribution）
* 独立同分布（independent and indentically distributed）：每个样本都是独立地从一个分布上个采样获得的
* 归纳（induction）：从具体事实归纳出一般规律
* 演绎（deduction）：从一般到特化

### 更多机器学习案例
#### 计算机视觉（CV - Computer Vision）
* 图像分类


```
原始图像 --> 机器学习模型 --> 类别
```

#### 自然语言处理（NLP）
#### 语音识别（Speech Recognition）
#### 决策（Decision Making）

###
