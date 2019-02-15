# Scikit-learn
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Scikit-learn](#scikit-learn)
	- [开始使用 Scikit-learn](#开始使用-scikit-learn)
		- [学习类问题的分类](#学习类问题的分类)
	- [更多推荐阅读](#更多推荐阅读)

<!-- /TOC -->

**Scikit-learn**（简称`sklearn`）是开源的 _Python_ 机器学习库，它基于`Numpy`和`Scipy`，包含大量数据挖掘和分析的工具，例如数据预处理、交叉验证、算法与可视化算法等。

从功能上来讲，Sklearn基本功能被分为分类，回归，聚类，数据降维，模型选择，数据预处理。

从机器学习任务的步骤来讲，Sklearn可以独立完成机器学习的六个步骤：
* **选择数据**：将数据分成三组，分别是训练数据、验证数据和测试数据。
* **模拟数据**：使用训练数据来构建使用相关特征的模型。
* **验证模型**：使用验证数据接入模型。
* **测试模型**：使用测试数据检查被验证的模型的表现。
* **使用模型**：使用完全训练好的模型在新数据上做预测。
* **调优模型**：使用更多数据、不同的特征或调整过的参数来提升算法的性能表现。

## 学习类问题的分类
* **监督学习**：训练数据中有我们想要预测的属性，也就是说对每一组 _输入_ 数据，都有对应的 _输出_。问题可以分为两类：
	* **分类**：数据属于有限多个类别，希望从已标记数据中学习如何预测未标记数据的类别。
		* 例子：手写数字的识别（0-9共10个类别）。
	* **回归**：每组数据对应的输出是一个或多个连续变量。
		* 例子：是根据鲑鱼长度作为其年龄和体重。
* **无监督学习**：训练数据无对应的输出值。
	* 例子：数据聚类、降维。

## 开始使用 Scikit-learn
用于导入`Scikit-learn`库的名称是`sklearn`：
``` python
import sklearn
```

### 加载示例数据集
这里我们通过手写数字的识别作为例子，先加载数据：
``` python
from sklearn import datasets
digits = datasets.load_digits()
```

看看数据的大小：共1797行，每个数字图片是8*8的，所以有64列：
``` python
print(digits.data.shape)		# => (1797, 64)
print(digits.target.shape)	# => (1797,)
```

### 学习和预测
**问题描述**：对输入的图像，预测其表示的数字。 <br/>
**解决方案**：输入训练集合，训练集合包括 10 个可能类别（数字 0 到 9）的样本，在这些类别上拟合一个 _估计器_ (`estimator`)，预测未知样本所属的类别。

#### 构建估计器
选择不同的估计器，就好比选择了不同的解决方案。估计器的一个例子是`sklearn.svm.SVC()`，它实现了支持向量分类。例子如下：
``` python
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
```

#### 选择模型参数
在上面的代码里，我们手动给定了模型参数，实际上可以使用 **网络搜索**和**交叉验证** 等工具来寻找比较好的值。

#### 训练模型

``` python
clf.fit(digits.data[:-1], digits.target[:-1])
```

#### 预测未知数据
``` python
clf.predict(digits.data[-1:])
```


## 更多推荐阅读
* [Scikit-learn速查表](Scikit_Learn_Cheat_Sheet_Python.pdf)
* [Scikit-learn文档](https://scikit-learn.org/stable/documentation.html)

[回到目录](#scikit-learn)
