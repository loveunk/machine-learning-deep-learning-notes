# Scikit-learn
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Scikit-learn](#scikit-learn)
	- [学习类问题的分类](#学习类问题的分类)
	- [开始使用 Scikit-learn](#开始使用-scikit-learn)
		- [加载示例数据集](#加载示例数据集)
		- [学习和预测](#学习和预测)
			- [构建估计器](#构建估计器)
			- [选择模型参数](#选择模型参数)
			- [训练模型](#训练模型)
			- [预测未知数据](#预测未知数据)
		- [保存模型](#保存模型)
		- [约定](#约定)
			- [类型转换](#类型转换)
			- [再次训练和更新参数](#再次训练和更新参数)
			- [多分类与多标签拟合](#多分类与多标签拟合)
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
print(digits.data.shape)    # => (1797, 64)
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
clf.predict(digits.data[-1:]) # => array([8])
```

打印最后一张出来看看
``` python
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
plt.show()
```
<p align="center">
<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_digits_last_image_001.png" />
</p>

### 保存模型
Python 的内置的持久化模块joblib将模型保存:
``` python
from joblib import dump, load
s = dumps(clf, "filename.joblib")   # 保持此前fit的模型
clf2 = load(s)                      # 加载之前存的模型
clf2.predict(X[0:1])                # 做预测
```

### 约定
#### 类型转换
除非特别指定，输入将被转换为 float64

#### 再次训练和更新参数
``` python
import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = SVC()
clf.set_params(kernel='linear').fit(X, y) # 默认内核 rbf 被改为 linear
clf.predict(X_test)

clf.set_params(kernel='rbf', gamma='scale').fit(X, y) # 改回到 rbf 重新训练
clf.predict(X_test)
```

#### 多分类与多标签拟合
``` python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)

classif = OneVsRestClassifier(estimator=SVC(gamma='scale',
                                            random_state=0))
print(classif.fit(X, y).predict(X))
```
上述将输出
```
[[1 1 0 0 0]
 [1 0 1 0 0]
 [0 1 0 1 0]
 [1 0 1 0 0]
 [1 0 1 0 0]]
```

## 更多推荐阅读
* [Scikit-learn速查表](Scikit_Learn_Cheat_Sheet_Python.pdf)
* [Scikit-learn官方文档（英文）](https://scikit-learn.org/stable/documentation.html)
* [Scikit-learn中文文档](https://www.kancloud.cn/luponu/sklearn-doc-zh/889724)
* [Scikit-learn与TensorFlow机器学习实战](https://hand2st.apachecn.org/)

[回到目录](#scikit-learn)
