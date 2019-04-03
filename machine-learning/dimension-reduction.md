# 数据降维

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [数据降维](#数据降维)
	- [数据降维的动机](#数据降维的动机)
		- [数据降维](#数据降维)
		- [数据可视化](#数据可视化)
	- [PCA 主成分分析问题](#pca-主成分分析问题)
		- [从压缩数据中恢复](#从压缩数据中恢复)
		- [选择主成分的数量](#选择主成分的数量)
		- [PCA应用建议](#pca应用建议)

<!-- /TOC -->

## 数据降维的动机

### 数据降维
数据降维主要有两点好处：
1. 数据压缩，因而使用更少的内存或存储空间
2. 加速学习算法

一个简单的例子如下，把二维（ _x<sub>1</sub>_,  _x<sub>2</sub>_）映射到图中直线上，因而可以用一维数据来表示：

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/8274f0c29314742e9b4f15071ea7624a.png" />
</p>

稍微复杂点的例子，把三位数据映射到一个平面上，因而可以用二维坐标来表示：

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/67e2a9d760300d33ac5e12ad2bd5523c.jpg" />
</p>

类似的处理过程可以用来把任何维度 (_m_) 的数据降到任何想要的维度 (_n_)，例如将1000维的特征降至100维。

### 数据可视化
如果我们能将数据可视化，降维可以帮助我们：

例如有许多国家的数据，每一个特征向量都有50个特征（如GDP，人均GDP，平均寿命等）。如果要将50维数据可视化是不现实的。
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/789d90327121d3391735087b9276db2a.png" />
</p>

而使用降维的方法将其降至2维，我们便可以将其可视化了。
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/ec85b79482c868eddc06ba075465fbcf.png" />
</p>

## PCA 主成分分析问题
上面介绍了降维，那如何降维是合理的了？

PCA是其中一个很常见的方法。

原理是当把所有的数据都投射到新的方向向量上时，希望投射平均均方误差(MSE) 尽可能小。
方向向量是一个经过原点的向量，而投射误差是从特征向量向该方向向量作垂线的长度。如下图中蓝色线段所示：

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/a93213474b35ce393320428996aeecd9.jpg" />
</p>

**问题描述：**
将 _n_ 维数据降至 _k_ 维，目标是找到向量 _u<sup>(1)</sup>_ , _u<sup>(2)</sup>_ ,..., _u<sup>(k)</sup>_ 以最小化总体投射误差(MSE)。

对于上图的例子，看起来是不是很像线性回归？
但PCA和线性回归是不同的算法。PCA最小化的是投射误差（Projected Error），而线性回归最小化的是预测误差。线性回归的目的是预测结果，而主成分分析不作任何预测。下左图是线性回归的误差（垂直于横轴投影），下右图是PCA的误差（垂直于红线投影）：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/7e1389918ab9358d1432d20ed20f8142.png" />
</p>

PCA将 _n_ 个特征降维到 _k_ 个，可以用来数据压缩，如果100维的向量最后用10维来表示，那么压缩率为90%。同样图像处理领域的KL变换使用PCA做图像压缩。但PCA要保证降维后，还要保证数据的特性损失最小。

PCA的一大好处是**对数据进行降维**。可以对新求出的“主元”向量的重要性进行排序，根据需要取前面最重要的部分，将后面的维数省去，可以达到降维从而简化模型或是对数据进行压缩的效果。同时最大程度的保持了原有数据的信息。

此外，**PCA是完全无参数限制的**。在PCA的计算过程中不需要人为设定参数或是根据任何经验模型对计算进行干预，最后的结果只与数据相关，与用户是独立的。
> 但，这点同时也是缺点。如果用户对观测对象有一定的先验知识，例如掌握了数据的一些特征，却无法通过参数化等方法对处理过程进行干预，可能无法得到预期的效果。

PCA减少 _n_ 维到 _k_ 维：

1. 均值归一化（Mean Normalization）。
计算所有特征的均值 _μ<sub>j</sub>_，令 _x<sub>j</sub>=x<sub>j</sub> - μ<sub>j</sub>_ 。
如果特征是在不同的数量级上，我们还需要将其除以标准差 _σ<sup>2</sup>_ 。
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_j^{(i)}&space;=&space;\frac{x_j^{(i)}&space;-&space;\mu_j^{(i)}}{s_j}" title="x_j^{(i)} = \frac{x_j^{(i)} - \mu_j^{(i)}}{s_j}" />
</p>

2. 计算协方差矩阵（covariance matrix） _Σ_ ：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\sum=\dfrac{1}{m}\sum^{n}_{i=1}\left(x^{(i)}\right)\left(x^{(i)}\right)^T" title="\sum=\dfrac{1}{m}\sum^{n}_{i=1}\left(x^{(i)}\right)\left(x^{(i)}\right)^T" />
</p>

3. 计算协方差矩阵 _Σ_ 的特征向量（eigenvectors）:

在`Python`里我们可以利用 **奇异值分解（singular value decomposition）** 来求解:

``` python
import numpy as np
a = np.diag((1, 2, 3))
U, S, vh = np.linalg.svd(a) # ((3, 3), (3,), (3, 3))
```

其中 _U_ 是特征向量、 _S_ 是特征值。其实 _S_ 是按照特征值从大到小排序的，U的每一列 _u<sub>j</sub>_ 与对应位置的 _s<sub>j</sub>_ 对应的特征向量。其中 _U<sup>T</sup>U = I_。

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/0918b38594709705723ed34bb74928ba.png" />
</p>

所以，如果要把数据从 _n_ 维映射到 _k_ 维，只需要取特征向量 _U_ 的前 _k_ 维度列向量，构成映射矩阵 _U<sub>reduce</sub> = U[:, k]_。

_z = U<sup>T</sup><sub>reduce</sub> * x_ 即为映射后的数据，其中 _x_ 为原始数据。

### 从压缩数据中恢复
给定 _z<sup>(i)</sup>_，可能是100维，怎么得到到原来的表示 _x<sup>(i)</sup>_ ，也许本来是1000维的数组。

在压缩过数据后，可以采用如下方法近似地获得原有的特征：
<p align="center">
<i>x<sub>approx</sub>=U<sub>reduce</sub> z</i>
</p>

因为从 _x_ 得到 _z_ 的过程可以看做是 _x_ 在空间 _U<sup>T</sup><sub>reduce</sub>_ 上的映射。
而从 _z_ 得到 _x<sub>approx</sub>_ 的过程可以看做反向的映射，也就是在空间 _(U<sup>T</sup><sub>reduce</sub>)<sup>-1</sup> = U<sub>reduce</sub>_ （A的逆矩阵）上的映射。

下图中为一个恢复的例子：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/66544d8fa1c1639d80948006f7f4a8ff.png" />
</p>

**关于PCA更多的推导和证明：**请见[这里](../../math/pca.md)

### 选择主成分的数量

主要成分分析是减少投射的平均均方误差 MSE。
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\dfrac{1}{m}\sum^{m}_{i=1}\left|x^{\left(i\right)}&space;-&space;x^{\left(i\right)}_{approx}\right|^2" title="\dfrac{1}{m}\sum^{m}_{i=1}\left|x^{\left(i\right)} - x^{\left(i\right)}_{approx}\right|^2" />
</p>

训练集的方差（Variance）为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\dfrac{1}{m}\sum^{m}_{i=1}\left|x^{\left(i\right)}\right|^2" title="\dfrac{1}{m}\sum^{m}_{i=1}\left|x^{\left(i\right)}\right|^2" />
</p>

通常是选择 _k_ 值，使 MSE 与 Variance 的比例尽可能小的情况下选择尽可能小：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\text{arg}\underset{k}{\min}&space;=&space;\dfrac{\dfrac{1}{m}\sum^{m}_{i=1}\left|x^{\left(i\right)}-x^{\left(i\right)}{approx}\right|^{2}}{\dfrac{1}{m}\sum^{m}_{i=1}\left|x^{(i)}\right|^2}" title="\text{arg}\underset{k}{\min} = \dfrac{\dfrac{1}{m}\sum^{m}_{i=1}\left|x^{\left(i\right)}-x^{\left(i\right)}{approx}\right|^{2}}{\dfrac{1}{m}\sum^{m}_{i=1}\left|x^{(i)}\right|^2}" />
</p>

这个阈值（threshold）通常取值 0.01 （1%）。

如果希望比例小于1%，意味着原本数据的偏差有99%都保留下来了，如果选择保留95%的偏差，便能非常显著地降低模型中特征的维度了。

可以先令 _k=1_ ，然后执行PCA，获得 _U<sub>reduce</sub>_ 和 _z_ ，然后计算比例是否小于1%。如果不是的话再令 _k=2_ ，如此类推，直到找到可以使得比例小于1%的最小 _k_ 值。

还有一些更好的方式来选择 _k_ ，当使用 `numpy.linalg.svd()` 函数时，将获得三个参数：
```U,S,V = numpy.linalg.svd(sigma)```。

其中的 _S_ 是一个 _n×n_ 的矩阵，只有对角线上有值，而其它单元都是0（如下图）。
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/a4477d787f876ae4e72cb416a2cb0b8a.jpg" />
</p>

可以用这个矩阵来计算平均均方误差与训练集方差的比例：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\text{arg}\underset{k}{\min}&space;=&space;1&space;-&space;\dfrac{\Sigma^{k}_{i=1}S_{ii}}{\Sigma^{m}_{i=1}S_{ii}}\leq0.01" title="\text{arg}\underset{k}{\min} = 1 -\dfrac{\Sigma^{k}_{i=1}S_{ii}}{\Sigma^{m}_{i=1}S_{ii}}\leq0.01" />
</p>

即：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\frac{\Sigma^{k}_{i=1}s_{ii}}{\Sigma^{n}_{i=1}s_{ii}}\geq0.99" title="\frac{\Sigma^{k}_{i=1}s{ii}}{\Sigma^{n}_{i=1}s{ii}}\geq0.99" />
</p>

通过`svd()`得到的 _s<sub>ii</sub>_ 来计算上面的MSE与Variance比例很很方便的。


### PCA应用建议
假使正在针对一张100×100像素的图片做CV的机器学习，总共10000个特征。

* 第一步是运用主要成分分析将数据压缩至1000个特征
* 然后对训练集运行学习算法
* 在预测时，采用之前学习而来的 _U<sub>reduce</sub>_ 将输入的特征 _x_ 转换成特征向量 _z_ ，然后再进行预测

注：如果我们有交叉验证集合测试集，也采用对训练集学习而来的 _U<sub>reduce</sub>_ 。

错误的PCA用法：

* 将其用于减少过拟合（减少了特征的数量）。
  非常不好，不如尝试正则化处理。
  原因在于PCA只是近似地丢弃掉一些特征，它并不考虑任何与结果变量有关的信息，因此可能会丢失非常重要的特征。
  然而当我们进行正则化处理时，会考虑到结果变量，不会丢掉重要的数据。
* 默认地将PCA作为学习过程中的一部分，虽然PCA很多时候有效果，最好是从所有原始特征开始，只在有必要的时候（算法运行太慢或者用太多内存）才考虑采用PCA。

## Jupyter Notebook编程练习

- 推荐访问Google Drive的共享，直接在Google Colab在线运行ipynb文件：
  - [Google Drive: 7.kmeans_and_PCA](https://drive.google.com/drive/folders/1VNdwdcxeRGViyg9lsz8TyOVq39VhjiYg?usp=sharing)
- 不能翻墙的朋友，可以访问GitHub下载：
  - [GitHub: 7.kmeans_and_PCA](https://github.com/loveunk/ml-ipynb/tree/master/7.kmeans_and_PCA)


[回到顶部](#数据降维)
