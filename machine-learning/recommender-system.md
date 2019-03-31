# 推荐系统

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [推荐系统](#推荐系统)
	- [问题描述](#问题描述)
	- [基于内容的推荐系统](#基于内容的推荐系统)
		- [代价函数](#代价函数)
		- [协同过滤](#协同过滤)
		- [协同过滤算法](#协同过滤算法)
		- [向量化：低秩矩阵分解](#向量化低秩矩阵分解)
		- [均值归一化](#均值归一化)

<!-- /TOC -->

## 问题描述

为什么要讲推荐系统：

1. 推荐系统是机器学习中的一个重要的应用。
   * 在行业里，尤其是硅谷有很多公司试图建立好的推荐系统。如果你考虑像Amazon，Netflix或eBay等网站或系统试图推荐新产品给用户。如Amazon推荐新书，Netflix推荐新电影给你，等等。
   * 这些推荐系统，根据你过去买过什么书，或过去评价过什么电影来判断。对收入的增加非常有帮助。因此，对推荐系统性能的改善，将对这些企业的有实质性和直接的影响。
   * 在机器学习相关的学术界，推荐系统问题实际上受到很少的关注，或者说在学术界它占了很小的份额。但是，在许多有能力构建这些系统的科技企业中，推荐系统似乎占据很高的优先级。
2. 讨论机器学习中的一些大思想。对机器学习来说，特征是很重要的，所选择的特征，将对学习算法的性能有很大的影响。
   * 在机器学习中有一种大思想，它针对一些问题，可能并不是所有的问题，而是一些问题，有算法可以为你自动学习一套好的特征。
   * 不要试图手动设计。有一些设置，你可以有一个算法，仅仅学习其使用的特征，推荐系统就是类型设置的一个例子。还有很多其它的，通过推荐系统，你将领略一小部分特征学习的思想，至少，你将能够了解到这方面的一个例子，机器学习中的大思想也是这样。

例子：

假使一个电影供应商，有 5 部电影和 4 个用户，要求用户为电影打分。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/c2822f2c28b343d7e6ade5bd40f3a1fc.png" />
</p>

前三部电影是爱情片，后两部则是动作片。
可以看出**Alice**和**Bob**更倾向与爱情片， 而 **Carol** 和 **Dave** 似乎更倾向与动作片。
并且没有用户给所有的电影都打过分。
希望构建一个算法来预测每个人可能会给没看过的电影打多少分，并以此作为推荐的依据。

定义一些标记：
* _n<sub>u</sub>_ 代表用户的数量
* _n<sub>m</sub>_ 代表电影的数量
* _r(i,j)_ 如果用户j给电影 _i_ 评过分则 _r(i,j)=1_
* _y<sup>(i,j)</sup>_ 代表用户 _j_ 给电影 _i_ 的评分
* _m<sub>j</sub>_ 代表用户 _j_ 评过分的电影的总数

## 基于内容的推荐系统

在一个基于内容的推荐系统算法中，假设希望推荐的东西有一些数据，这些数据是这些东西的特征。

例子中，假设每部电影都有两个特征，如 _x<sub>1</sub>_ 代表电影的浪漫程度， _x<sub>2</sub>_ 代表电影的动作程度。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/747c1fd6bff694c6034da1911aa3314b.png" />
</p>

则每部电影都有一个特征向量，如 _x<sup>(1)</sup>_ 是第一部电影的特征向量为[0.9 0]。

下面要基于这些特征来构建一个推荐系统算法。
假设采用线性回归模型，可以针对每个用户训练一个线性回归模型，如 _θ<sup>(1)</sup>_ 是第一个用户的模型的参数。

则有：
* _θ<sup>(j)</sup>_ 用户 _j_ 的参数向量
* _x<sup>(i)</sup>_ 电影 _i_ 的特征向量

对于用户 _j_ 和电影 _i_ ，预测评分为： _(θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup>_

### 代价函数

针对用户 _j_，该线性回归模型的代价为预测误差的平方和，加上正则化项：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min_{\theta&space;(j)}\frac{1}{2}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2&plus;\frac{\lambda}{2}\left(\theta_{k}^{(j)}\right)^2" title="\min_{\theta (j)}\frac{1}{2}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2+\frac{\lambda}{2}\left(\theta_{k}^{(j)}\right)^2" />
</p>

其中 _i:r(i,j)_ 表示只计算那些用户 _j_ 评过分的电影。在一般的线性回归模型中，误差项和正则项应该都是乘以 _1/2m_ ，在这里将 _m_ 去掉。并且不对方差项 _θ<sub>0</sub>_ 进行正则化处理。

上面的代价函数只是针对一个用户的，为了学习所有用户，将所有用户的代价函数求和：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min_{\theta^{(1)},...,\theta^{(n_u)}}&space;\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2&plus;\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2" title="\min_{\theta^{(1)},...,\theta^{(n_u)}} \frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2" />
</p>

如果要用梯度下降法来求解最优解，计算代价函数的偏导数后得到梯度下降的更新公式为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\theta_k^{(j)}:=\theta_k^{(j)}-\alpha\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)}&space;\quad&space;(\text{for}&space;\,&space;k&space;=&space;0)" title="\theta_k^{(j)}:=\theta_k^{(j)}-\alpha\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)} \quad (\text{for} \, k = 0)" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\theta_k^{(j)}:=\theta_k^{(j)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)}&plus;\lambda\theta_k^{(j)}\right)&space;\quad&space;(\text{for}&space;\,&space;k\neq&space;0)" title="\theta_k^{(j)}:=\theta_k^{(j)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)}+\lambda\theta_k^{(j)}\right) \quad (\text{for} \, k\neq 0)" />
</p>

### 协同过滤

在之前的基于内容的推荐系统中，对于每一部电影，都掌握了可用的特征，使用这些特征训练出了每一个用户的参数。相反地，如果拥有用户的参数，可以学习得出电影的特征。
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\mathop{min}\limits_{x^{(1)},...,x^{(n_m)}}\frac{1}{2}\sum_{i=1}^{n_m}\sum_{j{r(i,j)=1}}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2&plus;\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2" title="\mathop{min}\limits_{x^{(1)},...,x^{(n_m)}}\frac{1}{2}\sum_{i=1}^{n_m}\sum_{j{r(i,j)=1}}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2" />
</p>

但是如果既没有用户的参数，也没有电影的特征，这两种方法都不可行了。协同过滤算法可以同时学习这两者。

优化目标便改为同时针对 _x_ 和 _θ_ 进行。
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J(x^{(1)},...x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})\\&space;=\frac{1}{2}\sum_{(i:j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2&plus;\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(j)})^2&plus;\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2" title="J(x^{(1)},...x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})\\ =\frac{1}{2}\sum_{(i:j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2" />
</p>

对代价函数求偏导数的结果如下：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_k^{(i)}:=x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\theta_k^{j}&plus;\lambda&space;x_k^{(i)}\right)" title="x_k^{(i)}:=x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\theta_k^{j}+\lambda x_k^{(i)}\right)" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\theta_k^{(i)}:=\theta_k^{(i)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}x_k^{(i)}&plus;\lambda&space;\theta_k^{(j)}\right)" title="\theta_k^{(i)}:=\theta_k^{(i)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}x_k^{(i)}+\lambda \theta_k^{(j)}\right)" />
</p>

注：在协同过滤从算法中，通常不使用方差项，如果需要的话，算法会自动学得。
协同过滤算法使用步骤如下：

1. 初始 _x<sup>(1)</sup>, x<sup>(1)</sup>, ..., x<sup>(nm)</sup>, θ<sup>(1)</sup>, θ<sup>(2)</sup>, ..., θ<sup>(n<sub>u</sub>)</sup>_ 为一些随机小值
2. 使用梯度下降算法最小化代价函数
3. 在训练完算法后，预测 _(θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup>_ 为用户 _j_ 给电影 _i_ 的评分

通过这个学习过程获得的特征矩阵包含了有关电影的重要数据，这些数据不总是人能读懂的，但是可以用这些数据作为给用户推荐电影的依据。

例如，如果一位用户正在观看电影 _x<sup>(i)</sup>_ ，算法可以寻找另一部电影 _x<sup>(j)</sup>_ ，依据两部电影的特征向量之间的距离 _‖x<sup>(i)</sup>-x<sup>(j)</sup>‖_ 的大小。

### 协同过滤算法

协同过滤优化目标：
* 给定 _x<sup>(1)</sup>, ..., x<sup>(n<sub>m</sub>)</sup>_ ，估计 _θ<sup>(1)</sup>, ..., θ<sup>(n<sub>u</sub>)</sup>_ ：

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min_{\theta^{(1)},...,\theta^{(n_u)}}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2&plus;\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2" title="\min_{\theta^{(1)},...,\theta^{(n_u)}}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2" />
</p>

给定 _θ<sup>(1)</sup>, ..., θ<sup>(n<sub>u</sub>)</sup>_ ，估计 _x<sup>(1)</sup>, ..., x<sup>(n<sub>m</sub>)</sup>_ ：

同时最小化 _x<sup>(1)</sup>, ..., x<sup>(n<sub>m</sub>)</sup>_ 和 _θ<sup>(1)</sup>, ..., θ<sup>(n<sub>u</sub>)</sup>_ ：

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})\\&space;=\frac{1}{2}\sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2&plus;\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2&plus;\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2" title="J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})\\ =\frac{1}{2}\sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min_{x^{(1)},...,x^{(n_m)}&space;\theta^{(1)},...,\theta^{(n_u)}}J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})" title="\min_{x^{(1)},...,x^{(n_m)} \theta^{(1)},...,\theta^{(n_u)}}J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})" />
</p>

### 向量化：低秩矩阵分解

有关该算法的向量化实现，以及说说有关该算法你可以做的其他事情。

举例子：
1. 当给出一件产品时，你能否找到与之相关的其它产品。
2. 一位用户最近看上一件产品，有没有其它相关的产品，你可以推荐给他。

我将要做的是：实现一种选择的方法，写出协同过滤算法的预测情况。

有关于五部电影的数据集，我将要做的是，将这些用户的电影评分，进行分组并存到一个矩阵中。

有五部电影，以及四位用户，那么这个矩阵 _Y_ 就是一个5行4列的矩阵，它将这些电影的用户评分数据都存在矩阵里：

| **Movie**            | **Alice (1)** | **Bob (2)** | **Carol (3)** | **Dave (4)** |
| -------------------- | ------------- | ----------- | ------------- | ------------ |
| Love at last         | 5             | 5           | 0             | 0            |
| Romance forever      | 5             | ?           | ?             | 0            |
| Cute puppies of love | ?             | 4           | 0             | ?            |
| Nonstop car chases   | 0             | 0           | 5             | 4            |
| Swords vs. karate    | 0             | 0           | 5             | ?            |

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/42a92e07b32b593bb826f8f6bc4d9eb3.png" />
</p>

推出评分：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/c905a6f02e201a4767d869b3791e8aeb.png" />
</p>

找到相关影片：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/0a8b49da1ab852f2996a02afcaca2322.png" />
</p>

现在已经对特征参数向量进行了学习，就会有一个很方便的方法来度量两部电影之间的相似性。
例如：电影 _i_ 有一个特征向量 _x<sup>(i)</sup>_ ，是否能找到一部不同的电影 _j_ ，保证两部电影的特征向量之间的距离 _x<sup>(i)</sup>_ 和 _x<sup>(j)</sup>_ 很小，那就能很有力地表明电影 _i_ 和电影 _j_ 在某种程度上有相似，至少在某种意义上，某些人喜欢电影 _i_ ，或许更有可能也对电影 _j_ 感兴趣。
总结一下，当用户在看某部电影 _i_ 的时候，如果你想找5部与电影非常相似的电影，为了能给用户推荐5部新电影，你需要做的是找出电影 _j_ ，在这些不同的电影中与要找的电影 _i_ 的距离最小，这样你就能给你的用户推荐几部不同的电影了。

通过这个方法，希望你能知道，如何进行一个向量化的计算来对所有的用户和所有的电影进行评分计算。同时希望你也能掌握，通过学习特征参数，来找到相关电影和产品的方法。

### 均值归一化

来看下面的用户评分数据：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/54b1f7c3131aed24f9834d62a6835642.png" />
</p>

如果新增一个用户 **Eve**，并且 **Eve** 没有为任何电影评分，那么以什么为依据为**Eve**推荐电影呢？

首先需要对结果 _Y_ 矩阵进行均值归一化处理，将每一个用户对某一部电影的评分减去所有用户对该电影评分的平均值：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/9ec5cb55e14bd1462183e104f8e02b80.png" />
</p>
然后利用这个新的 _Y_ 矩阵来训练算法。
如果要用新训练出的算法来预测评分，则需要将平均值重新加回去，预测 _(θ<sup>(j)</sup>)<sup>T</sup>x<sup>(i)</sup> + μ<sub>i</sub>_ ，对于**Eve**，新模型会认为她给每部电影的评分都是该电影的平均分。

## Jupyter Notebook编程练习

- 推荐访问Google Drive的共享，直接在Google Colab在线运行ipynb文件：
  - [Google Drive: 8.anomaly_detection_and_recommendation](https://drive.google.com/drive/folders/1DECp5ajQ9bs7oMQ7Ob0AbKXgkz1zS9zY?usp=sharing)
- 不能翻墙的朋友，可以访问GitHub下载：
  - [GitHub: 8.anomaly_detection_and_recommendation](https://github.com/loveunk/ml-ipynb/blob/master/8.anomaly_detection_and_recommendation)

[回到顶部](#推荐系统)
