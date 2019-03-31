# 异常检测
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [异常检测](#异常检测)
	- [高斯分布](#高斯分布)
	- [基于高斯分布的异常检测算法](#基于高斯分布的异常检测算法)
	- [开发和评价异常检测系统](#开发和评价异常检测系统)
	- [异常检测与监督学习对比](#异常检测与监督学习对比)
	- [选择特征](#选择特征)
		- [误差分析](#误差分析)
		- [异常检测误差分析](#异常检测误差分析)
	- [多元高斯分布](#多元高斯分布)
	- [使用多元高斯分布进行异常检测](#使用多元高斯分布进行异常检测)

<!-- /TOC -->


异常检测(Anomaly detection)，是机器学习算法的一个常见应用。
它虽然主要用于非监督学习问题，但从某些角度看，它又类似于一些监督学习问题。

什么是异常检测呢？举个例子：
* 比如飞机制造商空客公司（AirBus），在生产的飞机引擎从生产线上流出时，需要进行QA(质量测试)，作为这个测试的一部分，测量了飞机引擎的一些特征变量，比如引擎运转时产生的热量，或者引擎的振动等等。

这样就有了一个数据集，从 _x<sup>(1)</sup>_ 到 _x<sup>(m)</sup>_ ，如果生产了 _m_ 个引擎的话，将这些数据绘制成图表，看起来就是这个样子：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/fe4472adbf6ddd9d9b51d698cc750b68.png" />
</p>

上图中的每个点（红叉），都是无标签数据。
这样，异常检测问题可以定义如下：
假设有一天，一个新飞机引擎从生产线下线，而新飞机引擎有特征变量 _x<sub>test</sub>_ 。
所谓的异常检测问题是：希望知道这个新的飞机引擎是否有某种异常，或者说，希望判断这个引擎是否需要进一步测试。
因为，如果它看起来像一个正常的引擎，那么可以直接将它运送到客户那里，而不需要进一步的测试。

给定数据集 _x<sup>(1)</sup>,x<sup>(2)</sup>,..,x<sup>(m)</sup>_ ，假使数据集是正常的，希望知道新的数据 _x<sub>test</sub>_ 是不是异常。
即这个测试数据不属于该组数据的几率多大。
所构建的模型应能根据该测试数据的位置告诉其属于一组数据的可能性 _p(x)_ 。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/65afdea865d50cba12d4f7674d599de5.png" />
</p>

上图中，在蓝色圈内的数据属于该组数据的可能性较高，而越是偏远的数据，其属于该组数据的可能性就越低。

这种方法称为密度估计，表达如下：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?flag(x_{test})&space;=&space;\begin{cases}&space;\text{anomaly}&space;&if&space;\quad&space;p(x)&space;<&space;\varepsilon&space;\\&space;\text{normal}&space;&if&space;\quad&space;p(x)&space;\ge&space;\varepsilon&space;\end{cases}" title="flag(x_{test}) = \begin{cases} \text{anomaly} &if \quad p(x) < \varepsilon \\ \text{normal} &if \quad p(x) \ge \varepsilon \end{cases}" />
</p>

异常检测主要用来识别欺骗。
* 例如在线采集而来的有关用户的数据，一个特征向量中可能会包含如：用户多久登录一次，访问过的页面，在论坛发布的帖子数量，甚至是打字速度等。尝试根据这些特征构建一个模型，可以用这个模型来识别那些不符合该模式的用户。

再一个例子是检测一个数据中心，特征可能包含：内存使用情况，被访问的磁盘数量，CPU的负载，网络的通信量等。根据这些特征可以构建一个模型，用来判断某些计算机是不是有可能出错了。

## 高斯分布
通常如果认为变量 _x_ 符合高斯分布 _x ~ N(mu,σ<sup>2</sup>)_ 则其概率密度函数为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?p(x,\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)" title="p(x,\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)" />
</p>

可以利用已有的数据来预测总体中的 _μ_ 和 _σ<sup>2</sup>_ 的计算方法如下：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\mu=\frac{1}{m}\sum\limits_{i=1}^{m}x^{(i)}" title="\mu=\frac{1}{m}\sum\limits_{i=1}^{m}x^{(i)}" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\sigma^2=\frac{1}{m}\sum\limits_{i=1}^{m}(x^{(i)}-\mu)^2" title="\sigma^2=\frac{1}{m}\sum\limits_{i=1}^{m}(x^{(i)}-\mu)^2" />
</p>

高斯分布样例：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/fcb35433507a56631dde2b4e543743ee.png" />
</p>

机器学习中对于方差通常只除以 _m_ 而非统计学中的 _(m-1)_ 。
在实际使用中，是使用 _1/m_ 还是 _1/(m-1)_？
* 区别很小，只要有一个还算大的训练集，在机器学习领域大部分人更习惯使用 _1/m_。
* 这两个版本在理论特性和数学特性上稍有不同，但是在实际使用中，他们的区别甚小，几乎可以忽略不计。

## 基于高斯分布的异常检测算法
对于给定的数据集 _x<sup>(1)</sup>,x<sup>(2)</sup>,...,x<sup>(m)</sup>_。

首先要针对每一个特征计算 _μ_ 和 _σ<sup>2</sup>_ 的估计值。

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\mu_j&=\frac{1}{m}\sum\limits_{i=1}^{m}x_j^{(i)}\\&space;\sigma_j^2&=\frac{1}{m}\sum\limits_{i=1}^m(x_j^{(i)}-\mu_j)^2&space;\end{align*}" title="\begin{align*} \mu_j&=\frac{1}{m}\sum\limits_{i=1}^{m}x_j^{(i)}\\ \sigma_j^2&=\frac{1}{m}\sum\limits_{i=1}^m(x_j^{(i)}-\mu_j)^2 \end{align*}" />
</p>

一旦获得了平均值和方差的估计值，给定新的一个训练实例，根据模型计算 _p(x)_ ：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?p(x)=\prod\limits_{j=1}^np(x_j;\mu_j,\sigma_j^2)=\prod\limits_{j=1}^1\frac{1}{\sqrt{2\pi}\sigma_j}exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})" title="p(x)=\prod\limits_{j=1}^np(x_j;\mu_j,\sigma_j^2)=\prod\limits_{j=1}^1\frac{1}{\sqrt{2\pi}\sigma_j}exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})" />
</p>

当 _p(x)<ϵ_ 时，为异常。

下图这个例子是一个由两个特征的训练集，以及特征的分布情况：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/ba47767a11ba39a23898b9f1a5a57cc5.png" />
</p>

下面的三维图表表示的是密度估计函数， _z_ 轴为根据两个特征的值所估计 _p(x)_ 值：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/82b90f56570c05966da116c3afe6fc91.jpg" />
</p>

选择一个 _ϵ_ ，将 _p(x)=ϵ_ 作为判定边界，当 _p(x) > ϵ_ 时预测数据为正常数据，否则为异常。

## 开发和评价异常检测系统
异常检测算法是一个非监督学习算法，意味着无法根据结果变量 _y_ 的值来判断数据是否真的是异常的。
需要另一种方法来帮助检验算法是否有效。
当开发一个异常检测系统时，从带标记（异常或正常）的数据着手，
从其中选择一部分正常数据用于构建训练集，然后用剩下的正常数据和异常数据混合的数据构成交叉检验集和测试集。

**例如：**
有10000台正常引擎的数据，有20台异常引擎的数据。这样分配数据：
* 6000台正常引擎的数据作为训练集
* 2000台正常引擎和10台异常引擎的数据作为交叉检验集
* 2000台正常引擎和10台异常引擎的数据作为测试集

**评价方法**：
* 根据测试集数据，估计特征的平均值和方差并构建 _p(x)_ 函数
* 对交叉检验集，尝试使用不同的 _ϵ_ 值作为阀值，并预测数据是否异常，根据 _F1_ 值或者查准率与查全率的比例来选择 _ϵ_
* 选出 _ϵ_ 后，针对测试集进行预测，计算异常检验系统的 _F1_ 值，或者查准率与查全率之比

## 异常检测与监督学习对比
异常检测系统也使用了带标记的数据，与监督学习（例如分类）有些相似，下面的对比可以帮你选择用**监督学习**还是**异常检测**：

| 异常检测                                | 监督学习                                     |
| ----------------------------------- | ---------------------------------------- |
| 非常少量的正向类（异常数据 _y=1_）, 大量的负向类（_y=0_） | 同时有大量的正向类和负向类                            |
| 许多不同种类的异常。根据少量的正向类数据来训练算法。   | 有足够多的正向类实例，足够用于训练，未来遇到的正向类实例可能与训练集中的非常近似。 |
| 未来遇到的异常可能与已掌握的异常、非常的不同。          |                                          |
| 例如： 检测欺诈行为，检测数据中心的计算机运行状况 | 例如：邮件过滤器 天气预报 肿瘤分类                       |


## 选择特征

对于异常检测，使用的特征非常关键，那如何选择特征？

之前讨论的情况是默认特征符合高斯分布，如果数据的分布不是高斯分布，异常检测也可以工作。

但是最好先把数据转换成高斯分布，例如对数函数： _x=log(x+c)_。
* 其中 _c_ 为非负常数；或者 _x = x<sup>c</sup>_ ， _c_ 为0-1之间的一个分数，等方法。
* 在 `Python` 中，通常用 `np.log1p()`函数，`log1p()` 就是 `log(x+1)` ，可以避免出现负数结果，反向函数是`np.expm1()`。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/0990d6b7a5ab3c0036f42083fe2718c6.jpg" />
</p>

### 误差分析
一个常见的问题是一些异常数据可能会有较高的 _p(x)_ 值，因而被算法认为是正常的。
这种情况下误差分析有帮助，可以分析哪些被算法错误预测为正常的数据。

可能能从问题中发现需要增加一些新的特征，增加这些新特征后获得的新算法能够帮助更好地进行异常检测。

### 异常检测误差分析
通常可以通过将一些相关的特征进行组合，来获得一些更好的特征（异常数据的该特征值异常地大或小）

例如，在检测数据中心的计算机状况的例子中，可以用CPU负载与网络通信量的比例作为一个新的特征，如果该值异常地大，便有可能意味着该服务器是陷入了一些问题中。

总结：
* 介绍了如何选择特征，以及对特征进行一些小小的转换，让数据更像正态分布，然后再把数据输入异常检测算法。
* 也介绍了建立特征时，进行的误差分析方法，来捕捉各种异常的可能。

希望通过这些方法，能够了解如何选择好的特征变量，从而帮助异常检测算法，捕捉到各种不同的异常情况。

## 多元高斯分布
假使有两个相关的特征，而且这两个特征的值域范围比较宽。
这种情况下，一般的高斯分布模型可能不能很好地识别异常数据。
其原因在于，一般的高斯分布模型尝试的是去同时抓住两个特征的偏差，因此创造出一个比较大的判定边界。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/598db991a7c930c9021cec5f6ab9beb9.png" />
</p>

上图中是两个相关特征，洋红色的线（根据 _ε_ 的不同其范围可大可小）是一般的高斯分布模型获得的判定边界，很明显绿色的 _X_ 所代表的数据点很可能是异常值，但是其 _p(x)_ 值却仍然在正常范围内。多元高斯分布将创建像图中蓝色曲线所示的判定边界。

在一般的高斯分布模型中，计算 _p(x)_ 的方法是：通过分别计算每个特征对应的几率然后将其累乘起来，在多元高斯分布模型中，将构建特征的协方差矩阵，用所有的特征一起来计算 _p(x)_ 。

首先计算所有特征的平均值，然后再计算协方差矩阵：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?p(x)=\prod_{j=1}^np(x_j;\mu,\sigma_j^2)=\prod_{j=1}^n\frac{1}{\sqrt{2\pi}\sigma_j}exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})" title="p(x)=\prod_{j=1}^np(x_j;\mu,\sigma_j^2)=\prod_{j=1}^n\frac{1}{\sqrt{2\pi}\sigma_j}exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\mu&=\frac{1}{m}\sum_{i=1}^mx^{(i)}\\&space;\Sigma&=\frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu)(x^{(i)}-\mu)^T=\frac{1}{m}(X-\mu)^T(X-\mu)&space;\end{align*}" title="\begin{align*}\mu&=\frac{1}{m}\sum_{i=1}^mx^{(i)}\\ \Sigma&=\frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu)(x^{(i)}-\mu)^T=\frac{1}{m}(X-\mu)^T(X-\mu) \end{align*}" />
</p>

其中 _μ_ 是一个向量，其每一个单元都是原特征矩阵中一行数据的均值。最后计算多元高斯分布：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?p\left(&space;x&space;\right)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)" title="p(x)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)" />
</p>

其中
* _|Σ|_ 是行列式，在 `Python` 中用 `numpy.linalg.det(a)` 计算。
* _Σ<sup>-1</sup>_ 是逆矩阵

下面看看协方差矩阵是如何影响模型的：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/29df906704d254f18e92a63173dd51e7.jpg" />
</p>

上图是5个不同的模型，从左往右依次分析：
1. 一般的高斯分布模型
2. 通过协方差矩阵，令特征1拥有较小偏差，同时保持特征2的偏差
3. 通过协方差矩阵，令特征2拥有较大偏差，同时保持特征1的偏差
4. 通过协方差矩阵，在不改变两个特征原有偏差的基础上，增加两者间的正相关性
5. 通过协方差矩阵，在不改变两个特征原有偏差的基础上，增加两者间的负相关性

**多元高斯分布模型与原高斯分布模型的关系**
* 一元高斯分布模型是多元高斯分布模型的一个子集
* 即像上图中的第1、2、3，3个例子所示，如果协方差矩阵只在对角线的单位上有非零的值时，即为原本的高斯分布模型

原高斯分布模型和多元高斯分布模型的比较：

| 原高斯分布模型 | 多元高斯分布模型 |
|--------------------------------------------------------------|------------------------------------------------------------------------------------|
| 不能捕捉特征间的相关性，但可以通过将特征进行组合的方法来解决 | 自动捕捉特征之间的相关性 |
| 计算代价低，能适应大规模的特征 | 计算代价较高 训练集较小时适用 |
|  | 必须m>n，否则协方差矩阵Σ不可逆的，通常需要m>10n。 特征冗余也会导致协方差矩阵不可逆 |

## 使用多元高斯分布进行异常检测
多元高斯分布该如何应用到异常检测？
先回顾一下多元高斯分布和多元正态分布：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/3dbee365617e9264831400e4de247adc.png" />
</p>

上面的多元高斯分布公式中有两个参数， _μ_ 和 _Σ_ 。
其中 _μ_ 是 _n_ 维向量。 _Σ_ 是协方差矩阵， _n×n_ 的矩阵。
如果改变 _μ_ 和 _Σ_ 这两个参数，可以得到不同的高斯分布。

接下来考虑参数拟合或参数估计问题：

假设有一组样本 _x<sup>(1)</sup>, x<sup>(2)</sup>, ..., x<sup>(m)</sup>_， 是一个 _n_ 维向量。假设样本来自一个多元高斯分布。
该如何尝试估计参数 _μ_ 和 _Σ_ 以及标准公式？

如之前讨论， _μ_ 是训练样本的平均值，_Σ_ 是协方差矩阵。
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\mu&=\frac{1}{m}\sum_{i=1}^mx^{(i)}\\&space;\Sigma&=\frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu)(x^{(i)}-\mu)^T=\frac{1}{m}(X-\mu)^T(X-\mu)&space;\end{align*}" title="\begin{align*}\mu&=\frac{1}{m}\sum_{i=1}^mx^{(i)}\\ \Sigma&=\frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu)(x^{(i)}-\mu)^T=\frac{1}{m}(X-\mu)^T(X-\mu) \end{align*}" />
</p>

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/d1a228f2bec262f2206379ed844c7f4a.png" />
</p>

1. 首先，根据训练集计算 _μ_ 和 _Σ_；
2. 其次，拟合模型计算 _p(x)_；
3. 比较 _p(x)_ 和 _ϵ_；

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/015cee3a224dde6da0181215cf91a23d.png" />
</p>

如图
* 该分布在中央最多，越到外面的圈的范围越小。
* 并在该点是出路这里的概率非常低。
* 原始模型与多元高斯模型的关系如图：

几个二元高斯分布和的协方差矩阵 _Σ_ 的关系图如下：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/7104dd2548f1251e4c423e059d1d2594.png" />
</p>

如上图，如果多元高斯分布模型的 _Σ_ 仅对角线上的元素为非零值，那么它就是与原始高斯分布模型一样。

**原始模型和多元高斯分布比较**
<table>
<tr>
<th>原始模型 </th>
<th>多元高斯分布模型</th>
</tr>
<tr>
<td> <i>p(x<sub>1</sub>;μ<sub>1</sub>,σ<sub>1</sub><sup>2</sup>)× ... p(x<sub>n</sub>;μ<sub>n</sub>,σ<sub>n</sub><sup>2</sup>)</i> </td>
<td><img src="https://latex.codecogs.com/gif.latex?p(x;&space;\mu,&space;\Sigma)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)" title="p(x; \mu, \Sigma)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)" /></td>
</tr>
<tr><td>手动选择特征，通常需要将多个特征组合使用（例如<i>x<sub>3</sub> = x<sub>1</sub>/x<sub>2</sub></i></td><td>自动发现特征间的关系</td></tr>
<tr><td>相对少的计算量（使用与大量样本）</td><td>计算代价高（协方差矩阵是 <i>n*n</i>，尤其是算逆矩阵需要大量计算）</td></tr>
<tr><td><i>m</i>很小时也适用</td><td>需要 <i>m > n</i>，否则协方差矩阵不可逆</td></tr>
</table>

## Jupyter Notebook编程练习

- 推荐访问Google Drive的共享，直接在Google Colab在线运行ipynb文件：
  - [Google Drive: 8.anomaly_detection_and_recommendation](https://drive.google.com/drive/folders/1DECp5ajQ9bs7oMQ7Ob0AbKXgkz1zS9zY?usp=sharing)
- 不能翻墙的朋友，可以访问GitHub下载：
  - [GitHub: 8.anomaly_detection_and_recommendation](https://github.com/loveunk/ml-ipynb/blob/master/8.anomaly_detection_and_recommendation)


[回到顶部](#异常检测)
