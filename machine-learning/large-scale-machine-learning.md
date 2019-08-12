# 大规模机器学习
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [大规模机器学习](#大规模机器学习)
	- [大型数据集的学习](#大型数据集的学习)
		- [确认大规模的训练集是否必要](#确认大规模的训练集是否必要)
	- [随机梯度下降法 Stochastic Gradient Descent (SGD)](#随机梯度下降法-stochastic-gradient-descent-sgd)
	- [小批量梯度下降 Mini-Batch Gradient Descent](#小批量梯度下降-mini-batch-gradient-descent)
	- [随机梯度下降收敛](#随机梯度下降收敛)
	- [在线学习 Online Learning](#在线学习-online-learning)
	- [MapReduce和数据并行](#MapReduce和数据并行)

<!-- /TOC -->
## 大型数据集的学习

一个例子：现在有一个低方差（Low Variance）模型，增加数据集的规模可以帮助你获得更好的结果。应该怎样应对一个有100万条记录的训练集？

以线性回归模型为例，每次梯度下降迭代，都需要计算训练集的误差的平方和，如果学习算法需要20次迭代，将带来是很大的计算代价。

### 确认大规模的训练集是否必要

也许只用1000个samples也能获得好的效果，可以绘制学习曲线来帮助判断。

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/bdf069136b4b661dd14158496d1d1419.png" />
</p>

* 上图左，体现了高方差（variance），增加数据是有用的。
* 上图右，体现了高偏差（bias），通常再增加单纯的数据帮助不大。需要增加特征，或者换模型了。

## 随机梯度下降法 Stochastic Gradient Descent (SGD)
在随机梯度下降法中，定义代价函数为一个单一训练实例的代价：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?cost\left(\theta,\left(x^{(i)},{y}^{(i)}\right)\right)=\frac{1}{2}\left(&space;h_{\theta}\left(x^{(i)}\right)-y^{{(i)}}\right)^{2}" title="cost\left(\theta,\left(x^{(i)},{y}^{(i)}\right)\right)=\frac{1}{2}\left( h_{\theta}\left(x^{(i)}\right)-y^{{(i)}}\right)^{2}" />
</p>

随机梯度下降算法为：首先对训练集随机洗牌（Shuffle）

`for i = 1:m, repeat`
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\theta:={\theta}_j-\alpha\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}&space;\right){x_j}^{(i)}" title="\theta:={\theta}_j-\alpha\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)} \right){x_j}^{(i)}" />
</p>

随机梯度下降算法在每一次计算之后便更新参数 _θ_ ，而不需要先将所有的训练集求和。
梯度下降算法还没有完成一次迭代时，随机梯度下降算法便已走出了很远。
但是这样的算法存在的问题是，不是每一步都是朝着“正确”的方向。
因此算法虽然会逐渐走向全局最小值的位置，但可能无法到最小值的那一点，而在最小值点附近徘徊。

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/9710a69ba509a9dcbca351fccc6e7aae.jpg" />
</p>

## 小批量梯度下降 Mini-Batch Gradient Descent
小批量梯度下降算法(Mini-Batch Gradient Descent)是介于批量梯度下降算法(Gradient Descent)和随机梯度下降算法(SGD)之间的算法。

每计算常数 _b_ 次训练实例，便更新一次参数 _θ_ 。

```
for i = 1: m, repeat
  for i = 1:b, repeat
```
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\theta_j:=\theta_j-\alpha\frac{1}{b}\sum\limits_{k=i}^{i&plus;b-1}\left(h_{\theta}\left(x^{(k)}\right)-y^{(k)}\right)x_j^{(k)}" title="\theta_j:=\theta_j-\alpha\frac{1}{b}\sum\limits_{k=i}^{i+b-1}\left(h_{\theta}\left(x^{(k)}\right)-y^{(k)}\right)x_j^{(k)}" />
</p>

```
  i += b
```

通常会令 _b_ 在`2-512`之间（2的倍数）。
好处是，可以用向量化的方式循环 _b_ 个训练实例，如果用的线性代数函数库比较好，能够支持平行处理，那么算法的总体表现将不受影响（与随机梯度下降相同）。

关于Batch Size的取值可以参考[这篇文章](https://software.intel.com/en-us/articles/cifar-10-classification-using-intel-optimization-for-tensorflow)。当然还要结合GPU显存大小来综合考虑。通常小Batch size可以提高网络的泛化能力。

## 随机梯度下降收敛
关于随机梯度下降算法(SGD)的调试，以及学习率 _α_ 的选取。

在批量梯度下降中，可以令代价函数 _J_ 为迭代次数的函数，绘制图表，根据图表来判断梯度下降是否收敛。
但是，在大规模的训练集的情况下，这是不现实的，因为计算代价太大了。

在随机梯度下降中，在每一次更新 _θ_ 之前都计算一次代价，然后每 _x_ 次迭代后，求出这 _x_ 次对训练实例计算代价的平均值，然后绘制这些平均值与 _x_ 次迭代的次数之间的函数图表。

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/76fb1df50bdf951f4b880fa66489e367.png"/>
</p>

当绘制这样的图时，可能会得到一个颠簸不平但不会明显减少的函数图像（如上面左下图中蓝线所示）。
* 可以增加 _α_ 来使得函数更加平缓，也许便能看出下降的趋势了（如上面左下图中红线所示）；
* 或者可能函数图表仍然是颠簸不平且不下降的（如左下图洋红色线所示），那么模型本身可能存在一些错误。

如果曲线如右下方所示，不断上升，那么可能会需要选择一个较小的学习率 _α_。

也可以令学习率随着迭代次数的增加而减小，例如令：

_α = (const1/(iterationNumber + const2))_

随着不断地靠近全局最小值，通过减小学习率，迫使算法收敛而非在最小值附近徘徊。但是通常不需要这样做便能有非常好的效果。

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/f703f371dbb80d22fd5e4aec48aa9fd4.jpg" />
</p>

## 在线学习 Online Learning
许多大型网站，使用不同版本的在线学习机算法，从大批的涌入又离开网站的用户身上进行学习。特别是，如果有一个由连续的用户流引发的连续的数据流，你能做的是使用一个在线学习机制，从数据流中学习用户的偏好，然后使用这些信息来优化一些关于网站的决策。

假定你有一个提供运输服务的公司，用户们来向你询问把包裹从A地运到B地的服务，同时假定你有一个网站，让用户们可多次登陆，然后他们告诉你，他们想从哪里寄出包裹，以及包裹要寄到哪里去，也就是出发地与目的地，然后你的网站开出运输包裹的的服务价格。比如，我会收取50来运输你的包裹，我会收取20之类的，然后根据你开给用户的这个价格，用户有时会接受这个运输服务，那么这就是个正样本，有时他们会走掉，然后他们拒绝购买你的运输服务，所以，假定想要一个学习算法来帮助，优化给用户开出的价格。

一个算法来从中学习的时候来模型化问题在线学习算法指的是对数据流而非离线的静态数据集的学习。许多在线网站都有持续不断的用户流，对于每一个用户，网站希望能在不将数据存储到数据库中便顺利地进行算法学习。

假使正在经营一家物流公司，每当一个用户询问从地点A至地点B的快递费用时，给用户一个报价，该用户可能选择接受（ _y=1_ ）或不接受（ _y=0_ ）。

现在，希望构建一个模型，来预测用户接受报价使用物流服务的可能性。因此报价是一个特征，其他特征为距离，起始地点，目标地点以及特定的用户数据。模型的输出是: _p(y=1)_ 。

在线学习的算法与随机梯度下降算法有些类似，对单一的实例进行学习，而非对一个提前定义的训练集进行循环。

不断重复： 

* _θ<sub>j</sub> := θ<sub>j</sub> - α(h<sub>θ</sub>(x) - y) x<sub>j</sub>_, (for _j=0:n_ )

一旦对一个数据的学习完成了，便可以丢弃该数据，不需要再存储它了。
这种方式的好处在于，算法可以很好的适应用户的倾向性，可以针对用户的当前行为不断地更新模型以适应该用户。

每次交互事件并不只产生一个数据集，例如，一次给用户提供3个物流选项，用户选择2项，实际上可以获得3个新的训练实例，因而算法可以一次从3个实例中学习并更新模型。

这些问题中的任何一个都可以被归类到标准的，拥有一个固定的样本集的机器学习问题中。
或许，你可以运行一个你自己的网站，尝试运行几天，然后保存一个数据集，一个固定的数据集，然后对其运行一个学习算法。
但是这些是实际的问题，在这些问题里，你会看到大公司会获取如此多的数据，真的没有必要来保存一个固定的数据集，取而代之的是可以使用一个在线学习算法来连续学习，从这些用户不断产生的数据中来学习。

这就是在线学习机制，所使用的这个算法与随机梯度下降算法非常类似，唯一的区别的是，不会使用一个固定的数据集，会做的是获取一个用户样本，从那个样本中学习，然后丢弃那个样本并继续下去，而且如果你对某一种应用有一个连续的数据流，这样的算法可能会非常值得考虑。

当然，在线学习的一个优点就是，如果有一个变化的用户群，又或者你在尝试预测的事情，在缓慢变化，就像你的用户的品味在缓慢变化，这个在线学习算法，可以慢慢地调试你所学习到的假设，将其调节更新到最新的用户行为。

## MapReduce和数据并行
映射化简和数据并行对于大规模机器学习问题而言是非常重要的概念。之前提到，如果用批量梯度下降算法来求解大规模数据集的最优解，需要对整个训练集进行循环，计算偏导数和代价，再求和，计算代价非常大。

如果能够将数据集分配给少量数台计算机，让每一台计算机处理数据集的一个子集，然后将计所的结果汇总在求和。这样的方法叫做`映射简化`（MapReduce）。

具体而言，如果任何学习算法能够表达为，对训练集的函数的求和，那么便能将这个任务分配给多台计算机（或者同一台计算机的不同CPU 核心），以达到加速处理的目的。

例如，有400个训练实例，可以将批量梯度下降的求和任务分配给4台计算机进行处理：

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/919eabe903ef585ec7d08f2895551a1f.jpg" />
</p>

很多高级的线性代数函数库已经能够利用多核CPU的多个核心来并行地处理矩阵运算，这也是算法的向量化实现如此重要的缘故（比调用循环快）。

[回到顶部](#大规模机器学习)
