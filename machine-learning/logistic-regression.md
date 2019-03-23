# 逻辑回归

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [逻辑回归](#逻辑回归)
	- [Hypothesis 表示](#hypothesis-表示)
	- [边界判定](#边界判定)
	- [代价函数](#代价函数)
	- [梯度下降算法](#梯度下降算法)
	- [多类别分类：一对多](#多类别分类一对多)
	- [正则化 Regularization](#正则化-regularization)
		- [过拟合的问题](#过拟合的问题)
		- [代价函数](#代价函数)
		- [正则化线性回归](#正则化线性回归)
			- [正则化与逆矩阵](#正则化与逆矩阵)
		- [正则化的逻辑回归模型](#正则化的逻辑回归模型)

<!-- /TOC -->

逻辑回归 (Logistic Regression) 的算法，这是目前最流行使用最广泛的学习算法之一。

首先介绍几种分类问题：
* 垃圾邮件分类：垃圾邮件（是或不是）？
* 在线交易分类：欺诈性的（是或不是）？
* 肿瘤：恶性 / 良性

先从二元的分类问题开始讨论。将因变量(dependent variable)可能属于的两个类分别称为
* 负向类（negative class）和
* 正向类（positive class）

则 因变量 _y ∈ { 0,1 }_ ，其中 0 表示负向类，1 表示正向类。

对于肿瘤分类是否为良性的问题：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/f86eacc2a74159c068e82ea267a752f7.png" />
</p>

对于二分类问题，_y_ 取值为 0 或者1，但如果使用的是线性回归，那么 _h_ 的输出值可能 远大于1，或者远小于。但数据的标签应该取值0 或者1。所以在接下来的要研究一种新算法**逻辑回归算法**，这个算法的性质是：它的输出值永远在0到 1 之间。

逻辑回归算法是分类算法。可能因为算法的名字中出现“回归”让人感到困惑，但逻辑回归算法实际上是一种分类算法，它适用于标签 _y_ 取值离散的情况，如：1 0 0 1。

## Hypothesis 表示
逻辑回归的输出变量范围始终在0和1之间。
逻辑回归模型的假设是： _h<sub>θ</sub> = g(θ<sup>T</sup>X)_ ，其中： _X_ 代表特征向量 _g_ 代表逻辑函数（Logistic function）。

逻辑函数是一个常用的逻辑函数为S形函数（Sigmoid function）:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?g\left(&space;z&space;\right)=\frac{1}{1&plus;{{e}^{-z}}}" title="g\left( z \right)=\frac{1}{1+{{e}^{-z}}}" />
</p>

整合上式，得到：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?h_\theta&space;\left(&space;x&space;\right)=\frac{1}{1&plus;{{e}^{-\theta^TX}}}" title="h_\theta \left( x \right)=\frac{1}{1+{{e}^{-\theta^TX}}}" />
</p>

逻辑函数的示意图：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/1073efb17b0d053b4f9218d4393246cc.jpg" />
</p>

Python实现：
``` python
import numpy as np
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
```

_h<sub>θ</sub>(x)_ 的作用，给定输入变量，计算输出变量 = 1的可能性（estimated probability），即 _h<sub>θ</sub>(x) = P(y=1 | x;θ)_。
* 例如对一个肿瘤的样本，计算得到 _h<sub>θ</sub>(x) = 0.7_，也就是有70%的可能是恶性的。

## 边界判定
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/6590923ac94130a979a8ca1d911b68a3.png" />
</p>

* 当 _h<sub>θ</sub>(x) >= 0.5_ 时，预测 _y=1_。
* 当 _h<sub>θ</sub>(x) < 0.5_ 时，预测 _y=0_ 。

可以总结为：
* _z=0_ 时 _g(z)=0.5_
* _z>0_ 时 _g(z)>0.5_
* _z<0_ 时 _g(z)<0.5_

又 _z=θ<sup>T</sup>x_ ，即： _θ<sup>T</sup>x>=0_ 时，预测 _y=1_  _θ<sup>T</sup>x<0_ 时，预测 _y=0_

举个例子：
* 假设现在有一个模型，参数 _θ_ 是向量[-3 1 1]。则当 _-3+x<sub>1</sub>+x<sub>2</sub> >= 0_ ，即 _x<sub>1</sub>+x<sub>2</sub> >= 3_ 时，模型将预测 _y=1_ 。我们可以绘制直线 _x<sub>1</sub>+x<sub>2</sub>=3_ ，这条线便是我们模型的分界线，将预测为1的区域和预测为0的区域分隔开。
如下图：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/58d098bbb415f2c3797a63bd870c3b8f.png" />
</p>

分类的示意图如下：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/f71fb6102e1ceb616314499a027336dc.jpg" />
</p>

上面的例子还是很明显的，来一个复杂一点的：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/197d605aa74bee1556720ea248bab182.jpg" />
</p>

上图中的数据需要用曲线才能分隔 _y=0_ 区域和 _y=1_ 区域。
_y = 0_ 区域接近一个圆形，选用二次多项式： _h<sub>θ</sub>(x)=g(θ<sub>0</sub>+θ<sub>1</sub>x<sub>1</sub>+θ<sub>2</sub>x<sub>2</sub>+θ<sub>3</sub>x<sub>1</sub><sup>2</sup>+θ<sub>4</sub>x<sub>2</sub><sup>2</sup>)_。到这里还未讲过如何自动选取参数，先假设参数向量是[-1 0 0 1 1]，则我们得到的判定边界恰好是圆点在原点且半径为1的圆形。

## 代价函数
对于一个模型，如何选取参数 _θ_ 了？
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/f23eebddd70122ef05baa682f4d6bd0f.png" />
</p>

首先要定义用来拟合参数的优化目标或者叫代价函数，这便是监督学习问题中的逻辑回归模型的拟合问题。

对于线性回归模型，我们定义的代价函数是所有模型误差的平方和（ _J(θ)=1/(2m) Σ (h<sub>θ</sub>(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>_ ）。理论上来说，我们也可以对逻辑回归模型沿用这个定义，但是问题在于，当我们将 _h<sub>θ</sub>(x)=(1+e<sup>-θ<sup>T</sup>x</sup>)<sup>-1</sup>_ 带入到这样定义了的代价函数中时，我们得到的代价函数将是一个非凸函数（non-convex function）。

凸函数和非凸函数的示意如下：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/8b94e47b7630ac2b0bcb10d204513810.jpg" />
</p>

加入代价函数 _J_ 是非凸函数，意味着有许多局部最小值（见上图左图），这将影响梯度下降算法寻找全局最小值。

因此重新定义代价函数为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J\left(&space;\theta&space;\right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left(&space;{h_\theta}\left(&space;{x}^{\left(&space;i&space;\right)}&space;\right),{y}^{\left(&space;i&space;\right)}&space;\right)}" title="J\left( \theta \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left( {h_\theta}\left( {x}^{\left( i \right)} \right),{y}^{\left( i \right)} \right)}" />
</p>

其中 _Cost()_ 的定义为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?Cost\left(h_\theta(x),y)\right)&space;=&space;\begin{cases}&space;-log\left(h_\theta(x)&space;\right&space;)&space;&&space;\text{&space;if&space;}&space;y=1&space;\\&space;-log\left(1-h_\theta(x)&space;\right&space;)&space;&&space;\text{&space;if&space;}&space;y=0&space;\end{cases}" title="Cost\left(h_\theta(x),y)\right) = \begin{cases} -log\left(h_\theta(x) \right ) & \text{ if } y=1 \\ -log\left(1-h_\theta(x) \right ) & \text{ if } y=0 \end{cases}" />
</p>

 _h<sub>θ</sub>(x)_ 与 _Cost(h<sub>θ</sub>(x),y)_ 之间的关系如下图所示：
 <p align="center">
 <img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/ffa56adcc217800d71afdc3e0df88378.jpg" />
 </p>

这样构建的 _Cost(h<sub>θ</sub>(x),y)_ 函数的特点是：当实际的 _y=1_ 且 _h<sub>θ</sub>(x)_ 也为1时误差为0，当 _y=1_ 但 _h<sub>θ</sub>(x)_ 不为1时误差随着 _h<sub>θ</sub>(x)_ 变小而变大；当实际的 _y=0_ 且 _h<sub>θ</sub>(x)_ 也为0时代价为0，当 _y=0_ 但 _h<sub>θ</sub>(x)_ 不为0时误差随着 _h<sub>θ</sub>(x)_ 的变大而变大。

将上式整合如下：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?Cost\left(&space;{h_\theta}\left(&space;x&space;\right),y&space;\right)=-y&space;log\left(&space;{h_\theta}\left(&space;x&space;\right)&space;\right)-(1-y)&space;log\left(&space;1-{h_\theta}\left(&space;x&space;\right)&space;\right)" title="Cost\left( {h_\theta}\left( x \right),y \right)=-y log\left( {h_\theta}\left( x \right) \right)-(1-y) log\left( 1-{h_\theta}\left( x \right) \right)" />
</p>

从而得到
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;J\left(&space;\theta&space;\right)&space;&=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log&space;\left(&space;{h_\theta}\left(&space;{{x}^{(i)}}&space;\right)&space;\right)-\left(&space;1-{{y}^{(i)}}&space;\right)\log&space;\left(&space;1-{h_\theta}\left(&space;{{x}^{(i)}}&space;\right)&space;\right)]}\\&space;&=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log&space;\left(&space;{h_\theta}\left(&space;{{x}^{(i)}}&space;\right)&space;\right)&plus;\left(&space;1-{{y}^{(i)}}&space;\right)\log&space;\left(&space;1-{h_\theta}\left(&space;{{x}^{(i)}}&space;\right)&space;\right)]}&space;\end{align*}" title="\begin{align*} J\left( \theta \right) &=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}\\ &=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]} \end{align*}" />
</p>

对于这个代价函数，为了拟合 _θ_ ，目标函数为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\underset{\theta}{\mathbf{min}}J(\theta)" title="\underset{\theta}{\mathbf{min}}J(\theta)" />
</p>

对于新的样本 _x_，输出为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?h_\theta(x)&space;=&space;\frac{1}{1&plus;e^{-\theta^Tx}}" title="h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}" />
</p>

Python代码计算Cost的示例，两种方法是同样的效果：
``` python
import numpy as np

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def cost1(theta, X, y): # 这是第一种方法
  first = - y.T @ np.log(sigmoid(X @ theta))
  second = (1 - y.T) @ np.log(1 - sigmoid(X @ theta))
  return ((first - second) / (len(X))).item()


def cost2(theta, X, y): # 这是第二种方法
  first = np.multiply(-y, np.log(sigmoid(X @ theta)))
  second = np.multiply((1 - y), np.log(1 - sigmoid(X @ theta)))
  return np.sum(first - second) / (len(X))
```

## 梯度下降算法

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\text{Repeat&space;\{}&space;&&space;\\&space;&\theta_j&space;:=&space;\theta_j&space;-&space;\alpha&space;\frac{1}{m}\sum\limits_{i=1}^{m}&space;\left(&space;h_\theta&space;\left(&space;x^{\left(&space;i&space;\right)}&space;\right)&space;-&space;y^{\left(&space;i&space;\right)}&space;\right)&space;x_{j}^{(i)}&space;\\&space;&\text{(simultaneously&space;update&space;all)}&space;\\&space;\mathbf{\}}&space;\end{align*}" title="\begin{align*} \text{Repeat \{} & \\ &\theta_j := \theta_j - \alpha \frac{1}{m}\sum\limits_{i=1}^{m} \left( h_\theta \left( x^{\left( i \right)} \right) - y^{\left( i \right)} \right) x_{j}^{(i)} \\ &\text{(simultaneously update all)} \\ \mathbf{\}} \end{align*}" />
</p>

推导过程：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?{{y}^{(i)}}\log&space;\left(&space;{h_\theta}\left(&space;{{x}^{(i)}}&space;\right)&space;\right)&plus;\left(&space;1-{{y}^{(i)}}&space;\right)\log&space;\left(&space;1-{h_\theta}\left(&space;{{x}^{(i)}}&space;\right)&space;\right)\\&space;={{y}^{(i)}}\log&space;\left(&space;\frac{1}{1&plus;{{e}^{-{\theta^T}{{x}^{(i)}}}}}&space;\right)&plus;\left(&space;1-{{y}^{(i)}}&space;\right)\log&space;\left(&space;1-\frac{1}{1&plus;{{e}^{-{\theta^T}{{x}^{(i)}}}}}&space;\right)\\&space;=-{{y}^{(i)}}\log&space;\left(&space;1&plus;{{e}^{-{\theta^T}{{x}^{(i)}}}}&space;\right)-\left(&space;1-{{y}^{(i)}}&space;\right)\log&space;\left(&space;1&plus;{{e}^{{\theta^T}{{x}^{(i)}}}}&space;\right)" title="{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)\\ ={{y}^{(i)}}\log \left( \frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} \right)\\ =-{{y}^{(i)}}\log \left( 1+{{e}^{-{\theta^T}{{x}^{(i)}}}} \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1+{{e}^{{\theta^T}{{x}^{(i)}}}} \right)" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\frac{\partial}{\partial{\theta_{j}}}J\left(\theta\right)&=\frac{\partial}{\partial{\theta_{j}}}[-\frac{1}m\sum\limits_{i=1}^m{[-{{y}^{(i)}}\log\left(1&plus;{e^{-{\theta^T}{x^{(i)}}}}\right)-\left(1-{y^{(i)}}\right)\log\left(1&plus;{{e}^{\theta^Tx^{(i)}}}\right)]}]\\&space;&=-\frac{1}{m}\sum\limits_{i=1}^m{y^{(i)}}\frac{x_j^{(i)}}{1&plus;{e^{{\theta^T}{x^{(i)}}}}}-\left(1-y^{(i)}&space;\right)\frac{x_j^{(i)}{e^{{\theta^T}x^{(i)}}}}{1&plus;{e^{{\theta^T}x^{(i)}}}}]\\&space;&=-\frac{1}{m}\sum\limits_{i=1}^m{\frac{{y^{(i)}}x_j^{(i)}-x_j^{(i)}{e^{{\theta^T}{x^{(i)}}}}&plus;{{y}^{(i)}}x_j^{(i)}{{e}^{{\theta^T}{x^(i)}}}}{1&plus;{e^{{\theta^T}{x^{(i)}}}}}}\\&space;&=-\frac{1}{m}\sum\limits_{i=1}^m{(y^{(i)}-\frac{{e^{\theta^Tx^{(i)}}}}{1&plus;{e^{{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}=-\frac{1}{m}\sum\limits_{i=1}^m{(y^{(i)}-\frac{1}{1&plus;{e^{{-\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}\\&space;&=\frac{1}{m}\sum\limits_{i=1}^m{[{h_\theta}\left(x^{(i)}&space;\right)-{y^{(i)}}]x_j^{(i)}}&space;\end{align*}" title="\begin{align*} \frac{\partial}{\partial{\theta_{j}}}J\left(\theta\right)&=\frac{\partial}{\partial{\theta_{j}}}[-\frac{1}m\sum\limits_{i=1}^m{[-{{y}^{(i)}}\log\left(1+{e^{-{\theta^T}{x^{(i)}}}}\right)-\left(1-{y^{(i)}}\right)\log\left(1+{{e}^{\theta^Tx^{(i)}}}\right)]}]\\ &=-\frac{1}{m}\sum\limits_{i=1}^m{y^{(i)}}\frac{x_j^{(i)}}{1+{e^{{\theta^T}{x^{(i)}}}}}-\left(1-y^{(i)} \right)\frac{x_j^{(i)}{e^{{\theta^T}x^{(i)}}}}{1+{e^{{\theta^T}x^{(i)}}}}]\\ &=-\frac{1}{m}\sum\limits_{i=1}^m{\frac{{y^{(i)}}x_j^{(i)}-x_j^{(i)}{e^{{\theta^T}{x^{(i)}}}}+{{y}^{(i)}}x_j^{(i)}{{e}^{{\theta^T}{x^(i)}}}}{1+{e^{{\theta^T}{x^{(i)}}}}}}\\ &=-\frac{1}{m}\sum\limits_{i=1}^m{(y^{(i)}-\frac{{e^{\theta^Tx^{(i)}}}}{1+{e^{{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}=-\frac{1}{m}\sum\limits_{i=1}^m{(y^{(i)}-\frac{1}{1+{e^{{-\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}\\ &=\frac{1}{m}\sum\limits_{i=1}^m{[{h_\theta}\left(x^{(i)} \right)-{y^{(i)}}]x_j^{(i)}} \end{align*}" />
</p>

虽然得到的梯度下降算法表面上看上去与线性回归的梯度下降算法一样，但是这里 _h<sub>θ</sub> = g(θ<sup>T</sup>X)_ 的与线性回归中不同。另外，在运行梯度下降算法之前，进行特征缩放依旧是非常必要的。

然而梯度下降并不是唯一使用的算法，还有其他一些更高级、更复杂的算法。例如共轭梯度法（Conjugate gradient）、BFGS (变尺度法) 和L-BFGS (限制变尺度法) 就是其中更高级的优化算法，它们需要你计算代价函数 _J(θ)_ 和导数项，然后会它们帮你最小化代价函数。

这Conjugate Gradient、BFGS、L-BFGS等算法有许多优点：
1. 不需要手动选择学习率 _α_。只用给出计算导数项和代价函数的方法，因为算法内有一个智能的内部循环，称为线性搜索(line search)算法，它可以自动尝试不同的学习速率 ，并自动选择一个好的学习速率 ，它甚至可以为每次迭代选择不同的学习速率。
2. 这些算法实际上在做更复杂的事情，不仅仅是选择一个好的学习速率，所以它们往往最终比梯度下降收敛得快多了，不过关于它们到底做什么的详细讨论，已经超过了这里讨论的范围。

下面是一个关于使用Conjugate Gradient的完整例子，并plot出training和prediction data
``` python
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

""" X是训练集的数据 """
X_train = np.array([[1.,  1.],
              [1.,  2.],
              [-1., -1.],
              [-1., -2.]])
""" y是训练集的label """
y_train = np.array([1, 1, 0, 0])

""" 处理训练集X，补上x_0 """
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

"""Sigmoid 函数公式 """
def sigmoid(z):
  return 1/(1 + np.exp(-z))

""" 目标函数，也就是待最小化的 Cost function """
def cost(theta, X, y):
  first = - y.T @ np.log(sigmoid(X @ theta))
  second = (1 - y.T) @ np.log(1 - sigmoid(X @ theta))
  return ((first - second) / (len(X))).item()

def hypothesis(X, theta):
  return sigmoid(X @ theta)

def cost_wrapper(theta):
  return cost(theta, X_train, y_train)

def hypothesis_wrapper(theta):
  return hypothesis(X_train, theta)

""" 目标函数的梯度 """
def gradient(theta):
  gradient_sum = (hypothesis_wrapper(theta) - y_train).T @ X_train
  return gradient_sum / X_train.shape[0]

theta_train = np.array([1, 1.,2.])

theta_opt = optimize.minimize(cost_wrapper, theta_train,
                              method='CG', jac=gradient)
print(theta_opt)

""" 构造预测集数据 """
delta = 0.2
px = np.arange(-3.0, 3.0, delta)
py = np.arange(-3.0, 3.0, delta)
px, py = np.meshgrid(px, py)
px = px.reshape((px.size, 1))
py = py.reshape((py.size, 1))
pz = np.hstack((np.hstack((np.ones((px.size, 1)), px)), py))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
""" plot训练集 """
ax.scatter(X_train[:, 1], X_train[:, 2], y_train,
           color='red', marker='^', s=200, label='Traning Data')
""" plot预测集, 二分类时在hypothesis外加上 np.around """
ax.scatter(px, py, (hypothesis(pz, theta_opt.x)),
           color='gray', label='Prediction Data')
ax.legend(loc=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('classification')
plt.show()
```

## 多类别分类：一对多
上述讨论都是针对二分类问题，那如何使用逻辑回归 (logistic regression) 解决多类别分类问题?具体来说，是一个叫做"一对多" (one-vs-all) 的分类算法。

* **例子1**：假如现在需要一个学习算法自动地将邮件归到不同文件夹，或者说自动地加上标签。那么，我们就有了这样一个分类问题：其类别有四个，分别用 _y=1_、_y=2_、_y=3_、_y=4_ 来代表。

* **例子2**：是有关药物诊断的，如果一个病人因为鼻塞来到你的诊所，他可能并没有生病，用 _y=1_ 这个类别来代表；或者患了感冒，用 _y=2_ 来代表；或者得了流感用 _y=3_ 来代表。

* **例子3**：如果你正在做有关天气的机器学习分类问题，那么你可能想要区分哪些天是晴天、多云、雨天、或者下雪天，对上述的例子，可以取一个很小的数值，一个相对"谨慎"的数值，比如1 到3、1到4或者其它数值。

对于一个多类分类问题，我们的数据集或许看起来像下图的右图：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/54d7903564b4416305b26f6ff2e13c04.png" />
</p>


用3种不同的符号来代表3个类别，问题是给出3个类型的数据集，我们如何得到一个学习算法来进行分类呢？

下面将介绍如何进行一对多的分类工作，有时这个方法也被称为"一对余"方法。

如下图，训练集有3个类别，用三角形表示 _y=1_ ，方框表示 _y=2_ ，叉叉表示 _y=3_ 。我们下面要做的就是使用一个训练集，将其分成3个二元分类问题。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/450a83c67732d254dbac2aeeb8ab910c.png" />
</p>

我们先从用三角形代表的类别1开始，实际上我们可以创建一个，新的"伪"训练集，类型2和类型3定为负类，类型1设定为正类，我们创建一个新的训练集，如下图所示的那样，我们要拟合出一个合适的分类器。见下图：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/b72863ce7f85cd491e5b940924ef5a5f.png" />
</p>

这里的三角形是正样本，而圆形代表负样本。可以这样想，设置三角形的值为1，圆形的值为0，下面我们来训练一个标准的逻辑回归分类器，这样我们就得到一个正边界。

为了能实现这样的转变，我们将多个类中的一个类标记为正向类（ _y=1_ ），然后将其他所有类都标记为负向类，这个模型记作 _h<sub>θ</sub><sup>(1)</sup>(x)_ 。接着，类似地第我们选择另一个类标记为正向类（ _y=2_ ），再将其它类都标记为负向类，将这个模型记作 _h<sub>θ</sub><sup>(2)</sup>(x)_ ,依此类推。最后我们得到一系列的模型简记为： _h<sub>θ</sub><sup>(i)</sup>(x)=p(y=i|x;θ)_ 其中： _i=(1,2,3....k)_

最后，在做预测时，将所有的分类机都运行一遍，然后对每一个输入变量，选择最高可能性的输出变量。

现在要做的就是训练这个逻辑回归分类器： _h<sub>θ</sub><sup>(i)</sup>(x)_ ，其中 _i_ 对应每一个可能的 _y=i_ ，最后，为了做出预测，我们给出输入一个新的 _x_ 值，用这个做预测。我们要做的就是在我们三个分类器里面输入 _x_ ，然后我们选择一个让 _h<sub>θ</sub><sup>(i)</sup>(x)_ 最大的 _i_ ，即 _max h<sub>θ</sub><sup>(i)</sup>(x)_ 。

## 正则化 Regularization
### 过拟合的问题
在这之前，已经介绍了线性回归和逻辑回归，它们能够有效地解决许多问题。但当将它们应用到某些特定的机器学习问题时，可能遇到过拟合(over-fitting)的问题，过拟合可能会导致这些算法的效果很差。

如果我们有非常多的特征，我们通过学习得到的假设可能能够非常好地适应训练集（代价函数可能几乎为0），但是可能会不能推广到新的数据。下图是一个回归问题的例子：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/72f84165fbf1753cd516e65d5e91c0d3.jpg" />
</p>

上图中，第一个是线性模型，明显欠拟合（under-fitting），因为不能很好地适应训练集；第三个模型是一个四次方的模型，过于强调拟合原始数据，而丢失了算法的本质：预测新数据。可以看出，若给出一个新的值使之预测，它将表现的很差，是过拟合，虽然能非常好地适应训练集，但在新输入变量进行预测时可能效果不好；而中间的模型最合适。

分类问题中也存在这样的问题：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/be39b497588499d671942cc15026e4a2.jpg" />
</p>

就以多项式理解， _x_ 的次数越高，拟合的越好，但相应的预测的能力就可能变差。
问题是，如果我们发现了过拟合问题，应该如何处理？
* 丢弃一些不能帮助正确预测的特征。可以是手工选择保留哪些特征，或者使用一些模型选择的算法来处理（例如PCA）
* 正则化。保留所有的特征，但是减少参数的大小（magnitude）。

### 代价函数
上面的回归问题中如果我们的模型是： _h<sub>θ</sub>(x)=θ<sub>0</sub>+θ<sub>1</sub>x<sub>1</sub>+θ<sub>2</sub>x<sub>2</sub><sup>2</sup>+θ<sub>3</sub>x<sub>3</sub><sup>3</sup>+θ<sub>4</sub>x<sub>4</sub><sup>4</sup>_。可以从之前事例中看出，是高次项导致过拟合，所以如果能让这些高次项的系数接近于0的话，就能很好的拟合了（避免过拟合）。所以要做的是在一定程度上减小参数 _θ_ 的值，这是正则化的基本方法。我们决定要减少 _θ<sub>3</sub>_ 和 _θ<sub>4</sub>_ 的大小，要做的便是修改代价函数，在其中 _θ<sub>3</sub>_ 和 _θ<sub>4</sub>_ 设置一点惩罚。这样做的话，在尝试最小化代价时也需要将这个惩罚纳入考虑中，并最终导致选择较小一些的 _θ<sub>3</sub>_ 和 _θ<sub>4</sub>_ 。修改后的代价函数如下：

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\underset{\theta}{\mathop{\min}}\frac{1}{2m}\left[\sum\limits_{i=1}^{m}{{\left({{h}_{\theta}}\left(x^{(i)}\right)-y^{(i)}\right)}^2}&plus;1000\theta_3^2&plus;10000\theta_4^2\right]" title="\underset{\theta}{\mathop{\min}}\frac{1}{2m}\left[\sum\limits_{i=1}^{m}{{\left({{h}_{\theta}}\left(x^{(i)}\right)-y^{(i)}\right)}^2}+1000\theta_3^2+10000\theta_4^2\right]" />
</p>

通过这个代价函数选择出的 _θ<sub>3</sub>_ 和 _θ<sub>4</sub>_ 对预测结果的影响就比之前要小许多。假如有非常多的特征，并不知道哪些特征我们要惩罚，可以对所有的特征进行惩罚，并且让代价函数最优化的软件来选择这些惩罚的程度。这样的结果是得到了一个较为简单的能防止过拟合问题的假设：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J\left(\theta\right)=\frac{1}{2m}\left[\sum\limits_{i=1}^m{{({h_\theta}({x^{(i)}})-{y}^{(i)})}^2}&plus;\lambda\sum\limits_{j=1}^n{\theta_j^2}\right]" title="J\left(\theta\right)=\frac{1}{2m}\left[\sum\limits_{i=1}^m{{({h_\theta}({x^{(i)}})-{y}^{(i)})}^2}+\lambda\sum\limits_{j=1}^n{\theta_j^2}\right]" />
</p>

其中 _\lambda_ 又称为**正则化参数**（**RegularizationParameter**）。注：根据惯例，不对 _θ<sub>0</sub>_ 惩罚。经过正则化处理的模型与原模型的可能对比如下图所示：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/ea76cc5394cf298f2414f230bcded0bd.jpg" />
</p>

如果选择的正则化参数 _λ_ 过大，则会把所有的参数都最小化了，导致模型变成 _h<sub>θ</sub>(x)=θ<sub>0</sub>_ ，也就是上图中红色直线所示的情况，造成欠拟合。那为什么增加的一项 _λ=Σ<sup>n</sup><sub>j=1</sub>θ<sub>j</sub><sup>2</sup>_ 可以使 _θ_ 的值减小呢？

因为如果我们令 _λ_ 的值很大的话，为了使`CostFunction`尽可能的小，所有的 _θ_ 的值（不包括 _θ<sub>0</sub>_ ）都会在一定程度上减小。但若 _λ_ 的值太大了，那么 _θ_ （不包括 _θ<sub>0</sub>_ ）都会趋近于0，这样我们所得到的只能是一条平行于 _x_ 轴的直线。所以对于正则化，要取一个合理的 _λ_ 值，才能更好的应用正则化。

回顾一下代价函数，为了使用正则化，让我们把这些概念应用到到线性回归和逻辑回归中去，那么我们就可以让他们避免过度拟合了。

### 正则化线性回归
对于线性回归的求解，我们之前推导了两种学习算法：一种基于梯度下降，一种基于正规方程。

正则化线性回归的代价函数为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J\left(&space;\theta&space;\right)=\frac{1}{2m}\left[\sum\limits_{i=1}^{m}{({{{h_\theta}({x^{(i)}})-{y^{(i)}})}^2}&plus;\lambda&space;\sum\limits_{j=1}^n{\theta_j^2}}\right]" title="J\left( \theta \right)=\frac{1}{2m}\left[\sum\limits_{i=1}^{m}{({{{h_\theta}({x^{(i)}})-{y^{(i)}})}^2}+\lambda \sum\limits_{j=1}^n{\theta_j^2}}\right]" />
</p>

如果我们要用梯度下降法令这个代价函数最小化，因为我们不对 _θ<sub>0</sub>_ 进行正则化，所以梯度下降算法将分两种情形：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;{\theta_0}&:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^m{({h_\theta}(x^{(i)})-{y^{(i)}})x_0^{(i)}}\\&space;{\theta_j}&:={\theta_j}-a\frac{1}{m}\left[\sum\limits_{i=1}^m{({h_\theta}(x^{(i)})-{y^{(i)}})x_j^{\left(i\right)}}&plus;\lambda{\theta_j}\right]&space;\text{,&space;for&space;}&space;j=1,2,...,n&space;\end{align*}" title="\begin{align*} {\theta_0}&:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^m{({h_\theta}(x^{(i)})-{y^{(i)}})x_0^{(i)}}\\ {\theta_j}&:={\theta_j}-a\frac{1}{m}\left[\sum\limits_{i=1}^m{({h_\theta}(x^{(i)})-{y^{(i)}})x_j^{\left(i\right)}}+\lambda{\theta_j}\right] \text{, for } j=1,2,...,n \end{align*}" />
</p>

上式可以调整为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?{\theta_j}:={\theta_j}(1-\alpha\frac{\lambda}{m})-a\frac{1}{m}\sum\limits_{i=1}^m{\left({h_\theta}(x^{(i)})-{y^{(i)}}\right)x_j^{\left(i\right)}}" title="{\theta_j}:={\theta_j}(1-\alpha\frac{\lambda}{m})-a\frac{1}{m}\sum\limits_{i=1}^m{\left({h_\theta}(x^{(i)})-{y^{(i)}}\right)x_j^{\left(i\right)}}" />
</p>

可以看出，正则化线性回归的梯度下降算法的变化在于，每次都在原有算法更新规则的基础上令值减少了一个额外的值。

我们同样也可以利用正规方程来求解正则化线性回归模型，方法如下所示：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/71d723ddb5863c943fcd4e6951114ee3.png" />
</p>

图中的矩阵尺寸为 _(n+1)*(n+1)_ 。

#### 正则化与逆矩阵
之前讲正规方程时讲过，要求 _X<sup>T</sup>X_ 必须是可逆的，那对于不可逆的情况，可以用 `pinv()`来计算伪逆。

而正则化刚好解决了这个问题，如果 _λ > 0_，可以证明 _X<sup>T</sup>X + λI_ 是可逆的。

##### 证明过程
正定矩阵（positive defined matrix）有一个性质，所有的正定矩阵都是可逆的。因此我们可以通过证明 _X<sup>T</sup>X + λI_ 是正定阵来证明其可逆。
（此外，正定矩阵特征值全部大于零（半正定矩阵特征值全部大于等于零）

1. _X<sup>T</sup>X_ 是半正定的（positive semi-definited matrix）：

	根据半正定的定义，只需证明，对于不为零的向量 _z_，有  _z<sup>T</sup>(X<sup>T</sup>X)z=(z<sup>T</sup>X<sup>T</sup>)(Xz)=(Xz)<sup>T</sup>(Xz) ≥ 0_ ；
	_Xz_ 记做 _u_，那么 _(Xz)<sup>T</sup>(Xz) = u<sup>T</sup>u_，根据向量点乘的性质得 _u<sup>T</sup>u ≥ 0_。

	所以X<sup>T</sup>X 是半正定的。

2. _X<sup>T</sup>X + λI_ 是正定的：
	类似上面的证明 _z<sup>T</sup>(X<sup>T</sup>X + λI) z = (z<sup>T</sup>X<sup>T</sup>)(Xz) + z<sup>T</sup>λIz_。
	根据第1点的证明，可知 _(Xz)<sup>T</sup>(Xz) ≥ 0_。
	而对于非零向量z，有 _z<sup>T</sup>λIz > 0_。
	所以 _z<sup>T</sup>(X<sup>T</sup>X + λI) z > 0_。
	根据正定矩阵定义，**_X<sup>T</sup>X + λI_ 是正定的。所以 _X<sup>T</sup>X + λI_ 是可逆的**。

关于正定矩阵和半正定矩阵的几何理解，推荐看[这里](https://www.zhihu.com/question/22098422/answer/35874276)。

### 正则化的逻辑回归模型
针对逻辑回归问题，我们在之前的课程已经学习过两种优化算法：
1. 梯度下降法来优化代价函数 _J(θ)_
2. 更高级的优化算法，这些高级优化算法需要你自己设计代价函数 _J(θ)_

例如对于下图的数据，当有很多features时，容易导致过拟合。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/2726da11c772fc58f0c85e40aaed14bd.png" />
</p>

类似线性回归正则的处理，我们也给代价函数增加一个正则化的表达式，得到代价函数：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J\left(\theta\right)=\frac{1}{m}\sum\limits_{i=1}^m{[-{y^{(i)}}\log\left({h_\theta}\left({x^{(i)}}\right)\right)-\left(1-{y^{(i)}}\right)\log\left(1-{h_\theta}\left({x^{(i)}}\right)\right)]}&plus;\frac{\lambda}{2m}\sum\limits_{j=1}^n{\theta_j^2}" title="J\left(\theta\right)=\frac{1}{m}\sum\limits_{i=1}^m{[-{y^{(i)}}\log\left({h_\theta}\left({x^{(i)}}\right)\right)-\left(1-{y^{(i)}}\right)\log\left(1-{h_\theta}\left({x^{(i)}}\right)\right)]}+\frac{\lambda}{2m}\sum\limits_{j=1}^n{\theta_j^2}" />
</p>

要最小化该代价函数，求导得梯度下降算法：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}{\theta_0}&:={\theta_0}-\frac{\alpha}{m}\sum\limits_{i=1}^{m}{({h_\theta}({x^{(i)}})-{y^{(i)}})x_0^{(i)}}\\&space;{\theta_j}&:={\theta_j}-\frac{\alpha}{m}\left[\sum\limits_{i=1}^{m}{({h_\theta}({x^{(i)}})-{y^{(i)}})x_{j}^{\left(i\right)}}&plus;\lambda{\theta_j}\right]\text{&space;for&space;}j=1,2,...,n\end{align*}" title="\begin{align*}{\theta_0}&:={\theta_0}-\frac{\alpha}{m}\sum\limits_{i=1}^{m}{({h_\theta}({x^{(i)}})-{y^{(i)}})x_0^{(i)}}\\ {\theta_j}&:={\theta_j}-\frac{\alpha}{m}\left[\sum\limits_{i=1}^{m}{({h_\theta}({x^{(i)}})-{y^{(i)}})x_{j}^{\left(i\right)}}+\lambda{\theta_j}\right]\text{ for }j=1,2,...,n\end{align*}" />
</p>

注：看上去同线性回归一样，但是知道 _h<sub>θ</sub>(x)=g(θ<sup>T</sup>X)_ ，所以与线性回归不同。


## Jupyter Notebook编程练习
* 推荐访问Google Drive的共享，直接在Google Colab在线运行ipynb文件：
  * [Google Drive: 2.logistic_regression](https://drive.google.com/drive/folders/1rsDEXuYeGnFnWiR2d1NKHRJTLRN24XYg)
* 不能翻墙的朋友，可以访问GitHub下载：
  * [GitHub: 2.logistic_regression](https://github.com/loveunk/ml-ipynb/tree/master/2.logistic_regression)

[回到顶部](#逻辑回归)
