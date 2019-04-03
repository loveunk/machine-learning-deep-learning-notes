# 线性回归
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [线性回归](#线性回归)
	- [单变量线性回归 (Linear Regression with One Variable)](#单变量线性回归-linear-regression-with-one-variable)
		- [模型表示](#模型表示)
		- [代价函数](#代价函数)
		- [梯度下降](#梯度下降)
		- [梯度下降的直观理解](#梯度下降的直观理解)
		- [梯度下降的线性回归](#梯度下降的线性回归)
	- [多变量线性回归 (Linear Regression with Multiple Variables)](#多变量线性回归-linear-regression-with-multiple-variables)
		- [多维特征](#多维特征)
		- [多变量梯度下降](#多变量梯度下降)
			- [梯度下降法实践1：特征缩放](#梯度下降法实践1特征缩放)
				- [数据的标准化 (Normalization)](#数据的标准化-normalization)
			- [梯度下降法实践2：学习率 (Learning Rate)](#梯度下降法实践2学习率-learning-rate)
		- [特征和多项式回归](#特征和多项式回归)
	- [正规方程 Normal Equations](#正规方程-normal-equations)
		- [对比梯度下降和正规方程](#对比梯度下降和正规方程)
			- [正规方程及不可逆性](#正规方程及不可逆性)
	- [Jupter Notebook编程练习](#jupter-notebook编程练习)

<!-- /TOC -->
## 单变量线性回归 (Linear Regression with One Variable)
### 模型表示
**例子**：预测住房价格

**数据集**：已知一个数据集，包含某个城市的住房价格。每个样本包括房屋尺寸、售价。<br/>
**问题**：根据不同房屋尺寸所售出的价格，画出数据集。如果房子是1250平方尺，你要告诉他们这房子能卖多少钱。

首先，你可以构建一个模型（假设是条直线，如下图）。根据数据模型，你可以告诉你的朋友，他的房子大约值220000美元。
<p align="center">
<img src="https://raw.githubusercontent.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/8e76e65ca7098b74a2e9bc8e9577adfc.png" />
</p>

上述例子是个**监督学习**的例子，同时是一个**回归问题**。**回归**指根据之前的数据预测出一个准确的输出值，对于这个例子预测的值是价格。

这个数据集可以表示为下表：

| 房屋大小 (_x_) | 价格 (_y_) |
| :---:         |     :---:      |
| 2104 | 460 |
| 1416 | 232 |
| 1534 | 315 |
| 852  | 178 |
| ...  | ... |

我们用如下的符号来描述这个问题：
* _m_： 代表训练集中样本的数量（下文也将用 _m_ 表示训练样本数量）
* _x_： 代表特征/输入变量
* _y_： 代表目标变量/输出变量
* (_x, y_)： 代表训练集中的一个样本
* (_x<sup>(i)</sup>, y<sup>(i)</sup>_)：代表第 _i_ 个观察样本
* _h_：代表学习算法的解决方案或函数也称为假设（hypothesis）

这个监督学习的工作方式如下：
```
训练集 → 学习算法
            ↓
房屋大小  →  h  → 预测价格
```

上述步骤总结为：
1. 把训练集（房屋大小和价格）输入到学习算法；
2. 学习算法计算出函数 _h_。函数输入是房屋大小 (_x_)，输出 _y_ 值对应房子的价格，因此 _h_ 是一个从 _x_ 到 _y_ 的函数映射；
3. 对于新要预测的样本 _x_，往 _h_ 输入 _x_ 值可得对应的 _y_ 值。

那么，对于我们的房价预测问题， _h_ 可能的表述如下：
* <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;h_\theta&space;\left(&space;x&space;\right)=\theta_{0}&space;&plus;&space;\theta_{1}x" title="h_\theta \left( x \right)=\theta_{0} + \theta_{1}x" />，因为只含有一个特征/输入变量，因此这样的问题叫作单变量线性回归问题。

下一步是如何确定参数 _θ<sub>0</sub>_ 和 _θ<sub>1</sub>_。在预测房价这个例子， _θ<sub>1</sub>_ 是直线的斜率，  _θ<sub>0</sub>_ 是在 _y_ 轴上的截距。

选择的参数决定了 _h_ 相对与训练集的准确程度。

定义**建模误差**（modeling error）为模型所预测的值与训练集中实际值之间的差距（下图中蓝线所指）。
<p align="center">
<img src="https://raw.githubusercontent.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/6168b654649a0537c67df6f2454dc9ba.png" />
</p>

### 代价函数
为衡量 _h_ 的性能，回归任务中常见的方法是定义代价函数（Cost Function）：
* 均方误差（MES: Mean Squared Error）
  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?J(\theta_0,\theta_1)&space;=&space;\frac{1}{2m}\sum\limits_{i=1}^m&space;\left(&space;h_{\theta}(x^{(i)})-y^{(i)}&space;\right)^{2}" title="J(\theta_0,\theta_1) = \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}" />
  </p>
* 选取参数以最小化 _J_，从而优化 _h_

我们绘制一个等高线图，三个坐标分别为 _θ<sub>0</sub>_ 和 _θ<sub>1</sub>_、_J(θ<sub>0</sub>, θ<sub>1</sub>)_，可以看到在三维空间中存在一个使得 _J(θ<sub>0</sub>, θ<sub>1</sub>)_ 最小的点：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/27ee0db04705fb20fab4574bb03064ab.png" />
</p>

下右图是把代价函数呈现为等高线图（Contour Plot），以便我们观察 _θ<sub>0</sub>_ 和 _θ<sub>1</sub>_ 对 _J(θ<sub>0</sub>, θ<sub>1</sub>)_ 的影响。
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/86c827fe0978ebdd608505cd45feb774.png" />
</p>

根据上图，人工的方法很容易找到 代价函数最小值时对应的 _θ<sub>0</sub>_ 和 _θ<sub>1</sub>_，但我们真正需要的是一种有效的算法，能够自动地找出这些使代价函数 _J_ 取最小值的参数  _θ<sub>0</sub>_ 和 _θ<sub>1</sub>_。也就是下面要降到的[梯度下降](#梯度下降)。

### 梯度下降
**梯度下降**是一种用来求函数最小值的算法，我们将使用梯度下降算法来求出代价函数 _J(θ<sub>0</sub>, θ<sub>1</sub>)_ 的最小值。

**梯度下降的思想**：开始时随机选择一个参数的组合 _(θ<sub>0</sub>, θ<sub>1</sub>, ..., θ<sub>n</sub>)_ ，计算代价函数；然后寻找下一个能让代价函数值下降最多的参数组合。持续这么做直到找到一个**局部最小值**（Local minimum），因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是**全局最小值**（Global minimum），选择不同的初始参数组合，可能会找到不同的局部最小值。

如下图所示，不同的起始参数导致了不同的局部最小值。
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/db48c81304317847870d486ba5bb2015.jpg" />
</p>

为了理解梯度下降，可以想象一下你正站立在山的一点上（上图中的红色起始点），并且希望用最短的时间下山。在梯度下降算法中，要做的就是旋转360度，看看周围，并问自己要在某个方向上，用小碎步尽快下山。这些小碎步需要朝什么方向？如果我们站在山坡上的这一点，看一下周围，你会发现最佳的下山方向，按照自己的判断迈出一步；重复上面的步骤，从新的位置，环顾四周，并决定从什么方向将会最快下山，然后又迈进了一小步，并依此类推，直到你接近局部最低点的位置。

批量梯度下降（batch gradient descent）算法可以抽象为公式：

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/7da5a5f635b1eb552618556f1b4aac1a.png" />
</p>

其中 _α_ 是**学习率**（Learning rate），它决定了沿着能让代价函数下降程度最大的方向向下迈出的步子有多大；在批量梯度下降中，每一次都同时让所有的参数减去学习速率乘以代价函数的导数。

上面的公式展开如下：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/ef4227864e3cabb9a3938386f857e938.png" />
</p>

**重点**：更新上述式子需要同时更新一组参数 _(θ<sub>0</sub>, θ<sub>1</sub>, ..., θ<sub>n</sub>)_ ，之后再开始下一轮迭代。 这里先不解释为什么需要同时更新。但请记住，同时更新是梯度下降中常用方法。之后会讲到，同步更新也是更自然的实现方法。人们谈到梯度下降时，意思就是同步更新。

### 梯度下降的直观理解
考虑上图中关于梯度下降的公式，其中求导，是取红点的切线，就是下图中红色的直线，其与函数相切于红色的点。红色直线的斜率，正好是下图红色三角形的高度除以这个水平长度，这条线有一个正斜率，也就是说它有正导数，因此，为了得到更新的 _J_，_θ<sub>1</sub>_ 更新后等于 _θ<sub>1</sub>_ 减去一个正数乘以 _α_。

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/ee916631a9f386e43ef47efafeb65b0f.png" />
</p>

 _α_ 的取值有怎么的影响？
* 如果 _α_ 太小，即学习速率太小，结果是红点一点点挪动，努力去接近最低点，需要很多步才能到达最低点。同样会需要很多步才能到达全局最低点。（如下图-左图）
* 如果 _α_ 太大，梯度下降法可能会越过最低点，甚至可能无法收敛，下一次迭代又移动了一大步，一次次越过最低点，直到你发现实际上离最低点越来越远，所以如果 _α_ 太大，它会导致无法收敛，甚至发散。（如下图-右图）

<p align="center">
<img src="img/gradient-descent-learning-rate-alpha-effect.png" />
</p>

现在，还有一个问题，需要思考，如果我们预先把 _θ<sub>1</sub>_ 放在一个局部的最低点，下一步梯度下降法会怎样工作？

假设你将 _θ<sub>1</sub>_ 初始化在局部最低点，它已经在一个局部的最优处或局部最低点。结果是局部最优点的导数将等于零，因为它是那条切线的斜率。使得 _θ<sub>1</sub>_ 不再改变，也就是新的 _θ<sub>1</sub>_ 等于原来的 _θ<sub>1</sub>_ ，因此，如果参数已经处于局部最低点，那么梯度下降法更新其实什么都没做，它不会改变参数的值。这也解释了为什么即使学习速率 _α_ 保持不变时，梯度下降也可以收敛到局部最低点。

我们再来看下代价函数 _J(θ)_ ，如下图

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/4668349e04cf0c4489865e133d112e98.png" />
</p>

随着接近最低点，导数越来越接近零，所以，梯度下降一步后，新的导数会变小一点点。再梯度下降一步，在这个绿点，会用一个稍微跟刚才在那个品红点时比，再小一点的一步，到了新的红色点，更接近全局最低点了，因此这点的导数会比在绿点时更小。所以，再进行一步梯度下降时，导数项是更小的， _θ<sub>1</sub>_ 更新的幅度就会更小。所以随着梯度下降法的运行，移动的幅度会自动变得越来越小，直到最终移动幅度非常小，这时已经收敛到局部极小值。

总结一下：
* 在梯度下降法中，当接近局部最低点时，梯度下降法会自动采取更小的幅度，这是因为当接近局部最低点时，导数值会自动变得越来越小，梯度下降将自动采取较小的幅度。所以实际上没有必要再另外减小 _α_。
* 你可以用它来最小化任何代价函数 _J(θ)_ ，不只是线性回归中的代价函数 _J(θ)_。

### 梯度下降的线性回归
这一节介绍如何将梯度下降和代价函数结合。

梯度下降算法和线性回归算法比较如图：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/5eb364cc5732428c695e2aa90138b01b.png" />
</p>

回顾一下之前的线性回归问题的代价函数：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J(\theta_0,\theta_1)&space;=&space;\frac{1}{2m}\sum\limits_{i=1}^m&space;\left(&space;h_{\theta}(x^{(i)})-y^{(i)}&space;\right)^{2}" title="J(\theta_0,\theta_1) = \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}" />
</p>

其偏导数为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;}{\partial&space;{{\theta_j}}}J(\theta_0,\theta_1)&space;=&space;\frac{\partial&space;}{\partial&space;{{\theta_j}}}&space;\frac{1}{2m}\sum\limits_{i=1}^m&space;\left(&space;h_{\theta}(x^{(i)})-y^{(i)}&space;\right)^{2}" title="\frac{\partial }{\partial {{\theta_j}}}J(\theta_0,\theta_1) = \frac{\partial }{\partial {{\theta_j}}} \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}" />
</p>

* 当 _j = 0_ 时：
  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;}{\partial&space;{{\theta_0}}}J(\theta_0,\theta_1)&space;=&space;\frac{\partial&space;}{\partial&space;{{\theta_0}}}&space;\frac{1}{2m}\sum\limits_{i=1}^m&space;\left(&space;h_{\theta}(x^{(i)})-y^{(i)}&space;\right)" title="\frac{\partial }{\partial {{\theta_0}}}J(\theta_0,\theta_1) = \frac{\partial }{\partial {{\theta_0}}} \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)" />
  </p>
* 当 _j = 1_ 时：
  <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;}{\partial&space;{{\theta_1}}}J(\theta_0,\theta_1)&space;=&space;\frac{\partial&space;}{\partial&space;{{\theta_1}}}&space;\frac{1}{2m}\sum\limits_{i=1}^m&space;\left(&space;h_{\theta}(x^{(i)})-y^{(i)}&space;\right)x^{(i)}" title="\frac{\partial }{\partial {{\theta_1}}}J(\theta_0,\theta_1) = \frac{\partial }{\partial {{\theta_1}}} \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right) x^{(i)}" />
  </p>

所以算法可以写为：

Repeat {
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;{\theta_{0}}&:={\theta_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{&space;\left({{h}_{\theta&space;}}({{x}^{(i)}})-{{y}^{(i)}}&space;\right)}\\&space;{\theta_{1}}&:={\theta_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{&space;\left({{h}_{\theta&space;}}({{x}^{(i)}})-{{y}^{(i)}}&space;\right)\cdot&space;{{x}^{(i)}}}&space;\end{aligned}" title="\begin{aligned} {\theta_{0}}&:={\theta_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{ \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}\\ {\theta_{1}}&:={\theta_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{ \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}}} \end{aligned}" />
</p>
}

上述算法有时也称为 **批量梯度下降（Batch Gradient Descent）**。**“批量”是**指在梯度下降的每一步中，我们都用到了所有的训练样本。
* 在梯度下降，计算微分求导项时，需要求和运算，所以，在每一个单独的梯度下降中，最终都要计算这样一个东西，这个项需要对所有 _m_ 个训练样本求和。
* 批量梯度下降法这个名字说明需要考虑所有这一"批"训练样本。事实上，也有其他类型的梯度下降法，不是"批量"型的，不考虑整个的训练集，每次只关注训练集中的一些小的子集。后续会介绍。

此外，也许你知道有一种计算代价函数最小值的数值解法，不需要梯度下降这种迭代算法。在后面我们也会谈到这个方法，可以在不需要多步梯度下降的情况下，解出代价函数的最小值，这中方法称为**正规方程(normal equations)**。实际上在数据量较大的情况下，梯度下降法比正规方程要更适用一些。

## 多变量线性回归 (Linear Regression with Multiple Variables)
### 多维特征
在之前的房价预测问题里，只考虑到房屋尺寸一个特征，这里我们考虑多个特征的问题。比如在房价预测问题中，引入房间数、楼层、年限等。

下表是一个示例数据：

| 房屋大小 | 房间数 | 楼层 | 年限 | 价格 (_y_) |
|:--------:|:------:|:----:|:----:|:----------:|
|   2104   |    5   |   1  |  45  |     460    |
|   1416   |    3   |   2  |  40  |     232    |
|   1534   |    3   |   2  |  30  |     315    |
|    852   |    2   |   1  |  36  |     178    |
|    ...   |   ...  |  ... |  ... |     ...    |

介绍更多的问题描述符号：
* _n_：特征的数量
* _x<sup>(i)</sup>_：第i个训练样本。如果样本用矩阵表示，那它对应就是矩阵的第 _i_ 行，也是一个向量。比如 _x<sup>(2)</sup> = [1416; 3; 2; 40; 232]_。
* _x<sub>j</sub><sup>(i)</sup>_：训练样本中的第 _i_ 个hang本中的第 _j_ 个特征，也就是矩阵里的 _i_ 行的 _j_ 列。

多变量的假设 $h$ 表示为：<img src="https://latex.codecogs.com/gif.latex?h_{\theta}\left(&space;x&space;\right)={\theta_{0}}&plus;{\theta_{1}}{x_{1}}&plus;{\theta_{2}}{x_{2}}&plus;...&plus;{\theta_{n}}{x_{n}}" title="h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}" />

上述公式中有 _n+1_ 个参数和 _n_ 个变量，为了简化公式，引入 _x<sub>0</sub> = 1_，则上式写作：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;h_{\theta}&space;\left(&space;x&space;\right)&={\theta_{0}}{x_{0}}&plus;{\theta_{1}}{x_{1}}&plus;{\theta_{2}}{x_{2}}&plus;...&plus;{\theta_{n}}{x_{n}}\\&space;&={\theta^{T}}X&space;\end{aligned}" title="\begin{aligned} h_{\theta} \left( x \right)&={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}\\ &={\theta^{T}}X \end{aligned}" />
</p>

其中，_T_ 代表矩阵转置

在多变量线性回归中，我们也构建一个代价函数，则这个代价函数是所有建模误差的平方和，即：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J\left({\theta_{0}},{\theta_{1}}...{\theta_{n}}\right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left(h_{\theta}\left({x}^{\left(i\right)}\right)-{y}^{\left(i\right)}\right)}^{2}}}" title="J\left({\theta_{0}},{\theta_{1}}...{\theta_{n}}\right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left(h_{\theta}\left({x}^{\left(i\right)}\right)-{y}^{\left(i\right)}\right)}^{2}}}" />
</p>

计算代价函数的Python代码如下：
``` python
def compute_cost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
```

### 多变量梯度下降
多变量梯度下降的目标和单变量线性回归问题中一样，要找出使得代价函数最小的一系列参数。多变量线性回归的批量梯度下降算法为：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/41797ceb7293b838a3125ba945624cf6.png" />
</p>

求导后得到：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/dd33179ceccbd8b0b59a5ae698847049.png" />
</p>

可以看出，有 _n_ 个特诊的梯度下降算法和算法 单特征的梯度下降算法的区别是 _θ_ 变量的个数及在每一步更新 _θ_ 变量的个数。

#### 梯度下降法实践1：特征缩放
**在面对多维特征问题时，要保证这些特征都具有相近的尺度，将使梯度下降算法更快地收敛。**

以房价问题为例，假如有两个特征，房屋尺寸和房间数量。尺寸值为 0-2000平方英尺，房间数量取值0-5，以两个参数分别为横纵坐标，绘制代价函数的等高线图能，看出图像会显得很扁，梯度下降算法需要非常多次的迭代才能收敛。如下图：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/966e5a9b00687678374b8221fdd33475.jpg" />
</p>

解决的方法是尝试将所有特征的尺度都尽量缩放到-1到1之间。如图右图：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/b8167ff0926046e112acf789dba98057.png" />
</p>

##### 数据的标准化 (Normalization)
对于X数据的标准化公式：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?x_n=\frac{x_n-\mu_n}{s_n}" title="x_n=\frac{x_n-\mu_n}{s_n}" />
</p>

其中 _μ<sub>n</sub>_ 是平均值，_s<sub>n</sub>_ 是标准差。Python示例代码如下：
``` python
import numpy as np
X = np.random.rand(5, 2)

mu = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = X - mu
X = X / std
```

#### 梯度下降法实践2：学习率 (Learning Rate)
梯度下降算法收敛所需要的迭代次数根据模型的不同而不同，虽不能提前预知，但可以画出迭代次数和代价函数 _J_ 的图表来观测算法在何时趋于收敛。如下图所示，可以看到 _J_ 是随着迭代次数增加而不断的减小。当迭代次数达到300之后， _J_ 降低的趋势已经非常小了，说明已经收敛。

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/cd4e3df45c34f6a8e2bb7cd3a2849e6c.jpg" />
</p>

也有一些自动测试是否收敛的方法，例如将代价函数的变化值与某个阀值（例如0.001）进行比较，如果比阈值小，就认为已经收敛。但通常看上面这样的图表更好。

梯度下降算法的迭代受到学习率 _α_ 影响：
* 如果学习率 _α_ 过小，则达到收敛所需的迭代次数会非常高；
* 如果学习率 _α_ 过大，每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛。

通常可以考虑尝试些学习率： _α = 0.01，0.03，0.1，0.3，1，3，10_

最后，大的原则是，**有效的  _α_ 是可以让 _J_ 随着迭代不断变小。但太小的 _α_ 会导致收敛的很慢**。

### 特征和多项式回归
仍然是以房价预测为例：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/8ffaa10ae1138f1873bc65e1e3657bd4.png" />
</p>

按照此前的线性回归模型可得：
_h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub> × frontage + θ<sub>2</sub> × depth_

定义：
* _x<sub>1</sub> = frontage_（临街宽度）
* _x<sub>2</sub> = depth_（纵向深度）
* _x = frontage*depth = area_（面积）

则：_h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x_。

线性回归并不适用于所有数据，有时我们需要曲线来适应我们的数据，比如一个二次方模型：_h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sup>2</sup>_ 或者三次方模型： _h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sup>2</sup> + θ<sub>3</sub>x<sup>3</sup>_。如下图所示：

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/3a47e15258012b06b34d4e05fb3af2cf.jpg" />
</p>

通常我们需要先观察数据然后再决定模型的类型。此外，可以令 _x2 = x<sup>2</sup>_，_x3 = x<sup>3</sup>_，从而将多项式回归转换为线性回归。

## 正规方程 Normal Equations
我们都在使用梯度下降算法，但是对于某些线性回归问题，正规方程方法是更好的解决方案。如：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/a47ec797d8a9c331e02ed90bca48a24b.png" />
</p>

正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：
<img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial}{\partial{\theta_{j}}}J\left(&space;{\theta_{j}}&space;\right)=0" title="\frac{\partial}{\partial{\theta_{j}}}J\left( {\theta_{j}} \right)=0" />
。 假设我们的训练集特征矩阵为 _X_（包含了 _x<sub>0</sub> = 1_）并且我们的训练集结果为向量 _y_，则利用正规方程解出向量：
<p align="center">
	<i>θ = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y</i>
</p>

推导如下：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\inline&space;\begin{aligned}J&=\left(&space;y-x\theta&space;\right)&space;^{T}\left(&space;y-X\theta&space;\right)&space;\\&space;\dfrac{\partial&space;J}{\partial&space;\theta}&=2x^{T}\left(&space;y-x\theta&space;\right)&space;\\&space;\dfrac{\partial&space;J}{\partial&space;\theta}&=0\Leftrightarrow&space;X^{T}X\theta&space;=x^{T}y\\&space;&\Rightarrow&space;\theta&space;=\left(&space;X^{T}X\right)&space;^{-1}X^{T}y&space;\end{aligned}" title="\begin{aligned}J&=\left( y-x\theta \right) ^{T}\left( y-X\theta \right) \\ \dfrac{\partial J}{\partial \theta}&=2x^{T}\left( y-x\theta \right) \\ \dfrac{\partial J}{\partial \theta}&=0\Leftrightarrow X^{T}X\theta =x^{T}y\\ &\Rightarrow \theta =\left( X^{T}X\right) ^{-1}X^{T}y \end{aligned}" />
</p>

举个例子：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/261a11d6bce6690121f26ee369b9e9d1.png" />
</p>

运用正规方程方法求解参数：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/b62d24a1f709496a6d7c65f87464e911.jpg" />
</p>

上面的正规方程用Python的实现如下：
``` python
import numpy as np

def normal_equation(X, y):
	theta = np.linalg.pinv(X.T @ X) @ X.T @ y
```

**注意**：对于不可逆的矩阵 _X_（通常因为特征之间不独立，如同时包含英尺为单位的尺寸和米为单位的尺寸两个特征，也有可能是特征数量大于训练集的数量），正规方程方法是不能用。

### 对比梯度下降和正规方程
| 梯度下降             | 正规方程                                     |
| ---------------- | ---------------------------------------- |
| 需要选择学习率 _α_  | 不需要                                      |
| 需要多次迭代           | 一次运算得出                                   |
| 当特征数量 _n_ 很大时也适用 | 需要计算 _(X<sup>T</sup>X)<sup>-1_ 如果特征数量n较大则运算代价大，因为矩阵逆的计算时间复杂度为 _O(n<sup>3</sup>)_，通常来说当 _n_ 小于10000 时还是可以接受 |
| 适用于各种类型的模型       | 只适用于线性模型，不适合逻辑回归模型等其他模型                  |

#### 正规方程及不可逆性
我们称那些不可逆矩阵为奇异或退化矩阵。有两种情况可能导致矩阵的不可逆：
* 在 _m_ 小于或等于 _n_
	* 我们会使用一种叫做正则化（Regularization）的线性代数方法删除某些特征
* 特征之间线性相关：
	* 可以删除这两个重复特征里的其中一个

## Jupyter Notebook编程练习
* 推荐访问Google Drive的共享，直接在Google Colab在线运行ipynb文件：
	* [Google Drive: 1.linear_regression](https://drive.google.com/open?id=1VzVxUSOwRYJogJJpdt4EZNuVtJCJZ7FF)
* 不能翻墙的朋友，可以访问GitHub下载：
	* [GitHub: 1.linear_regression](https://github.com/loveunk/ml-ipynb/tree/master/1.linear_regression)

[回到顶部](#线性回归)
