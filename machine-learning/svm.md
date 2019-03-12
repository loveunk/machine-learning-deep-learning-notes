# 支持向量机

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [支持向量机](#支持向量机)
	- [优化目标](#优化目标)
	- [大边界](#大边界)
	- [大边界分类背后的数学](#大边界分类背后的数学)
	- [核函数](#核函数)
	- [使用SVM](#使用svm)
	- [什么时候使用SVM](#什么时候使用svm)

<!-- /TOC -->

支持向量机（Support Vector Machine）是一个广泛应用于工业界和学术界的算法。
与逻辑回归和神经网络相比，SVM在学习复杂非线性方程时提供了一种更为清晰，更加强大的方式。

## 优化目标
为了描述SVM，我们先从逻辑回归开始展示如何一点一点修改来得到SVM：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/3d12b07f13a976e916d0c707fd03153c.png" />
</p>

在逻辑回归中我们已经熟悉了这里的假设函数形式，和右边的S型激励函数。
然而，为了解释一些数学知识。将用 _z_ 表示 _θ<sup>T</sup>x_ 。

看看逻辑回归做什么：
1. 对一个 _y=1_ 的样本，希望 _hθ(x)_ 趋近1。
    * 因为想要正确地将此样本分类，这就意味着当 _hθ(x)_ 趋近于1时， _θ<sup>T</sup>x >> 0_ 。
    * 因为由于 _z_ 表示 _θ<sup>T</sup>x_ ，当 _z >> 0_ ，即到了上图右图，逻辑回归的输出将趋近于1。
2. 对一个 _y=0_ 的样本。希望 _hθ(x)_ 趋近0。
    * 这对应于 _θ<sup>T</sup>x << 0_ ，或者就是 _z << 0_，因为对应的假设函数的输出值趋近0。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/66facb7fa8eddc3a860e420588c981d5.png" />
</p>

对于逻辑回归的代价函数，考虑两种情况：
1. 是 _y_ 等于1的情况
   * _y = - log((1/(1 + e<sup>-z</sup>)))​_
2. 是 _y_ 等于0的情况
   * _y = - (1-y) log(1- (1/(1 + e<sup>-z</sup>)))​_

**现在开始建立SVM，我们从这里开始：**

对代价函数 _-log(1-((1)/(1+e<sup>-z</sup>)))_ 做一点修改（如上图中紫色的曲线）。
由两条线段组成，即位于右边的水平部分和位于左边的直线部分。
* 左边的函数称为 _cost<sub>1</sub>(z)_
* 右边的函数称为 _cost<sub>0</sub>(z)_ 。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/59541ab1fda4f92d6f1b508c8e29ab1c.png" />
</p>

因此，对于SVM，我们得到了这里的最小化问题，即:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min_{\theta}C\sum&space;^{m}_{i=1}\left[&space;y^{(i)}cost_{1}\left(&space;\theta&space;^{T}x^{(i)}\right)&space;&plus;&space;(1-y^{(i)})cost_{1}\left(&space;\theta&space;^{T}x^{(i)}\right)\right]&space;&plus;\dfrac&space;{1}{2}\sum&space;^{n}_{i=1}\theta&space;^{2}_{j}" title="\min_{\theta}C\sum ^{m}_{i=1}\left[ y^{(i)}cost_{1}\left( \theta ^{T}x^{(i)}\right) + (1-y^{(i)})cost_{1}\left( \theta ^{T}x^{(i)}\right)\right] +\dfrac {1}{2}\sum ^{n}_{i=1}\theta ^{2}_{j}" />
</p>

和逻辑回归相比，有几点不同：
1. SVM代价函数公式里多了 _C_，少了 _λ/m_。如果把 _C_ 当做是 _1/λ_，实际是优化目标等效的。因为对于一个最小化问题，在目标公式上乘除一个常量或加减一个常量都不影响求最优解。
2. 当获得参数 _θ_ 时，直接用 _θ<sup>T</sup>x_ 预测 _y_ 的值。
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?h_\theta(x)=\begin{cases}1&space;&\text{&space;if&space;}&space;\theta^Tx\ge0\\0&space;&\text{otherwise&space;}\end{cases}" title="h_\theta(x)=\begin{cases}1 &\text{ if } \theta^Tx\ge0\\0 &\text{otherwise }\end{cases}" />
</p>

## 大边界
从直观的角度看看优化目标，实际上是在做什么，以及SVM的假设函数将会学习什么，同时也会谈谈如何做些许修改，学习更加复杂、非线性的函数。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/cc66af7cbd88183efc07c8ddf09cbc73.png" />
</p>

这是SVM模型的代价函数，横轴表示 _z_ ：
* 左边显示关于 _z_ 的代价函数 _cost<sub>1</sub>(z)_ ，用于正样本
* 右边显示关于 _z_ 的代价函数 _cost<sub>0</sub>(z)_ ，用于负样本

现在考虑一下，最小化这些代价函数的必要条件是什么。
* 对一个正样本 _y=1_ ，只有 _z >= 1_ 时，代价函数 _cost<sub>1</sub>(z) = 0_，等同希望 _θ<sup>T</sup>x >= 1_ ；
* 对一个负样本 _y=0_ ，只有 _z <= 1_ 时，代价函数 _cost<sub>0</sub>(z) = 0_，等同希望 _θ<sup>T</sup>x <= 1_

这是SVM的一个性质：
* 如果 _y = 1_ ，则仅要求 _θ<sup>T</sup>x >= 0_，就能将样本恰当分出，这是因为如果 _θ<sup>T</sup>x > 0_ ，模型代价函数值为0。
* 如果 _y = 0_，仅需要 _θ<sup>T</sup>x <= 0_ 就会将负例正确分离，

但是，**SVM的要求更高**，不仅要能正确分开样本。即
* 对于正例，不仅要求 _θ<sup>T</sup>x > 0_，需要的是比0值大很多，比如大于等于1；
* 对于负样本，不仅要求 _θ<sup>T</sup>x < 0_，还希望比0小很多，希望它小于等于-1

这就相当于在SVM中嵌入了一个额外的安全因子，或者说**安全的间距因子**。

让我们看一下，在SVM中，这个因子会导致什么结果。
具体而言，接下来考虑一个特例。将这个常数 _C_ 设置成一个非常大的值。比如我们假设 _C = 100000_，然后观察SVM会给出什么结果？

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/12ebd5973230e8fdf279ae09e187f437.png" />
</p>

如果 _C_ 非常大，则最小化代价函数的时候，我们将会很希望找到一个使第一项为0的最优解。因此，让我们尝试在代价项的第一项为0的情形下理解该优化问题。比如我们可以把$C$设置成了非常大的常数，这将给我们一些关于SVM模型的直观感受。
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min_{\theta}C\sum&space;^{m}_{i=1}\left[&space;y^{(i)}cost_{1}\left(&space;\theta&space;^{T}x^{(i)}\right)&space;&plus;&space;(1-y^{(i)})cost_{1}\left(&space;\theta&space;^{T}x^{(i)}\right)\right]&space;&plus;\dfrac&space;{1}{2}\sum&space;^{n}_{i=1}\theta&space;^{2}_{j}" title="\min_{\theta}C\sum ^{m}_{i=1}\left[ y^{(i)}cost_{1}\left( \theta ^{T}x^{(i)}\right) + (1-y^{(i)})cost_{1}\left( \theta ^{T}x^{(i)}\right)\right] +\dfrac {1}{2}\sum ^{n}_{i=1}\theta ^{2}_{j}" />
</p>


这将遵从以下的约束：
* _θ<sup>T</sup>x<sup>(i)</sup> >= 1_ ，如果 _y<sup>(i)</sup>_ 是等于1的
* _θ<sup>T</sup>x<sup>(i)</sup> <= -1_ ，如果样本 _i_ 是一个负样本

这样当你求解这个优化问题的时候，当你最小化这个关于变量 _θ_ 的函数的时候，你会得到一个非常有趣的决策边界。

---
具体而言，如果你考察这样一个数据集，其中有正样本，也有负样本，可以看到这个数据集是线性可分的。我的意思是，存在一条直线把正负样本分开。当然有多条不同的直线，可以把正样本和负样本完全分开。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/01105c3afd1315acf0577f8493137dcc.png" />
</p>

上图的例子，画了很多条决策边界，都可以将正样本和负样本分开。但绿色和红色的仅仅是勉强分开，看起来都不是特别好的选择。

而SVM将会选择这个黑色的决策边界，相较于之前我用粉色或者绿色画的决策界。
这条黑色的看起来好得多，黑线看起来是更稳健的决策界。在分离正样本和负样本上它显得的更好。

数学上来讲，这条黑线有更大的距离，这个距离叫做**间距（margin）**。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/e68e6ca3275f433330a7981971eb4f16.png" />
</p>

当画出这两条额外的蓝线，看到黑色的决策界和训练样本之间有更大的最短距离。
这个距离叫做SVM的间距，而这是SVM具有鲁棒性的原因，因为它努力用一个最大间距来分离样本。因此SVM有时被称为**大间距分类器**，而这其实是求解上面优化问题的结果。

---
上面的例子中将大间距分类器中的正则化因子常数 _C_ 设置的非常大（100000），因此对这样的一个数据集，也许我们将选择这样的决策界，从而最大间距地分离开正样本和负样本。那么在让代价函数最小化的过程中，我们希望找出在 _y=1_ 和 _y=0_ 两种情况下都使得代价函数中左边的这一项尽量为零的参数。如果我们找到了这样的参数，则我们的最小化问题便转变成：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/f4b6dee99cfb4352b3cac5287002e8de.png" />
</p>

事实上，SVM现在要比这个大间距分类器所体现得更成熟，尤其是当你使用大间距分类器的时候，你的学习算法会受异常点(outlier)的影响。比如我们加入一个额外的正样本。如下图：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/b8fbe2f6ac48897cf40497a2d034c691.png" />
</p>

为了将样本用最大间距分开，也许最终会得到一条上图粉色的决策界
它仅仅基于一个异常值，就将决策界从黑线变到粉线，显然不合理。
但如果 _C_ 设置的小一点，则最终仍会得到这条黑线。

_C_ 的作用类似于 _1/λ_ ， _λ_ 是我们之前使用过的正则化参数。这只是 _C_ 非常大的情形，或者等价地 _λ_ 非常小的情形。
当 _C_ 不是非常非常大的时候，它可以忽略掉一些异常点的影响，得到更好的决策界。甚至当数据不是线性可分的时候，SVM也可以给出好的结果。


回顾 _C = <sup>1</sup>/<sub>λ</sub>_ ，因此：
* _C_ 较大时，相当于 _λ_ 较小，可能会导致过拟合，高方差。
* _C_ 较小时，相当于 _λ_ 较大，可能会导致低拟合，高偏差。

## 大边界分类背后的数学
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/55e05845c636c8c99c03e6e29337d8c4.png" />
</p>

先复习一下向量内积：假设有两个向量， _u_ 和 _v_ ，两个都是二维向量， _u<sup>T</sup>v_ 的结果 _u<sup>T</sup>v_ 也叫做向量 _u_ 和 _v_ 之间的内积。

向量 _u_
* 在横轴上，取值为 _u<sub>1</sub>_ ；在纵轴上，取值为 _u<sub>2</sub>_。
* _‖u‖_ 表示 _u_ 的范数，即 _u_ 的长度，也是向量 _u_ 的欧几里得长度。
* _‖u‖=sqrt(u<sub>1</sub><sup>2</sup>+u<sub>2</sub><sup>2</sup>)_ ，这是向量 _u_ 的长度，是一个实数。

向量 _v_
* _v_ 是另一个向量，它的两个分量 _v1_ 和 _v2_ 是已知的

_u_ 和 _v_ 之间的内积
* 计算内积方法1：
  * 将向量 _v_ 投影到向量 _u_ 上，我们做一个直角投影，或者说一个90度投影将其投影到 _u_ 上，接下来我度量这条红线的长度。
称这条红线的长度为 _p_ ，因此 _p_ 就是长度，或者说是向量 _v_ 投影到向量 _u_ 上的量，我将它写下来， _p_ 是 _v_ 投影到向量 _u_ 上的长度，因此可以将 _u<sup>T</sup>v=p·‖u‖_ ，或者说 _u_ 的长度。

* 计算内积的方法2：
  * _u<sup>T</sup>v_ 是 _[u1 u2]_ 这个一行两列的矩阵乘以 _v_ 。因此可以得到 _u1×v1 + u2×v2_ 。
  * 其中 _u<sup>T</sup>v = v<sup>T</sup>u_
  * 等式中 _u_ 的范数是一个实数， _p_ 也是一个实数，因此 _u<sup>T</sup>v_ 就是两个实数正常相乘。
    * 如果 _u_ 和 _v_ 间的夹角小于90度，那么 _p_ 是正值。
    * 如果这个夹角大于90度，则 _p_ 是负值。

根据线性代数的知识，方法2和方法1会给出同样的结果：
* _u<sup>T</sup>v = v<sup>T</sup>u = p·‖u‖_

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/44ad37bce4b7e03835095dccbd2a7b7a.png" />
</p>
---
根据此前讨论，下图列出SVM的目标函数：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/03bd4b3ff69e327f7949c3d2a73eed8a.png" />
</p>

先忽略掉截距，令 _θ<sub>0</sub>=0_ ，这样更容易画示意图。
将特征数 _n_ 置为2，仅有两个特征 _x<sub>1</sub>,x<sub>2</sub>_

看一下SVM的优化目标函数。当仅有两个特征，即 _n=2_ 时，需要最小化 _((1)/(2))(θ<sub>1</sub><sup>2</sup>+θ<sub>2</sub><sup>2</sup>)_ 。

现在我将要看看这些项： _θ<sup>T</sup>x_ 更深入地理解它们的含义。
给定参数向量 _θ_ 给定一个样本 _x_ ，这等于什么呢?

_θ_ 和 _x<sup>(i)</sup>_ 就类似于 _u_ 和 _v_ 的向量，如下面的示意图：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/4510b8fbc90ba2b233bb6996529f2df1.png" />
</p>

上面的示意图里，用一红叉表示正样本 _x<sup>(i)</sup>​_
*（水平轴上取值为 _x<sub>1</sub><sup>(i)</sup>​_ ，在竖直轴上取值为 _x<sub>2</sub><sup>(i)</sup>​_）。

将 _θ<sub>1</sub>​_ 画在横轴这里，将 _θ<sub>2</sub>​_ 画在纵轴这里，那么内积 _θ<sup>T</sup>x<sup>(i)</sup>_

使用之前的方法，计算的方式是我将训练样本投影到参数向量 _θ_ ，然后看这个线段的长度，图中红色。
将它称为 _p<sup>(i)</sup>_ 用来表示这是第 _i_ 个训练样本在参数向量 _θ_ 上的投影。根据之前的内容，知道
* _θ<sup>T</sup>x<sup>(i)</sup> = p × ‖θ‖_，也等于
* _θ<sub>1</sub> · x<sub>1</sub><sup>(i)</sup>+θ<sub>2</sub> · x<sub>2</sub><sup>(i)</sup>_ 。

这两种方式是等价的，都可以用来计算 _θ_ 和 _x<sup>(i)</sup>_ 之间的内积。

表达的意思是：
这个 _θ<sup>T</sup>x<sup>(i)</sup> >= 1_ 或者 _θ<sup>T</sup>x<sup>(i)</sup> < -1_ 的约束是可以被 _p<sup>(i)</sup> · ‖θ‖ >= 1_ 代替。

---
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/912cb43058cee46ddf51598b7538968c.png" />
</p>

现在考虑上图中的训练样本。继续使用之前的简化，即 _θ<sub>0</sub>=0_ ，来看下SVM会选择什么样的决策界。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/20725ba601c1c90d1024e50fc24c579b.png" />
</p>
上图左图中绿色直线可以是一种边界的选择。假设SVM选择了这个决策边界。当然这是不好的选择，因为它的间距很小。这个决策界离训练样本的距离很近。我们来看一下为什么SVM不会选择它。

对于这样选择的参数 _θ_ ，可以看到参数向量 _θ_ 事实上是和决策界是90度正交的，因此这个绿色的决策界对应着一个参数向量 _θ_ 。其中 _θ<sub>0</sub>=0_ 的简化仅仅意味着决策界必须通过原点 _(0,0)_ 。

比如这个样本，假设它是第一个样本 _x<sup>(1)</sup>_ ，如果考察这个样本到参数 _θ_ 的投影，投影是图中很短的红线段，所以 _p<sup>(1)</sup>_ 非常小。
类似地，这个样本如果它恰好是 _x<sup>(2)</sup>_ ，我的第二个训练样本，则它到 _θ_ 的投影是图中粉色段，同样线段很短。即第二个样本到我的参数向量 _θ_ 的投影。
注意的是，_p<sup>(2)</sup>_ 事实上是一个负值， _p<sup>(2)</sup>_ 在相反的方向，这个向量和参数向量 _θ_ 的夹角大于90度，所以 _p<sup>(2)</sup>_ 的值小于0。

发现这些 _p<sup>(i)</sup>_ 将会是非常小的数，因此当考察优化目标函数时，对正样本而言，需要 _p<sup>(i)</sup> · ‖θ‖ >= 1_ ，但如果 _p<sup>(i)</sup>_ 都非常小，意味着需要 _θ_ 的范数非常大。
因为如果 _p<sup>(1)</sup>_ 很小，而希望 _p<sup>(1)</sup> · ‖θ‖ >= 1_，令其实现的唯一的办法就是希望 _θ_ 的范数大。
类似地，对于负样本而言我们需要 _p<sup>(2)</sup> · ‖θ‖ <= -1_ 。我们已经在这个样本中看到 _p<sup>(2)</sup>_ 会是一个非常小的数，因此唯一的办法就是 _θ_ 的范数变大。

但是我们的目标函数是希望找到一个参数 _θ_ ，它的范数是小的。这是矛盾的，因此，这个决策边界对应的参数向量 _θ_ 不是个好的选择。

---
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/5eab58ad9cb54b3b6fda8f6c96efff24.png" />
</p>

相反的，看一个不同的决策边界。
比如说，SVM选择了上图右图的决策边界，那条绿色的垂直线，情况会很不同。
绿色的决策界有一个垂直于它的向量 _θ_ 。
如果考察你的数据在横轴 _x_ 上的投影，比如之前提到的样本，_x<sup>(1)</sup>_ ，当将它投影到横轴 _x_ 上，或说投影到 _θ_ 上，就会得到这样 _p<sup>(1)</sup>_ （右图红色线段）。
另一个样本，_x<sup>(2)</sup>_ 做同样的投影，得到 _p<sup>(2)</sup>_ 的长度是负值（右图粉色线段）。
注意到 _p<sup>(1)</sup>_ 和 _p<sup>(2)</sup>_ 投影长度长多了。
如果仍然要满足这些约束， _P<sup>(i)</sup> · ‖θ‖ > 1_，则因为 _p<sup>(1)</sup>_ 变大了， _‖θ‖_ 就可以变小了。

因此意味着通过选择右图中的决策界，而非左图那个，SVM可以使参数 _θ_ 的范数变小很多。也就是让代价函数变小。
因此，如果我们想令 _θ_ 的范数变小，就能让SVM选择右边的决策界。

以上，就是SVM如何能有效地产生大间距分类的原因。

推广到有用非常多个样本的训练集，同样目标是希望正样本和负样本投影到 _θ_ 上的 _p_ 值大。做到这点的唯一方式是选择类似右图绿线做决策界。这是大间距决策界来区分开正样本和负样本这个间距的值。间距值是 _p<sup>(1)</sup>,p<sup>(2)</sup>,p<sup>(3)</sup>_ 等等。
通过让间距变大，即通过 _p<sup>(1)</sup>,p<sup>(2)</sup>,p<sup>(3)</sup>_ 等等的值，SVM最终可以找到一个较小的 _θ_ 范数。
这正是SVM中最小化目标函数的目的。

**以上就是为什么SVM最终会找到大间距分类器的原因。因为它试图极大化这些 _p<sup>(i)</sup>_ 的绝对值，它们是训练样本到决策边界的距离。**

---
注意，之前的论述自始至终有一个简化假设，就是参数 _θ<sub>0</sub> = 0_ 。

就像之前提到的， _θ<sub>0</sub>=0_ 的意思是我们让决策界通过原点。
如果你令 _θ<sub>0</sub> != 0_，含义是希望决策界不通过原点。
这里不做全部的推导。实际上，SVM产生大间距分类器的结论证明同样成立，证明方式是非常类似的，是上述证明的推广。

即便 _θ<sub>0</sub>_ 不等于0，SVM要做的是优化目标函数对应着 _C_ 值非常大的情况，但是可以说明的是，即便 _θ<sub>0</sub> != 0_，SVM仍然会找到正样本和负样本之间的大间距分隔。

## 核函数

之前讨论过可以使用```高级数的多项式模型```来解决无法用```直线```进行分隔的分类问题，如下图的分类问题：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/529b6dbc07c9f39f5266bd0b3f628545.png" />
</p>

为了获得上图所示的判定边界，我们的模型可能是 _θ<sub>0</sub>+θ<sub>1</sub>x1+θ<sub>2</sub>x2+θ<sub>3</sub>x1x2+θ<sub>4</sub>x1<sup>2</sup>+θ<sub>5</sub>x2<sup>2</sup>+ ..._ 的形式。

我们可以用一系列的新的特征 _f_ 来替换模型中的每一项。例如令： _f<sub>1</sub> = x<sub>1</sub>, f<sub>2</sub> = x<sub>2</sub>, f<sub>3</sub> = x<sub>1</sub>x<sub>2</sub>, f<sub>4</sub> = x<sub>1</sub><sup>2</sup>, f<sub>5</sub> = x<sub>2</sub><sup>2</sup>_

...得到 _h<sub>θ</sub>(x) = θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + ... + θ<sub>n</sub>f<sub>n</sub>_ 。然而，除了对原有的特征进行组合以外，有没有更好的方法来构造 _f<sub>1</sub>, f<sub>2</sub>, f<sub>3</sub>_ ？

我们可以利用核函数来计算出新的特征。

---
给定一个训练样本 _x_ ，我们利用 _x_ 的各个特征与我们预先选定的地标(landmarks) _l<sup>(1)</sup>, l<sup>(2)</sup>, l<sup>(3)</sup>_ 的近似程度来选取新的特征 _f<sub>1</sub>, f<sub>2</sub>, f<sub>3</sub>_ 。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/2516821097bda5dfaf0b94e55de851e0.png" />
</p>

例如：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?f_1=similarity(x,{l^{(1)}})=e(-\frac{{{\left\|x-{l^{(1)}}\right\|}^2}}{2\sigma^2})" title="f_1=similarity(x,{l^{(1)}})=e(-\frac{{{\left\|x-{l^{(1)}}\right\|}^2}}{2\sigma^2})" />
</p>

其中：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?{\left\|x-l^{(1)}\right\|}^{2}=\sum_{j=1}^n{(x_j-l_j^{(1)})}^{2}" title="{\left\|x-l^{(1)}\right\|}^{2}=\sum_{j=1}^n{(x_j-l_j^{(1)})}^{2}" />
</p>

是实例 _x_ 中所有特征与地标 _l<sup>(1)</sup>_ 之间的距离的和。上例中的 _similarity(x, l<sup>(1)</sup>)_ 就是核函数，具体来讲，是**高斯核函数（Gaussian Kernel）_，但是还有其他类型的核函数存在。（但是和高斯分布没什么关系，只是看起来像）

地标的不同决定了 _x_ 和 _l_ 之间的距离：
* 如果距离趋近0，那么 _f ≈ e<sup>-0</sup> = 1_
* 如果距离非常大，那么 _f ≈ e<sup>∞</sup> = 0_

假设我们的训练样本含有两个特征 _[x<sub>1</sub> x<sub>2</sub>]_，给定地标 _l<sup>(1)</sup>_ 与不同的 _σ_，见下图：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/b9acfc507a54f5ca13a3d50379972535.jpg" />
</p>

图中水平左边为 _x<sub>1</sub>,x<sub>2</sub>_，z轴代表 _f_。可以总结出：
* 当 _x_ 与 _l<sup>(1)</sup>_ 重合时 _f_ 才有最大值。
* 随着x的变化，f值改变的速率受到 _σ<sub>2</sub>_ 的控制

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/3d8959d0d12fe9914dc827d5a074b564.jpg" />
</p>

举个例子，在上图中，有三个样本点：
1. 红色点的位置，因其离 _l<sup>(1)</sup>_ 更近，但是离 _l<sup>(2)</sup>_ 和 _l<sup>(3)</sup>_ 较远，因此 _f<sub>1</sub>_ 接近1，而 _f<sub>2</sub>_ , _f<sub>3</sub>_ 接近0。因此 _h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + θ<sub>1</sub>f<sub>3</sub> > 0_ ，因此预测 _y=1_ 。
2. 同理可以求出，对于离 _l<sup>(2)</sup>_ 较近的绿色点，也预测 _y=1_
3. 但是对于蓝绿色的点，因为其离三个地标都较远，预测 _y=0_ 。

图中红色的封闭曲线所表示的范围，便是依据一个单一的训练样本和选取的地标所得出的判定边界。预测时，采用的特征不是训练样本本身的特征，而是通过核函数计算出的新特征 _f<sub>1</sub>, f<sub>2</sub>, f<sub>3</sub>_ 。

---
如何选择地标？

通常是根据训练集的数量选择地标的数量，即如果训练集中有 _m_ 个样本，则选取 _m_ 个地标，并且令: _l<sup>(1)</sup> = x<sup>(1)</sup>, l<sup>(2)</sup> = x<sup>(2)</sup>, ....., l<sup>(m)</sup> = x<sup>(m)</sup>_ 。

这样做的好处在于：现在我们得到的新特征是建立在原有特征与训练集中所有其他特征之间距离的基础之上的，即：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/eca2571849cc36748c26c68708a7a5bd.png" />
</p>

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/ea31af620b0a0132fe494ebb4a362465.png" />
</p>

下面将核函数运用到SVM中，修改SVM假设为：
* 给定 _x_ ，计算新特征 _f_ ，当 _θ<sup>T</sup> f >= 0_ 时，预测 _y = 1_ ，否则反之。
* 相应地修改代价函数为： _Σ<sub>j=1</sub><sup>n=m</sup>θ<sub>j</sub><sup>2</sup>=θ<sup>T</sup>θ_

> 理论上讲，也可以在逻辑回归中使用核函数，但是上面使用 _M_ 来简化计算的方法不适用与逻辑回归，因为计算将非常耗时。

在此不介绍最小化SVM的代价函数的方法，可以使用软件包（如liblinear, libsvm等）。在用这些软件包最小化代价函数之前，通常需要编写核函数，如果我们使用高斯核函数，那么在使用之前进行特征缩放是非常必要的。

下面是SVM的两个参数 _C_ 和 _sigma_ 的影响：
 _C=1/λ_
 * _C_ 较大时，相当于 _λ_ 较小，可能会导致过拟合，高方差；
 * _C_ 较小时，相当于 _λ_ 较大，可能会导致低拟合，高偏差；
 * _σ_ 较大时，可能会导致低方差，高偏差；
 * _σ_ 较小时，可能会导致低偏差，高方差。

## 使用SVM
上面已经讨论了SVM理论，那如何运用SVM？
已经介绍了SVM算法，但不建议自己写软件求解参数 _θ_ ，就像今天很少人或者其实没有人考虑自己写代码来转换矩阵，或求一个数的平方根等。只是知道如何调用库函数来实现。
同样的，解决SVM最优化问题的软件很复杂，且已经有研究者做了很多年数值优化。
因此你提出好的软件库和好的软件包来做这样一些事儿。
然后强烈建议用高优化软件库中的一个，而不是尝试自己落实一些数据。
有许多好的软件库，用得最多的两个是liblinear和libsvm，但还要其他很多。

在高斯核函数之外还有其他选择，如：
* 多项式核函数（PolynomialKernel）
* 字符串核函数（Stringkernel）
* 卡方核函数（chi-squarekernel）
* 直方图交集核函数（histogramintersectionkernel）
* 等等...

这些核函数的目标也都是根据训练集和地标之间的距离来构建新特征，这些核函数需要满足Mercer's定理，才能被SVM的优化软件正确处理。

---
**多类分类问题**

假设利用之前介绍的一对多方法来解决多类分类问题。
如果一共有 _k_ 个类，则需要 _k_ 个模型，以及 _k_ 个参数向量 _θ_ 。
也可以训练 _k_ 个SVM来解决多类分类问题。
但是大多数SVM软件包都有内置的多类分类功能。

尽管不用写SVM的优化软件，但需要：
1. 提出参数 _C_ 的选择。我们在之前讨论过误差/方差在这方面的性质。

2. 选择内核参数或想要使用的相似函数，其中一个选择是：
    * 选择不需要任何内核参数，没有内核参数的理念，也叫线性核函数。因此，如果有人说他使用了线性核的SVM，这就意味这他使用了不带有核函数的SVM。

## 什么时候使用SVM
从逻辑回归模型，得到了SVM模型，在两者之间，应该如何选择呢？

下面是一些普遍使用的准则：
**_n_ 为特征数， _m_ 为训练样本数**

1. _n >> m_ 而言，即训练集数据量不够支持训练一个复杂的非线性模型，我们选用逻辑回归模型或者不带核函数的SVM。（可能没有足够的数据训练一个非线性函数，但是对于线性函数是ok的）

2. _n_ 较小， _m_ 中等大小，例如 _n_ 在1-1000之间，而 _m_ 在10-10000之间，使用高斯核函数的SVM。

3. _n_ 较小， _m_ 较大，例如 _n_ 在1-1000之间，而 _m_ 大于50000，则使用SVM会非常慢，解决方案是创造、增加更多的特征，然后使用逻辑回归或不带核函数的SVM。

> 神经网络在以上三种情况下都可能会有较好的表现，但是训练神经网络可能非常慢，**选择SVM的原因主要在于它的代价函数是凸函数，不存在局部最小值。**

SVM包会工作得很好，但它们仍然有些慢。
当有非常大的训练集，用核函数的SVM可能很慢。
可以考虑创建更多的特征变量 _n_ ，然后用逻辑回归或不带核函数的SVM。

对于逻辑回归或者不带核函数的SVM，经常把他俩把它们放在一起讨论是有原因的：
* 逻辑回归和不带核函数的SVM它们都是非常相似的算法，不管是逻辑回归还是不带核函数的SVM，通常都会做相似的事情，并给出相似的结果。但是根据你实现的情况，其中一个可能会比另一个更有效。在另一些问题上，可能一个有效，另一个效果不好。
* 但随着SVM的复杂度增加，当使用不同的内核函数学习复杂的非线性函数时，如当有多达1万（10,000）的样本时，也可能是5万（50,000），特征变量的数量相当大。不带核函数的SVM就会表现得相当突出。

---
那**神经网络**呢？
对于所有的这些问题，一个设计得好的神经网络可能会非常有效。
但有个 **缺点是神经网络训练起来可能会特别慢。**

一个非常好的SVM实现包，它可能会运行得比较快比神经网络快很多。
此外，SVM的优化问题是一种凸优化问题。因此，好的SVM优化软件包总是会找到全局最小值，或者接近它的值。对于SVM你不需要担心局部最优。

在实际应用中，局部最优不是神经网络所需要解决的一个重大问题，所以这是你在使用SVM的时候不需要太去担心的一个问题。

**算法确实很重要，但是通常更加重要的是数据量、有多熟练是否擅长做误差分析和排除学习算法，指出如何设定新的特征变量和找出其他能决定学习算法的变量等方面，通常这些方面会比你使用逻辑回归还是SVM这方面更加重要。**

当然SVM仍然被广泛认为是一种最强大的学习算法。实际上SVM与逻辑回归、神经网络一起构建了一个很好的机器学习算法武器库。

## 阅读推荐
* [支持向量机SVM - by Jerrylead](http://www.cnblogs.com/jerrylead/archive/2011/03/13/1982639.html)
* [支持向量机系列 - by pluskid](http://blog.pluskid.org/?page_id=683)

[回到顶部](#支持向量机)