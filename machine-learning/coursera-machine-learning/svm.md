# 支持向量机 Support Vector Machine (SVM)

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [支持向量机 Support Vector Machine (SVM)](#支持向量机-support-vector-machine-svm)

<!-- /TOC -->

支持向量机（Support Vector Machine）是一个广泛应用于工业界和学术界的算法。
与逻辑回归和神经网络相比，SVM在学习复杂非线性方程时提供了一种更为清晰，更加强大的方式。

为了描述支持向量机，我们先从逻辑回归开始展示如何一点一点修改来得到支持向量机：
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

**现在开始建立支持向量机，我们从这里开始：**

对代价函数 _-log(1-((1)/(1+e<sup>-z</sup>)))_ 做一点修改（如上图中紫色的曲线）。
由两条线段组成，即位于右边的水平部分和位于左边的直线部分。
* 左边的函数称为 _cost<sub>1</sub>(z)_
* 右边的函数称为 _cost<sub>0</sub>(z)_ 。

现在就开始构建支持向量机。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/59541ab1fda4f92d6f1b508c8e29ab1c.png" />
</p>

$$ \min_{\theta}C\sum ^{m}_{i=1}\left[ y^{(i)}cost_{1}\left( \theta ^{T}x^{(i)}\right) + (1-y^{(i)})cost_{1}\left( \theta ^{T}x^{(i)}\right)\right] +\dfrac {1}{2}\sum ^{n}_{i=1}\theta ^{2}_{j} $$

<p align="center">
<img src="https://raw.github.com/loveunk/lateximg/master/tex/2275bfb7d01c4f41561e348acc6cbf90.svg?invert_in_darkmode&sanitize=true" />
</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\min_{\theta}C\sum&space;^{m}_{i=1}\left[&space;y^{(i)}cost_{1}\left(&space;\theta&space;^{T}x^{(i)}\right)&space;&plus;&space;(1-y^{(i)})cost_{1}\left(&space;\theta&space;^{T}x^{(i)}\right)\right]&space;&plus;\dfrac&space;{1}{2}\sum&space;^{n}_{i=1}\theta&space;^{2}_{j}" title="\min_{\theta}C\sum ^{m}_{i=1}\left[ y^{(i)}cost_{1}\left( \theta ^{T}x^{(i)}\right) + (1-y^{(i)})cost_{1}\left( \theta ^{T}x^{(i)}\right)\right] +\dfrac {1}{2}\sum ^{n}_{i=1}\theta ^{2}_{j}" />
</p>

这是我们在逻辑回归中使用代价函数 _J(θ)_ 。也许这个方程看起来不是非常熟悉。这是因为之前有个负号在方程外面，但是，这里我所做的是，将负号移到了表达式的里面，这样做使得方程看起来有些不同。对于支持向量机而言，实质上我们要将这替换为 _\cost<sub>1</sub>(z)_ ，也就是 _\cost<sub>1</sub>(θ<sup>T</sup>x)_ ，同样地，我也将这一项替换为 _\cost<sub>0</sub>(z)_ ，也就是代价 _\cost<sub>0</sub>(θ<sup>T</sup>x)_ 。这里的代价函数 _\cost<sub>1</sub>_ ，就是之前所提到的那条线。此外，代价函数 _\cost<sub>0</sub>_ ，也是上面所介绍过的那条线。因此，对于支持向量机，我们得到了这里的最小化问题，即:
