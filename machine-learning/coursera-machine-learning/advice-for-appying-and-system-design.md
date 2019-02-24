# 打造实用的机器学习系统
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [打造实用的机器学习系统](#打造实用的机器学习系统)
	- [应用机器学习算法的建议](#应用机器学习算法的建议)
		- [评估一个假设函数 Evaluating a Hypothesis](#评估一个假设函数-evaluating-a-hypothesis)
		- [模型选择和交叉验证集 Model Selection](#模型选择和交叉验证集-model-selection)
		- [偏差(Bias)和方差(Variance)](#偏差bias和方差variance)
		- [正则化和偏差/方差](#正则化和偏差方差)
		- [学习曲线](#学习曲线)
		- [总结：决定下一步做什么](#总结决定下一步做什么)
	- [机器学习系统设计](#机器学习系统设计)

<!-- /TOC -->

## 应用机器学习算法的建议
这部分介绍如果改进机器学习系统性能的一些建议。

对一个线性回归模型，在得到学习参数后，如果要将假设函数放到一组新的房屋样本上测试。如果发现在预测房价时产生了巨大的误差，现在你的问题是要想改进这个算法，接下来应该怎么办？可能有如下办法：
1. 一种办法是使用更多的训练样本
2. 尝试选用更少的特征集
   * 如果你有一系列特征比如 _x<sub>1</sub>,x<sub>2</sub>,x<sub>3</sub>_ 等等。也许可以从这些特征中仔细挑选一小部分来防止过拟合
3. 尝试选用更多的特征
   * 也许目前的特征集，对你来讲并不是很有帮助
4. 尝试增加多项式特征的方法
   * 比如 _x<sub>1</sub>_ 的平方， _x<sub>2</sub>_ 的平方， _x<sub>1</sub>,x<sub>2</sub>_ 的乘积
5. 减小正则化参数 _λ_ 的值
6. 增大正则化参数 _λ_ 的值

**不应该随机选择上面的某种方法来改进我们的算法，而是运用一些机器学习诊断法来决定上面哪些方法对算法是有效的。**


### 评估一个假设函数 Evaluating a Hypothesis
对于一个训练集，在计算算法的参数的时候，考虑的是选择参数以使训练误差（Cost function）最小化，有人认为得到一个非常小的训练误差一定是一件好事，但仅仅是因为这个假设具有很小的训练误差，并不能说明它是一个好的假设函数。尤其是考虑到过拟合的假设函数，所以所以这这个方法推广到新的训练集上是不适用的。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/f49730be98810b869951bbe38b6319ba.png" />
</p>

对于这个简单的例子，我们可以对假设函数 _h(x)_ 进行画图，然后观察图形趋势，但对于特征变量不止一个的这种一般情况，还有像有很多特征变量的问题，想要通过画出假设函数来进行观察，就会变得很难甚至是不可能实现。

因此，我们需要另一种方法来评估我们的假设函数过拟合检验。


为了检验算法是否过拟合，我们将数据分成训练集（Training set）和测试集（Test set），通常用70%的数据作为训练集，用剩下30%的数据作为测试集。

**注意**训练集和测试集均要含有各种类型的数据，通常我们要对数据进行“洗牌”（随机排序），然后再分成训练集和测试集。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/9c769fd59c8a9c9f92200f538d1ab29c.png" />
</p>

测试集评估在通过训练集让我们的模型学习得出其参数后，对测试集运用该模型，我们有两种方式计算误差：

1. 对于线性回归模型，利用测试集数据计算代价函数 _J_:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J_{test}{(\theta)}=-\frac{1}{{m}_{test}}\sum\limits_{i=1}^{m_{test}}&space;\left(h_{\theta}(x^{(i)}_{test})&space;-&space;y^{(i)}_{test}\right)^2" title="J_{test}{(\theta)}=-\frac{1}{{m}_{test}}\sum\limits_{i=1}^{m_{test}} \left(h_{\theta}(x^{(i)}_{test}) - y^{(i)}_{test}\right)^2" />
</p>

2. 对于逻辑回归模型:
   * 利用测试集数据计算代价函数 _J_
   <p align="center">
   <img src="https://latex.codecogs.com/gif.latex?J_{test}{(\theta)}=-\frac{1}{{m}_{test}}\sum\limits_{i=1}^{m_{test}}\left[\log{h_{\theta}(x^{(i)}_{test})}&plus;(1-{y^{(i)}_{test}})\log{h_{\theta}x^{(i)}_{test})}\right]" title="J_{test}{(\theta)}=-\frac{1}{{m}_{test}}\sum\limits_{i=1}^{m_{test}}\left[\log{h_{\theta}(x^{(i)}_{test})}+(1-{y^{(i)}_{test}})\log{h_{\theta}x^{(i)}_{test})}\right]" />
   </p>

   * 计算对于每一个测试集样本的误分类的比率，然后求平均：
   <p align="center">
   <img src="https://latex.codecogs.com/gif.latex?err(h_\theta(x),&space;y)\begin{cases}1&space;&&space;\text{&space;if&space;}&space;h_\theta(x)&space;\ge&space;0.5,&space;y&space;=&space;0\text{,&space;or}&space;\\&space;&&space;\text{&space;if&space;}&space;h_\theta(x)&space;<&space;0.5,&space;y&space;=&space;1\\0&space;&&space;\text{&space;otherwise&space;}\end{cases}" title="err(h_\theta(x), y)\begin{cases}1 & \text{ if } h_\theta(x) \ge 0.5, y = 0\text{, or} \\ & \text{ if } h_\theta(x) < 0.5, y = 1\\0 & \text{ otherwise }\end{cases}" />
   </p>

### 模型选择和交叉验证集 Model Selection
假设要在10个不同次数的二项式模型间选择：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/1b908480ad78ee54ba7129945015f87f.jpg" />
</p>

显然越高次数的多项式模型越能够适应我们的训练数据集，但也可能意味着过拟合（Overfitting），应该选择一个更能适应一般情况的模型。

这时需要使用交叉验证集（Cross-validation set）来帮助选择模型。​
* 交叉验证集通常是独立于训练集和测试集的
* 通常用6/2/2的划分，即60%的数据作为训练集，20%的数据作为交叉验证集，20%的数据作为测试集

模型选择方法：
1. 使用训练集训练出10个模型
1. 用10个模型分别对交叉验证集计算得出交叉验证误差（代价函数的值）
1. 选取代价函数值最小的模型
1. 用步骤3中选出的模型对测试集计算得出推广误差（代价函数的值）

训练误差（Training error）：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J_{train}(\theta)=\frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2" title="J_{train}(\theta)=\frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2" />
</p>

交叉验证误差（Cross Validation error）：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J_{cv}(\theta)=\frac{1}{2m_{cv}}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)}_{cv})-y^{(i)}_{cv})^2" title="J_{cv}(\theta)=\frac{1}{2m_{cv}}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)}_{cv})-y^{(i)}_{cv})^2" />
</p>

测试误差（Test error）：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J_{test}(\theta)=\frac{1}{2m_{test}}\sum\limits_{i=1}^{m_{test}}(h_{\theta}(x^{(i)}_{cv})-y^{(i)}_{cv})^2" title="J_{test}(\theta)=\frac{1}{2m_{test}}\sum\limits_{i=1}^{m_{test}}(h_{\theta}(x^{(i)}_{cv})-y^{(i)}_{cv})^2" />
</p>

虽然现实中有人用测试集来计算交叉验证误差，并用同样的测试集来计算测试误差，但这并不是一个好的做法，除非有大量的训练数据.. Anyway，不推荐这种做法，还是将数据集分成训练数据、交叉验证数据和测试数据吧。

### 偏差(Bias)和方差(Variance)
如果一个机器学习算法表现不理想，多半出现两种情况：
* 要么是偏差比较大，要么是方差比较大。
* 换句话说，出现的情况要么是欠拟合，要么是过拟合问题

例如下图，依次对应的是欠拟合、正常、过拟合：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/20c6b0ba8375ca496b7557def6c00324.jpg" />
</p>

为了分析模型性能，通常会将训练集和交叉验证集的代价函数误差与多项式的次数绘制在同一张图表上来帮助分析：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/bca6906add60245bbc24d71e22f8b836.png" />
</p>

上图可看出：
* 对于训练集，当 _d_ 较小时，模型拟合程度更低，误差较大；随着 _d_ 的增长，拟合程度提高，误差减小。
* 对于交叉验证集，当 _d_ 较小时，模型拟合程度低，误差较大；但是随着 _d_ 的增长，误差呈现先减小后增大的趋势，转折点是我们的模型开始过拟合训练数据集的时候。

根据上面的图表，我们还可以总结判断高方差、高偏差的方法:
* 训练集误差和交叉验证集误差近似时：
  * 偏差/欠拟合
* 交叉验证集误差远大于训练集误差时：
  * 方差/过拟合

### 正则化和偏差/方差
在训练模型时，一般会用正则化方法来防止过拟合。但是可能会正则化程度太高或太小，即在选择λ的值时也需要思考与此前选择多项式模型次数类似的问题。如下图是不同的 _λ_ 对应不同拟合程度：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/2ba317c326547f5b5313489a3f0d66ce.png" />
</p>

通常会尝试一系列的 _λ_ 值，以测试最佳选择：
  1. 使用训练集训练出12个不同程度正则化的模型
  1. 用12个模型分别对交叉验证集计算的出**交叉验证误差**
  1. 选择得出**交叉验证误差最小**的模型
  1. 运用步骤3中选出模型对测试集计算得出推广误差
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/8f557105250853e1602a78c99b2ef95b.png" />
</p>

我们也可以同时将训练集和交叉验证集模型的代价函数误差与λ的值绘制在一张图表上（如下图），可以看出训练集误差和 _λ_ 的关系如下：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/38eed7de718f44f6bb23727c5a88bf5d.png" />
</p>

总结：
* 当 _λ_ 较小时，训练集误差较小（过拟合）而交叉验证集误差较大
* 随 _λ_ 的增加，训练集误差不断增加（欠拟合），而交叉验证集误差则是先减小后增加

### 学习曲线
学习曲线是用来判断某一个学习算法是否处于偏差、方差问题，它是学习算法的一个很好的**合理检验**（**sanity check**）。

学习曲线是将训练集误差和交叉验证集误差作为训练集样本数量（ _m_ ）的函数绘制的图表。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/973216c7b01c910cfa1454da936391c6.png" />
</p>

例如，如果我们有100行数据，我们从1行数据开始，逐渐学习更多行。
**思想是**：当训练较少数据时，训练模型将能非常完美地适应较少的训练数据，但是训练的模型不能很好地适应交叉验证集或测试集。

如何利用学习曲线识别```高偏差/欠拟合```：
* 作为例子，尝试一条直线来拟合下面的数据，可以看出，无论训练集增长到多大误差都不会有太大改观（始终保持很高的error）
* **即：在高偏差/欠拟合的情况下，增加训练集不一定有帮助。**

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/4a5099b9f4b6aac5785cb0ad05289335.jpg" />
</p>

如何利用学习曲线识别```高方差/过拟合```：
* 假设我们使用一个非常高次的多项式模型，并且正则化非常小，可以看出，当交叉验证集误差远大于训练集误差时，往训练集增加更多数据可以提高模型的效果。
* **即：在高方差/过拟合的情况下，增加训练集可能提高算法效果。**
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/2977243994d8d28d5ff300680988ec34.jpg" />
</p>

### 总结：决定下一步做什么
1. 增加训练样本数 _m_ ——解决高方差
1. 减少特征的数量 _n_ ——解决高方差
1. 获得更多的特征 _n_ ——解决高偏差
1. 增加多项式特征 _n_ ——解决高偏差
1. 减少正则化程度 _λ_ ——解决高偏差
1. 增加正则化程度 _λ_ ——解决高方差

神经网络的方差和偏差：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/c5cd6fa2eb9aea9c581b2d78f2f4ea57.png" />
</p>

神经网络神经元个数选择：
* 使用较小的神经网络类似于参数较少的情况，容易导致高偏差和欠拟合，但计算代价较小
* 使用较大的神经网络，类似于参数较多的情况，容易导致高方差和过拟合，虽然计算代价比较大，但是可以通过正则化手段来调整而更加适应数据
* **通常选择较大的神经网络并采用正则化处理会比采用较小的神经网络效果要好**

神经网络隐藏层层数的选择：
* 通常从一层开始逐渐增加层数
* 为了更好地作选择，可以把数据分为训练集、交叉验证集和测试集，针对不同隐藏层层数的神经网络训练神经网络，然后选择交叉验证集代价最小的神经网络。




## 机器学习系统设计
