# 打造实用的机器学习系统

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

## 机器学习系统设计
