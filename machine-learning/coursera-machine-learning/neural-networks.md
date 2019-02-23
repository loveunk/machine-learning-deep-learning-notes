# 神经网络
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [神经网络](#神经网络)
	- [为什么需要神经网络](#为什么需要神经网络)
	- [神经元和大脑](#神经元和大脑)
	- [模型表示](#模型表示)
		- [神经元模型：逻辑单元](#神经元模型逻辑单元)
		- [前向传播](#前向传播)
		- [神经网络架构](#神经网络架构)
	- [神经网络应用](#神经网络应用)
	- [神经网络解决多分类问题](#神经网络解决多分类问题)
	- [代价函数 Cost Function](#代价函数-cost-function)
	- [反向传播算法](#反向传播算法)

<!-- /TOC -->

## 为什么需要神经网络
我们此前已经学过线性回归和非线性回归，那为什么还需要神经网络了？

因为无论是线性回归还是逻辑回归都有这样一个缺点，即：当特征太多时，计算的负荷会非常大。举个例子：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/5316b24cd40908fb5cb1db5a055e4de5.png" />
</p>

当使用 _x<sub>1</sub>_ , _x<sub>2</sub>_ 的多次项式进行预测时，回归算法可以很好地工作。之前已经看到过，使用非线性的多项式项，能够建立更好的分类模型。

但是，如果数据集有非常多的特征，例如大于100，当希望用这100个特征构建一个非线性的多项式模型，将是数量惊人的特征组合，即便我们只采用两两特征的组合 _(x<sub>1</sub>x<sub>2</sub>+x<sub>1</sub>x<sub>3</sub>+x<sub>1</sub>x<sub>4</sub>+...+x<sub>2</sub>x<sub>3</sub>+x<sub>2</sub>x<sub>4</sub>+...+x<sub>99</sub>x<sub>100</sub>)_ ，也会有接近5000个组合而成的特征。特征的数量会以 _O(n)_ 的复杂度来增加。这对于逻辑回归需要计算的特征太多了！

假设希望训练一个模型识别视觉对象（例如识别图片上是否是汽车）。一种方法是我们利用很多汽车的图片和很多非汽车的图片，然后利用这些图片上一个个像素的值（饱和度或亮度）来作为特征。

假如只选用灰度图片，每个像素则只有一个值（非RGB值），可以将每个图片的像素作为特征来训练逻辑回归算法，并利用像素值来判断图片上是否是汽车。

假如采用的是50*50像素的图片，则有2500个特征，如果进一步采用两两特征组合构成一个多项式模型，则有 _2500<sup>2</sup>/2 ≈ 3百万_ 个特征。复杂度太高了！逻辑回归已经不能很好处理，这时候就需要神经网络。

## 神经元和大脑
神经网络是一种很古老的算法，发明它最初的目的是制造模拟大脑的机器。

神经网络兴起于二十世纪八九十年代，应用得非常广泛。但在90年代的后期应用减少了。但最近，又东山再起。
* 其中一个原因是神经网络计算量偏大。由于近年计算机运行速度变快，能运行起大规模的神经网络。正由这个原因和其他技术因素，如今神经网络对许多应用来说是最先进的技术。

人类大脑可以学数学、微积分，而且能处理各种不同的事情。似乎如果你想要模仿大脑，你得写很多不同的软件来模拟所有这些五花八门的事情。不过能不能假设大脑做所有这些，不同事情的方法，不需要用上千个不同的程序。相反，大脑处理的方法，只需要一个单一的学习算法就可以了？尽管这只是一个假设，先来介绍一些这方面的证据：
* 神经系统科学家通过神经重接实验证实人体有同一块脑组织可以处理光、声或触觉信号，那么也许存在一种学习算法，可以同时处理视觉、听觉和触觉，而不是需要运行上千个不同的程序。也许需要做的就是找出一些近似的或实际的大脑学习算法，然后实现它大脑通过自学掌握如何处理这些不同类型的数据。

## 模型表示

为了构建神经网络模型，我们先思考大脑中的神经网络是怎样的：每个神经元都被认为是一个处理单元，即神经核（Nucleus），它含有许多输入，即树突（Dendrite），并且有一个输出，即轴突（Axon）。神经网络是大量神经元相互链接并通过电脉冲来交流的一个网络。如下图：

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/3d93e8c1cd681c2b3599f05739e3f3cc.jpg" />
</p>

神经元利用微弱的电流进行沟通。这些弱电流也称作动作电位。所以如果神经元想要传递一个消息，它会通过它的轴突，发送一段微弱电流给其他神经元。

下图是一条连接到输入神经，或者连接另一个神经元树突的神经。右上角的神经元 A 通过轴突把消息传递给左下角的神经元 B，B有可能会把消息再传给其他神经元。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/7dabd366525c7c3124e844abce8c2dd6.png" />
</p>

这就是所有人类思考的模型：神经元把收到的消息进行计算，并向其他神经元传递消息。这也是感觉和肌肉运转的原理。
> 如果你想活动一块肌肉，就会触发一个神经元给你的肌肉发送脉冲，并引起你的肌肉收缩。如果一些感官：比如说眼睛想要给大脑传递一个消息，那么它就像这样发送电脉冲给大脑的。

### 神经元模型：逻辑单元

神经网络模型建立在很多神经元之上，每一个神经元又是一个个学习模型。这些神经元（也叫激活单元，activation unit）采纳一些特征作为输出，并且根据本身的模型提供一个输出。下图是一个以逻辑回归模型作为自身学习模型的神经元示例，在神经网络中，参数又可被成为权重（weight）。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/c2233cd74605a9f8fe69fd59547d3853.jpg" />
</p>

其中 _x<sub>1</sub>_ , _x<sub>2</sub>_ , _x<sub>3</sub>_ 是输入单元（input units），我们将原始数据输入给它们。 _a<sub>1</sub>_ , _a<sub>2</sub>_ , _a<sub>3</sub>_ 是中间单元，它们负责将数据进行处理，然后呈递到下一层。最后是输出单元，它负责计算 _h<sub>θ</sub>(x)_ 。
**注意**：这里 _h<sub>θ</sub>(x)=1/(1+e<sup>-θ<sup>T</sup>X</sup>)_

神经元的神经网络，效果如下：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/fbb4ffb48b64468c384647d45f7b86b5.png" />
</p>

神经网络模型是许多逻辑单元按照不同层级组织起来的网络，每一层的输出变量都是下一层的输入变量。上图为一个3层的神经网络，第一层成为输入层（**Input Layer**），最后一层称为输出层（**Output Layer**），中间一层成为隐藏层（**Hidden Layers**）。我们为每一层都增加一个偏差单位（**Bias unit**），如下图所示。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/8293711e1d23414d0a03f6878f5a2d91.jpg" />
</p>

### 前向传播
神经网络模型的一些符号描述：
* _a<sub>i</sub><sup>(j)</sup>_ 代表第 _j_ 层的第 _i_ 个激活单元。
* _Θ<sup>(j)</sup>_ 代表从第 _j_ 层映射到第 _j+1_ 层时的权重的矩阵，例如 _θ<sup>(1)</sup>_ 代表从第一层映射到第二层的权重的矩阵。
  * 其尺寸为：**以第 _j+1_ 层的激活单元数量为行数，以第 _j_ 层的激活单元数加1为列数的矩阵。例如：上图所示的神经网络中 _θ<sup>(1)</sup>_ 的尺寸为3*4**（列加1是因为要对应 _x<sub>0</sub>_, _a<sub>0</sub>_ 这样的bias，搞清楚权重矩阵的大小很重要！）。

那么对于上图中的神经网络，可得：
* _a<sub>1</sub><sup>(2)</sup>=g(Θ<sub>10</sub><sup>(1)</sup>x0+Θ<sub>11</sub><sup>(1)</sup>x1+Θ<sub>12</sub><sup>(1)</sup>x2+Θ<sub>13</sub><sup>(1)</sup>x3)_
* _a<sub>2</sub><sup>(2)</sup>=g(Θ<sub>20</sub><sup>(1)</sup>x0+Θ<sub>21</sub><sup>(1)</sup>x1+Θ<sub>22</sub><sup>(1)</sup>x2+Θ<sub>23</sub><sup>(1)</sup>x3)_
* _a<sub>3</sub><sup>(2)</sup>=g(Θ<sub>30</sub><sup>(1)</sup>x0+Θ<sub>31</sub><sup>(1)</sup>x1+Θ<sub>32</sub><sup>(1)</sup>x2+Θ<sub>33</sub><sup>(1)</sup>x3)_
* _h<sub>Θ</sub>(x)=g(Θ<sub>10</sub><sup>(2)</sup>a<sub>0</sub><sup>(2)</sup>+Θ<sub>11</sub><sup>(2)</sup>a<sub>1</sub><sup>(2)</sup>+Θ<sub>12</sub><sup>(2)</sup>a<sub>2</sub><sup>(2)</sup>+Θ<sub>13</sub><sup>(2)</sup>a<sub>3</sub><sup>(2)</sup>)_

上面进行的讨论只是将特征矩阵中的一行（一个训练实例）喂给了神经网络，我们需要将整个训练集都喂给我们的神经网络算法来学习模型。

每一个 _a_ 都是由上一层所有的 _x_ 和每一个 _x_ 所对应的决定的。
我们把这样从左到右的算法称为前向传播算法（**Forward Propagation**）。

把 _x_ , _Θ_ , _a_ 分别用矩阵表示：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/20171101224053.png" />
</p>

我们可以得到 _a = Θ · X_ 。
> 当然上式这非常取决于X本身是如何排列的，通常是X的每一行是一个样本，每一列是一个样本特征。

前向传播算法使用循环来计算，当然利用向量化的方法（Vectorized Computation）会使得计算更为简便。以上面的神经网络为例。

此外，为了简化公式，定义另外一个符号 _z_：
* _z<sup>(j)</sup> = Θ<sup>(j-1)</sup> · a<sup>(j-1)</sup>_

那么上面的例子可以描述如下：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/303ce7ad54d957fca9dbb6a992155111.png" />
</p>

我们令 _z<sup>(2)</sup>=Θ<sup>(1)</sup>x_ ，则 _a<sup>(2)</sup>=g(z<sup>(2)</sup>)_ ，计算后添加 _a<sub>0</sub><sup>(2)</sup>=1_ 。展开过程也就是：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/2e17f58ce9a79525089a1c2e0b4c0ccc.png" />
</p>

我们令 _z<sup>(3)</sup>=Θ<sup>(2)</sup>a<sup>(2)</sup>_ ，则 _h<sub>Θ</sub>(x)=a<sup>(3)</sup>=g(z<sup>(3)</sup>)_ 。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/43f1cb8a2a7e9a18f928720adc1fac22.png" />
</p>

上面的过程只讨论了对训练集中一个训练实例的计算。如果要对整个训练集计算，需要将训练集特征矩阵进行转置，使得同一个实例的特征都在同一列里。即
*  _z<sup>(2)</sup>=Θ<sup>(1)</sup>X<sup>T</sup>_
* _a<sup>(2)</sup>=g(z<sup>(2)</sup>)_

----
为了更好了了解神经网络的原理，我们把上面示例网络的左半部分遮住：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/6167ad04e696c400cb9e1b7dc1e58d8a.png" />
</p>

可以看到，右半部分其实就是一个逻辑回归模型：以 _a<sub>0</sub>,a<sub>1</sub>,a<sub>2</sub>,a<sub>3</sub>_ 作为输入，并按照LogisticRegression的方式输出 _h<sub>θ</sub>(x)_ 。

其实神经网络就像是逻辑回归，只不过我们把逻辑回归中的输入向量 _[x<sub>1</sub> ~ x<sub>3</sub>]_ 变成了中间层的 _[a<sub>1</sub><sup>(2)</sup> ~ a<sub>3</sub><sup>(2)</sup>]_ ,即: _h<sub>θ</sub>(x)=g(Θ<sub>0</sub><sup>(2)</sup>a<sub>0</sub><sup>(2)</sup>+Θ<sub>1</sub><sup>(2)</sup>a<sub>1</sub><sup>(2)</sup>+Θ<sub>2</sub><sup>(2)</sup>a<sub>2</sub><sup>(2)</sup>+Θ<sub>3</sub><sup>(2)</sup>a<sub>3</sub><sup>(2)</sup>)_ 我们可以把 _a<sub>0</sub>, a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub>_ 看成更为高级的特征值，也就是 _x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>_ 的进化体，并且它们是由 _x_ 与 _Θ_ 决定的，因为是梯度下降的，所以 _a_ 是变化的，并且变得越来越厉害，所以这些更高级的特征值远比仅仅将 _x_ 次方厉害，也能更好的预测新数据。这就是神经网络相比于逻辑回归和线性回归的优势。

### 神经网络架构
神经网络架构是不同层（Layer）之间的链接方式。包括：
* 有多少层
* 每一层有多少激活单元

第一层是输入层（Input Layer），最后一层是输出层（Output Layer），中间的是影藏层（Hidden Layer）。

## 神经网络应用
神经网络能够通过学习得出其自身的一系列特征。在普通的逻辑回归中，我们被限制为使用数据中的原始特征 _x<sub>1</sub>,x<sub>2</sub>,...,x<sub>n</sub>_ ，我们虽然可以使用一些二项式项来组合这些特征，但是我们仍然受到这些原始特征的限制。在神经网络中，原始特征只是输入层，在我们上面三层的神经网络例子中，第三层也就是输出层做出的预测利用的是第二层的特征，而非输入层中的原始特征，我们可以认为第二层中的特征是神经网络通过学习后自己得出的一系列用于预测输出变量的新特征。

神经网络中，单层神经元（无中间层）的计算可用来表示逻辑运算，比如逻辑与(AND)、逻辑或(OR)。

**逻辑与（AND）**

下图中左半部分是神经网络的设计与输出层表达式，右边上部分是Sigmod函数，下半部分是真值表。

我们可以用这样的一个神经网络表示AND 函数：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/809187c1815e1ec67184699076de51f2.png" />
</p>

其中 _θ<sub>0</sub>=-30,θ<sub>1</sub>=20,θ<sub>2</sub>=20_ 我们的输出函数 _h<sub>θ</sub>(x)_ 即为： _h<sub>Θ</sub>(x)=g(-30+20x<sub>1</sub>+20x<sub>2</sub>)_

所以我们有： _h<sub>Θ</sub>(x) ≈ x<sub>1</sub> AND x<sub>2</sub>_

**逻辑或（OR）**

OR与AND整体一样，区别只在于 _Θ_ 的取值不同。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/aa27671f7a3a16545a28f356a2fb98c0.png" />
</p>

---
二元逻辑运算符（BINARY LOGICAL OPERATORS）当输入特征为布尔值（0或1）时，我们可以用一个单一的激活层可以作为二元逻辑运算符，为了表示不同的运算符，我们只需要选择不同的权重即可。

下图的神经元（三个权重分别为-30，20，20）可以被视为作用同于逻辑与（AND）：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/57480b04956f1dc54ecfc64d68a6b357.jpg" />
</p>
下图的神经元（三个权重分别为-10，20，20）可以被视为作用等同于逻辑或（OR）：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/7527e61b1612dcf84dadbcf7a26a22fb.jpg" />
</p>
下图的神经元（两个权重分别为 10，-20）可以被视为作用等同于逻辑非（NOT）：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/1fd3017dfa554642a5e1805d6d2b1fa6.jpg" />
</p>

可以利用神经元来组合更为复杂的神经网络以实现复杂的运算。例如要实现XNOR功能（输入的两个值必须一样，均为1或均为0），即 _XNOR=(x<sub>1</sub>,AND,x<sub>2</sub>) OR((NOT,x<sub>1</sub>) AND(NOT,x<sub>2</sub>))_
1. 首先构造一个能表达 _(NOT,x<sub>1</sub>)AND(NOT,x<sub>2</sub>)_ 部分的神经元：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/4c44e69a12b48efdff2fe92a0a698768.jpg" />
</p>

2. 然后将表示AND的神经元和表示 _(NOT,x<sub>1</sub>)AND(NOT,x<sub>2</sub>)_ 的神经元以及表示OR的神经元进行组合：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/432c906875baca78031bd337fe0c8682.jpg" />
</p>

这样就得到了一个 _XNOR_ 运算符功能的神经网络。按这种思路你可以逐渐构造出越来越复杂的函数和特征值。这就是神经网络的厉害之处。

---
一个神经网络做手写数字识别的演示视频

<p align="center">
  <a href="https://www.youtube.com/watch?v=yxuRnBEczUU" target="_blank">
    <img src="https://img.youtube.com/vi/yxuRnBEczUU/0.jpg" />
  </a>
</p>

（上面的视频是Youtube的，如果无法翻墙的，可访问[爱奇艺上的链接](https://www.iqiyi.com/w_19rue4wsdl.html)）

## 神经网络解决多分类问题
当分类问题不止两种分类时（ _y=1,2,3…._ ），比如如果我们要训练一个神经网络算法来识别路人、汽车、摩托车和卡车。
* 在输出层我们应该有4个值。例如，第一个值为1或0用于预测是否是行人，第二个值用于判断是否为汽车。
* 输入向量 _x_ 有三个维度，两个中间层，输出层4个神经元分别用来表示4类，也就是每一个数据在输出层都会出现 _[abcd]<sup>T</sup>_ ，且 _a,b,c,d_ 中仅有一个为1，表示当前类。

下面是该神经网络的可能结构示例：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/f3236b14640fa053e62c73177b3474ed.jpg" />
</p>

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/685180bf1774f7edd2b0856a8aae3498.png" />
</p>

神经网络算法的输出结果为四种可能情形之一：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/5e1a39d165f272b7f145c68ef78a3e13.png" />
</p>

## 代价函数 Cost Function
首先介绍一些符号计法：
* 假设神经网络的训练样本有 _m_ 个
* 每个包含一组输入 _x_ 和一组输出信号 _y_
* _L_ 表示神经网络层数
* _S<sub>I</sub>_ 表示每层的神经元个数（ _S<sub>l</sub>_ 表示输出层神经元个数）， _S<sub>L</sub>_ 代表最后一层中处理单元的个数

将神经网络的分类定义为两种情况：二类分类和多类分类，
* 二类分类： _S<sub>L</sub>=0, y=0, or, 1_ 表示哪一类；
*  _K_ 类分类： _S<sub>L</sub>=k, y<sub>i</sub>=1_ 表示分到第 _i_ 类； _(k>2)_

也可以参考下图：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/8f7c28297fc9ed297f42942018441850.jpg" />
</p>

我们回顾逻辑回归问题中我们的代价函数为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J\left(\theta\right)=\frac{1}{m}\sum\limits_{i=1}^m{[-{y^{(i)}}\log\left({h_\theta}\left({x^{(i)}}\right)\right)-\left(1-{y^{(i)}}\right)\log\left(1-{h_\theta}\left({x^{(i)}}\right)\right)]}&plus;\frac{\lambda}{2m}\sum\limits_{j=1}^n{\theta_j^2}" title="J\left(\theta\right)=\frac{1}{m}\sum\limits_{i=1}^m{[-{y^{(i)}}\log\left({h_\theta}\left({x^{(i)}}\right)\right)-\left(1-{y^{(i)}}\right)\log\left(1-{h_\theta}\left({x^{(i)}}\right)\right)]}+\frac{\lambda}{2m}\sum\limits_{j=1}^n{\theta_j^2}" />
</p>

在逻辑回归中，我们只有一个输出变量，又称标量（scalar），也只有一个因变量 _y_ ，但是在神经网络中，我们可以有很多输出变量，我们的 _h<sub>θ</sub>(x)_ 是一个维度为 _K_ 的向量，并且我们训练集中的因变量也是同样维度的一个向量，因此我们的代价函数会比逻辑回归更加复杂一些，为：
<img src="https://latex.codecogs.com/gif.latex?h_\theta\left(x\right)\in&space;\mathbb{R}^{K}&space;{\left({h_\theta}\left(x\right)\right)}_{i}={i}^{th}&space;\text{output}" title="h_\theta\left(x\right)\in \mathbb{R}^{K} {\left({h_\theta}\left(x\right)\right)}_{i}={i}^{th} \text{output}" />

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}J(\Theta)=&-\frac{1}{m}\sum\limits_{i=1}^m\sum\limits_{k=1}^K{\left[{y_k^{(i)}}\log\left({h_\Theta}\left({x^{(i)}}\right)\right)_k&plus;\left(1-{y_k^{(i)}}\right)\log\left(1-{h_\theta}\left({x^{(i)}}\right)\right)_k\right]}\\&space;&&plus;\frac{\lambda}{2m}\sum\limits_{l=1}^{L-1}\sum\limits_{i=1}^{s_l}\sum\limits_{j=1}^{s_{l&plus;1}}{\left(\Theta_{ji}^{(l)})\right)^2}\end{align*}" title="\begin{align*}J(\Theta)=&-\frac{1}{m}\sum\limits_{i=1}^m\sum\limits_{k=1}^K{\left[{y_k^{(i)}}\log\left({h_\Theta}\left({x^{(i)}}\right)\right)_k+\left(1-{y_k^{(i)}}\right)\log\left(1-{h_\theta}\left({x^{(i)}}\right)\right)_k\right]}\\ &+\frac{\lambda}{2m}\sum\limits_{l=1}^{L-1}\sum\limits_{i=1}^{s_l}\sum\limits_{j=1}^{s_{l+1}{\left(\Theta_{ji}^{(l)})\right)^2}\end{align*}" />
</p>

但神经网络代价函数的思想还是和逻辑回归代价函数是一样的，希望通过代价函数来观察算法预测的结果与真实情况的误差有多大，唯一不同的是，对于每一行特征，我们都会给出 _K_ 个预测，基本上我们可以利用循环，对每一行特征都预测 _K_ 个不同结果，然后在利用循环在 _K_ 个预测中选择可能性最高的一个，将其与 _y_ 中的实际数据进行比较。

**注意**：正则化的那一项排除了每一层 _Θ<sub>0</sub>​_ 的和。最里层的循环 _j​_ 循环所有的行（由 _s<sub>l</sub>​_ +1层的激活单元数决定），循环 _i​_ 则循环所有的列，由该层（ _s<sub>l</sub>​_ 层）的激活单元数所决定。即： _h<sub>Θ</sub>(x)​_ 与真实值之间的距离为每个样本-每个类输出的加和，对参数进行正则化（Regularization）的Bias项处理所有参数的平方和。
（注意，_Θ_ 是 以第 _s<sub>l+1</sub>_ 层的激活单元数量为行数，以第 _s<sub>l</sub>+1_ 为列数的矩阵，公式里 i = 1开始，相当于把 _Θ<sub>[:,0]</sub>​_ 忽略了，而 _Θ_ 的行数 _j_ 本身就是从1开始的。）

## 反向传播算法

之前我们在计算神经网络预测结果的时候我们采用了一种正向传播方法，我们从第一层开始正向一层一层进行计算，直到最后一层的 _h<sub>θ</sub>(x)_ 。现在，为了计算代价函数的偏导数 _((∂)/(∂Θ<sup>(l)</sup><sub>ij</sub>)) · J(Θ)_ ，我们需要采用一种反向传播算法，也就是首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层。以一个例子来说明反向传播算法。假设我们的训练集只有一个样本 _(x<sup>(1)</sup>,y<sup>(1)</sup>)_ ，我们的神经网络是一个四层的神经网络，其中 _K=4，S<sub>L</sub>=4，L=4_ ：

前向传播算法：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/2ea8f5ce4c3df931ee49cf8d987ef25d.jpg" />
</p>
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/6a0954ad41f959d7f272e8f53d4ee2de.jpg" />
</p>

由于反向传播使用梯度下降法，需要计算平方误差函数（MSE）对网络权重的导数：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?E=\dfrac{1}{2}(t-y)^{2}" title="E=\dfrac{1}{2}(t-y)^{2}" />
</p>

其中 _E_ 为平方误差，_t_ 为训练样本的目标输出，_y_ 为输出神经元的实际输出。
激活函数 _g_ 一般是非线性可微函数。常用作激活函数的是逻辑函数：

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?g(z)={\frac{1}{1&plus;e^{{-z}}}}" title="g(z)={\frac{1}{1+e^{{-z}}}}" />
</p>

其导数的形式为：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?{\frac{\partial&space;g}{\partial&space;z}}=g(1-g)" title="{\frac{\partial g}{\partial z}}=g(1-g)" />
</p>

首先，我们定义 _δ<sub></sub>j<sup>(l)</sup>_ 为第 _(l)_ 层第 _j_ 个神经元的 误差（Error）。
根据链式法则，可得：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\delta^{(l)}=\dfrac&space;{\partial&space;E}{2a^{(l)}}\cdot&space;\dfrac&space;{\partial&space;a^{(l)}}{\partial&space;z^{(1)}}\cdot\dfrac{\partial&space;z^{(l)}}{\partial\theta^{(l)}}" title="\delta^{(l)}=\dfrac {\partial E}{2a^{(l)}}\cdot \dfrac {\partial a^{(l)}}{\partial z^{(1)}}\cdot\dfrac{\partial z^{(l)}}{\partial\theta^{(l)}}" />
</p>

1. 从最后一层的误差开始计算，误差是激活单元的预测（ _a<sup>(4)</sup>_ ）与实际值（ _y<sup></sup>k_ ）之间的误差（ _k=1:k_ ）。则
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\delta^{\left(4\right)}=\dfrac{\partial&space;E}{\partial&space;a^{(4)}}=a^{(4)}-y" title="\delta^{\left(4\right)}=\dfrac{\partial E}{\partial a^{(4)}}=a^{(4)}-y" />
</p>

2. 利用这个误差值来计算前一层的误差： **_δ<sup>(3)</sup>=(Θ<sup>(3)</sup>)<sup>T</sup>δ<sup>(4)</sup> * g'(z<sup>(3)</sup>)_**，其中：
  * _g'(z<sup>(3)</sup>)_ 是 _S_ 形函数的导数， _g'(z<sup>(3)</sup>)=a<sup>(3)</sup> * (1-a<sup>(3)</sup>)_ 。
  * _(θ<sup>(3)</sup>)<sup>T</sup>δ<sup>(4)</sup>_ 则是权重导致的误差的和。

3. 下一步是继续计算第二层的误差： **_δ<sup>(2)</sup>=(Θ<sup>(2)</sup>)<sup>T</sup>δ<sup>(3)</sup> * g'(z<sup>(2)</sup>)_**

4. 第一层是输入变量，不存在误差。

我们有了所有的误差的表达式后，便可以计算代价函数的偏导数了，假设 _λ=0_ ，即我们不做任何正则化处理时有：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_{j}^{(l)}&space;\delta_{i}^{l&plus;1}" title="\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_{j}^{(l)} \delta_{i}^{l+1}" />
</p>


[回到顶部](#神经网络)
