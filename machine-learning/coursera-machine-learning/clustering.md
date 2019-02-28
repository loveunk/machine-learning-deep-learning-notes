# 聚类算法

在机器学习绪论中讲过，聚类算法属于无监督算法。
聚类算法在工作中比较常见。其中比较基础的算法包括K-Means，DBScan等等。

## K-Means

K-Means（K-均值）是很普及的一种的聚类算法，算法接受一个未标记的数据集，然后将数据聚类成多个不同的组。

K-Means是一个迭代算法，假设我们想要将数据聚类成 _n_ 个组，其方法为：
1. 首先选择 _K_ 个随机的点，称为聚类中心（Cluster centroids）；
2. 对于数据集中的每一个数据，分别计算其与 _K_ 个中心点的距离，选择距离最近的中心点。将该数据与此中心点关联起来。所有与同一个中心点关联的所有点聚成一类。
3. 计算每一组的平均值，将该组所关联的中心点移动到平均值的位置。

重复上述步骤2-3直到中心点不再变化。

下面几幅图是一个示例。

第一步，随机选择三个初始点（蓝色叉的位置），并依次计算每个数据点距离哪个初始点的位置最近。其实被聚类的数据分别标识红色绿色和蓝色：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/ff1db77ec2e83b592bbe1c4153586120.jpg" />
</p>

重新计算了一次中心点，并且重新对每个数据划分类之后，再次计算了中心点。结果如下图：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/acdb3ac44f1fe61ff3b5a77d5a4895a1.jpg" />
</p>

之后，再次对每个数据计算其所属分类，并重新计算中心点，重复这个过程两次后就得到下图的结果。可以看到分类效果还是不错的。
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/fe6dd7acf1a1eddcd09da362ecdf976f.jpg" />
</p>

下面，将算法用代码表示：
用 _μ<sup>1</sup>_ , _μ<sup>2</sup>_ ,..., _μ<sup>k</sup>_ 来表示聚类中心，用 _c<sup>(1)</sup>_ , _c<sup>(2)</sup>_ ,..., _c<sup>(m)</sup>_ 来存储与第 _i_ 个实例数据最近的聚类中心的索引，K-均值算法的伪代码如下：

```
Repeat {
  for i = 1 to m
    c(i) := index (from 1 to K) of cluster centroid closest to x(i)

  for k = 1 to K
    mu_k := average (mean) of points assigned to cluster k
}
```

算法分为两个步骤
1. 第一个`for`循环是赋值步骤，即：
   * 对于每一个样例 _i_ ，计算其应该属于的类
2. 第二个for循环是聚类中心的移动，即：
   * 对于每一个类 _K_ ，重新计算该类的质心


## 优化目标
K-Means优化目标是最小化所有的数据点与其关联的聚类中心点之间的距离之和，因此K-Means的代价函数（又称畸变函数 Distortion function）为：

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?J(c^{(1)},...,c^{(m)},\mu_1,...,\mu_K)=\dfrac{1}{m}\sum^{m}_{i=1}\left|X^{\left(&space;i\right)}-\mu{c^{(i)}}\right|^{2}" title="J(c^{(1)},...,c^{(m)},\mu_1,...,\mu_K)=\dfrac{1}{m}\sum^{m}_{i=1}\left|X^{\left( i\right)}-\mu{c^{(i)}}\right|^{2}" />
</p>

其中 _μ<sub>c<sup>(i)</sup></sub>_ 代表与 _x<sup>(i)</sup>_ 最近的聚类中心点。

优化目标是找出使得代价函数最小的 _c<sup>(1)</sup>_ , _c<sup>(2)</sup>_ ,..., _c<sup>(m)</sup>_ 和 _μ<sup>1</sup>_ , _μ<sup>2</sup>_ ,..., _μ<sup>k</sup>_。

K-Means算法，第一个循环是用于减小 _c<sup>(i)</sup>_ 引起的代价，而第二个循环则是用于减小 _μ<sub>i</sub>_ 引起的代价。迭代的过程一定会是每一次迭代都在减小代价函数，不然便是出现了错误。

## 随机初始化

在运行K-Means算法之前，首先要随机初始化所有的聚类中心：
1. 我们应该选择 _K < m_ ，即聚类中心点的个数要小于所有训练集实例的数量
2. 随机选择 _K_ 个训练实例，然后令 _K_ 个聚类中心分别与这 _K_ 个训练实例相等

K-Means的一个问题在于，它有可能会停留在一个局部最小值处，而这取决于初始化的情况。例如下图的情况：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/d4d2c3edbdd8915f4e9d254d2a47d9c7.png" />
</p>

为了解决 局部最小化 的问题，通常需要多次运行K-K-Means算法，每次都重新进行随机初始化，最后再比较多次运行K-Means的结果，选择代价函数最小的结果。
这种方法在 _K_ 较小的时候（2-10）还是可行的，但是**如果 _K_ 较大，这么做也可能不会有明显地改善**。

## 选择聚类数

没有最好的选择聚类数的方法，通常是需要根据不同的问题，人工进行选择。

选择的时候思考运用K-Means算法聚类的动机是什么，然后选择能最好服务于该目的标聚类数。

一个可能的方法叫作“肘部法则（Elbow method）”：
主要过程是改变 _K_ 值。运行 _K_ 个聚类的方法。

意味着，所有的数据都会分到一个聚类里，然后计算成本函数或者计算畸变函数 _J_ 。
 _K_ 代表聚类数字。

<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/f3ddc6d751cab7aba7a6f8f44794e975.png" />
</p>

应用这种方法，可能会得到一条类似于左上图这样的曲线。像一个人的肘部。
这种模式，它的畸变值会迅速下降，从1到2，从2到3之后，你会在3的时候达到一个肘点。
在此之后，畸变值下降的非常慢，看起来使用3个cluster来聚类是正确的，因为那个点是曲线的肘点，畸变值下降得很快， _K=3_ 之后就下降得很慢。

当应用“肘部法则”时，如果得到了一个像上图左图，那是一种用来选择聚类个数的合理方法。

更多的时候划分为多少个Clusters，取决于实际的应用场景：
> 制造T-恤的例子中，要将用户按照身材聚类，可以分成3个尺寸: _S,M,L_ ，也可以分成5个尺寸 _XS,S,M,L,XL_ ，这样的选择是建立在“聚类后制造的T-恤是否能较好地适合客户”这个问题的基础上。
