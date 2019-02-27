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
