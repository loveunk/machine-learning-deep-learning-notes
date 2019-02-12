# PCA (Principal Component Analysis) - 主成分分析
PCA是一种数据线性降维的方法，在学习PCA之前，先回顾一些基础知识。

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [PCA (Principal Component Analysis) - 主成分分析](#pca-principal-component-analysis-主成分分析)
	- [Varianes & Covariances 方差和协方差](#varianes-covariances-方差和协方差)
		- [Variance 方差](#variance-方差)
		- [Covariance 协方差](#covariance-协方差)
		- [Rules 方差规则](#rules-方差规则)
	- [Product](#product)
		- [Dot product 点积](#dot-product-点积)
			- [Algebraic definition 代数定义](#algebraic-definition-代数定义)
			- [Geometric definition 几何定义](#geometric-definition-几何定义)
		- [Inner product 内积](#inner-product-内积)
			- [Inner product properties 内积性质](#inner-product-properties-内积性质)
			- [Inner product of functions 函数内积](#inner-product-of-functions-函数内积)
			- [Inner product of random variables 随机变量内积](#inner-product-of-random-variables-随机变量内积)
	- [Projection 投影](#projection-投影)
		- [Projection onto 1D subspaces 投影到一维空间](#projection-onto-1d-subspaces-投影到一维空间)
		- [Projections onto higher-dimentional subspaces 投影到高维空间](#projections-onto-higher-dimentional-subspaces-投影到高维空间)
	- [PCA](#pca)
		- [PCA derivation - PCA推导](#pca-derivation-pca推导)
		- [PCA algorithm PCA算法](#pca-algorithm-pca算法)
			- [Steps of PCA PCA步骤](#steps-of-pca-pca步骤)
			- [High-dimentional PCA - 高维空间PCA](#high-dimentional-pca-高维空间pca)
	- [References](#references)

<!-- /TOC -->

## Varianes & Covariances 方差和协方差
### Variance 方差
<p align="center"><img src="/pca/tex/2c0023d97e0a0cd74c1618cc74f59e4a.svg?invert_in_darkmode&sanitize=true" align=middle width=266.968284pt height=47.60747145pt/></p>
<p align="center"><img src="/pca/tex/9f69ad824f6a6e2c37c1e6776151211d.svg?invert_in_darkmode&sanitize=true" align=middle width=141.76182405pt height=19.726228499999998pt/></p>

### Covariance 协方差
<p align="center"><img src="/pca/tex/87cc7434857420b0abe965cef61647ab.svg?invert_in_darkmode&sanitize=true" align=middle width=414.02777295pt height=17.031940199999998pt/></p>

For 2D data, the Covariance matrix is as follow
<p align="center"><img src="/pca/tex/d0d2ae380c0d5a39d1d9ba0f8452a6a1.svg?invert_in_darkmode&sanitize=true" align=middle width=173.8088847pt height=39.452455349999994pt/></p>

### Rules 方差规则
* <img src="/pca/tex/8807d0472c4891ea38773c7e8fa97e55.svg?invert_in_darkmode&sanitize=true" align=middle width=156.70361354999997pt height=24.65753399999998pt/>
* <img src="/pca/tex/200f2e9e884fc1ac0451588e3164cbbb.svg?invert_in_darkmode&sanitize=true" align=middle width=156.45072464999998pt height=26.76175259999998pt/>

For matrix <img src="/pca/tex/c40a373054f74758866fb2c35baf329d.svg?invert_in_darkmode&sanitize=true" align=middle width=196.09738005pt height=24.65753399999998pt/>
* <img src="/pca/tex/ab091ab2d66cde1f6b86c44b9f80a9d7.svg?invert_in_darkmode&sanitize=true" align=middle width=201.58937039999998pt height=27.6567522pt/>

## Product
### Dot product 点积
#### Algebraic definition 代数定义
<p align="center"><img src="/pca/tex/312d2f414d1bd623f930852764eb9972.svg?invert_in_darkmode&sanitize=true" align=middle width=186.04350929999998pt height=48.18280005pt/></p>

#### Geometric definition 几何定义
<p align="center"><img src="/pca/tex/d1550d19537fa87033aae6c3ac37bbe6.svg?invert_in_darkmode&sanitize=true" align=middle width=160.5096801pt height=18.7598829pt/></p>

### Inner product 内积
定义：对于 <img src="/pca/tex/a0a901384136988a9d6d78e56ddbdbf5.svg?invert_in_darkmode&sanitize=true" align=middle width=58.68325154999999pt height=22.465723500000017pt/>，内积 <img src="/pca/tex/97b1a397fad4dd310d999b59a0255d0d.svg?invert_in_darkmode&sanitize=true" align=middle width=183.67926885pt height=24.65753399999998pt/>，内积具有如下性质：
* Bilinear
  * <img src="/pca/tex/51a7774b54a0c8674fa6daf43423c868.svg?invert_in_darkmode&sanitize=true" align=middle width=203.02496114999997pt height=24.65753399999998pt/>
  * <img src="/pca/tex/12c1b6734072212ed337826ac49018a9.svg?invert_in_darkmode&sanitize=true" align=middle width=203.77074299999998pt height=24.65753399999998pt/>
* Positive definite
  *  <img src="/pca/tex/99c8d938e3892d7c04d582304b66934c.svg?invert_in_darkmode&sanitize=true" align=middle width=210.44457719999997pt height=24.65753399999998pt/>
* Symmetric
  * <img src="/pca/tex/4a48cdf0f489164ef3ba7020880d066a.svg?invert_in_darkmode&sanitize=true" align=middle width=98.1886521pt height=24.65753399999998pt/>

如果定义 <img src="/pca/tex/b7911fe0e9dbfaf20f81aa5ad5a26229.svg?invert_in_darkmode&sanitize=true" align=middle width=100.78174589999998pt height=27.6567522pt/>，当<img src="/pca/tex/6cba520138110bd6f4fe5ebaf7498303.svg?invert_in_darkmode&sanitize=true" align=middle width=42.762416399999985pt height=22.465723500000017pt/>，则其和x，y的点积一致，否则不同。

#### Inner product properties 内积性质
* <img src="/pca/tex/8dc33a58bec631b6a60db1eea1137a61.svg?invert_in_darkmode&sanitize=true" align=middle width=117.41999444999998pt height=24.65753399999998pt/>
* <img src="/pca/tex/ad0051711b5ac0c26c3d1f0bba3e10da.svg?invert_in_darkmode&sanitize=true" align=middle width=152.98308959999997pt height=24.65753399999998pt/>
* <img src="/pca/tex/bf49d4b9fcd341637305404332117c24.svg?invert_in_darkmode&sanitize=true" align=middle width=135.63155759999998pt height=24.65753399999998pt/>

计算角度
* <img src="/pca/tex/6e22d99d884fa03d39ae405e5030a15f.svg?invert_in_darkmode&sanitize=true" align=middle width=105.72848549999998pt height=33.20539859999999pt/>

#### Inner product of functions 函数内积
Example:
<p align="center"><img src="/pca/tex/f94f1f13be99495500d81c95bc1f92fe.svg?invert_in_darkmode&sanitize=true" align=middle width=176.99632334999998pt height=41.27894265pt/></p>
In this example, <img src="/pca/tex/7732d8a998f72cec1479ba40d01e0b0b.svg?invert_in_darkmode&sanitize=true" align=middle width=355.4589686999999pt height=24.65753399999998pt/>

#### Inner product of random variables 随机变量内积
Example:
<p align="center"><img src="/pca/tex/0d95067884b91d03b1610d2f7331d2e2.svg?invert_in_darkmode&sanitize=true" align=middle width=118.17536279999999pt height=16.438356pt/></p>
where <img src="/pca/tex/de97e079871785548f8ce73f6866993d.svg?invert_in_darkmode&sanitize=true" align=middle width=260.96807549999994pt height=29.424786600000015pt/> and <img src="/pca/tex/4a37536969a66463b2b6b4b051ea2cb7.svg?invert_in_darkmode&sanitize=true" align=middle width=80.24925479999999pt height=24.65753399999998pt/>

## Projection 投影
### Projection onto 1D subspaces 投影到一维空间
<p align="center">
  <img src="img/projection-onto-1d-subspace.png" width="300" />
</p>

投影后的向量 <img src="/pca/tex/730f68f8efd606b60ba487aefb60aea2.svg?invert_in_darkmode&sanitize=true" align=middle width=40.14480194999999pt height=24.65753399999998pt/> 具有如下两点属性:
1. <img src="/pca/tex/81b39ee97368c07ac07842ef24c9a9d4.svg?invert_in_darkmode&sanitize=true" align=middle width=145.829211pt height=24.65753399999998pt/>. (as <img src="/pca/tex/eedd82c01c75f648480cef748fc81d04.svg?invert_in_darkmode&sanitize=true" align=middle width=72.10811849999999pt height=24.65753399999998pt/>)
2. <img src="/pca/tex/15d243b4874282a70cf8a1c0e228e388.svg?invert_in_darkmode&sanitize=true" align=middle width=126.91393439999999pt height=24.65753399999998pt/> (orthogonality)

Then, we get
<p align="center"><img src="/pca/tex/4ec4208967db241519476d7e440e95aa.svg?invert_in_darkmode&sanitize=true" align=middle width=108.09674865pt height=40.33452225pt/></p>
推导如下：
<p align="center"><img src="/pca/tex/a82fc21279570d7873a5d291b7e38c2f.svg?invert_in_darkmode&sanitize=true" align=middle width=230.12533169999998pt height=190.65162105pt/></p>

### Projections onto higher-dimentional subspaces 投影到高维空间
<p align="center">
  <img src="img/projection-onto-2d-subspace.png" width="300" />
</p>

投影后的向量 <img src="/pca/tex/730f68f8efd606b60ba487aefb60aea2.svg?invert_in_darkmode&sanitize=true" align=middle width=40.14480194999999pt height=24.65753399999998pt/> 具有如下两点属性:
1. <img src="/pca/tex/c4301450c4c98622fdce3bc427a96073.svg?invert_in_darkmode&sanitize=true" align=middle width=198.16092614999997pt height=32.256008400000006pt/>
2. <img src="/pca/tex/482dd2e16eec3358f4da18af4514cbef.svg?invert_in_darkmode&sanitize=true" align=middle width=221.54285669999996pt height=24.65753399999998pt/> (orthogonality)

where <img src="/pca/tex/4862bc7dff092e4d615aac59f705af07.svg?invert_in_darkmode&sanitize=true" align=middle width=75.94082099999999pt height=77.26096289999998pt/>, <img src="/pca/tex/746ccb5a2c0d33096aaec720e2925c4c.svg?invert_in_darkmode&sanitize=true" align=middle width=107.81621894999998pt height=27.94539330000001pt/>

推导如下：
<p align="center"><img src="/pca/tex/808301ef48677b2fa6fed66bef3a1a80.svg?invert_in_darkmode&sanitize=true" align=middle width=278.2362759pt height=188.51225415pt/></p>

## PCA
### PCA derivation - PCA推导
**问题描述**：对于点集合 <img src="/pca/tex/500a0e9bbdb59bc7107074da65cc7e1f.svg?invert_in_darkmode&sanitize=true" align=middle width=168.23207445pt height=27.6567522pt/>，
定义是低维空间坐标系 <img src="/pca/tex/b8b1f48f2faeaf683afe0d69b875188c.svg?invert_in_darkmode&sanitize=true" align=middle width=112.3823085pt height=24.65753399999998pt/> 。其中 <img src="/pca/tex/51d7540736efae5d5b0123af77ff8f0a.svg?invert_in_darkmode&sanitize=true" align=middle width=53.72357759999999pt height=22.465723500000017pt/>， <img src="/pca/tex/d3aa71141bc89a24937c86ec1d350a7c.svg?invert_in_darkmode&sanitize=true" align=middle width=11.705695649999988pt height=22.831056599999986pt/> 是正交基，<img src="/pca/tex/3d13090ef3ed1448f3c4dc166d06ab4d.svg?invert_in_darkmode&sanitize=true" align=middle width=13.948864049999989pt height=22.831056599999986pt/>是正交基系数。
希望找到一个映射集合 <img src="/pca/tex/29b35d554c6c42b8e2ddb82722130105.svg?invert_in_darkmode&sanitize=true" align=middle width=55.12774079999999pt height=27.6567522pt/>。有如下 **公式(_A_)**：
<p align="center"><img src="/pca/tex/187f12da59988b1e47702ffd47870587.svg?invert_in_darkmode&sanitize=true" align=middle width=101.3470095pt height=47.806078649999996pt/></p>

假设使用的是点积，<img src="/pca/tex/0060675245d6343a9bcdc3efc9137c76.svg?invert_in_darkmode&sanitize=true" align=middle width=56.51844989999999pt height=22.831056599999986pt/> 和 <img src="/pca/tex/d3aa71141bc89a24937c86ec1d350a7c.svg?invert_in_darkmode&sanitize=true" align=middle width=11.705695649999988pt height=22.831056599999986pt/> 正交，那么得到 **公式(_B_)**：
<p align="center"><img src="/pca/tex/af10c86f1351a41ba6bb94104f052715.svg?invert_in_darkmode&sanitize=true" align=middle width=76.27068735pt height=18.7141317pt/></p>

<img src="/pca/tex/25950a2e429f93f60687aa0568aed862.svg?invert_in_darkmode&sanitize=true" align=middle width=122.46196049999999pt height=27.6567522pt/> 是 <img src="/pca/tex/7da75f4e61cdeabf944740206b511812.svg?invert_in_darkmode&sanitize=true" align=middle width=14.132466149999988pt height=22.465723500000017pt/> 在低维空间<img src="/pca/tex/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode&sanitize=true" align=middle width=13.29340979999999pt height=22.465723500000017pt/>上的投影的坐标值，称为coordinates或code。可得
<p align="center"><img src="/pca/tex/adc9c68353e0ed32c71d7f9986963b47.svg?invert_in_darkmode&sanitize=true" align=middle width=75.56098275000001pt height=17.8466442pt/></p>

对于PCA问题，其**优化目标**为：样本点到新的超平面上的距离足够近，等于最小化下面的成本函数，**公式(_C_)**：
<p align="center"><img src="/pca/tex/e124d6ed48be0c32dd59b4a747753f2d.svg?invert_in_darkmode&sanitize=true" align=middle width=166.8845706pt height=47.60747145pt/></p>

因此可得 **公式(_D_)**：
<p align="center"><img src="/pca/tex/5790eb8e55dc8f39e4e337038e886fd0.svg?invert_in_darkmode&sanitize=true" align=middle width=165.43991145pt height=36.2778141pt/></p>

**公式(_E_)**：
<p align="center"><img src="/pca/tex/38a795eda848f8cf9b136da4027c599a.svg?invert_in_darkmode&sanitize=true" align=middle width=68.13308864999999pt height=37.0084374pt/></p>

由(D), (E)可得

<p align="center"><img src="/pca/tex/373b7d7b36b859f6286b1246924191a1.svg?invert_in_darkmode&sanitize=true" align=middle width=215.6812515pt height=151.20189974999997pt/></p>

由(A), (B)可得
<p align="center"><img src="/pca/tex/21cd3604bc6f8dac91959455eea2f451.svg?invert_in_darkmode&sanitize=true" align=middle width=297.84528509999996pt height=124.93263584999998pt/></p>

**公式(_F_)**：
<p align="center"><img src="/pca/tex/3237c0592f14f17f6177cf5f43a54685.svg?invert_in_darkmode&sanitize=true" align=middle width=339.50969745pt height=59.1786591pt/></p>

由(_C_), (_F_)可得
<p align="center"><img src="/pca/tex/693fb7bf62bfdbc978b5e8a6f1079c6d.svg?invert_in_darkmode&sanitize=true" align=middle width=342.14835105pt height=346.56774075pt/></p>

**公式(_G_)**：
<p align="center"><img src="/pca/tex/572ef034ce558fbede1b8cd24235def3.svg?invert_in_darkmode&sanitize=true" align=middle width=127.2014238pt height=50.2924224pt/></p>

上式等于将数据的协方差矩阵 _S_ 投影到子空间 <img src="/pca/tex/f10f982c502c8d0f7703d188d11194e5.svg?invert_in_darkmode&sanitize=true" align=middle width=47.01774825pt height=27.6567522pt/> 中，因此 <img src="/pca/tex/d530021e5e2a93bd8ddb91ea547ec699.svg?invert_in_darkmode&sanitize=true" align=middle width=56.92671929999998pt height=24.65753399999998pt/> 等于投影到该子空间后的数据的方差最小化。

由(G)构造拉格朗日函数，其中 <img src="/pca/tex/48a5d5590c242598ccdc7dba996bd880.svg?invert_in_darkmode&sanitize=true" align=middle width=147.47822924999997pt height=27.6567522pt/> ，得到**公式(_H_)**：
<p align="center"><img src="/pca/tex/eec7f9ab4c2cc45a22d654f31be37b50.svg?invert_in_darkmode&sanitize=true" align=middle width=350.07059834999995pt height=99.85425615pt/></p>

由(_G_), (_H_)可得
<p align="center"><img src="/pca/tex/600eaf3c1faf71ea31c86790cacc1fc8.svg?invert_in_darkmode&sanitize=true" align=middle width=101.1381591pt height=50.2924224pt/></p>

所以在忽略的子空间里要选那些比较小的特征值，在主子空间选那些大的特征值。

This nicely aligns with properties of the covariance matrix. The eigen vectors of the covariance matrix are orthogonal to each other because of symmetry and the eigen vector belonging to the largest eigen value points in the direction of the data with the largest variance and the variance in that direction is given by the corresponding eigen value.

### PCA algorithm PCA算法
#### Steps of PCA PCA步骤
1. **数据预归一化** (normalization)
	1. **每列数据减该列平均值(mean)**, to avoid numerial problems
	2. **每列数据除该列标准差(std)**，使数据无单位（unit-free）且方差为1
	<p align="center"><img src="/pca/tex/0fa38c7ad69c18b991ec87f48037e468.svg?invert_in_darkmode&sanitize=true" align=middle width=130.1462943pt height=39.860229749999995pt/></p>
2. 计算数据**协方差矩阵**（covariance matrix）和**该矩阵**对应的**特征值**、**特征向量**（eigenvalues, eigenvectors）
	* <img src="/pca/tex/2d02b5fb9f2f4b64ecb656edab259639.svg?invert_in_darkmode&sanitize=true" align=middle width=161.56178939999998pt height=27.6567522pt/>
	* _B_ 是由特征向量作为列的矩阵，其中特征向量对应的是最大的特征值

#### High-dimentional PCA - 高维空间PCA
对于 <img src="/pca/tex/7e773f53227121ebad7ef454d6b9a93c.svg?invert_in_darkmode&sanitize=true" align=middle width=145.5927825pt height=78.37837259999999pt/> 如果 <img src="/pca/tex/f5accc43a9e4e0703d65ff8bebb3190a.svg?invert_in_darkmode&sanitize=true" align=middle width=54.63679979999999pt height=22.465723500000017pt/>，那么 _X_ 的协方差矩阵 _S_ 的秩为 _N_。那么 _S_ 有 _D-N+1_ 个特征值为0，其非满秩矩阵。

 下面考虑如何把 _S_ 转换为满秩矩阵 _E_：
 <p align="center"><img src="/pca/tex/731fac87d2ea7b56ca6b97ee2008e7eb.svg?invert_in_darkmode&sanitize=true" align=middle width=155.27622824999997pt height=57.34033469999999pt/></p>

其中 <img src="/pca/tex/83562e501a80dd5df6b6c0e433f10afa.svg?invert_in_darkmode&sanitize=true" align=middle width=61.118593799999985pt height=22.831056599999986pt/>，在变换后，_E_ 为满秩矩阵，由PCA 的计算方法可以得到 _E_ 对应的特征向量 <img src="/pca/tex/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode&sanitize=true" align=middle width=11.76470294999999pt height=14.15524440000002pt/>，但这里需要计算 _S_ 对应的特征向量。再次变换上式：
<p align="center"><img src="/pca/tex/5679dc38151a63db1329aa4c9f87bbd3.svg?invert_in_darkmode&sanitize=true" align=middle width=183.56396684999999pt height=55.428486299999996pt/></p>

所以 _S_ 的特征向量为 <img src="/pca/tex/681039f030ab2ab9dfed008b8b687f2a.svg?invert_in_darkmode&sanitize=true" align=middle width=37.02896339999999pt height=27.6567522pt/>。

## References
1. [PCA chapter of "Mathematics for Machine Learning"](https://mml-book.github.io/book/chapter10.pdf)
