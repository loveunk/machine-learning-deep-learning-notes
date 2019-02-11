# PCA (Principal Component Analysis) - 主成分分析
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [PCA (Principal Component Analysis) - 主成分分析](#pca-principal-component-analysis-主成分分析)
	- [Varianes & Covariances 方差 & 协方差](#varianes-covariances-方差-协方差)
		- [Variance 方差](#variance-方差)
		- [Covariance 协方差](#covariance-协方差)
		- [Rules 方差规则](#rules-方差规则)
	- [Product](#product)
		- [Dot product](#dot-product)
			- [Algebraic definition 代数定义](#algebraic-definition-代数定义)
			- [Geometric definition 几何定义](#geometric-definition-几何定义)
		- [Inner product 内积](#inner-product-内积)
			- [Inner product properties](#inner-product-properties)
			- [Inner product of functions](#inner-product-of-functions)
			- [Inner product of random variables](#inner-product-of-random-variables)
	- [Projection 投影](#projection-投影)
		- [Projection onto 1D subspaces 投影到一维空间](#projection-onto-1d-subspaces-投影到一维空间)
		- [Projections onto higher-dimentional subspaces 投影到高维空间](#projections-onto-higher-dimentional-subspaces-投影到高维空间)
	- [PCA derivation - PCA推导](#pca-derivation-pca推导)
	- [PCA algorithm - PCA算法](#pca-algorithm-pca算法)
		- [Steps of PCA - PCA步骤](#steps-of-pca-pca步骤)
		- [高维空间的PCA](#高维空间的pca)
	- [References](#references)

<!-- /TOC -->
PCA是一种数据线性降维的方法，在学习PCA之前，先回顾一些基础知识。

## Varianes & Covariances 方差 & 协方差
### Variance 方差
$$ Var[X] = \frac{1}{N} \sum_{n=1}^{N}(x_n - \mu)^2, \mu = E[X] $$
$$ Std[X] = \sqrt{Var[X]} $$

### Covariance 协方差
$$ Cov[X, Y] = E[(X-\mu_x)(Y-\mu_y)], \mu_x = E[X], \mu_y = E[Y] $$

For 2D data, the Covariance matrix is as follow
$$ \begin{bmatrix} var\left[ X\right] & cov\left[X, Y\right] \\ cov\left[ X,Y\right] & var\left[ Y\right] \end{bmatrix} $$

### Rules 方差规则
* $Var[D] = Var[D + a]$
* $Var[\alpha D] = \alpha^2 Var[D]$

For matrix $D = \{x_1, x_2, ..., x_n\}, x \in R^p$
* $Var[AD + b] = A Var[D] A^T$

## Product
### Dot product
#### Algebraic definition 代数定义
$$ x^Ty = \sum_{d=1}^{D} x_d y_d, x,y\in R^D $$

#### Geometric definition 几何定义
$$ x^Ty = ||x|| \cdot ||y|| cos(\theta) $$

### Inner product 内积
定义：对于 $x , y\in V$，内积 $\langle x, y \rangle的定义为x, y 到实数R的映射: V\times V -> R$，内积具有如下性质：
* Bilinear
  * $\langle \lambda x + z, y \rangle = \lambda \langle x, y \rangle + \langle z, y \rangle$
  * $\langle x, \lambda y + z\rangle = \lambda \langle x, y \rangle + \langle x, z \rangle$
* Positive definite
  *  $\langle x, x \rangle \geq 0, \langle x,x\rangle = 0 \Leftrightarrow x = 0$
* Symmetric
  * $\langle x, y \rangle = \langle y, x \rangle$

如果定义 $\langle x, y \rangle = x^TAy$，当$A=I$，则其和x，y的点积一致，否则不同。

#### Inner product properties
* $||\lambda x|| = |\lambda| \cdot ||x||$
* $||x + y|| \leq ||x|| + ||y||$
* $|\langle x, y\rangle| \leq ||x|| \cdot ||y||$

计算角度
* $cos(w) = \frac{\langle x, y\rangle}{|x|\cdot|y|}$

#### Inner product of functions
Example:
$$ \langle u, v \rangle = \int ^{b}_{a}u\left( x\right) v\left( x\right)dx $$
In this example, $u(x) = sin(x), v(x) = cos(x), f(x) = sin(x)cos(x)$

#### Inner product of random variables
Example:
$$ \langle x, y \rangle = cov [x, y] $$
where $||x|| = \sqrt{cov[x,x]} = \sqrt{var[x]} = \sigma(x)$ and $||y|| = \sigma(y)$

## Projection 投影
### Projection onto 1D subspaces 投影到一维空间
<p align="center">
  <img src="img/projection-onto-1d-subspace.png" width="300" />
</p>

投影后的向量 $\pi_u(x)$ 具有如下两点属性:
1. $\exists \lambda \in \mathbb{R}: \pi _{u}\left( x\right) =\lambda b$. (as $\pi_u(x) \in \mathbb{U}$)
2. $\langle b, \pi_u(x)-x\rangle = 0$ (orthogonality)

Then, we get
$$ \pi_u(x) = \frac{bb^T}{||b||^2}x $$
推导如下：
$$ \begin{aligned}
&\Rightarrow \langle b,\pi _{u}\left( X\right) -x\rangle =0\\ &\Leftrightarrow \langle b,\pi _{u}\left( X\right) \rangle -\langle b,x\rangle =0\\
&\Leftrightarrow \langle b,\lambda b\rangle -\langle b,x\rangle =0\\ &\Leftrightarrow \lambda \left\| b\right\| ^{2}-\langle b,x\rangle =0\\ &\Leftrightarrow \lambda =\dfrac {\langle b, x\rangle }{\left\| b\right\| ^{2}}\\
&\Rightarrow \pi _{u}\left( x\right) =\lambda b = \frac{b^Txb}{||b||^2} = \frac{bb^T}{||b||^2}x
\end{aligned} $$

### Projections onto higher-dimentional subspaces 投影到高维空间
<p align="center">
  <img src="img/projection-onto-2d-subspace.png" width="300" />
</p>

投影后的向量 $\pi_u(x)$ 具有如下两点属性:
1. $\exists \lambda \in \mathbb{R}: \pi _{u}\left( x\right) =\sum_{i=1}^M\lambda_i b_i$
2. $\langle \pi_u(x)-x, b_i\rangle = 0, i=1,...,M$ (orthogonality)

where $\lambda =\begin{bmatrix} \lambda _{xi} \\ \vdots \\ \lambda _{m} \end{bmatrix}$, $B = \begin{bmatrix}b_1 | ... | b_M\end{bmatrix}$

推导如下：
$$ \begin{aligned}
&\Rightarrow \pi _{u}\left( x\right) =B\lambda \\
&\Leftrightarrow \langle B\lambda -X,b_{i}\rangle =0\\
&\Leftrightarrow \lambda ^{T}B^{T}bi-X^{T}b_{i}=0,i=1,2,\ldots ,M\\
&\Leftrightarrow \lambda ^{T}B^{T}B-X^{T}B=0\\
&\Leftrightarrow \lambda ^{T}=X^{T}B\left( B^{T}B\right) ^{-1}\\
&\Leftrightarrow \lambda =\left( B^{T}B\right) ^{-1}B^{T}X \\
&\Rightarrow \pi _{u} =B\lambda = B\left( B^{T}B\right) ^{-1}B^{T}X
\end{aligned} $$

## PCA derivation - PCA推导
**问题描述**：对于点集合 $\mathcal{X} = {x_1, ..., x_N}, x_i \in \mathbb{R}^D$，
定义是低维空间坐标系 $B=(b_1,...,b_M)$ 。其中 $M < D$， $b_i$ 是正交基，$\beta_i$是正交基系数。
希望找到一个映射集合 $\tilde{x} \in \mathbb{R}^M$。
有如下推导：

$$ \tilde{x}_n = \sum_{i=1}^D\beta_{in}b_i \tag{A} $$

假设使用的是点积，$\beta_{D(D\neq i)}$ 和 $b_i$ 正交，那么
$$ \beta_{in} = x_n^Tb_i \tag{B} $$

$\mathcal{z}_n = B^TX \in \mathbb{R}^M$ 是 $\mathcal{X}$ 在低维空间$B$上的投影的坐标值，称为coordinates或code。可得
$$ \tilde{x} = BB^T\mathcal{x} $$

对于PCA问题，其**优化目标**为：样本点到新的超平面上的距离足够近，等于最小化下面的成本函数：
$$\mathcal{J} = \dfrac{1}{N}\sum_{n=1}^{N}||x_n - \tilde{x}_n||^2 \tag{C} $$

因此
$$ \dfrac {\partial J}{\partial \tilde{x}_{n}} = -\dfrac {2}{N}\left( x_{n}-\tilde{x}_{n}\right) ^{T} \tag{D} $$

$$ \dfrac {\partial \tilde{x}_n}{\partial \beta_{in}} = b_i \tag{E} $$

由(D), (E)可得

$$ \begin{aligned}
\dfrac {\partial J}{\partial \beta_{in}} &= -\frac{2}{N}(x_n - \sum_{j=1}^M\beta_{jn}b_j)^T \\&= -\frac{2}{N}(x_n^Tb_i - \beta_{in}b_i^Tb_i)^T \\ &= -\dfrac{2}{N}(x_n^Tb_i - \beta_{in}) \\
&= 0
\end{aligned}$$

由(A), (B)可得
$$ \begin{aligned}
\tilde{x}_n &= \sum_{j=1}^M b_j(b_j^Tx_n) = \left(\sum_{j=1}^M b_j b_j^T \right)x_n \\
x_n &= \left(\sum_{j=1}^M b_j b_j^T \right)x_n + \left(\sum_{j=M+1}^D b_jb_j^T \right)x_n
\end{aligned}
$$

$$ x_n - \tilde{x}_n = \left(\sum_{j=M+1}^D b_jb_j^T \right)x_n = \sum_{j=M+1}^D (b_j^Tx_n)b_j
\tag{F} $$

由(_C_), (_F_)可得
$$
\begin{aligned}
\mathcal{J} &= \dfrac{1}{N}\sum_{n=1}^{N}||\sum_{j=M+1}^D (b_j^Tx_n)b_j||^2 \\
&= \dfrac{1}{N}\sum_{n=1}^{N}\sum_{j=M+1}^D (b_j^T x_n)^2 \\
&= \dfrac{1}{N}\sum_{n=1}^{N}\sum_{j=M+1}^D b_j^T x_n x_n^T b_j \\
&= \sum_{j=M+1}^{D} b_j^T \left( \underset{\mathcal{S} = cov[x, x]}{\underbrace{\dfrac{1}{N} \sum_{n=1}^N x_n x_n^T }}\right) b_j \\
&= \sum_{j=M+1}^D b_j^T \mathcal{S} b_j = trace \left(\left(\sum_{j=M+1}^D b_j^T b_j\right) \mathcal{S}\right)
\end{aligned}$$

$$\mathcal{J}  = \sum_{j=M+1}^D b_j^T \mathcal{S} b_j \tag{G}$$

上式等于将数据的协方差矩阵 _S_ 投影到子空间 $\mathbb{R}^{D-M}$ 中，因此 $min(\mathcal{J})$ 等于投影到该子空间后的数据的方差最小化。

由(G)构造拉格朗日函数，其中 $b_i \in \mathbb{R}^{M}, b_j \in \mathbb{R}^{D-M}$ ：
$$ \begin{aligned}
L &= b_j^{T}Sb_j+\lambda \left( 1-b_j^{T}b_j\right) \\
&\Rightarrow
\begin{cases}
\dfrac {\partial L}{\partial \lambda }=1-b_j^{T}b=0 \\
\dfrac {\partial L}{\partial b_j}=2b_j^{T}s-2\lambda b_j^{T}=0
\end{cases}
\Leftrightarrow
\begin{cases}
b_j^T b_j = 1 \\
b_j^T s=\lambda b_j^T
\end{cases}
\end{aligned} \tag{F}$$

由(G), (F)可得
$$\mathcal{J} = \sum_{j=M+1}^D \lambda_j$$

所以在忽略的子空间里要选那些比较小的特征值，在主子空间选那些大的特征值。

This nicely aligns with properties of the covariance matrix. The eigen vectors of the covariance matrix are orthogonal to each other because of symmetry and the eigen vector belonging to the largest eigen value points in the direction of the data with the largest variance and the variance in that direction is given by the corresponding eigen value.

## PCA algorithm - PCA算法
### Steps of PCA - PCA步骤
1. **数据预归一化** (normalization)
	1. **每列数据减该列平均值(mean)**, to avoid numerial problems
	2. **梅列数据除该列标准差(std)**，使数据无单位（unit-free）且方差为1
	$$x_*^{(d)} \leftarrow \dfrac{x_*^{(d)} - \mu^{(d)}}{\sigma^{(d)}}$$
2. 计算数据**协方差矩阵**（covariance matrix）和**该矩阵**对应的**特征值**、**特征向量**（eigenvalues, eigenvectors）
	* $\tilde{x}_* = \pi_u(x_*) = BB^Tx_*$
	* _B_ 是由特征向量作为列的矩阵，其中特征向量对应的是最大的特征值

### High-dimentional PCA - 高维空间PCA
对于 $X = \begin{bmatrix} x_1^T \\ \vdots \\ x_N^T \end{bmatrix}
 \in \mathbb{R}^{N \times D}$ 如果 $N \ll D$，那么 _X_ 的协方差矩阵 _S_ 的秩为 _N_。那么 _S_ 有 _D-N+1_ 个特征值为0。

 下面考虑如何把 _S_ 转换为满秩矩阵：
 $$ \underset{E \in \mathbb{R} ^{N\times N}}{\underbrace{\dfrac {1}{N}XX^{T}}}
 \underset {c_{i}}{\underbrace{xb_{i}}}
 = \lambda _{i}  \underset {c_{i}}{\underbrace{xb_{i}}} $$

## References
1. [PCA chapter of "Mathematics for Machine Learning"](https://mml-book.github.io/book/chapter10.pdf)
