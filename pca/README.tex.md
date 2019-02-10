# PCA - Principal Component Analysis
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [PCA - Principal Component Analysis](#pca-principal-component-analysis)
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
	- [PCA derivation](#pca-derivation)
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

### Inner product of functions
Example:
$$ \langle u, v \rangle = \int ^{b}_{a}u\left( x\right) v\left( x\right)dx $$
In this example, $u(x) = sin(x), v(x) = cos(x), f(x) = sin(x)cos(x)$

### Inner product of random variables
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

## PCA derivation
**问题描述**：对于点集合 $\mathcal{X} = {x_1, ..., x_N}, x_i \in \mathbb{R}^D$，希望找到一个映射集合 $\tilde{x} \in \mathbb{R}^M (M < D)$, $b_i$是正交基, $\beta_i$是正交基系数。有如下推导：
1. $x_n = \sum_{i=1}^D\beta_{in}b_i$
2. $\beta_{in} = x_n^Tb_i$（这是假设使用的是点积，那么 $\beta_{D(D\neq i)}$ 和 $b_i$ 正交）
3. $B=(b_1,...,b_M)$是低维空间坐标系

那么得到如下表达，其中 $\mathcal{z}_n = B^TX \in \mathbb{R}^M$ 是X在低维空间$B$上的投影，称为coordinates或code。
$$ \tilde{x} = BB^TX $$

**优化目标**:样本点到新的超平面上的距离足够近
$$ min \mathcal{J} = \dfrac{1}{N}\sum_{n=1}^{N}||x_n - \tilde{x}_n||^2$$

## References
1. [PCA chapter of "Mathematics for Machine Learning"](https://mml-book.github.io/book/chapter10.pdf)
