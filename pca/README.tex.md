# PCA

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
