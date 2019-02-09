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
