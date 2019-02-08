# PCA

# Varianes & Covariances 方差 & 协方差
## Variance
$$ Var[X] = \frac{1}{N} \sum_{n=1}^{N}(x_n - \mu)^2, \mu = E[X] $$
$$ Std[X] = \sqrt{Var[X]} $$

## Covariance
$$ Cov[X, Y] = E[(X-\mu_x)(Y-\mu_y)], \mu_x = E[X], \mu_y = E[Y] $$

For 2D data, the Covariance matrix is as follow
$$ \begin{bmatrix} var\left[ X\right] & cov\left[X, Y\right] \\ cov\left[ X,Y\right] & var\left[ Y\right] \end{bmatrix} $$

## Rules
* $Var[D] = Var[D + a]$
* $Var[\alpha D] = \alpha^2 Var[D]$

For matrix $D = \{x_1, x_2, ..., x_n\}, x \in R^p$
* $Var[AD + b] = A Var[D] A^T$
