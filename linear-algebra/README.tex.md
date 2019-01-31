# Linear Algebra Notes

## Vectors
### Basic rules
$$ r + s  = s + r $$
$$ r \cdot s = s \cdot  r $$
$$ r \cdot  (s + t) = r \cdot s + r \cdot t $$

#### Cosine rule
$$ (r-s)^2 = r^2 + s^2 - 2r \cdot\ s \cdot\ cos\theta$$

#### Projection
##### Scalar projection
$$ r \cdot s = |r| \times|s| \times cos\theta $$
$$ proj_r^s  = \frac{r \cdot s}{|r|} $$

> <p align="center"><img src="./img/vector-projection-r-s.png" width="300" /> </p>

> 可以通过向量点乘的原理的来理解这一点，假设$r$是在坐标系$i$上的向量（$r_j = 0$）。那么 $r \cdot s = r_i s_i + r_j s_j = r_i s_i = \vert r \vert s_i$，其中 $s_i = \vert s \vert \cdot cos \theta$，所以 $r \cdot s = \vert r \vert \cdot \vert s \vert \cdot cos \theta$

##### Vector projection
$s$往$r$上的投影向量如下，同样可以用上图来解释
$$ proj_r^s  = \frac{r \cdot s}{|r| \times |r|} r$$

### Changing the reference frame
#### Change basis
for vector $r$ in the axis $(e_1, e_2)$，project its cordinates to $(b_1, b_2)$，the new value is of $r$ is
$$[\frac{r \cdot b_1}{|b_1|^2} , \frac{r \cdot b_2}{|b_2|^2} ]^T$$

> <p align="center"><img src="./img/vector-change-basis.png" width="300" /></p>

> In the aboeve example, $r = \begin{bmatrix} 2 \\ 0.5 \end{bmatrix}$.

##### Python code to calculate $r$
``` python
import numpy as np;
def change_basis(v, b1, b2):
    return [np.dot(v, b1)/np.inner(b1,b1), (np.dot(v, b2)/np.inner(b2,b2))]

v, b1, b2 = np.array([1,  1]), np.array([1,  0]), np.array([0,  2])

change_basis(v, b1, b2)
```

#### Linear independent
if $r$ is indenpdent to s, $r \ne \alpha \cdot s$, for any $\alpha$

## Matrices
#### Transformation
矩阵 $E = [e_1 e_2]$ 和一个向量$v$相乘可以理解为把$v$在$e_1, e_2$的坐标系上重新投影
$$ \begin{bmatrix}
1 & 0\\ 0
 & 1
\end{bmatrix}
\cdot \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix} $$

> <p align="center"><img src="./img/matrix-transformation.png" width="400"/></p>

#### Relationship between the matrix and rotaion angle $\theta$
The transformation matrix $= \begin{bmatrix} cos\theta & sin\theta \\ -sin\theta & cos\theta \end{bmatrix}$


### Matrix inverse
#### Going from Gaussian elimination to finding the inverse matrix
$$A^{-1}A = I$$

#### Determinant (行列式)
Matrix $A$'s determinant is denoted as $det(A)$ or $|A|$.
For matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, $|A| = ad - cd$

> <p align="center"><img src="./img/matrix-determinant.png" width="400"/></p>

>一个矩阵的行列式就是一个平行多面体的（定向的）体积，这个多面体的每条边对应着对应矩阵的列。 ------ 俄国数学家阿诺尔德（Vladimir Arnold）《论数学教育》

If $det(A) = 0$, then the invert matrix cannot be calculated.

#### Matrix dot product
$$\begin{bmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ a_{21} & a_{22} & \ldots & a_{2n} \\  \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{n2} & \ldots & a_{mn} \end{bmatrix}
\cdot
\begin{bmatrix} b_{11} & a_{12} & \ldots & b_{1n} \\ b_{21} & b_{22} & \ldots & b_{2n} \\ \vdots &  \vdots & \ddots & \vdots \\ b_{m1} & b_{n2} & \ldots & b_{mn} \end{bmatrix}
= \begin{bmatrix} c_{11} & c_{12} & \ldots & c_{1p} \\ c_{21} & c_{22} & \ldots & c_{2p} \\  \vdots & \vdots & \ddots & \vdots \\ c_{m1} & c_{m2} & \ldots & c_{mp} \end{bmatrix}$$

#### Matrices changing basis
For matrix $A$ and $B$, $A \cdot B$ can be treated as changing $B$'s basis to that as $A$.

Transform (rotate) $R$ in $B$'s coordinates: $B^{-1} R B$
> <p align="center"><img src="./img/transformation-in-a-changed-basis.png" width="300"/></p>


### Orthogonal matrices 正交矩阵
If $A$ is an orthogonal matrix, then $AA^T  = I$, so $A^T = A^{-1}$

### The Gram–Schmidt process
如果内积空间上的一组向量能够组成一个子空间，那么这一组向量就称为这个子空间的一个基。Gram－Schmidt正交化提供了一种方法，能够通过这一子空间上的一个基得出子空间的一个正交基，并可进一步求出对应的标准正交基。
$$\begin{aligned}
\beta _{1}=v_{1}, e_{1}=\dfrac {v_{1}}{\left| v_{1}\right| }\\
\beta _{2}=V_{2}-\left( v_{2}.e_{1}\right) e_{1},e_{2}=\dfrac {V_{2}}{\left| v_{2}\right| }\\
\beta _{3}=v_{3}-\left( v_{3}\cdot e_{1}\right) e_{1}-\left( v_{3}\cdot e_{2}\right) e_{2},e_{3}=\dfrac {V_{3}}{\left| v_{3}\right| }\\
\vdots \\
\beta _{n}=v_{n}-\sum ^{n-1}_{i=1}\left( v_{n}e_{i}\right) e_{i},e_{n}=\dfrac {V_{n}}{\left| v_{n}\right| }
\end{aligned} $$

After above process, $\beta_ij = 0$, for any $i,j$.

### Reflecting in a plane
$$ r' = E \cdot T_E \cdot E^{-1} \cdot r $$

Where $E$ is calculated via the gram-schmidt process, $T_E$ is the transformation matrix in the basic plane. $E^{-1} \cdot r$ stands for coverting $r$ to $E$'s plane, $T_E \cdot E^{-1} \cdot r$ stands for doing $T_E$ transformation in $E$'s plane. Finally, $E$ goes back to the original plane.

<p align="center"><img src="./img/matrix-reflecting-in-a-plane.png" width="300"/></p>

### Eigenvectors and Eigenvalues 特征向量和特征值
For matrix $A$, $A$'s eigenvector $v$ should satisfies $Av=\lambda v$, where $\lambda$ is a $scalar$ and it's the eigenvalue.
> <p align="center"><img src="./img/eigenvector-eigenvalues-example.png"/></p>
> In this shear mapping the red arrow changes direction but the blue arrow does not. The blue arrow is an eigenvector of this shear mapping because it does not change direction, and since its length is unchanged, its eigenvalue is 1.

According to the definition of eigenvector, we can have $det(A - \lambda  I) = 0$, e.g., $A=\begin{pmatrix} a & b \\ c & d \end{pmatrix}$, then $\det \left( \begin{pmatrix} a &b \\ c & d \end{pmatrix}-\begin{pmatrix} \lambda & 0 \\ 0 & \lambda \end{pmatrix} \right) =0$, then  $\lambda ^{2}-\left( a+d\right) \lambda +ad-bc=0$, we get $\lambda$ and use it to calculate the eigenvector.

#### Changing the Eigenbasis
$C=\begin{pmatrix} x_{1} & x_{2} & x_{3} \\ \vdots & \vdots & \vdots \end{pmatrix}$, $D=\begin{pmatrix} \lambda_1 & 0 & 0 \\ 0 & \lambda_{2} & 0 \\ 0 & 0 & \lambda_{3} \end{pmatrix}$, $T^n=\begin{pmatrix} a^n & 0 & 0 \\ 0 & b^n & 0 \\ 0 & 0 & c^n \end{pmatrix}$
$T=CDC^{-1}$
$T^2=CDC^{-1}CDC^{-1}=CD^2C^{-1}$
$T^n=CD^nC^{-1}$
> <p align="center"><img src="./img/Eigenbasis-example.png" width="300"/></p>

$C$ is from the eigenvectors. $D$ is constructed with eigenvalues.

For example, $T=\begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}$, $C=\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$,  $C^{-1}=\begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}$, $D = \begin{pmatrix} 1 & 0 \\ 0 & 2\end{pmatrix}$.
$T^2=\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}^{2}\begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 3 \\ 0 & 4 \end{pmatrix}$

## One more thing
Here's a [website](https://webdemo.myscript.com/views/math/index.html) to translate handwriting formula to latex.
