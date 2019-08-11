# 线性代数 Linear Algebra
这一章节总结了线性代数的一些基础知识，包括向量、矩阵及其属性和计算方法。

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [线性代数 Linear Algebra](#线性代数-linear-algebra)
	- [向量 Vectors](#向量-vectors)
		- [性质 Basic rules](#性质-basic-rules)
			- [向量点积 Cosine rule](#向量点积-cosine-rule)
			- [投影 Projection](#投影-projection)
				- [标量投影 Scalar projection](#标量投影-scalar-projection)
				- [向量投影 Vector projection](#向量投影-vector-projection)
		- [转换参考系](#转换参考系)
			- [向量基变更 Vector change basis](#向量基变更-vector-change-basis)
				- [计算 _r_ 的Python 代码](#计算-r-的python-代码)
		- [Linear independent 线性无关](#linear-independent-线性无关)
	- [Matrices 矩阵](#matrices-矩阵)
		- [Transformation 矩阵变换](#transformation-矩阵变换)
			- [矩阵与旋转角度 _θ_ 之间的关系](#矩阵与旋转角度-之间的关系)
		- [矩阵秩 Matrix Rank](#矩阵秩-matrix-rank)
		- [逆矩阵 Matrix inverse](#逆矩阵-matrix-inverse)
			- [高斯消元法到找到逆矩阵](#高斯消元法到找到逆矩阵)
		- [行列式 Determinant](#行列式-determinant)
		- [矩阵乘法 Matrix multiplication](#矩阵乘法-matrix-multiplication)
		- [矩阵基变更 Matrices changing basis](#矩阵基变更-matrices-changing-basis)
		- [正交矩阵 Orthogonal matrices](#正交矩阵-orthogonal-matrices)
		- [格拉姆-施密特正交化 The Gram–Schmidt process](#格拉姆-施密特正交化-the-gramschmidt-process)
		- [Reflecting in a plane](#reflecting-in-a-plane)
		- [特征向量和特征值 Eigenvectors and Eigenvalues](#特征向量和特征值-eigenvectors-and-eigenvalues)
			- [改变特征 Changing the Eigenbasis](#改变特征-changing-the-eigenbasis)
			- [特征值的属性](#特征值的属性)
	- [推荐阅读](#推荐阅读)

<!-- /TOC -->

## 向量 Vectors
### 性质 Basic rules
* _r + s = s + r_
* _r · s = s · r_
* _r · (s + t)=r · s + r · t_

#### 向量点积 Cosine rule
_(r - s)<sup>2</sup> = r<sup>2</sup> + s<sup>2</sup> - 2r · s · cosθ_

#### 投影 Projection
##### 标量投影 Scalar projection
_r · s =|r| × |s| × cosθ_

<img src="https://latex.codecogs.com/gif.latex?proj_r^s=\frac{r\cdot&space;s}{|r|}" title="proj_r^s=\frac{r\cdot s}{|r|}" />

> <p align="center"><img src="./img/vector-projection-r-s.png" width="300" /> </p>

> 可以通过向量点乘的原理的来理解这一点，假设 _r_ 是在坐标系 _i_ 上的向量（ _r<sub>j</sub>=0_ ）。那么 _r · s = r<sub>i</sub>s<sub>i</sub> + r<sub>j</sub>s<sub>j</sub> = r<sub>i</sub>s<sub>i</sub> = |r|s<sub>i</sub>_ ，其中 _s<sub>i</sub> = |s| · cosθ_ ，所以 _r · s =|r| · |s| · cosθ_

##### 向量投影 Vector projection
 _s_ 往 _r_ 上的投影向量如下，同样可以用上图来0解释

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?proj_r^s=\frac{r\cdot&space;s}{|r|\times|r|}r" title="proj_r^s=\frac{r\cdot s}{|r|\times|r|}r" />
</p>

### 转换参考系
#### 向量基变更 Vector change basis
对于在坐标系 _(e<sub>1</sub>, e<sub>2</sub>)_ 上的向量 _r_，把它的坐标点映射到 _(b<sub>1</sub>,b<sub>2</sub>)_ ，_r_ 在新的坐标系中的坐标点是

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\left[\frac{r\cdot&space;b_1}{|b_1|^2},\frac{r\cdot&space;b_2}{|b_2|^2}\right]^T" title="\left[\frac{r\cdot b_1}{|b_1|^2},\frac{r\cdot b_2}{|b_2|^2}\right]^T" />
</p>

> <p align="center"><img src="./img/vector-change-basis.png" width="300" /></p>

> 在上面的例子中，$r = \begin{bmatrix} 2 \\ 0.5 \end{bmatrix}$.

##### 计算 _r_ 的Python 代码
``` python
import numpy as np;
def change_basis(v, b1, b2):
    return [np.dot(v, b1)/np.inner(b1,b1), (np.dot(v, b2)/np.inner(b2,b2))]

v, b1, b2 = np.array([1,  1]), np.array([1,  0]), np.array([0,  2])

change_basis(v, b1, b2)
```

### Linear independent 线性无关
如果 _r_ 和 _s_ 是线性无关的，对于任何 _α_， _r ≠ α · s_。

## Matrices 矩阵
### Transformation 矩阵变换
矩阵 _E=[e<sub>1</sub> e<sub>2</sub>]_ 和一个向量 _v_ 相乘可以理解为把 _v_ 在 _e<sub>1</sub>, e<sub>2</sub>_ 的坐标系上重新投影

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}1&0\\0&1\end{bmatrix}\cdot\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}x\\y\end{bmatrix}" title="\begin{bmatrix}1&0\\0&1\end{bmatrix}\cdot\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}x\\y\end{bmatrix}" />
</p>

> <p align="center"><img src="./img/matrix-transformation.png" width="400"/></p>

#### 矩阵与旋转角度 _θ_ 之间的关系
转换矩阵为
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}cos\theta&sin\theta\\-sin\theta&cos\theta\end{bmatrix}" title="\begin{bmatrix}cos\theta&sin\theta\\-sin\theta&cos\theta\end{bmatrix}" />
</p>

### 矩阵秩 Matrix Rank
矩阵 _A_ 的列秩是 _A_ 的线性无关的纵列的极大数目。行秩是 _A_ 的线性无关的横行的极大数目。其列秩和行秩总是相等的，称作矩阵 _A_ 的秩。通常表示为 r(_A_)或rank(_A_)。

### 逆矩阵 Matrix inverse
#### 高斯消元法到找到逆矩阵
$$A^{-1}A = I$$

### 行列式 Determinant
矩阵 _A_ 的行列式表示为 _det(A)_ 或 _|A|_ .

对于矩阵 <img src="https://latex.codecogs.com/gif.latex?A=\begin{bmatrix}a&b\\c&d\end{bmatrix}" title="A=\begin{bmatrix}a&b\\c&d\end{bmatrix}" />  _|A|=a d-c d_

> <p align="center"><img src="./img/matrix-determinant.png" width="400"/></p>

>一个矩阵的行列式就是一个平行多面体的（定向的）体积，这个多面体的每条边对应着对应矩阵的列。 ------ 俄国数学家阿诺尔德（Vladimir Arnold）《论数学教育》

行列式 _det(A) = 0_ 的方阵一定是不可逆的。

### 矩阵乘法 Matrix multiplication
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}a_{11}&a_{12}&\ldots&a_{1n}\\a_{21}&a_{22}&\ldots&a_{2n}\\&space;\vdots&\vdots&\ddots&\vdots\\a_{m1}&a_{m2}&\ldots&a_{mn}\end{bmatrix}&space;\cdot&space;\begin{bmatrix}&space;b_{11}&a_{12}&\ldots&b_{1p}\\b_{21}&b_{22}&&space;\ldots&b_{2p}\\&space;\vdots&\vdots&\ddots&\vdots\\b_{n1}&b_{n2}&&space;\ldots&b_{np}\end{bmatrix}=\begin{bmatrix}&space;c_{11}&c_{12}&\ldots&c_{1p}\\c_{21}&c_{22}&&space;\ldots&c_{2p}\\&space;\vdots&\vdots&\ddots&\vdots\\c_{m1}&c_{m2}&space;&\ldots&c_{mp}\end{bmatrix}" title="\begin{bmatrix}a_{11}&a_{12}&\ldots&a_{1n}\\a_{21}&a_{22}& \ldots&a_{2n}\\ \vdots&\vdots&\ddots&\vdots\\a_{m1}&a_{m2}& \ldots&a_{mn}\end{bmatrix} \cdot \begin{bmatrix} b_{11}&a_{12}&\ldots&b_{1p}\\b_{21}&b_{22}& \ldots&b_{2p}\\ \vdots&\vdots&\ddots&\vdots\\b_{n1}&b_{n2}& \ldots&b_{np}\end{bmatrix} =\begin{bmatrix} c_{11}&c_{12}&\ldots&c_{1p}\\c_{21}&c_{22}& \ldots&c_{2p}\\ \vdots&\vdots&\ddots&\vdots\\c_{m1}&c_{m2} &\ldots&c_{mp}\end{bmatrix}" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}" title="c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}" />
</p>

### 矩阵基变更 Matrices changing basis
对于矩阵 _A_ 和 _B_ , _A · B_ 可以认为是把 _B_ 的坐标系变换到 _A_ 中。

Transform (rotate) _R_ in _B_'s coordinates:  _B<sup>-1</sup>RB_
> <p align="center"><img src="./img/transformation-in-a-changed-basis.png" width="300"/></p>

### 正交矩阵 Orthogonal matrices
**正交矩阵**是一个方块矩阵 _A_，其元素为实数，而且行向量与列向量皆为正交的单位向量，使得该矩阵的转置矩阵为其逆矩阵。

如果 _A_ 是正交矩阵，那么 _AA<sup>T</sup>=I_ ， _A<sup>T</sup>=A<sup>-1</sup>_ 。

### 格拉姆-施密特正交化 The Gram–Schmidt process
如果内积空间上的一组向量能够组成一个子空间，那么这一组向量就称为这个子空间的一个基。Gram－Schmidt正交化提供了一种方法，能够通过这一子空间上的一个基得出子空间的一个正交基，并可进一步求出对应的标准正交基。

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\beta_1&=v_1,&e_1=\dfrac{\beta_1}{\left|\beta_1\right|}\\&space;\beta_2&=v_2-\left(v_2\cdot&space;e_1\right)e_1,&e_2=\dfrac{\beta_2}{\left|\beta_2\right|}\\&space;\beta_3&=v_3-\left(v_3\cdot&space;e_1\right)e_1-\left(v_3\cdot&space;e_2\right)e_2,&e_3=\dfrac{\beta_3}{\left|\beta_3\right|}\\&space;\vdots\\&space;\beta_n&=v_n-\sum^{n-1}_{i=1}\left(v_n\cdot&space;e_i\right)e_i,&e_n=\dfrac{\beta_n}{\left|\beta_n\right|}&space;\end{aligned}" title="\begin{aligned} \beta_1&=v_1,&e_1=\dfrac{\beta_1}{\left|\beta_1\right|}\\ \beta_2&=v_2-\left(v_2\cdot e_1\right)e_1,&e_2=\dfrac{\beta_2}{\left|\beta_2\right|}\\ \beta_3&=v_3-\left(v_3\cdot e_1\right)e_1-\left(v_3\cdot e_2\right)e_2,&e_3=\dfrac{\beta_3}{\left|\beta_3\right|}\\ \vdots\\ \beta_n&=v_n-\sum^{n-1}_{i=1}\left(v_n\cdot e_i\right)e_i,&e_n=\dfrac{\beta_n}{\left|\beta_n\right|} \end{aligned}" />
</p>

经过上述过程后，对于任何 _i, j_ ， _β<sub>i</sub> β<sub>j</sub> = 0_ 。

### Reflecting in a plane
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?r'=E\cdot&space;T_E\cdot&space;E^{-1}\cdot&space;r" title="r'=E\cdot T_E\cdot E^{-1}\cdot r" />
</p>

Where $E$ is calculated via the gram-schmidt process, $T_E$ is the transformation matrix in the basic plane. $E^{-1} \cdot r$ stands for coverting $r$ to $E$'s plane, $T_E \cdot E^{-1} \cdot r$ stands for doing $T_E$ transformation in $E$'s plane. Finally, $E$ goes back to the original plane.

<p align="center"><img src="./img/matrix-reflecting-in-a-plane.png" width="300"/></p>
### 特征向量和特征值 Eigenvectors and Eigenvalues
对于一个给定的方阵 _A_，它的特征向量（eigenvector）_v_ 经过这个线性变换之后，得到的新向量仍然与原来的 _v_ 保持在同一条直线上，但其长度或方向也许会改变。
它们满足： _**A**v = **λ**v_。

**λ** 为标量，即特征向量的长度在该线性变换下缩放的比例，称 **λ**  为其特征值。

> <p align="center"><img src="./img/eigenvector-eigenvalues-example.png"/></p>
> 在上面这个图像变换的例子中，红色箭头改变方向，但蓝色箭头不改变方向。蓝色箭头是此剪切映射的特征向量，因为它不会改变方向，并且由于其长度不变，因此其特征值为1。

根据线性方程组理论，为了使这个方程有非零解，矩阵 _A_ 的行列式  _det(A - λI)=0_ 必须是零。

例如，矩阵 _A_ 为<img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}&space;a&space;&&space;b&space;\\&space;c&space;&&space;d&space;\end{pmatrix}" title="\begin{pmatrix} a & b \\ c & d \end{pmatrix}" />，那么
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\det\left(\begin{pmatrix}a&b\\c&d\end{pmatrix}-\begin{pmatrix}\lambda&space;&&space;0\\0&\lambda\end{pmatrix}\right)=0" title="\det\left(\begin{pmatrix}a&b\\c&d\end{pmatrix}-\begin{pmatrix}\lambda & 0\\0&\lambda\end{pmatrix}\right)=0" />
</p>

_λ<sup>2</sup>-(a+d)λ+ad-bc=0_ ，得到 _λ_ 并计算特征向量。

#### 改变特征 Changing the Eigenbasis
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;C&=\begin{pmatrix}x_1&x_2&x_3\\&space;\vdots&\vdots&\vdots\end{pmatrix},&space;D=\begin{pmatrix}\lambda_1&0&0\\0&\lambda_2&0\\0&0&\lambda_3\end{pmatrix},&space;T^n=\begin{pmatrix}a^n&0&0\\0&b^n&0\\0&0&c^n\end{pmatrix}\\&space;T&=CDC^{-1}\\&space;T^2&=CDC^{-1}CDC^{-1}=CD^2C^{-1}\\&space;T^n&=CD^nC^{-1}&space;\end{align*}" title="\begin{align*} C&=\begin{pmatrix}x_1&x_2&x_3\\ \vdots&\vdots&\vdots\end{pmatrix}, D=\begin{pmatrix}\lambda_1&0&0\\0&\lambda_2&0\\0&0&\lambda_3\end{pmatrix}, T^n=\begin{pmatrix}a^n&0&0\\0&b^n&0\\0&0&c^n\end{pmatrix}\\ T&=CDC^{-1}\\ T^2&=CDC^{-1}CDC^{-1}=CD^2C^{-1}\\ T^n&=CD^nC^{-1} \end{align*}" />
</p>

<p align="center"><img src="./img/Eigenbasis-example.png" width="300"/></p>
其中，_C_ 是**特征向量**(eigenvectors)，$D$由**特征值**(eigenvalues)构成.

一个例子：
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;T&=\begin{pmatrix}1&1\\0&2\end{pmatrix},C=\begin{pmatrix}1&1\\0&1\end{pmatrix},C^{-1}=\begin{pmatrix}&space;1&space;&&space;-1&space;\\0&1\end{pmatrix},D=\begin{pmatrix}1&0\\0&2\end{pmatrix}\\&space;T^2&=\begin{pmatrix}1&1\\0&1\end{pmatrix}\begin{pmatrix}1&0\\0&2\end{pmatrix}^{2}\begin{pmatrix}&space;1&-1\\0&1\end{pmatrix}=\begin{pmatrix}1&3\\0&4\end{pmatrix}&space;\end{align*}" title="\begin{align*} T&=\begin{pmatrix}1&1\\0&2\end{pmatrix},C=\begin{pmatrix}1&1\\0&1\end{pmatrix},C^{-1}=\begin{pmatrix} 1 & -1 \\0&1\end{pmatrix},D=\begin{pmatrix}1&0\\0&2\end{pmatrix}\\ T^2&=\begin{pmatrix}1&1\\0&1\end{pmatrix}\begin{pmatrix}1&0\\0&2\end{pmatrix}^{2}\begin{pmatrix} 1&-1\\0&1\end{pmatrix}=\begin{pmatrix}1&3\\0&4\end{pmatrix} \end{align*}" />
</p>

#### 特征值的属性
如 _λ_ 为 _A_ 的特征值， _x_ 是 _A_ 的属于 _λ_ 的特征向量：
* _λ_ 也是 _A<sup>T</sup>_ 的特征值；
* _λ<sup>m</sup>_ 也是 _A<sup>m</sup>_ 的特征值（m是任意常数）；
* _A_ 可逆时，_λ<sup>-1</sup>_ 是 _A<sup>-1</sup>_ 的特征值；

## 推荐阅读
1. [Mathematics for Machine Learning: Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning/)。

2. [矩阵的特征：特征值，特征向量，行列式，trace](https://zhuanlan.zhihu.com/p/25955676)

3. [理解矩阵](https://blog.csdn.net/myan/article/details/647511)

4. [强大的矩阵奇异值分解(SVD)及其应用](https://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html)

[回到顶部](#linear-algebra-线性代数)
