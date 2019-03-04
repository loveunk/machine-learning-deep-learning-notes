# Linear Algebra 线性代数
这一章节总结了线性代数的一些基础知识，包括向量、矩阵及其属性和计算方法。

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Linear Algebra 线性代数](#linear-algebra-线性代数)
	- [Vectors 向量](#vectors-向量)
		- [Basic rules 性质](#basic-rules-性质)
			- [Cosine rule 向量点积](#cosine-rule-向量点积)
			- [Projection 投影](#projection-投影)
				- [Scalar projection 标量投影](#scalar-projection-标量投影)
				- [Vector projection 向量投影](#vector-projection-向量投影)
		- [Changing the reference frame](#changing-the-reference-frame)
			- [Vector change basis 向量基变更](#vector-change-basis-向量基变更)
				- [Python code to calculate r](#python-code-to-calculate-r)
		- [Linear independent 线性无关](#linear-independent-线性无关)
	- [Matrices 矩阵](#matrices-矩阵)
		- [Transformation 矩阵变换](#transformation-矩阵变换)
			- [Relationship between the matrix and rotaion angle θ](#relationship-between-the-matrix-and-rotaion-angle-theta)
		- [Matrix Rank 矩阵秩](#matrix-rank-矩阵秩)
		- [Matrix inverse 逆矩阵](#matrix-inverse-逆矩阵)
			- [Going from Gaussian elimination to finding the inverse matrix](#going-from-gaussian-elimination-to-finding-the-inverse-matrix)
		- [Determinant 行列式](#determinant-行列式)
		- [Matrix multiplication 矩阵乘法](#matrix-multiplication-矩阵乘法)
		- [Matrices changing basis 矩阵基变更](#matrices-changing-basis-矩阵基变更)
		- [Orthogonal matrices 正交矩阵](#orthogonal-matrices-正交矩阵)
		- [The Gram–Schmidt process 格拉姆-施密特正交化](#the-gramschmidt-process-格拉姆-施密特正交化)
		- [Reflecting in a plane](#reflecting-in-a-plane)
		- [Eigenvectors and Eigenvalues 特征向量和特征值](#eigenvectors-and-eigenvalues-特征向量和特征值)
			- [Changing the Eigenbasis](#changing-the-eigenbasis)
	- [One more thing](#one-more-thing)

<!-- /TOC -->

## Vectors 向量
### Basic rules 性质
<p align="center"><img src="/linear-algebra/tex/c2e702aeb877bafd3edeecd31194c38e.svg?invert_in_darkmode&sanitize=true" align=middle width=93.2568813pt height=10.958925449999999pt/></p>
<p align="center"><img src="/linear-algebra/tex/79a09924276dfddb1b316b1562de2284.svg?invert_in_darkmode&sanitize=true" align=middle width=76.81846259999999pt height=7.305955799999999pt/></p>
<p align="center"><img src="/linear-algebra/tex/26a07f19206a8aaf20a57b43b25fb105.svg?invert_in_darkmode&sanitize=true" align=middle width=161.40341085pt height=16.438356pt/></p>

#### Cosine rule 向量点积
<p align="center"><img src="/linear-algebra/tex/e76df7ed7a962b1a7f0040e3000c560e.svg?invert_in_darkmode&sanitize=true" align=middle width=237.7182423pt height=18.312383099999998pt/></p>

#### Projection 投影
##### Scalar projection 标量投影
<p align="center"><img src="/linear-algebra/tex/4638742f5e0a8676951049ddd39aa57f.svg?invert_in_darkmode&sanitize=true" align=middle width=154.35462074999998pt height=16.438356pt/></p>
<p align="center"><img src="/linear-algebra/tex/137eef3148f2b6f46e0e9895dd9a8ae9.svg?invert_in_darkmode&sanitize=true" align=middle width=90.18891914999999pt height=33.81210195pt/></p>

> <p align="center"><img src="./img/vector-projection-r-s.png" width="300" /> </p>

> 可以通过向量点乘的原理的来理解这一点，假设<img src="/linear-algebra/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/>是在坐标系<img src="/linear-algebra/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>上的向量（<img src="/linear-algebra/tex/3cd2a27357b75c79b6fa6f5355a41346.svg?invert_in_darkmode&sanitize=true" align=middle width=44.47956314999999pt height=21.18721440000001pt/>）。那么 <img src="/linear-algebra/tex/1f52075e5d9534b10dc52df12f030fdf.svg?invert_in_darkmode&sanitize=true" align=middle width=223.76567400000002pt height=24.65753399999998pt/>，其中 <img src="/linear-algebra/tex/2e3672a24fbcd6fb067199052bdd3ceb.svg?invert_in_darkmode&sanitize=true" align=middle width=94.76667749999999pt height=24.65753399999998pt/>，所以 <img src="/linear-algebra/tex/57f0f45daa8ac4703678be80baa6d025.svg?invert_in_darkmode&sanitize=true" align=middle width=137.91620204999998pt height=24.65753399999998pt/>

##### Vector projection 向量投影
<img src="/linear-algebra/tex/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode&sanitize=true" align=middle width=7.7054801999999905pt height=14.15524440000002pt/>往<img src="/linear-algebra/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/>上的投影向量如下，同样可以用上图来解释
<p align="center"><img src="/linear-algebra/tex/7da1a727c4c6891a94326e5580f35867.svg?invert_in_darkmode&sanitize=true" align=middle width=126.68600174999999pt height=33.81210195pt/></p>

### Changing the reference frame
#### Vector change basis 向量基变更
for vector <img src="/linear-algebra/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/> in the axis <img src="/linear-algebra/tex/0558930693deb29567044c994aea539c.svg?invert_in_darkmode&sanitize=true" align=middle width=50.14851269999999pt height=24.65753399999998pt/>，project its cordinates to <img src="/linear-algebra/tex/c701722573d311c8a477072ac0fda939.svg?invert_in_darkmode&sanitize=true" align=middle width=48.94982894999999pt height=24.65753399999998pt/>，the new value is of <img src="/linear-algebra/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/> is
<p align="center"><img src="/linear-algebra/tex/fa0ee461fa76017e6e13b569c22d0b5b.svg?invert_in_darkmode&sanitize=true" align=middle width=102.2107449pt height=37.9216761pt/></p>

> <p align="center"><img src="./img/vector-change-basis.png" width="300" /></p>

> In the aboeve example, <img src="/linear-algebra/tex/a5b9f6119dc7fc37d8ebc607b7f311fe.svg?invert_in_darkmode&sanitize=true" align=middle width=68.14689915pt height=47.6716218pt/>.

##### Python code to calculate <img src="/linear-algebra/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/>
``` python
import numpy as np;
def change_basis(v, b1, b2):
    return [np.dot(v, b1)/np.inner(b1,b1), (np.dot(v, b2)/np.inner(b2,b2))]

v, b1, b2 = np.array([1,  1]), np.array([1,  0]), np.array([0,  2])

change_basis(v, b1, b2)
```

### Linear independent 线性无关
if <img src="/linear-algebra/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/> is indenpdent to s, <img src="/linear-algebra/tex/96f50302506523723f0ca2e748c38eca.svg?invert_in_darkmode&sanitize=true" align=middle width=59.94454454999999pt height=22.831056599999986pt/>, for any <img src="/linear-algebra/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>

## Matrices 矩阵
### Transformation 矩阵变换
矩阵 <img src="/linear-algebra/tex/bb701071518b9d3b853057a99a8230a6.svg?invert_in_darkmode&sanitize=true" align=middle width=74.18945159999998pt height=24.65753399999998pt/> 和一个向量<img src="/linear-algebra/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/>相乘可以理解为把<img src="/linear-algebra/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/>在<img src="/linear-algebra/tex/6e45973e67cf9bc08b5a479fe71e11c7.svg?invert_in_darkmode&sanitize=true" align=middle width=36.54116564999999pt height=14.15524440000002pt/>的坐标系上重新投影
<p align="center"><img src="/linear-algebra/tex/b7187936b57f3c1cd56549a16ef50559.svg?invert_in_darkmode&sanitize=true" align=middle width=137.51134155pt height=39.452455349999994pt/></p>

> <p align="center"><img src="./img/matrix-transformation.png" width="400"/></p>

#### Relationship between the matrix and rotaion angle <img src="/linear-algebra/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/>
The transformation matrix <img src="/linear-algebra/tex/7bce273b0aeb5fb88c237eb7447ec887.svg?invert_in_darkmode&sanitize=true" align=middle width=126.74521694999999pt height=47.6716218pt/>

### Matrix Rank 矩阵秩
矩阵 _A_ 的列秩是 _A_ 的线性无关的纵列的极大数目。行秩是 _A_ 的线性无关的横行的极大数目。其列秩和行秩总是相等的，称作矩阵 _A_ 的秩。通常表示为 r(_A_)或rank(_A_)。

### Matrix inverse 逆矩阵
#### Going from Gaussian elimination to finding the inverse matrix
<p align="center"><img src="/linear-algebra/tex/57c6618f56ff2c786286b279dd19e4bb.svg?invert_in_darkmode&sanitize=true" align=middle width=72.7396725pt height=14.202794099999998pt/></p>

### Determinant 行列式
Matrix <img src="/linear-algebra/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/>'s determinant is denoted as <img src="/linear-algebra/tex/2c97c2f8dfa812fdaa07606cd541d543.svg?invert_in_darkmode&sanitize=true" align=middle width=47.260430249999985pt height=24.65753399999998pt/> or <img src="/linear-algebra/tex/443abb7974801e87ec30c61efd42e490.svg?invert_in_darkmode&sanitize=true" align=middle width=21.46124639999999pt height=24.65753399999998pt/>.
For matrix <img src="/linear-algebra/tex/f6c53fa46a966e6e7aa70dfefbef46d5.svg?invert_in_darkmode&sanitize=true" align=middle width=85.28154194999999pt height=47.6716218pt/>, <img src="/linear-algebra/tex/fa5ebb003a655d4fd7dff791597d0c3a.svg?invert_in_darkmode&sanitize=true" align=middle width=96.38495294999998pt height=24.65753399999998pt/>

> <p align="center"><img src="./img/matrix-determinant.png" width="400"/></p>

>一个矩阵的行列式就是一个平行多面体的（定向的）体积，这个多面体的每条边对应着对应矩阵的列。 ------ 俄国数学家阿诺尔德（Vladimir Arnold）《论数学教育》

If <img src="/linear-algebra/tex/416cf1804dfbcd312914e6c17540bdd6.svg?invert_in_darkmode&sanitize=true" align=middle width=77.39727104999999pt height=24.65753399999998pt/>, then the invert matrix cannot be calculated.

### Matrix multiplication 矩阵乘法
<p align="center"><img src="/linear-algebra/tex/a0be816095d02111ac60726af6860782.svg?invert_in_darkmode&sanitize=true" align=middle width=536.41500495pt height=88.76800184999999pt/></p>
<p align="center"><img src="/linear-algebra/tex/51cd80371454688a7659c54b10e46863.svg?invert_in_darkmode&sanitize=true" align=middle width=109.1116257pt height=45.2741091pt/></p>

### Matrices changing basis 矩阵基变更
For matrix <img src="/linear-algebra/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and <img src="/linear-algebra/tex/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode&sanitize=true" align=middle width=13.29340979999999pt height=22.465723500000017pt/>, <img src="/linear-algebra/tex/df38d07e48ba29024be60b4c169f6c8f.svg?invert_in_darkmode&sanitize=true" align=middle width=37.49419079999999pt height=22.465723500000017pt/> can be treated as changing <img src="/linear-algebra/tex/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode&sanitize=true" align=middle width=13.29340979999999pt height=22.465723500000017pt/>'s basis to that as <img src="/linear-algebra/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/>.

Transform (rotate) <img src="/linear-algebra/tex/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.60847334999999pt height=22.465723500000017pt/> in <img src="/linear-algebra/tex/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode&sanitize=true" align=middle width=13.29340979999999pt height=22.465723500000017pt/>'s coordinates: <img src="/linear-algebra/tex/5b19e1ebba75efa63a892b3ebff78cf2.svg?invert_in_darkmode&sanitize=true" align=middle width=56.843740799999985pt height=26.76175259999998pt/>
> <p align="center"><img src="./img/transformation-in-a-changed-basis.png" width="300"/></p>


### Orthogonal matrices 正交矩阵
If <img src="/linear-algebra/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> is an orthogonal matrix, then <img src="/linear-algebra/tex/a0725bb56ad74c32c7ac40fa37f3bae8.svg?invert_in_darkmode&sanitize=true" align=middle width=65.44680779999999pt height=27.6567522pt/>, so <img src="/linear-algebra/tex/e0f997eff2a3c6dfee29b07a3b66628e.svg?invert_in_darkmode&sanitize=true" align=middle width=73.75738754999999pt height=27.6567522pt/>

### The Gram–Schmidt process 格拉姆-施密特正交化
如果内积空间上的一组向量能够组成一个子空间，那么这一组向量就称为这个子空间的一个基。Gram－Schmidt正交化提供了一种方法，能够通过这一子空间上的一个基得出子空间的一个正交基，并可进一步求出对应的标准正交基。
<p align="center"><img src="/linear-algebra/tex/98dde14fec114ec64d3cb02e9506b6b2.svg?invert_in_darkmode&sanitize=true" align=middle width=342.7458375pt height=215.03230814999998pt/></p>

After above process, <img src="/linear-algebra/tex/635109484d0cb7219d82c571275998d6.svg?invert_in_darkmode&sanitize=true" align=middle width=52.61801819999998pt height=22.831056599999986pt/>, for any <img src="/linear-algebra/tex/4fe48dde86ac2d37419f0b35d57ac460.svg?invert_in_darkmode&sanitize=true" align=middle width=20.679527549999985pt height=21.68300969999999pt/>.

### Reflecting in a plane
<p align="center"><img src="/linear-algebra/tex/e72bcaa0ca647c0608cd5fe51ddc0788.svg?invert_in_darkmode&sanitize=true" align=middle width=142.4154039pt height=16.66852275pt/></p>

Where <img src="/linear-algebra/tex/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode&sanitize=true" align=middle width=13.08219659999999pt height=22.465723500000017pt/> is calculated via the gram-schmidt process, <img src="/linear-algebra/tex/08376a7e30918bb08996d4dec5494dbc.svg?invert_in_darkmode&sanitize=true" align=middle width=19.889337599999987pt height=22.465723500000017pt/> is the transformation matrix in the basic plane. <img src="/linear-algebra/tex/90ef2fdf0a9392b85da459e8aafb9092.svg?invert_in_darkmode&sanitize=true" align=middle width=50.47557239999999pt height=26.76175259999998pt/> stands for coverting <img src="/linear-algebra/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/> to <img src="/linear-algebra/tex/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode&sanitize=true" align=middle width=13.08219659999999pt height=22.465723500000017pt/>'s plane, <img src="/linear-algebra/tex/844fe03b1a2d50af7233a576b824eae5.svg?invert_in_darkmode&sanitize=true" align=middle width=83.05878569999999pt height=26.76175259999998pt/> stands for doing <img src="/linear-algebra/tex/08376a7e30918bb08996d4dec5494dbc.svg?invert_in_darkmode&sanitize=true" align=middle width=19.889337599999987pt height=22.465723500000017pt/> transformation in <img src="/linear-algebra/tex/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode&sanitize=true" align=middle width=13.08219659999999pt height=22.465723500000017pt/>'s plane. Finally, <img src="/linear-algebra/tex/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode&sanitize=true" align=middle width=13.08219659999999pt height=22.465723500000017pt/> goes back to the original plane.

<p align="center"><img src="./img/matrix-reflecting-in-a-plane.png" width="300"/></p>

### Eigenvectors and Eigenvalues 特征向量和特征值
For matrix <img src="/linear-algebra/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/>, <img src="/linear-algebra/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/>'s eigenvector <img src="/linear-algebra/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/> should satisfies <img src="/linear-algebra/tex/e395fbf80746008225543fd047b0a866.svg?invert_in_darkmode&sanitize=true" align=middle width=60.95121449999999pt height=22.831056599999986pt/>, where <img src="/linear-algebra/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/> is a <img src="/linear-algebra/tex/06316c77132ca6158472939a99814319.svg?invert_in_darkmode&sanitize=true" align=middle width=45.29889209999998pt height=22.831056599999986pt/> and it's the eigenvalue.
> <p align="center"><img src="./img/eigenvector-eigenvalues-example.png"/></p>
> In this shear mapping the red arrow changes direction but the blue arrow does not. The blue arrow is an eigenvector of this shear mapping because it does not change direction, and since its length is unchanged, its eigenvalue is 1.

According to the definition of eigenvector, we can have <img src="/linear-algebra/tex/ef3dace1aa49e17a18cdfdbd01db821a.svg?invert_in_darkmode&sanitize=true" align=middle width=115.59351044999998pt height=24.65753399999998pt/>, e.g., <img src="/linear-algebra/tex/d758736b9b6005ac0db377ffc91d7393.svg?invert_in_darkmode&sanitize=true" align=middle width=92.13088334999999pt height=47.6716218pt/>, then <img src="/linear-algebra/tex/8b6468be2bab7b22da50af1084b0b339.svg?invert_in_darkmode&sanitize=true" align=middle width=217.70180520000002pt height=47.6716218pt/>, then  <img src="/linear-algebra/tex/2f51cf221fa1deed5c0dd650859e066c.svg?invert_in_darkmode&sanitize=true" align=middle width=201.23810189999998pt height=26.76175259999998pt/>, we get <img src="/linear-algebra/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/> and use it to calculate the eigenvector.

#### Changing the Eigenbasis
<img src="/linear-algebra/tex/60bce48e4007f5c7b3577611add60ac5.svg?invert_in_darkmode&sanitize=true" align=middle width=144.0547416pt height=57.53473770000001pt/>, <img src="/linear-algebra/tex/2d6bac04a6ebc21ac95923d734ca0132.svg?invert_in_darkmode&sanitize=true" align=middle width=148.5183975pt height=67.39784699999998pt/>, <img src="/linear-algebra/tex/e9fb3869d8c25f815ce92fd54cdfa222.svg?invert_in_darkmode&sanitize=true" align=middle width=154.1003079pt height=67.39784699999998pt/>
<img src="/linear-algebra/tex/2f156b11699f8585c0e9f11a810a88df.svg?invert_in_darkmode&sanitize=true" align=middle width=90.54901514999999pt height=26.76175259999998pt/>
<img src="/linear-algebra/tex/9b6e7e66adb0457535d6769ec5969c02.svg?invert_in_darkmode&sanitize=true" align=middle width=242.3434926pt height=26.76175259999998pt/>
<img src="/linear-algebra/tex/460a2dd88e9d934481b86fd3d30d0fa8.svg?invert_in_darkmode&sanitize=true" align=middle width=108.44489039999999pt height=26.76175259999998pt/>
> <p align="center"><img src="./img/Eigenbasis-example.png" width="300"/></p>

其中，<img src="/linear-algebra/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/> 是**特征向量**(eigenvectors)，<img src="/linear-algebra/tex/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=22.465723500000017pt/>由**特征值**(eigenvalues)构成.

>一个例子：
>
> <img src="/linear-algebra/tex/46b2bcebab075dce603c1a362cb73836.svg?invert_in_darkmode&sanitize=true" align=middle width=90.88472909999999pt height=47.6716218pt/>, <img src="/linear-algebra/tex/8761f77b33a88c75d7e458569181cf3e.svg?invert_in_darkmode&sanitize=true" align=middle width=91.92005459999999pt height=47.6716218pt/>,  <img src="/linear-algebra/tex/60e78040fd02c95d58bfe94a93fc84cb.svg?invert_in_darkmode&sanitize=true" align=middle width=122.35393995pt height=47.6716218pt/>, <img src="/linear-algebra/tex/0739b208cbbb3d03b3b6be832d720f45.svg?invert_in_darkmode&sanitize=true" align=middle width=93.06164834999998pt height=47.6716218pt/>.
> <img src="/linear-algebra/tex/f265938112144b63ab3223e48e92a9e2.svg?invert_in_darkmode&sanitize=true" align=middle width=317.04938595pt height=54.374859000000015pt/>

#### 特征值的属性
如 _λ_ 为 _A_ 的特征值， _x_ 是 _A_ 的属于 _λ_ 的特征向量：
* _λ_ 也是 _A<sup>T</sup>_ 的特征值；
* _λ<sup>m</sup>_ 也是 _A<sup>m</sup>_ 的特征值（m是任意常数）；
* _A_ 可逆时，_λ<sup>-1</sup>_ 是 _A<sup>-1</sup>_ 的特征值；

## 推荐阅读
1. [Mathematics for Machine Learning: Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning/)。
2. [矩阵的特征：特征值，特征向量，行列式，trace](https://zhuanlan.zhihu.com/p/25955676)
3. [理解矩阵](https://blog.csdn.net/myan/article/details/647511)

[回到顶部](#linear-algebra-线性代数)
