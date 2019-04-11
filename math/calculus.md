# 微积分 Calculus
这篇笔记总结了微积分的一些基础知识，包括导数、偏导数、泰勒展开式、拉格朗日乘数等等的基础知识。
内容部分参考[Mathematics for Machine Learning: Multivariate Calculus](https://www.coursera.org/learn/multivariate-calculus-machine-learning/)。

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [微积分 Calculus](#微积分-calculus)
	- [Derivative 导数](#derivative-导数)
		- [基本规则](#基本规则)
			- [线性法则 (Sum / Subtraction rule)](#线性法则-sum-subtraction-rule)
			- [指数法则 (Power rule)](#指数法则-power-rule)
			- [其他规则](#其他规则)
		- [乘积法则 (Product rule)](#乘积法则-product-rule)
		- [链式法则 (Chain rule)](#链式法则-chain-rule)
	- [偏导数 (Partial Derivative)](#偏导数-partial-derivative)
		- [定义](#定义)
	- [Jacobians - vectors of derivatives](#jacobians-vectors-of-derivatives)
	- [Hessian](#hessian)
- [Neural Networks](#neural-networks)
	- [Simple neural networks](#simple-neural-networks)
	- [Backpropagation](#backpropagation)
- [泰勒级数 (Taylor series)](#泰勒级数-taylor-series)
	- [多变量泰勒级数 Multivariable Taylor Series](#多变量泰勒级数-multivariable-taylor-series)
- [约束最大最小值的优化方法](#约束最大最小值的优化方法)
	- [牛顿-拉弗森方法 Newton-Raphson](#牛顿-拉弗森方法-newton-raphson)
	- [Gradient Descent](#gradient-descent)
	- [拉格朗日乘数 Lagrange multipliers](#拉格朗日乘数-lagrange-multipliers)
- [线性回归Linear Regression](#线性回归linear-regression)
- [非线性回归 Non-linear Regression](#非线性回归-non-linear-regression)
	- [快速下降法 Steepest Descent](#快速下降法-steepest-descent)

<!-- /TOC -->

## Derivative 导数
导数的定义:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\frac{df}{dx}&space;=&space;f'(x)&space;=&space;\lim&space;_{\Delta&space;x\rightarrow&space;0}\left(&space;\dfrac&space;{f\left(&space;x&space;&plus;&space;\Delta&space;x&space;\right)&space;-f(x)}{\Delta&space;x}\right)" title="\frac{df}{dx} = f'(x) = \lim _{\Delta x\rightarrow 0}\left( \dfrac {f\left( x + \Delta x \right) -f(x)}{\Delta x}\right)" />
</p>

### 导数基本规则
#### 线性法则 (Sum / Subtraction rule)
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}\dfrac{d}{dx}\left(f\left(x\right)&plus;g\left(x\right)\right)=\dfrac{df\left(x\right)}{dx}&plus;\dfrac{dg\left(x\right)}{dx}\end{aligned}" title="\begin{aligned}\dfrac{d}{dx}\left(f\left(x\right)+g\left(x\right)\right)=\dfrac{df\left(x\right)}{dx}+\dfrac{dg\left(x\right)}{dx}\end{aligned}" />
</p>

#### 指数法则 (Power rule)

函数为
<p align="center"><i>
f(x) = a x<sup>b</sup>
</i></p>

则
<p align="center"><i>
f'(x) = ab x<sup>(b-1)</sup>
</i></p>

#### 其他规则

| 函数                                     | 导数                                                     |
| ---------------------------------------- | -------------------------------------------------------- |
| _f(x) = (<sup> 1 </sup>/<sub> x </sub>)_ | _f'(x) =  - (<sup> 1 </sup>/<sub> x<sup>2</sup> </sub>)_ |
| _f(x) = e<sup>x</sup>_                   | _f'(x) = e<sup>x</sup>_                                  |
| _f(x) = log<sub>a</sub>(x)_              | _f'(x) = (<sup> 1 </sup>/<sub> xln(a) </sub>)_           |
| _f(x) = sin(x)_                          | _f'(x) = cos(x)_                                         |
| _f(x) = cos(x)_                          | _f'(x) = -sin(x)_                                        |

### 乘积法则 (Product Rule)
函数为 _f(x) · g(x) = f(x) g'(x) + f'(x) g(x)_，则：

<p align="center"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\begin&space;{aligned}&space;\lim&space;_{\Delta&space;x\rightarrow&space;0}(\Delta&space;A(x))&space;&&space;=&space;\lim_{\Delta&space;x\rightarrow&space;0}(f(x)&space;(g(x&plus;\Delta&space;x)&space;-&space;g(x))&space;&plus;&space;(f(x&plus;\Delta&space;x)&space;-&space;f(x)))&space;\\&space;&=&space;f(x)&space;g'(x)&space;&plus;&space;f'(x)&space;g(x)&space;\end{aligned}" title="\begin {aligned} \lim _{\Delta x\rightarrow 0}(\Delta A(x)) & = \lim_{\Delta x\rightarrow 0}(f(x) (g(x+\Delta x) - g(x)) + (f(x+\Delta x) - f(x))) \\ &= f(x) g'(x) + f'(x) g(x) \end{aligned}" /></p>

需要说明上面的等式忽略了 _(f(x+Δx) - f(x)) (g(x+Δx) - g(x))_ ，结合下图就可以更好理解，被忽略的部分是右下角白色的小框，随着  _lim<sub>Δx→0</sub>_ ，这部分可以忽略不计了。

<p align="center"><img src="./img/derivative-product-rule-explanation.png" width="300" /> </p>

### 链式法则 (Chain rule)
对于函数 _f(g), g(x)_，有：

<p align="center"><i>f(g(x))' = f'(g(x)) g'(x)</i></p>

可以想象成两个函数分别求导，再求乘积，例子如下图 

<p align="center"><img src="./img/derivative-chain-rule-explanation.png" width="300" /> </p>

## 偏导数 (Partial Derivative)
### 定义
一个多变量的函数的偏导数是它关于其中一个变量的导数，而保持其他变量恒定。

表示为：

<p align="center"><img src="https://latex.codecogs.com/gif.latex?\inline&space;f'_x,f_x,\partial_x&space;f,D_xf,D_1f,\frac{\partial}{\partial&space;x}f,\text{&space;or&space;}&space;\frac{\partial&space;f}{\partial&space;x}.\text{&space;or&space;}f_x(x,&space;y,\ldots),\frac{\partial&space;f}{\partial&space;x}(x,y,\ldots)" title="f'_x,f_x,\partial_x f,D_xf,D_1f,\frac{\partial}{\partial x}f,\text{ or } \frac{\partial f}{\partial x}.\text{ or }f_x(x, y,\ldots),\frac{\partial f}{\partial x}(x,y,\ldots)" /></p>

## 雅可比矩阵(Jacobians Matrix)

雅克比矩阵是一阶偏导数的排列成的矩阵。其行列式称为雅克比行列式。

假设一个函数由 _m_ 个实函数组成：_y<sub>1</sub>(x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>), ..., y<sub>n</sub>(x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)_，这些函数的偏导数可以组成 _m_ 行 _n_ 列的矩阵，矩阵就是雅克比矩阵 (Jacobians Matrix)：

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\left[&space;\begin{array}{ccc}{\frac{\partial&space;y_{1}}{\partial&space;x_{1}}}&space;&&space;{\cdots}&space;&&space;{\frac{\partial&space;y_{1}}{\partial&space;x_{n}}}&space;\\&space;{\vdots}&space;&&space;{\ddots}&space;&&space;{\vdots}&space;\\&space;{\frac{\partial&space;y_{m}}{\partial&space;x_{1}}}&space;&&space;{\cdots}&space;&&space;{\frac{\partial&space;y_{m}}{\partial&space;x_{n}}}\end{array}\right]" title="\left[ \begin{array}{ccc}{\frac{\partial y_{1}}{\partial x_{1}}} & {\cdots} & {\frac{\partial y_{1}}{\partial x_{n}}} \\ {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial y_{m}}{\partial x_{1}}} & {\cdots} & {\frac{\partial y_{m}}{\partial x_{n}}}\end{array}\right]" />
</p>

矩阵符号写作：

_J<sub>F</sub>(x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)_，或者 (<sup> ∂ (y<sub>1</sub>,  ... , y<sub>n</sub>) </sup>/<sub> ∂ (x<sub>1</sub>,  ... , x<sub>n</sub>) </sub>)

### 例子

函数如下：

* u(x, y) = x - 2y
* v(x, y) = 3y - 2x

则Jacobians Matrix为：

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;J&space;&=\left[\begin{array}{ll}{\frac{\partial&space;x}{\partial&space;r}}&{\frac{\partial&space;x}{\partial\theta}}&space;\\&space;{\frac{\partial&space;y}{\partial&space;r}}&{\frac{\partial&space;y}{\partial\theta}}\end{array}\right]=\left[\begin{array}{ll}{\cos(\theta)}&space;&&space;{-r\sin(\theta)}&space;\\&space;{\sin(\theta)}&{r\cos(\theta)}\end{array}\right]&space;\end{aligned}" title="\begin{aligned} J &=\left[\begin{array}{ll}{\frac{\partial x}{\partial r}}&{\frac{\partial x}{\partial\theta}} \\ {\frac{\partial y}{\partial r}}&{\frac{\partial y}{\partial\theta}}\end{array}\right]=\left[\begin{array}{ll}{\cos(\theta)} & {-r\sin(\theta)} \\ {\sin(\theta)}&{r\cos(\theta)}\end{array}\right] \end{aligned}" />
</p>
另一个例子：

* _x(r, θ) = r cos(θ)_
* _y(r, θ) = r sin(θ)_

则Jacobians Matrix为：

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;J&space;&=\left[\begin{array}{ll}{\frac{\partial&space;x}{\partial&space;r}}&{\frac{\partial&space;x}{\partial\theta}}&space;\\&space;{\frac{\partial&space;y}{\partial&space;r}}&{\frac{\partial&space;y}{\partial\theta}}\end{array}\right]=\left[\begin{array}{ll}{\cos(\theta)}&space;&&space;{-r\sin(\theta)}&space;\\&space;{\sin(\theta)}&{r\cos(\theta)}\end{array}\right]&space;\end{aligned}" title="\begin{aligned} J &=\left[\begin{array}{ll}{\frac{\partial x}{\partial r}}&{\frac{\partial x}{\partial\theta}} \\ {\frac{\partial y}{\partial r}}&{\frac{\partial y}{\partial\theta}}\end{array}\right]=\left[\begin{array}{ll}{\cos(\theta)} & {-r\sin(\theta)} \\ {\sin(\theta)}&{r\cos(\theta)}\end{array}\right] \end{aligned}" />
</p>

Jacobians Det为

<p align="center"><i>
 |J|  =  r(cos<sup>2</sup>(θ)  +  sin<sup>2</sup>(θ))  =  r
 </i></p>

## Hessian

# Neural Networks
## Simple neural networks
## Backpropagation

# 泰勒级数 (Taylor series)
当 x = 0, 可得
$$\sum ^{\infty }_{n=0}\dfrac {f^{\left( n\right) }\left( 0\right) }{n!}x^{n}$$
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\sum&space;^{\infty&space;}_{n=0}\dfrac&space;{f^{\left(&space;n\right)&space;}\left(&space;0\right)&space;}{n!}x^{n}" title="\sum ^{\infty }_{n=0}\dfrac {f^{\left( n\right) }\left( 0\right) }{n!}x^{n}" />
</p>

其中
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;f(x)&=f(p)\\&space;f(x)&=f(p)&plus;f'(p)(x-p)\\&space;f(x)&=f(p)&plus;f'(p)(x-p)&plus;\frac{1}{2}f''(p-p)(x-p)^2\\&space;f(x)&=\sum^{\infty&space;}_{n=0}\dfrac{f^{\left(n\right)}\left(p\right)}{n!}(x-p)^{n}&space;\end{aligned}" title="\begin{aligned} f(x)&=f(p)\\ f(x)&=f(p)+f'(p)(x-p)\\ f(x)&=f(p)+f'(p)(x-p)+\frac{1}{2}f''(p-p)(x-p)^2\\ f(x)&=\sum^{\infty }_{n=0}\dfrac{f^{\left(n\right)}\left(p\right)}{n!}(x-p)^{n} \end{aligned}" />
</p>

## 多变量泰勒级数 Multivariable Taylor Series

# 约束最大最小值的优化方法
## 牛顿-拉弗森方法 Newton-Raphson
## Gradient Descent
## 拉格朗日乘数 Lagrange multipliers

# 线性回归Linear Regression

# 非线性回归 Non-linear Regression
## 快速下降法 Steepest Descent
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\mathbf{J}=\left[\frac{\partial(\chi^2)}{\partial\mu},\frac{\partial(\chi^2)}{\partial\sigma}\right]" title="\mathbf{J}=\left[\frac{\partial(\chi^2)}{\partial\mu},\frac{\partial(\chi^2)}{\partial\sigma}\right]" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\chi^2=|\mathbf{y}-f(\mathbf{x};\mu,\sigma)|^2" title="\chi^2=|\mathbf{y}-f(\mathbf{x};\mu,\sigma)|^2" />
</p>

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial(\chi^2)}{\partial\mu}=-2(\mathbf{y}-f(\mathbf{x};\mu,\sigma))\cdot\frac{\partial&space;f}{\partial\mu}(\mathbf{x};\mu,\sigma)" title="\frac{\partial(\chi^2)}{\partial\mu}=-2(\mathbf{y}-f(\mathbf{x};\mu,\sigma))\cdot\frac{\partial f}{\partial\mu}(\mathbf{x};\mu,\sigma)" />
</p>

[回到顶部](#微积分-calculus)
