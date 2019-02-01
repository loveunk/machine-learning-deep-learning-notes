# Calculus Notes 微积分

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Calculus Notes 微积分](#calculus-notes-微积分)
	- [Derivative 导数](#derivative-导数)
		- [Rules](#rules)
			- [Sum / Subtraction rule 线性法则](#sum-subtraction-rule-线性法则)
			- [Power rule 指数法则](#power-rule-指数法则)
			- [Other rule 其他](#other-rule-其他)
		- [Product rule 乘积法则](#product-rule-乘积法则)
		- [Chain rule 链式法则](#chain-rule-链式法则)
	- [Partial Derivative 偏导数](#partial-derivative-偏导数)
		- [Definitions](#definitions)

<!-- /TOC -->

## Derivative 导数
Definition:
$$ \frac{df}{dx} = f'(x) = \lim _{\Delta x\rightarrow 0}\left( \dfrac {f\left( x + \Delta x \right) -f(x)}{\Delta x}\right) $$

### Rules
#### Sum / Subtraction rule 线性法则
$$ \begin{aligned} \dfrac {d}{dx}\left( f\left( x\right) +g\left( x\right) \right) = \dfrac {df\left( x\right) }{dx} +\dfrac {dg\left( x\right) }{dx}\end{aligned} $$

#### Power rule 指数法则
if $f(x) = ax^b$, then
$$ f'(x) = abx^{b-1} $$

#### Other rule 其他
* $f(x) =\dfrac {1}{x}, f'(x) = -\dfrac{1}{x^{2}}$
* $f(x) = e^{x},  f'(x) = e^x$
* $f(x) = sin(x), f'(x) = cos(x)$
* $f(x) = cos(x), f'(x) = -sin(x)$

### Product rule 乘积法则
* $f(x) \cdot g(x) = f(x) g'(x) + f'(x) g(x)$
> $$ \begin {aligned} \lim _{\Delta x\rightarrow 0}(\Delta A(x)) & = \lim_{\Delta x\rightarrow 0}(f(x) (g(x+\Delta x) - g(x)) + (f(x+\Delta x) - f(x))) \\ &= f(x) g'(x) + f'(x) g(x) \end{aligned}$$
> 需要说明上面的等式忽略了 $(f(x + \Delta x) - f(x))(g(x + \Delta x) - g(x))$，结合下图就可以更好理解，被忽略的部分是右下角白色的小框，随着 $lim _{\Delta x\rightarrow 0}$，这部分可以忽略不计了。
> <p align="center"><img src="./img/derivative-product-rule-explanation.png" width="300" /> </p>

### Chain rule 链式法则
* $f(g(x))' = f'(g(x)) g'(x)$
> 可以想象成两个函数分别求导，再求乘积，例子如下图 <p align="center"><img src="./img/derivative-chain-rule-explanation.png" width="300" /> </p>

## Partial Derivative 偏导数
### Definitions
A partial derivative of a function of several variables is its derivative with respect to one of those variables, with the others held constant. <br/>
一个多变量的函数的偏导数是它关于其中一个变量的导数，而保持其他变量恒定。

Denoted by
$f'_x, f_x, \partial_x f,\ D_xf, D_1f, \frac{\partial}{\partial x}f, \text{ or } \frac{\partial f}{\partial x}.$ or  $f_x(x, y, \ldots), \frac{\partial f}{\partial x} (x, y, \ldots)$
