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
<p align="center"><img src="/calculus/tex/08e96921bcaeaddd7a86d676ecda4c64.svg?invert_in_darkmode&sanitize=true" align=middle width=299.17671794999995pt height=39.452455349999994pt/></p>

### Rules
#### Sum / Subtraction rule 线性法则
<p align="center"><img src="/calculus/tex/5e22ef5be8fd8291c0e1383147c8ffa0.svg?invert_in_darkmode&sanitize=true" align=middle width=254.014332pt height=34.7253258pt/></p>

#### Power rule 指数法则
if <img src="/calculus/tex/c96be4c1fe26be551262160288e019fb.svg?invert_in_darkmode&sanitize=true" align=middle width=77.78044394999998pt height=27.91243950000002pt/>, then
<p align="center"><img src="/calculus/tex/6c6e14544feaeb76e4f5eb45dd5825b7.svg?invert_in_darkmode&sanitize=true" align=middle width=106.27368345pt height=18.88772655pt/></p>

#### Other rule 其他
* <img src="/calculus/tex/d5232b97a6802f7d17313b07dcce7b26.svg?invert_in_darkmode&sanitize=true" align=middle width=164.61630569999997pt height=43.42856099999997pt/>
* <img src="/calculus/tex/7ac9c567818db293dd86c778d60edd3b.svg?invert_in_darkmode&sanitize=true" align=middle width=150.7876029pt height=24.7161288pt/>
* <img src="/calculus/tex/3d7eeb2868aebdb290b8f30fb8925837.svg?invert_in_darkmode&sanitize=true" align=middle width=210.97660484999997pt height=27.77565449999998pt/>
* <img src="/calculus/tex/ff83fac3f8592cd90af956224edd9219.svg?invert_in_darkmode&sanitize=true" align=middle width=210.13244669999997pt height=24.7161288pt/>
* <img src="/calculus/tex/5fd832a490eb3bf1a35e49a22c928e75.svg?invert_in_darkmode&sanitize=true" align=middle width=222.91788089999997pt height=24.7161288pt/>

### Product rule 乘积法则
* <img src="/calculus/tex/ff09a5921d1f66cd7d3d69fd3514770c.svg?invert_in_darkmode&sanitize=true" align=middle width=250.9303929pt height=24.7161288pt/>
> <p align="center"><img src="/calculus/tex/4560ddacebc01e0c8b8fdddb716f47e6.svg?invert_in_darkmode&sanitize=true" align=middle width=500.94772034999994pt height=49.06842765pt/></p>
> 需要说明上面的等式忽略了 <img src="/calculus/tex/18921c57cacd3b9d8a07815c0ee8412f.svg?invert_in_darkmode&sanitize=true" align=middle width=277.34017739999996pt height=24.65753399999998pt/>，结合下图就可以更好理解，被忽略的部分是右下角白色的小框，随着 <img src="/calculus/tex/9aba8e632072b0a013789884b336e626.svg?invert_in_darkmode&sanitize=true" align=middle width=63.235812749999994pt height=22.831056599999986pt/>，这部分可以忽略不计了。
> <p align="center"><img src="./img/derivative-product-rule-explanation.png" width="300" /> </p>

### Chain rule 链式法则
* <img src="/calculus/tex/e0cd8211f1b6f71d6f0f655e3a3822ce.svg?invert_in_darkmode&sanitize=true" align=middle width=172.79128514999996pt height=24.7161288pt/>
> 可以想象成两个函数分别求导，再求乘积，例子如下图 <p align="center"><img src="./img/derivative-chain-rule-explanation.png" width="300" /> </p>

## Partial Derivative 偏导数
### Definitions
A partial derivative of a function of several variables is its derivative with respect to one of those variables, with the others held constant. <br/>
一个多变量的函数的偏导数是它关于其中一个变量的导数，而保持其他变量恒定。

Denoted by
<img src="/calculus/tex/0b1468f8c8af2024c8e55b39eb3e58a3.svg?invert_in_darkmode&sanitize=true" align=middle width=246.13603080000001pt height=30.648287999999997pt/> or  <img src="/calculus/tex/6572ae083bad5b8434549f6081c959ab.svg?invert_in_darkmode&sanitize=true" align=middle width=172.24168995pt height=30.648287999999997pt/>

## Jacobians - vectors of derivatives
## Hessian

# Neural Networks
## Simple neural networks
## Backpropagation

# Taylor series
When x = 0, we have
<p align="center"><img src="/calculus/tex/aeee31ee512cf115f775095fd93a5871.svg?invert_in_darkmode&sanitize=true" align=middle width=101.75946pt height=44.91258585pt/></p>
When
<p align="center"><img src="/calculus/tex/f3d7696510403d1867a0dd33e4600828.svg?invert_in_darkmode&sanitize=true" align=middle width=347.89482749999996pt height=135.38765625pt/></p>

## Multivariable Taylor Series

# Optimisation - min and max with constraints
## Newton-Raphson 牛顿-拉弗森方法
## Gradient Descent
## Lagrange multipliers 拉格朗日乘数

# Linear Regression

# Non-linear Regression
## Steepest Descent
<p align="center"><img src="/calculus/tex/7323a732de923a1aaf5745f76b8db0d5.svg?invert_in_darkmode&sanitize=true" align=middle width=144.40847685pt height=40.11819404999999pt/></p>

<p align="center"><img src="/calculus/tex/5181baed84b827deee3f108bef66e023.svg?invert_in_darkmode&sanitize=true" align=middle width=152.67284834999998pt height=18.312383099999998pt/></p>

<p align="center"><img src="/calculus/tex/9b9662cda0db60a22561474c88ef9bbf.svg?invert_in_darkmode&sanitize=true" align=middle width=287.8009431pt height=38.973783749999996pt/></p>
