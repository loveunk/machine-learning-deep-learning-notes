# Calculus Notes 微积分

## Derivative 倒数
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
