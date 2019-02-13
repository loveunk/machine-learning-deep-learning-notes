# NumPy

NumPy 是一个运行速度非常快的 Python 数学库，主要用于数组计算。
这里总结一些常用的功能，供查阅。

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [NumPy](#numpy)
	- [开始使用NumPy](#开始使用numpy)
	- [Numpy Array 数组](#numpy-array-数组)
		- [创建数组](#创建数组)
		- [访问数组](#访问数组)
		- [基本操作](#基本操作)
			- [`ndarray` 和一个数字运算](#ndarray-和一个数字运算)
			- [两个`ndarray`间的运算](#两个ndarray间的运算)
			- [统计函数](#统计函数)
			- [全局函数 `universal functions`](#全局函数-universal-functions)
	- [操作形状](#操作形状)
		- [拼接数组](#拼接数组)
		- [拆分数组](#拆分数组)
	- [拷贝和 视图 (Views)](#拷贝和-视图-views)
	- [函数和方法总结](#函数和方法总结)
	- [NumPy进阶](#numpy进阶)
		- [广播 Broadcasting](#广播-broadcasting)
	- [高级索引](#高级索引)
		- [用索引数组索引](#用索引数组索引)
		- [用布尔数组索引](#用布尔数组索引)
	- [线性代数](#线性代数)
	- [一些技巧](#一些技巧)
		- [自动塑形](#自动塑形)
		- [直方图 Histgram](#直方图-histgram)
	- [Reference](#reference)

<!-- /TOC -->

## 开始使用NumPy
对于使用 _Python_ 库，第一步必然是`import`：
``` python
import numpy as np
```

## Numpy Array 数组
_NumPy_ 的核心是数组 (`arrays`)。具体来说是多维数组 (`ndarrays`)。其中几个常用的属性和方法：
* `ndarray.ndim`：数组维度
* `ndarray.shape`：数组形状
* `ndarray.size`：所有元素的个数

### 创建数组
* 可以使用 `array` 函数从一个常规的 _Python_ 列表或元组创建一个数组。创建的数组类型是从原始序列中的元素推断出来的。
  ``` python
  np.array([1,2,3,4])
  ```
* array 将序列转化成高维数组
  ``` python
  np.array([(1.5,2,3), (4,5,6)])
  ```
* 数组的类型也能够在创建时具体指定
  ``` python
  np.array( [ [1,2], [3,4] ], dtype=complex )
  ```
* 使用函数创建
  * `zeros(shape)` 函数创建一个全是 0 的数组
  * `ones(shape)` 函数创建全是 1 的数组
  * `empty(shape)` 创建一个随机的数组。默认创建数组的类型是 float64
  * `arange(start, end, step)` 为了创建数字序列，返回一个数组而不是列表
  * `linspace(start, end, num)` 类似`arange()`，但它接收元素数量而不是步长作为参数

### 访问数组
* _indexing_ 索引
  * `nparray[i]`
* _slicing_ 切片
  * `nparray[i:j]`
    <p align = "center">
    <img src="http://ww2.sinaimg.cn/mw690/006faQNTgw1f6flkbesiyj30dw06cgm9.jpg" />
    </p>
  * 三个点(...) 用来表示数组访问所需的剩余所有冒号，例如
    * `x[1,2,...]` 等同 `x[1,2,:,:,:]`
    * `x[...,3]` 等同 `x[:,:,:,:,3]`
    * `x[4,...,5,:]` 等同  `x[4,:,:,5,:]`
* _iterating_ 迭代
  ``` python
  for row in b:           # loop 每行
    print(row)
  for element in b.flat:  # loop 每个元素
    print(element)
  ```

### 基本操作
#### `ndarray` 和一个数字运算
* `+` `-` `*` `/`：将每个元素和数字相加、相减、相乘、相除
* `** n`：将每个元素求n次方

#### 两个`ndarray`间的运算
* `*` ：按照元素位置相乘 (elmentwise multiply)
* `@` ：同`.dot()`，求向量点积、矩阵相乘

#### 统计函数
数组所有元素的和的一元操作。通过指定 axis 参数可以将操作应用于数组的某一具体 axis 。
* `ndarray.mean()`
* `ndarray.sum()`
* `ndarray.min()`
* `ndarray.max()`

#### 全局函数 `universal functions`
全局函数操作数组中每个元素，输出一个数组。
* `ndarray.sin()`

## 操作形状
`ndarray.reshape(shape)`
* 根据数组里的数据，返回一个数组，其形状为shape

`ndarray.resize(shape)`
* 类似`reshape`，但它直接修改`ndarray`本身

`ndarray.T`
* 转置矩阵

### 拼接数组
* `vstack()`：垂直拼接
* `hstack()`：水平拼接
数组可以通过不同的 axes 组合起来。
``` python
>>> a = np.floor(10*np.random.random((2,2)))
>>> a
array([[ 8.,  8.],
       [ 0.,  0.]])
>>> b = np.floor(10*np.random.random((2,2)))
>>> b
array([[ 1.,  8.],
       [ 0.,  4.]])
>>> np.vstack((a,b))
array([[ 8.,  8.],
       [ 0.,  0.],
       [ 1.,  8.],
       [ 0.,  4.]])
>>> np.hstack((a,b))
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
```

### 拆分数组
* `vsplit()`：垂直拆分
* `hsplit()`：水平拆分
``` python
>>> a = np.floor(10*np.random.random((2,12)))
>>> a
array([[ 9.,  5.,  6.,  3.,  6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 1.,  4.,  9.,  2.,  2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])
>>> np.hsplit(a,3)   # 水平拆分为3个数组
[array([[ 9.,  5.,  6.,  3.],
        [ 1.,  4.,  9.,  2.]]),
 array([[ 6.,  8.,  0.,  7.],
        [ 2.,  1.,  0.,  6.]]),
 array([[ 9.,  7.,  2.,  7.],
        [ 2.,  2.,  4.,  0.]])]
```

## 拷贝和 视图 (Views)
在操作数组的时候，数据有时拷贝到新的数组，有时候又不拷贝。
* 不拷贝
  * 简单的赋值不会拷贝任何数组对象和它们的数据。
  * _Python_ 将可变对象作为引用传递，函数调用不会产生拷贝。
* 视图(Views) 和浅拷贝(Shaow Copy)
  * 不同的数组对象可以分享相同的数据。`view` 方法创建了一个相同数据的新数组对象。
  * 切片数组返回一个 `view`
  ``` python
  >>> c = a.view()
  >>> c is a
  False
  >>> c.base is a                        # c is a view of the data owned by a
  True
  >>> c.flags.owndata
  False
  >>>
  >>> c.shape = 2,6                      # a's shape doesn't change
  >>> a.shape
  (3, 4)
  >>> c[0,4] = 1234                      # a's data changes
  ```
* 深拷贝 (Deep Copy)
  * `copy` 方法完全拷贝数组。
  ``` python
  >>> d = a.copy()                          # a new array object with new data is created
  >>> d is a
  False
  >>> d.base is a                           # d doesn't share anything with a
  False
  ```

## 函数和方法总结
* 数组创建 Array Creation
  * arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like

* 转换 Conversions
  * ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat

* 操作 Manipulations
  * array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack

* 探测 Questions
  * all, any, nonzero, where

* 排序 Ordering
  * argmax, argmin, argsort, max, min, ptp, searchsorted, sort

* 运算 Operations
  * choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum

* 基本统计 Basic Statistics
  * cov, mean, std, var

* 基本线性代数 Basic Linear Algebra
  * cross, dot, outer, linalg.svd, vdot

## NumPy进阶
### 广播 Broadcasting
广播允许全局函数 (`universal functions`) 输入不相同的形状的数组。
* 输入数组向维度(`ndim`)最大的看齐，对于小于`max(ndim)`的数组，在其shape前面补1
* 输出数组的shape是输入数组shape的各个轴上的最大值
* 如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错
* 当输入数组的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值
```
Image (3d array):  256 x 256 x 3
Scale (1d array):              3
Result (3d array): 256 x 256 x 3

A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5

A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  15 x 3 x 5
B      (1d array):  15 x 1 x 5
Result (2d array):  15 x 3 x 5
```

下面是 _NumPy_ 官方的几个说明图：
<p align="center">
<img src="https://www.numpy.org/devdocs/_images/theory.broadcast_1.gif" /><br/>
<img src="https://www.numpy.org/devdocs/_images/theory.broadcast_2.gif" /><br/>
<img src="https://www.numpy.org/devdocs/_images/theory.broadcast_3.gif" /><br/>
<img src="https://www.numpy.org/devdocs/_images/theory.broadcast_4.gif" />
</p>

## 高级索引
### 用索引数组索引
``` python
>>> a = np.arange(12)**2
>>> i = np.array([1,1,3,8,5]) # an array of indices
>>> a[i]
array([ 1,  1,  9, 64, 25])
>>>
>>> j = np.array([[3,4],
                  [9,7]])
>>> a[j]
array([[ 9, 16],
       [81, 49]])
>>> a = np.arange(12).reshape(3,4)
>>> a
array([[0, 1,  2,  3],
      [ 4, 5,  6,  7],
      [ 8, 9, 10, 11]])
>>> i = np.array([[0,1],  # indices for the first dim of a
...               [1,2]])
>>> j = np.array([[2,1],  # indices for the second dim
...               [3,3]])
>>>
>>> a[i,j]                # i and j must have equal shape
array([[ 2,  5],
      [ 7, 11]])
```

使用索引数组对数组赋值：
``` python
>>> a = np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> a[[1,3,4]] = 0
>>> a
array([0, 0, 2, 0, 0])
```

### 用布尔数组索引
我们可以通过一个布尔数组来索引目标数组，以此找出与布尔数组中值为True的对应的目标数组中的数据。
布尔数组的长度必须与目标数组对应的轴的长度一致。
``` python
>>> a = np.arange(12).reshape(3,4)
>>> b = a > 4
>>> b
array([[False, False, False, False],
       [False,  True,  True,  True],
       [ True,  True,  True,  True]], dtype=bool)
>>> a[b]
array([ 5,  6,  7,  8,  9, 10, 11])
```

选择性赋值：
``` python
>>> a[b] = 0
>>> a
array([[0, 1, 2, 3],
       [4, 0, 0, 0],
       [0, 0, 0, 0]])
```

## 线性代数
_NumPy_ 可以实现大量的矩阵操作，例如：
* `transpose(a)` ：返回矩阵转置
* `eye(n)`：创建单位矩阵
* `dot(a, b)`：求点积
* `trace(a)`：求对角线元素的和

`linalg`中的常用函数：
* `linalg.inv(a)`：求逆矩阵
* `linalg.det(a)`：求矩阵求行列式（标量）
* `linalg.norm(a)`：求矩阵范数（默认L2）
* `linalg.eig(a)`：求矩阵特征值和特征向量
* `linalg.solve(a, b)`：解线性方程

## 一些技巧
### 自动塑形
为了改变数组的维度，你可以省略一个可以自动被推算出来的大小的参数。
``` python
>>> a = np.arange(30)
>>> a.shape = 2,-1,3  # -1 means "whatever is needed"
>>> a.shape
(2, 5, 3)
```

### 直方图 Histgram
_NumPy_ 的 `histogram` 函数应用于数组，返回两个`vector`：数组的柱状图和 bins 的`vector`
``` python
(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
plt.plot(.5*(bins[1:]+bins[:-1]), n)
plt.show()
```
<p align="center">
<img src="https://docs.scipy.org/doc/numpy/_images/quickstart-2_01_00.png" />
</p>

## Reference
* [NumPy官方入门教程](https://docs.scipy.org/doc/numpy/user/quickstart.html)
* [Numpy与MATLAB的区别——写给Matlab用户](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html)

[回到目录](#numpy)
