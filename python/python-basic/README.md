# Python基础

本文为Python的一些基础知识总结，基于Python 3。
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Python基础](#python基础)
	- [Types 类型](#types-类型)
		- [Basic Types 基础变量类型](#basic-types-基础变量类型)
		- [Type conversion 类型转换](#type-conversion-类型转换)
	- [Grammar 基本语法](#grammar-基本语法)
		- [Expression 表达式](#expression-表达式)
			- [Mathematical Operations 数学运算](#mathematical-operations-数学运算)
		- [String operations 字符串操作](#string-operations-字符串操作)
			- [Define a string 定义字符串](#define-a-string-定义字符串)
			- [String slicing 字符串切片](#string-slicing-字符串切片)
			- [String Concatenation 字符串连接](#string-concatenation-字符串连接)
			- [String replication 字符串复制](#string-replication-字符串复制)
			- [String is immutable 字符串的值是不可变的](#string-is-immutable-字符串的值是不可变的)
			- [String functions 字符串常用函数](#string-functions-字符串常用函数)
	- [Lists and Tuple 列表和元组](#lists-and-tuple-列表和元组)
		- [Tuple 元组](#tuple-元组)
			- [Tuple properties 元组属性](#tuple-properties-元组属性)
			- [Tuple Nesting 元组嵌套](#tuple-nesting-元组嵌套)
		- [List 列表](#list-列表)
			- [List properties 列表属性](#list-properties-列表属性)
			- [Covert String to List 转换字符串到列表](#covert-string-to-list-转换字符串到列表)
			- [List Aliasing 列表别名](#list-aliasing-列表别名)
			- [List Clone 列表复制](#list-clone-列表复制)
	- [Dictionary 字典](#dictionary-字典)
	- [Sets 集合](#sets-集合)
	- [Conditions and Branching 条件和分支](#conditions-and-branching-条件和分支)
		- [Conditions 条件](#conditions-条件)
		- [Branching 分支](#branching-分支)
	- [Loops 循环](#loops-循环)
	- [Functions 函数](#functions-函数)
		- [Build in functions 内置函数](#build-in-functions-内置函数)
		- [Collecting arguments 多参数](#collecting-arguments-多参数)
		- [Scope 作用域](#scope-作用域)
	- [Objects and Classes 对象和类](#objects-and-classes-对象和类)
	- [File IO 文件操作](#file-io-文件操作)
		- [File open and close 文件打开关闭](#file-open-and-close-文件打开关闭)
		- [Reading Files 读文件](#reading-files-读文件)
		- [Writting Files 写文件](#writting-files-写文件)
		- [Delete a File or Folder 删除文件或目录](#delete-a-file-or-folder-删除文件或目录)
	- [函数变量](#函数变量)
		- [局部变量和全局变量](#局部变量和全局变量)
		- [模块导入变量](#模块导入变量)
	- [Reference](#reference)

<!-- /TOC -->

## Types 类型
### Basic Types 基础变量类型
* int
* float
* str
* bool

### Type conversion 类型转换
使用 _type(*)_ 用来获得变量类型，_help(*)_ 可以获得帮助说明。
``` python
float(7)  # => 7.0
int(7.24) # => 7
int('A')  # error, 不能强转非数字的字符
str(2)    # => "2"
bool(1)   # => True
```

## Grammar 基本语法
### Expression 表达式
#### Mathematical Operations 数学运算
Python 3中，**/** 和 **//** 代表的除法含义不同：
* **/** 表示浮点数除法
* **//** 为除法后的结果向下取整
``` python
f1 = 3 / 2  # => 1.5
f2 = 3 // 2 # => 1
```

### String operations 字符串操作
#### Define a string 定义字符串
* 对于简单字符串，可以使用单引号或双引号来表示 **""**, **''**
* 对于字符串中出现的相同引号，需要用 **\\** 来转义
* 可以使用三重引号来避免转义

``` python
s1 = "Kevin"
s2 = 'Kevin' # s2 同 s1

S3 = """Kevin and "K.K".""" # => Kevin and "K.K".
```
* 三重引号也可以用来实现多行注释
``` python
def add(x, y):
  '''Add two object(x, y) --> object(x + y)
  Return two var to one var
  '''
  return x + y
```

``` python
name[0]  # => 'K'
name
```

#### String slicing 字符串切片
Python `string`的语法定义为 `string[start:end:step]`，其中`start`默认为`0`，`end`默认为`len(string)`，`step`默认为`1`。

使用 `str[i:j]` 的方式来获取子字符串，其表示从索引`i`开始截止到（不含）索引`j`的字符
* `i >= -len(str)`
* `j > i`
``` python
s1 = "Kevin"
s1[0:2] # => "Ke"

s2 = "012345678"
s2[:5]      # => "01234"
s2[2:]      # => "2345678"
s2[2:5]     # => "234"
s2[2:7:2]   # => "246"
```

List的slicing:
``` python
a = [1, 2, 3]
a[1:]       # => [2, 3]
a[1:2]      # => [2] 注意这是取一个sub-list
a[1]        # => 2   而这是取一个元素
```

#### String Concatenation 字符串连接
可以用`str0 + str1`的方式连接字符
``` python
s0 = "I am "
s1 = "Kevin"
s2 = s0 + s1 # => "I am Kevin"
```
#### String replication 字符串复制
`num * string`的方式复制字符串为多次
``` python
s1 = "Kevin "
s2 = 3 * s1 # => "Kevin Kevin Kevin "
```

#### String is immutable 字符串的值是不可变的
不能改变一个字符串里的值，但可以对变量重新赋值。
``` python
s1 = "Kevin "
s1[0] = 'K' # ERROR!!!
s1 = "Wen"  # OK
```

#### String functions 字符串常用函数
* `str.upper()`
* `str.replace(from_str, to_str)`
* `str.find(sub_str)`，返回第一次出现的`index`，没找到返回`-1`

## Lists and Tuple 列表和元组
### Tuple 元组
* Tuple是有序的序列
* Tuple可以含有不同的基本类型，例如`string`、`int`、`float`、`bool`等
* 使用圆括号 `(*, *)`
* Tuple里可以只包含一个元素，但此时逗号不可以省略
```python
t = ('Kevin',) # 此时t是元组
s = ('Kevin')  # 此时s是字符串，相当于 s = 'Kevin'
```

#### Tuple properties 元组属性
* 可以使用索引的方式访问`tuple[i]`
* 合并：`tuple3 = tuple1 + tuple2`
* 支持`slicing`
* 同样`immutable`

#### Tuple Nesting 元组嵌套
`Tuple`的一个`item`可以是`Tuple`
``` python
tuple0 = (1, 2, ("pop", "rock"), (3, 4))
tuple0[2]     # => ("pop", "rock")
tuple0[2][1]  # => "rock"`
```

### List 列表
* List是有序的序列
* List可以含有不同的基本类型，例如`string`、`int`、`float`、`bool`等
* 使用方括号 `[*, *]`

#### List properties 列表属性
* 支持索引访问
* 合并：`list3 = list1 + list2`
  * 等于`list3 = list1`, `list3.extend(list2)`
  * **注意**：`append`与`extend`不同，被`append`的`item`是作为一个整体追加到`list`末尾，例如`L.append(["pop", 10])`后，`L`中多了一个元素`["pop", 10]`
* 支持`slicing`
* 支持Nesting 嵌套
* **`mutable`**，元素可被修改
  * 对某个元素重新赋值：`L[0] = 0`
  * `del(L[0])`即可删除某个`item`

#### Covert String to List 转换字符串到列表
`string.split()`：按照空格拆分字符串
``` python
"Kaikai is a dog".split()     # => ['Kaikai', 'is', 'a', 'dog']
"Kaikai,is,a,dog".split(",")  # => ['Kaikai', 'is', 'a', 'dog']
```

#### List Aliasing 列表别名
如果一个变量`B`指向另一个变量`A` `(B = A)`，那么`B`是`A`的别名。对`A`的任何改动，`B`都可以访问到。
```python
l1 = [1, 2, 3]
l2 = l1
l1[0] = 0
print(l1) # [0, 2, 3]
print(l2) # [0, 2, 3]
```

#### List Clone 列表复制
复制一个列表通过 `B = A[:]`的方式，`A`和`B`相互不影响
```python
l1 = [1, 2, 3]
l2 = l1[:]
l1[0] = 0
print(l1) # [0, 2, 3]
print(l2) # [1, 2, 3]
```

## Dictionary 字典
`Dictionary`中的每个元素由`(key, value)`的键值对组成。
* 使用花括号来表示`{}`来表示。
* `Key`是唯一的，并且不可修改
* `value`可以是可修改的/不可修改的，可重复的
* 每个`item`之间用逗号隔开
* 访问
  * 使用`dict[key]`的方式来访问`value`
* 新增
  * 直接对一个`dict[key]`赋值
* 删除
  * `del(dict[key])`
* 获取所有keys或values
  * `dict.keys()`
  * `dict.values()`

```python
dic = {"k1":1, "k2":"2", "k3":[3,3], "k4":(4,4), ('k5'):5}
```

## Sets 集合
`Sets` 是类似`list`和`tuple`的集合，可以包含任意元素。
* 使用花括号来表示`{}`来表示。
* **Unordered**：Sets是无序的
* **Unique**：Sets里的元素值是唯一的
* 创建
  * 通过类型转换：`set(a_list)`
* 新增
  * `s.add(item)`
* 删除
  * `s.remove(item)`
* 两个集合的交集
  * `s0 & s2`
* 两个集合的合集
  * `s0`
* 测试是否包含某个元素
  * `item in s`
* 测试是否是子集
  * `s0.issubset(s1)`

## Conditions and Branching 条件和分支
### Conditions 条件
* `==, <=, >=, >, <, !=`
  * 比较int、float、double、string、list等都可以
  * 尤其是`==`可用来比较值/元素是否相同
* `or` / `and`

### Branching 分支
* `if (...), elif(...), else`:

## Loops 循环
`range([start], end, [step])`
* 产生从`0`开始的一个序列直到`end -1`，每个值之间相差`step`， `step`可省略，默认为`1`，

`for i in range(N):`
* 循环`0, 1, ..., N-1`

`for i in range(1, 10, 2):`
* `1, 3, 5, 7, 9`

`while (condition):`

## Functions 函数
函数是有输入和输出的代码集合，主要目的是为了复用，同时让代码结构更清晰。 _Python_ 中函数的定义使用关键词`def function_name():`

### Build in functions 内置函数
* `len()`: 长度、元素个数
* `sum()`：元素求和
* `sorted()`：将collection的元素排序的结果返回，原collection不受影响

### Collecting arguments 多参数
参数前可以有1个或2个星号，这两种用法其实都是用来将任意个数的参数导入到python函数中。
* `def foo(param1, *agrs):`：
  * 将所以参数以元组(tuple)的形式导入
  ``` python
  def foo(param1, *param2):
    print(param1) # => 1
    print(param2) # => (2, 3, 4, 5)
  foo(1,2,3,4,5)
  ```
* `def bar(param1, **kwargs):`：
  * 将参数以字典的形式导入
  ``` python
  def bar(param1, **param2):
    print(param1) # => 1
    print(param2) # => {'a': 2, 'b': 3}
  bar(1, a=2, b=3)
  ```

### Scope 作用域
_Python_ 中变量区分局部和全局作用域，同 _C++_ / _Java_ 之类的语言。
如果在函数中希望定义全局变量，可以在变量前加关键字`global`，以便在全局可被访问。

## Objects and Classes 对象和类
* **类(Class)**: 用来描述具有相同的属性(Data attributes)和方法(methods)的对象的集合。
  * 定义：
    * `class ClassName ([ParentClassName]):`
  * 构造函数或初始化方法：
    * `def __init__(self, name, salary):`
* **对象(Object)**：通过类定义的数据结构实例。对象包括两个数据成员（类变量和实例变量）和方法。
  * 创建：
    * `varName = ClassName ([parameters])`
* **方法(Method)**：类中定义的函数

可以用`dir(object)`的方式列出类的属性。

## File IO 文件操作
### File open and close 文件打开关闭
1. 读写格式
	* `FileObject = open(file_path, mode)`
		* 创建：`"x"`，如果文件存在则返回失败
		* 只读：`"r"`
		* 覆写：`"w"`
		* 追加：`"a"`
		* 文本：`"t"` ，为默认值
		* 二进制：`"b"` ，例如读写图片

2. 关闭文件
	* `file.close()`
	* 推荐用`with open(path, mode) as file:`，在执行完with的作用域时自动调用`file.close()`。

### Reading Files 读文件
* `file.read([n_characters])`：
	* `n_characters == 0`: 读整个文件
	* 读`n_characters`个字符的内容
* `file.readline()` ：读一行
* `file.readlines()` ：读所有的行

``` python
with open("example.txt", "r") as file:
  content = file.read()
  print(content)
```

### Writting Files 写文件
* `file.write(string)`：
	* 写入一行内容

``` python
with open("example.txt", "w") as file:
  file.write("a line")
```

### Delete a File or Folder 删除文件或目录
* 删除一个文件或目录：
	* `os.remove("filename_or_foldername")`

## 函数变量

### 局部变量和全局变量

有其他语言经验的朋友对这两点一定很容易理解这两点。只是想提醒一下，避免`global`的全部变量，毕竟让code很难维护。下面说说模块导入变量。

### 模块导入变量

核心思想是把同一个全局模块的内容组织到一个py文件中，通过模块导入的方式共享到相同模块的代码中：

看例子，在`main.py`中：

```python
import global_abc
import another

def print_variables():
    print(global_abc.GLOBAL_A + ", " + global_abc.GLOBAL_B)

if __name__ == '__main__':
    print_variables()       # Hello, Python

    global_abc.print_name() # Kevin
    global_abc.modify_name()# change Kevin --> GoGo
    global_abc.print_name() # GoGo
    another.print_name_in_3rd_module()  # GoGo
```

在`global_abc.py`中：

```python
GLOBAL_A = 'Hello'
GLOBAL_B = 'Python'

name = 'Kevin'

def modify_name():
    global name
    name = 'GoGo'

def print_name():
    print(name)
```

在`another.py`中：

```python
import global_abc

def print_name_in_3rd_module():
    global_abc.print_name()
```

通过这种方式，可以在多个不同的文件间组织和共享变量。

> 这部分的内容的组织方式自参考了文章[2]。

## 其他

### Python 参数传递

Python函数参数前面单星号（*）和双星号（**）的区别

```python
'''单星号（*）：*agrs：将所以参数以元组(tuple)的形式导入'''
def foo(param1, *param2):
    print(param1, param2)

'''双星号（**）：**kwargs：将参数以字典的形式导入'''
def bar(param1, **param2):
    print(param1, param2)
    
foo(1,2,3,4,5)
# output: 1 (2, 3, 4, 5)

bar(1,a=2,b=3)
# output: 1 {'a': 2, 'b': 3}
```



## Reference

1. [Python 3 官方文档](https://docs.python.org/zh-cn/3/)
2. https://blog.csdn.net/Eastmount/article/details/48766861

[回到目录](#python基础)
