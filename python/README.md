# Python基础
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

<!-- /TOC -->

## Types 类型
### Basic Types 基础变量类型
* int
* float
* str
* bool

### Type conversion 类型转换
使用 _Type(*)_ 用来获得变量类型
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
* 但三重引号的字符串如果没有复制，其相当于多行注释
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
不能改变一个字符串里的值，但可以对变量重新负值。
``` python
s1 = "Kevin "
s1[0] = 'K' # ERROR!!!
s1 = "Wen"  # OK
```

#### String functions 字符串常用函数
* `str.upper()`
* `str.repleace(from_str, to_str)`
* `str.find(sub_str)`，返回第一次出现的`index`，没找到返回`-1`
