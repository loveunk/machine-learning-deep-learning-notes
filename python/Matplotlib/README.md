# Matplotlib

_Matplotlib_ 是一个 _Python_ 2D绘图库，可以生成各种硬拷贝格式和跨平台交互式环境的出版物质量数据。Matplotlib可用于 _Python_ 脚本，_Python_ 和 _IPython_ shell，_Jupyter_ 笔记本，Web应用程序服务器和四个图形用户界面工具包。



## 画图基本说明
这张图说明了图的各个部分
<p align="center">
<img src="https://matplotlib.org/_images/anatomy.png" />
</p>

## 简单的例子
``` python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")
plt.legend()
plt.show()
```
<p align="center">
<img src="https://www.matplotlib.org.cn/static/images/tutorials/sphx_glr_usage_003.png" />
</p>

## 多个子图
``` python
def my_plotter(ax, data1, data2, param_dict):
    out = ax.plot(data1, data2, **param_dict)
    return out

fig, (ax1, ax2) = plt.subplots(1, 2)
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})
```

<p align="center">
<img src="https://www.matplotlib.org.cn/static/images/tutorials/sphx_glr_usage_006.png" />
</p>

## 关于画图的一点意见
这里并未总结很多画图的知识点，因为对于画图，个人认为初期不用花时间系统的学习所有画图技巧。只用对各种图的类型有一个概念或印象，在有需求的时候再查资料学习不迟。结合我自身的经验，几年前因为发论文的需要，有大量各种的图需要制作，而彼时我对MATLAB画图是一点不懂，也是遇到问题就Google、查资料各个击破。当然最终论文发表是没问题的，甚至被同学讲画的很Fancy。

## Reference
* ![Matplotlib中文文档](https://www.matplotlib.org.cn/)

[回到目录](#matplotlib)
