# 综合案例：照片文字识别

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [综合案例：照片文字识别](#综合案例照片文字识别)
	- [图片文字识别](#图片文字识别)
	- [滑动窗口](#滑动窗口)
	- [获取更多数据](#获取更多数据)
	- [天花板分析：你最该关注哪部分子任务](#天花板分析你最该关注哪部分子任务)

<!-- /TOC -->

## 图片文字识别
从图像中提取文字是一个很常见的应用场景，比如Google此前发起的纸质书籍电子化的项目。

具体来讲，图像文字识别应用是指，从一张给定的图片中识别文字。
这比从一份扫描文档中识别文字要复杂的多。

如下图，是从一张街拍照片里提取店铺名等信息：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/095e4712376c26ff7ffa260125760140.jpg" />
</p>

思路如下：

1. **文字侦测（Text detection）**：将图片上的文字与其他环境对象分离开来

2. **字符切分（Character segmentation）**：将文字分割成一个个单一的字符

3. **字符分类（Character classification）**：确定每一个字符是什么。

可以用任务流程图（pipeline）来拆分问题，每一项子任务可以由一个单独的小队来负责解决：

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/610fffb413d8d577882d6345c166a9fb.png" />
</p>

## 滑动窗口
滑动窗口是用来从图像中提取对象的技术。

假使需要在一张图片中识别行人，主要步骤包括：
1. **训练模型**：用大量固定尺寸的图片（训练集）训练能够准确识别行人的模型。
2. **裁剪目标图像，并识别**：依据训练集的图片尺寸，在要做行人识别的图片上进行剪裁（设定一个窗口），然后将剪裁得到的切片交给模型，让模型判断是否为行人，然后在图片上滑动窗口重新进行剪裁，将新剪裁的切片也交给模型进行判断，如此循环直至将图片全部检测完。
3. **缩放窗口，重复上一步**：按比例放大剪裁的区域，再以新的尺寸对图片进行剪裁，将新剪裁的切片按比例缩小至模型所采纳的尺寸，交给模型进行判断，如此循环。

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/1e00d03719e20eeaf1f414f99d7f4109.jpg" />
</p>

滑动窗口技术也被用于文字识别：

**一、提取文字区域**
1. **训练模型**：训练模型能够区分字符与非字符
2. **滑动窗口识别字符，拼接字符区域并扩展**：用滑动窗口技术识别字符，一旦完成了字符的识别，将识别得出的区域进行扩展，然后将重叠的区域进行合并。
3. 以宽高比作为过滤条件，过滤掉高度比宽度更大的区域（认为单词的长度通常比高度要大）。

下图中绿色的区域是经过这些步骤后被认为是文字的区域，而红色的区域是被忽略的。
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/bc48a4b0c7257591643eb50f2bf46db6.jpg" />
</p>

**二、分割字符**
1. 训练一个模型来完成将文字分割成一个个字符的任务，需要的训练集由单个字符的图片和两个相连字符之间的图片来训练模型。
2. 使用滑动窗口技术来进行字符识别。
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/0a930f2083bbeb85837f018b74fd0a02.jpg" />
</p>

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/0bde4f379c8a46c2074336ecce1a955f.jpg" />
</p>


**三、字符分类阶段**
1. 利用神经网络、支持向量机或者逻辑回归算法训练一个分类器即可


## 获取更多数据

如果模型是低方差的，那获得更多的数据用于训练模型，是能够有更好的效果的。那怎样获得数据？

一、人工地制造

* 以文字识别应用为例，可以在字体网站下载各种字体，然后利用这些不同的字体配上各种不同的随机背景图片创造出一些用于训练的实例，这能够获得一个无限大的训练集。这是从零开始创造实例。

二、利用已有的数据，然后对其进行修改：

* 例如将已有的字符图片进行**扭曲、旋转、模糊**处理。
* 只要认为实际数据有可能和经过这样处理后的数据类似，便可以用这样的方法来创造大量的数据。

总结获得更多数据的几种方法：
* 人工数据合成
* 手动收集、标记数据
* 众包（Crowdsourcing）

## 天花板分析：你最该关注哪部分子任务

在机器学习的应用中，通常需要通过几个步骤才能进行最终的预测。
如何知道哪一部分最值得花时间和精力去改善呢？
这个问题可以通过**天花板分析**来回答。

回到文字识别应用中，**任务流程图**如下：
<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/610fffb413d8d577882d6345c166a9fb.png" />
</p>

流程图中每一部分的输出都是下一部分的输入。

在**天花板分析**中，选取一部分，手工提供100%正确的输出结果，然后看应用的整体效果提升了多少。假使例子中总体效果为72%的正确率。

如果令 `Text Detection`部分输出的结果100%正确，发现系统的总体效果从72%提高到了89%。这意味着很可能会希望投入时间精力来提高 `Text Detection`部分。

接着手动选择数据，让`Character Segmentation`输出的结果100%正确，发现系统的总体效果只提升了1%，这意味着，`Text Detection`部分可能已经足够好了。

最后手工选择数据，让`Character Recognition`输出的结果100%正确，系统的总体效果又提升了10%，这意味着可能也会应该投入更多的时间和精力来提高应用的总体表现。

<p align="center">
<img src="https://raw.github.com/loveunk/Coursera-ML-AndrewNg-Notes/master/images/f1ecee10884098f98032648da08f8937.jpg" />
</p>

总的思想是，确定哪个子模块对整体的性能影响最大。花最多的时间和人力在这个模块上。然后是下一个最值得投入的模块，依次类推。

[回到顶部](#综合案例：照片文字识别)
