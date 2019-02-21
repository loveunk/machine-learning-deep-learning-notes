# 逻辑回归 Logistic Regression

逻辑回归 (Logistic Regression) 的算法，这是目前最流行使用最广泛的学习算法之一。

首先介绍几种分类问题：
* 垃圾邮件分类：垃圾邮件（是或不是）？
* 在线交易分类：欺诈性的（是或不是）？
* 肿瘤：恶性 / 良性

先从二元的分类问题开始讨论。将因变量(dependent variable)可能属于的两个类分别称为
* 负向类（negative class）和
* 正向类（positive class）

则 因变量 _y ∈ { 0,1 }_ ，其中 0 表示负向类，1 表示正向类。

对于肿瘤分类是否为良性的问题：
<p align="center">
<img src="https://raw.github.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/images/f86eacc2a74159c068e82ea267a752f7.png" />
</p>

对于二分类问题，_y_ 取值为 0 或者1，但如果使用的是线性回归，那么 _h_ 的输出值可能 远大于1，或者远小于。但数据的标签应该取值0 或者1。所以在接下来的要研究一种新算法**逻辑回归算法**，这个算法的性质是：它的输出值永远在0到 1 之间。

逻辑回归算法是分类算法。可能因为算法的名字中出现“回归”让人感到困惑，但逻辑回归算法实际上是一种分类算法，它适用于标签 _y_ 取值离散的情况，如：1 0 0 1。

## Hypothesis 表示
