# 机器学习
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [机器学习](#机器学习)
	- [介绍](#介绍)
	- [线性回归](#线性回归)
	- [逻辑回归 Logistic Regression](#逻辑回归-logistic-regression)
	- [神经网络](#神经网络)
	- [打造实用的机器学习系统](#打造实用的机器学习系统)
	- [支持向量机SVM](#支持向量机svm)
	- [无监督学习](#无监督学习)
	- [异常检测与推荐系统](#异常检测与推荐系统)
	- [大规模机器学习](#大规模机器学习)
	- [应用案例照片文字识别](#应用案例照片文字识别)

<!-- /TOC -->

## 介绍
本章节将主要基于Coursera Andrew Ng的Machine Learning课程整理传统机器学习算法的内容。

## 线性回归
[线性回归](linear-regression.md)

单变量线性回归 (Linear Regression with One Variable)
- 模型表示
- 代价函数
- 梯度下降
  - 梯度下降的直观理解
  - 梯度下降的线性回归
- 多变量线性回归 (Linear Regression with Multiple Variables)

多变量线性回归 (Linear Regression with Multiple Variables)
- 多维特征
- 多变量梯度下降
  - 梯度下降法实践1 - 特征缩放
    - 数据的标准化 (Normalization)
  - 梯度下降法实践2 - 学习率 (Learning Rate)
- 特征和多项式回归
- 正规方程 Normal Equations
- 对比梯度下降和正规方程
  - 正规方程及不可逆性

## 逻辑回归 Logistic Regression
[线性回归](logistic-regression.md)
- Hypothesis 表示
- 边界判定
- 代价函数
- 梯度下降算法
- 多类别分类：一对多
- 正则化 Regularization
	- 过拟合的问题
	- 代价函数
	- 正则化线性回归
		- 正则化与逆矩阵
	- 正则化的逻辑回归模型

## 神经网络
[神经网络](neural-networks.md)
- 背景介绍
  - 为什么需要神经网络
  - 神经元和大脑
- 模型表示
  - 神经元模型：逻辑单元
  - 前向传播
    - 神经网络架构
  - 神经网络应用
    - 神经网络解决多分类问题
- 反向传播 Backpropagation
  - 代价函数 Cost Function
  - 反向传播算法
    - 反向传播算法的直观理解
  - 梯度检验 Gradient Checking
  - 随机初始化
- 总结
  - 网络结构
  - 训练神经网络
- 自动驾驶的例子

## 打造实用的机器学习系统
[打造实用的机器学习系统](advice-for-appying-and-system-design.md)
- 应用机器学习算法的建议
	- 评估一个假设函数 Evaluating a Hypothesis
	- 模型选择和交叉验证集 Model Selection
	- 偏差(Bias)和方差(Variance)
	- 正则化和偏差/方差
	- 学习曲线
	- 总结：决定下一步做什么
- 机器学习系统设计
	- 误差分析 Error Analysis
	- 类偏斜的误差度量
	- 查准率和查全率之间的权衡
	- 机器学习的数据

## 支持向量机SVM
[支持向量机 SVM](svm.md)
- 优化目标
- 大边界
- 大边界分类背后的数学
- 核函数
- 使用SVM
- 什么时候使用SVM

## 无监督学习
- [聚类算法](clustering.md)
  - K-Means聚类
  - DBScan聚类
- [数据降维](dimension-reduction.md)
	- 数据降维的动机
		- 数据降维
		- 数据可视化
	- PCA 主成分分析问题
		- 从压缩数据中恢复
		- 选择主成分的数量
		- PCA应用建议

## 异常检测与推荐系统
- [异常检测](anomaly-detection.md)
	- 高斯分布
	- 基于高斯分布的异常检测算法
	- 开发和评价异常检测系统
	- 异常检测与监督学习对比
	- 选择特征
		- 误差分析
		- 异常检测误差分析
	- 多元高斯分布
	- 使用多元高斯分布进行异常检测
- [推荐系统](recommender-system.md)
	- 问题描述
	- 基于内容的推荐系统
		- 代价函数
		- 协同过滤
		- 协同过滤算法
		- 向量化：低秩矩阵分解
		- 均值归一化

## 大规模机器学习
[大规模机器学习](large-scale-machine-learning.md)
- 大型数据集的学习
   - 确认大规模的训练集是否必要
- 随机梯度下降法 Stochastic Gradient Descent (SGD)
- 小批量梯度下降 Mini-Batch Gradient Descent
- 随机梯度下降收敛
- 在线学习 Online Learning
- 映射化简和数据并行

## 应用案例照片文字识别
[应用案例照片文字识别](photo-ocr.md)


[回到顶部](#机器学习)
