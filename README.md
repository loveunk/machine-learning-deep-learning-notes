# 深度学习（机器学习）学习路径

最近几年，尤其是自从2016年Alpha Go打败李世石事件后，人工智能技术受到了各行业极大关注。其中以机器学习技术中深度学习最受瞩目。主要原因是这些技术在科研领域和工业界的应用效果非常好，大幅提升了算法效率、降低了成本。因而市场对相关技术有了如此大的需求。

我在思考传统行业与这些新兴技术结合并转型的过程中，亦系统的回顾了深度学习及其相关技术。本文正是我在学习过程中所作的总结。我将按照我所理解的学习路径来呈现各部分内容，希望对你亦有帮助。欢迎一起交流。

主要分为如下几个部分：
* **数学基础**：包括微积分、线性代数、概率论等对理解机器学习算法有帮助的基本数学。
* **Python**：`Python`提供了非常丰富的工具包，非常适合学习者实现算法，也可以作为工业环境完成项目。主流的深度学习框架，例如当前最流行的两个AI框架`TensorFlow`、`PyTorch`都以Python作为首选语言。此外，主流的在线课程（比如Andrew Ng在Coursera的深度学习系列课程）用Python作为练习项目的语言。在这部分，我将介绍包括Python语言基础和机器学习常用的几个Library，包括`Numpy`、`Pandas`、`matplotlib`、`Scikit-Learn`等。
* **机器学习**：介绍主流的机器学习算法，比如线性回归、逻辑回归、神经网络、SVM、PCA、聚类算法等等。
* **深度学习**：介绍原理和常见的模型（比如`CNN`、`RNN`、`LSTM`、`GAN`等）和深度学习的框架（`TensorFlow`、`Keras`、`PyTorch`）。
* **强化学习**：TBD
* **实践项目**：这里将结合几个实际的项目来做比较完整的讲解。此外结合`Kaggle`、`阿里云天池`比赛来做讲解。
* **阅读论文**：如果你追求更高和更深入的研究时，看深度学习各细分领域的论文是非常必要的。

> 内容持续更新中，未完成的部分标识有TBD (To be done)。
> 文中涉及的公式部分是用[CodeCogs](https://codecogs.com/latex/eqneditor.php)的在线LaTeX渲染，如果公式未正确加载，可以尝试多刷新几次。

## 绪论
[机器学习绪论](machine-learning/machine-learning-intro.md)一文中总结了机器学习领域和其解决的问题介绍，建议先读此文，以便有一个系统认知。

## 数学基础
微积分和线性代数的基础是必须要掌握的，不然对于理解学习算法的原理会有困难。如果已经有一定的数学基础，可以先跳过这一部分，需要的时候再回来补。这里的Notes是基于Coursera中Mathematics for Machine Learning专题做的总结。
  * [Calculus 微积分](math/calculus.md)
  * [Linear Algebra 线性代数](math/linear-algebra.md)
  * 概率论 (TBD)
  * [PCA 主成分分析](math/pca.md)

## Python
如果有比较好的Python和机器学习相关Library的知识，对于学习算法过程中的代码可以快速理解和调试，一方面节省时间，另一方面也可以更聚焦在算法和模型本身上。
  * [Python](python/python-basic)
  * [Pandas](python/pandas)
  * [NumPy](python/numpy)
  * [Matplotlib](python/Matplotlib)
  * [Scikit-Learn](python/Sklearn)

## 机器学习算法
主要基于Machine Learning (Coursera, Andrew Ng) 的课程内容。
* [机器学习算法系列](machine-learning/README.md)
  * 内容参考包括：吴恩达Coursera系列、周志华《机器学习》、密西根大学Applied Machine Learning in Python
  * 每章节配套的[<img src="img/github32.png" width="18" target="_blank" />Jupyter Notebook练习](https://github.com/loveunk/ml-ipynb) 参考网络内容修订
* 目录结构：
  1. [绪论](machine-learning/machine-learning-intro.md)
  1. [线性回归](machine-learning/linear-regression.md)
  1. [逻辑回归](machine-learning/logistic-regression.md)
  1. [神经网络](machine-learning/neural-networks.md)
  1. [打造实用的机器学习系统](machine-learning/advice-for-appying-and-system-design.md)
  1. [支持向量机 SVM](machine-learning/svm.md)
  1. [聚类算法](machine-learning/clustering.md)
  1. [数据降维](machine-learning/dimension-reduction.md)
  1. [异常检测](machine-learning/anomaly-detection.md)
  1. [推荐系统](machine-learning/recommender-system.md)
  1. [大规模机器学习](machine-learning/large-scale-machine-learning.md)
  1. [应用案例照片文字识别](machine-learning/photo-ocr.md)
  1. [总结](machine-learning/ssummary.md)

## 深度学习
### Deep Learning 专题课程
主要基于Deep Learning (Coursera, Andrew Ng) 的专题课程 ，介绍深度学习的各种模型的原理。
* [深度学习](deep-learning/README.md)
  1. 深度学习基础
     - [深度学习基础](deep-learning/1.deep-learning-basic.md)
  2. 深度神经网络调参和优化
     - [深度学习的实践层面](deep-learning/2.improving-deep-neural-networks-1.practical-aspects.md)
     - [深度学习优化算法](deep-learning/2.improving-deep-neural-networks-2.optimization-algorithms.md)
     - [超参数调试、批量正则化和程序框架](deep-learning/2.improving-deep-neural-networks-3.pyperparameter-tuning.md)
  3. 深度学习的工程实践
     - [机器学习策略（1）](deep-learning/3.structuring-machine-learning-1.ml-strategy.md)
     - [机器学习策略（2）](deep-learning/3.structuring-machine-learning-2.ml-strategy.md)
  4. 卷积神经网络（CNN）
     - [卷积神经网络](deep-learning/4.convolutional-neural-network-1.foundations-of-cnn.md)
     - [深度卷积网络：实例探究](deep-learning/4.convolutional-neural-network-2.deep-convolutional-models.md)
     - [目标检测](deep-learning/4.convolutional-neural-network-3.object-detection.md)
     - [特殊应用：人脸识别和神经风格转换](deep-learning/4.convolutional-neural-network-4.face-recognition-and-neural-style-transfer.md)
  5. 序列模型（RNN、LSTM）
     - [循环序列模型（RNN）](deep-learning/5.sequence-model-1.recurrent-neural-netoworks.md)
     - [自然语言处理与词嵌入](deep-learning/5.sequence-model-2.nlp-and-word-embeddings.md)
     - [序列模型和注意力机制](deep-learning/5.sequence-model-3.sequence-models-and-attention-machanism.md)
  6. 进阶讨论（待补充）

### TensorFlow 
* 推荐吴恩达DeepLearning.ai和Coursera推出的系列TensoFlow课程。每门课均包括四周内容，Exercise基于Google Colab平台，讲师是来自Google Brain团队的Laurence Moroney：
  1. [《Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning》](https://www.coursera.org/learn/introduction-tensorflow)：TF入门
  2. [《Convolutional Neural Networks in TensorFlow》](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow)：CNN, Transfer Learning
  3. [还未发布]：TensorFlow.js
* 关于TensorFlow 2.0，推荐阅读[<img src="img/zhihu32.png" width="18" />《TensorFlow Dev Summit 2019》](https://zhuanlan.zhihu.com/p/60077966)以便对TensorFlow体系有个完整认知。
* [TensorFlow/Keras的例子](tensorflow)

### PyTorch
PyTorch同样是一个优秀的深度学习框架，发展势头不错，值得关注
* [<img src="img/zhihu32.png" width="18" />PyTorch的安装与Tutorial](https://zhuanlan.zhihu.com/p/60526007)
* [<img src="img/github32.png" width="18" />PyTorch 中文手册](https://github.com/zergtant/pytorch-handbook)

## 强化学习
* TBD

## 项目和竞赛
### 竞赛
* [Kaggle](competitions/kaggle.md)（全球赛、推荐的平台）
* [天池](https://tianchi.aliyun.com) - 阿里云（中国）
* [DataFountain](https://www.datafountain.cn/)（中国）
* [SODA](http://soda.shdataic.org.cn/) - 开放数据创新应用大赛（中国）

## 相关论文

对于一些问题的深入研究，最终是离不开阅读优秀论文，推荐如下GitHub：

* [<img src="img/github32.png" width="18" />深度学习论文的阅读路径](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)：适合深度学习领域新人，循序渐进带你读论文

* [<img src="img/github32.png" width="18" />Papers with code](https://github.com/zziz/pwc)：总结了近 10 年来顶会（包括NIPS/CVPR/ECCV/ICML）优秀论文和复现代码

## 工欲善其事，必先利其器
### 推荐的书

* 《机器学习》（别名《西瓜书》周志华）
* 《Deepleanrning》（别名《花书》作者Ian Goodfellow）
* 《Hands on Machine Learning with Scikit Learn Keras and TensorFlow》（已经出了第二版，作者Aurélien Géron）
* 非常推荐购买纸质书，关于电子版可参考这个的Repo：[<img src="img/github32.png" width="18" target="_blank" />机器学习/深度学习/Data Science相关的书籍](https://github.com/loveunk/Deep-learning-books)

### 推荐的实践环境
* Anaconda
  * [<img src="img/zhihu32.png" width="18" />Anaconda/Tensorflow-GPU安装总结](https://zhuanlan.zhihu.com/p/58607298)
* IDE
  * PyCharm（推荐） / VS Code（推荐）/ Atom 等：[<img src="img/zhihu32.png" width="18" />参考阅读《Python的几款IDE》](https://zhuanlan.zhihu.com/p/58178996)。

### 一些好用的工具
* 机器学习在线环境
  * [Google Colab](https://colab.research.google.com)：Jupyter环境。[<img src="img/zhihu32.png" width="18" />一篇介绍Google Colab的总结](https://zhuanlan.zhihu.com/p/57759598)。
* 科学上网
  * 内地朋友避免不了和China GFW斗智斗勇，建议花小钱省事省心。可看左耳耗子的文章[《科学上网》](https://github.com/haoel/haoel.github.io)。

## 写在最后

### 一点建议

对于此前不是机器学习/深度学习这个领域的朋友，不管此前在其他领域有多深的积累，还请以一个敬畏之心来对待。

* 持续的投入：三天打鱼两天晒网的故事，我们从小便知，不多说了；
* 系统的学习：一个学科，知识是一个体系，系统的学习才可以避免死角，或者黑洞；
* 大量的练习：毕竟机器学习/深度学习属于Engineering & Science的范畴，是用来解决实际的问题的。单纯的理论研究，如果没有实际的项目（包括研究项目）经验做支撑，理论可能不会有很大突破。

### 欢迎反馈
* 如果发现内容的错误，欢迎在GitHub提交Issue或者Pull Request
* 个人精力有限，欢迎感兴趣的朋友一起来完善和补充内容
* 欢迎Star 和Share 此Repository ​

[回到顶部](#深度学习机器学习学习路径)