# TensorFlow的几个完整例子

1. [tensorflow.keras.mnist.classifier.py](code/tensorflow.keras.mnist.classifier.py)：一个完整的mnist分类demo，其中涉及的技术点包括：
   1. TensorFlow dataset minist的加载
   2. 数据直方图打印
   3. 数据归一化
   4. label数据的 one hot vectors转换
   5. 数据集切分（train、test）
   6. CNN 模型创建
   7. 保存模型图片
   8. 图片数据增强
   9. 绘制训练集和验证集的loss和accuracy曲线
   10. 使用TensorBoard
   11. 对测试集做预测
   12. 对prediction的one-hot vector转换为数字
   13. 计算Precision、recall、F1等
2. [tensorflow.keras.save.load.model.py](code/tensorflow.keras.save.load.model.py)：讲述利用Keras api保存和加载model。
   其中涉及的技术点包括：
   1. 保存一个模型到存储
   2. 加载已有模型
   3. 使用已有的模型做分类
3. [tensorflow-2.0-Alpha0-helloworld.py](code/tensorflow-2.0-Alpha0-helloworld.py)：基于TensorFlow2.0版本的2个完整的mnist分类demo。涉及：
   1. TensorFlow dataset minist的加载
   2. 创建自定义Model
   3. 对测试集做预测

