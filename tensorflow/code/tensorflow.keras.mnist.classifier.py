"""
Author: Kevin
Github: github.com/loveunk

这是一个完整的mnist分类demo，其中涉及的技术点包括：
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

可以作为入门TensorFlow/Keras的例子。
测试环境：TensorFlow：1.13.1
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(13)
sns.set(style='white', context='talk', palette='deep')

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

# 看看数据的shape
print(X_train.shape)
print(Y_train.shape)

# 画一个数据集的例子来看看
plt.imshow(X_train[0][:,:])
plt.show()

# 打印数据的直方图
sns.countplot(Y_train)
plt.show()

# 归一化数据，让CNN更快
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 把label转换为one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                  Y_train,
                                                  test_size=0.1,
                                                  random_state=2)


# 创建CNN model
# 模型：
"""
  [[Conv2D->relu]*2 -> BatchNormalization -> MaxPool2D -> Dropout]*2 -> 
  [Conv2D->relu]*2 -> BatchNormalization -> Dropout -> 
  Flatten -> Dense -> BatchNormalization -> Dropout -> Out
"""
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='Same', activation='relu', input_shape = (28,28,1)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='Same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='Same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(10, activation="softmax"))

# 打印出model 看看
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
plt.imshow(mpimg.imread('model.png'))
plt.show()

# 定义Optimizer
optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# 编译model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# 设置学习率的动态调整
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                               patience=3,
                                                               verbose=1,
                                                               factor=0.5,
                                                               min_lr=0.00001)

# should add early_stopping to the model training callbacks later
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True)

# 设置epochs和batch size
epochs = 20
batch_size = 128

# 通过数据增强来防止过度拟合
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

# 训练模型
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs=epochs,
                              validation_data=(X_val, Y_val),
                              verbose=2,
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction, early_stopping, tf.keras.callbacks.TensorBoard(log_dir='./log_dir')])

# 画训练集和验证集的loss和accuracy曲线。可以判断是否欠拟合或过拟合
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()

# 对测试集做预测
results = model.predict(X_test)

# 把one-hot vector转换为数字
Y_pred = np.argmax(results, axis=1)

print("precision = ", precision_score(Y_test, Y_pred, average="macro"))
print("recall = ", recall_score(Y_test, Y_pred, average="macro"))
print("f1_score = ", f1_score(Y_test, Y_pred, average="macro"))
print("accuracy = ", accuracy_score(Y_test, Y_pred))
