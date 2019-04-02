"""
Author: Kevin
Link: www.kaikai.ai
Github: github.com/loveunk

这是基于TensorFlow2.0版本的2个完整的mnist分类demo，涉及：
1. TensorFlow dataset minist的加载
2. 创建自定义Model
3. 对测试集做预测

可以作为入门TensorFlow 2.0的例子。
测试环境：TensorFlow：2.0.0-alpha0
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


def tf2_helloworld_for_beginner():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test)


# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/quickstart/advanced.ipynb
def tf2_helloworld_for_advanced():
    # Load and prepare the MNIST dataset.
    # Convert the samples from integers to floating-point numbers:
    dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(32)
    mnist_test = mnist_test.map(convert_types).batch(32)

    # Build the tf.keras model using the Keras model subclassing API
    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10, activation='softmax')

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    model = MyModel()

    # Choose an optimizer and loss function for training:
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # Select metrics to measure the loss and the accuracy of the model.
    # These metrics accumulate the values over epochs and then print the overall result.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Train the model using tf.GradientTape:
    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            predictions = model(image)
            loss = loss_object(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, predictions)

    # Now test the model:
    @tf.function
    def test_step(image, label):
        predictions = model(image)
        t_loss = loss_object(label, predictions)

        test_loss(t_loss)
        test_accuracy(label, predictions)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        for image, label in mnist_train:
            train_step(image, label)

        for test_image, test_label in mnist_test:
            test_step(test_image, test_label)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

    # The image classifier is now trained to ~98% accuracy on this dataset.

print(tf.__version__)

tf2_helloworld_for_beginner()
# tf2_helloworld_for_advanced()

