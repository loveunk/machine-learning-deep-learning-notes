"""
Author: Kevin
Link: www.kaikai.ai
Github: github.com/loveunk

使用一个简单的conv2d网络测试CPU & GPU的性能对比

测试环境：TensorFlow：1.13.1
"""

import tensorflow as tf
import timeit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.device('/cpu:0'):
    random_image_cpu = tf.random_normal((100, 1000, 100, 3))
    net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
    net_cpu = tf.reduce_sum(net_cpu)

with tf.device('/gpu:0'):
    random_image_gpu = tf.random_normal((100, 1000, 100, 3))
    net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
    net_gpu = tf.reduce_sum(net_gpu)

sess = tf.Session(config=config)

# Test execution once to detect errors early.
try:
    sess.run(tf.global_variables_initializer())
except tf.errors.InvalidArgumentError:
    print(
        '如果出了这个Error表示GPU配置不成功！\n\n')
    raise


def cpu():
    sess.run(net_cpu)


def gpu():
    sess.run(net_gpu)


# Runs the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x1000x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time / gpu_time)))

sess.close()
