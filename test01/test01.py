import tensorflow as tf
import numpy as np


tensor = 0.8
a = tf.random_uniform([4, 3]) + tensor
x = tf.SparseTensor(indices=[[0, 0], [1, 1]], values=[1, 1], dense_shape=[3, 3])
# tf.floor(x, name=None) 是向下取整，3.6=>3.0；
# tf.ceil(x, name=None) 是向上取整，3.6=>4.0。
dropout_mask = tf.cast(tf.floor(a), dtype=tf.bool)
# dropout_mask必须是一维的，to_retain = [True, False, False, True]
pre_out = tf.sparse_retain(x, [True, True])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 这里随机输出是第二次结果
    # print(sess.run(a))
    print(sess.run(dropout_mask))
    print(sess.run(pre_out))
    # print(sess.run(tf.floor(a)))
