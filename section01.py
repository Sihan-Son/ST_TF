# Created by SIHAN
# DATE 2018-12-12
# TIME 오후 4:53

import tensorflow as tf
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = tf.add(a, b)

sess = tf.Session()

print(sess.run(c, feed_dict={a: 3, b: 4.5}))
