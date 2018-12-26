# Created by SIHAN
# DATE 2018-12-12
# TIME 오후 5:31

import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1
# X = [1, 2, 3]
# Y = [1, 2, 3]
#
# W = tf.placeholder(tf.float32)
#
# hypothesis = X * W
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# sess = tf.Session()
#
# sess.run(tf.global_variables_initializer())
#
# w_val = []
# cost_val = []
#
# for i in range(-30, 50):
#     feed_W = i * 0.1
#     curr_cost, curr_w = sess.run([cost, W], feed_dict={W: feed_W})
#     w_val.append(curr_w)
#     cost_val.append(curr_cost)
#
# plt.plot(w_val, cost_val)
# plt.show()
######

# #2
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypo = X * W
cost = tf.reduce_mean(tf.square(hypo - Y))

lr = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - lr * gradient
update = W.assign(descent)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(41):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
