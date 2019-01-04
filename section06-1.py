# Created by SIHAN
# DATE 2019-01-04
# TIME 오후 11:55

import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])
nb_class = 3

W = tf.Variable(tf.random_normal([4, nb_class]))
b = tf.Variable(tf.random_normal([nb_class]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

cost_graph = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(4000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        cost_tmp = sess.run(cost, feed_dict={X: x_data, Y: y_data})
        cost_graph.append(cost_tmp)
        if step % 200 == 0:
            print("%5d" % step, cost_tmp)

    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, "\n", sess.run(tf.argmax(a, 1)))

    print('--------------')

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, "\n", sess.run(tf.argmax(b, 1)))

    print('--------------')

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, "\n", sess.run(tf.argmax(c, 1)))

    print('--------------')

    all = sess.run(hypothesis, feed_dict={
        X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, "\n", sess.run(tf.argmax(all, 1)))

    plt.plot(cost_graph)
    plt.show()
