import tensorflow as tf
import numpy as np

g = tf.Graph()

with g.as_default():
    a = tf.Variable(tf.constant(4.))
    x_val = 5.
    x_data = tf.placeholder(dtype=tf.float32)

    with tf.name_scope("output"):
        multiplication = tf.multiply(a, x_data)

    with tf.name_scope("loss"):
        loss = tf.square(tf.subtract(multiplication, 50.))

    with tf.name_scope("train"):
        my_opt = tf.train.GradientDescentOptimizer(0.01)
        train_step = my_opt.minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(10):
            sess.run(train_step, feed_dict={x_data: x_val})
            a_val = sess.run(a)
            mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
            print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))
