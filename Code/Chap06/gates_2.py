import tensorflow as tf
import numpy as np

g = tf.Graph()

with g.as_default():
    a = tf.Variable(tf.constant(1.))
    b = tf.Variable(tf.constant(-1.8))
    x_val = 5.
    x_data = tf.placeholder(dtype=tf.float32)

    with tf.name_scope("output"):
        multiplication = tf.add(tf.multiply(a, x_data), b)

    with tf.name_scope("loss"):
        loss = tf.square(tf.subtract(multiplication, 50.))

    with tf.name_scope("train"):
        my_opt = tf.train.GradientDescentOptimizer(0.01)
        train_step = my_opt.minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(100):
            [a_val, b_val] = sess.run([a, b])
            sess.run(train_step, feed_dict={x_data: x_val})
            mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
            print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(mult_output))