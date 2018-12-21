import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)


def load_data():
    x_vals_column = np.transpose(np.matrix(x_vals))
    ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
    A = np.column_stack((x_vals_column, ones_column))
    b = np.transpose(np.matrix(y_vals))
    return A, b

A,b = load_data()

def input():
    A_tensor = tf.convert_to_tensor(A)
    b_tensor = tf.convert_to_tensor(b)
    return A_tensor, b_tensor

g = tf.Graph()

with g.as_default():
    with g.name_scope("train"):
        A_tensor, b_tensor = input()
        tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
        tA_A_inv = tf.matrix_inverse(tA_A)
        product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
        solution = tf.matmul(product, b_tensor)

    with g.name_scope("inference"):
        best_fit = tf.matmul(A_tensor, solution)

    with tf.Session() as sess:
        solution_eval = sess.run(solution)
        best_fit = sess.run(best_fit)
        print(solution_eval)
        slope = solution_eval[0][0]
        y_intercept = solution_eval[1][0]
        print('slope: ' + str(slope))
        print('intercept: ' + str(y_intercept))

    with tf.name_scope("draw"):
        plt.plot(x_vals, y_vals, 'o', label='Data')
        plt.plot(x_vals, best_fit, 'r-', label='Best', linewidth=3)
        plt.legend(loc='upper left')
        plt.show()
