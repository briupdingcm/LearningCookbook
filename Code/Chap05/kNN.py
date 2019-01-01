import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests

num_features = 10
def load_dataset():
    housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                      'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV0']
    cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS',
                      'TAX', 'PTRATIO', 'B', 'LSTAT']
    num_features = len(cols_used)
    housing_file = requests.get(housing_url)

    housing_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in housing_file.text.split('\n') if
                    len(y) >= 1]
    y_vals = np.transpose([np.array([y[13] for y in housing_data])])
    x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])
    x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)
    return x_vals, y_vals

x_vals, y_vals = load_dataset()

def split(indices):
    return x_vals[indices], y_vals[indices]

def choice():
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    return train_indices, test_indices

train_indices, test_indices = choice()
x_vals_train, y_vals_train = split(train_indices)
x_vals_test, y_vals_test = split(test_indices)

k = 4
batch_size = len(x_vals_test)

with tf.Session() as sess:
    x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
    x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
    y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    with tf.name_scope('simility'):
        distance = tf.reduce_mean(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), reduction_indices=2)

    with tf.name_scope('inference'):
        top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)


