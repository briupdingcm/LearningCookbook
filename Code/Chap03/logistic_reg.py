import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import  datasets
from sklearn.preprocessing import normalize
import requests

def load_dataset():
    birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')[5:]
    birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
    birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
    y_vals = np.array(x[1] for x in birth_data)
    x_vals = np.array(x[2:9] for x in birth_data)
    return x_vals, y_vals

x_vals, y_vals=load_dataset()

def input():
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals(train_indices)
    x_vals_test = x_vals(test_indices)
    y_vals_train = y_vals(train_indices)
    y_vals_test = y_vals(test_indices)
    return x_vals_train, y_vals_train, x_vals_test, y_vals_test


