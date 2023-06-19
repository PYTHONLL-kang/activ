import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from sklearn.model_selection import train_test_split

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

dnn_model = tf.keras.models.load_model('./model/dnn.h5')

def make_sequence(X, length):
    """
    args : 
        X(ndarray) : (len(X), 21)
        lenfth(int) : sequencial length of lstm

    returns :
        sequence : (len(X), length, 21)
        label : (len(X), 21)
    """
    X_data, y_data = [], []

    for idx in range(len(X)-length-1):
        X_data.append(X[idx:idx+length])
        y_data.append(X[idx+length+1])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data

def make_dataset(X, length, n=150, test_size=0.2):
    """
    args :
        X(ndarray) : (total length, 21)
        length(int) : sequential length of lstm
        n(int : default=150) : each length of dataset
        test_size(float) : train-test rate

    returns :
       train_data : (n, 60, 21)
       train_label : (n, 21)
       test_data : (n*test_size, 60, 21)
       test_label : (n*test_size, 21)
    """
    X_data, y_data = [], []

    for i in range(int(n/(1-test_size))):
        idx = rand.randint(0, len(X)-length-2)
        X_data.append(X[idx:idx+length])
        y_data.append(X[idx+length+1])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return train_test_split(X_data, y_data, test_size=test_size, shuffle=False)

def shift_data(origin, d, l=60):
    shift_d = tf.concat((origin[0][1::], d), axis=0).reshape(1, l, 21)
    return shift_d

class Node:
    def __init__(self, state):
        self.child = None
        self.state = state

class Tree:
    def __init__(self, leaf_num, model):
        self.root = None
        self.state_list = []
        self.leaf_num = leaf_num
        self.model_list = model
        self.len_model = len(self.model_list)

        self.size = 0
        self.depth = 1

    def _size(self):
        i = 0
        t = self.size

        while True:
            t -=  self.leaf_num**i
            i += 1

            if t == 0:
                break

        self.depth = i

    @tf.function
    def sort_by_value(self, value, child_state):
        for i in range(self.len_model):
            state = child_state[i, 0, -1].reshape(1, 21)
            value = value[i].assign(dnn_model(state)[0][0])

        argsort_order = tf.argsort(value)
        return argsort_order

    def push(self, state):
        node = Node(state)

        node.child = [Node for i in range(self.leaf_num)]
        child_state = np.zeros((self.len_model, 1, 60, 21))

        for i in range(self.len_model):
            child_state[i] = shift_data(state, self.model_list[i](state))

        sorting_child_state = self.sort_by_value(tf.Variable(tf.zeros(self.len_model)), child_state)
        for j in range(self.len_model):
            i = sorting_child_state[j]
            node.child[j] = Node(child_state[i])

        self.size += 1

        return node

    def post_order(self):
        self.state_list = []
        def _post_order(node):
            if node.child != None:
                for i in range(self.leaf_num):
                    _post_order(node.child[i])
            self.state_list.append(node.state[0][-1].reshape(1, 21))

        _post_order(self.root)

    def sort_by_level(self):
        self.post_order()
        self._size()
        arr = np.zeros((self.depth, self.leaf_num, 21))
        counter = np.zeros(self.depth, dtype=np.int8)

        c = self.depth -1
        for s in self.state_list:
            if counter[c] >= self.leaf_num:
                c -= 1

            arr[c][counter[c]] = s
            counter[c] += 1

            if c == 0:
                break

            if counter[c] % self.leaf_num == 0 and self.leaf_num != 1:
                arr[c-1][counter[c-1]] = s
                counter[c-1] += 1

        return arr

def grow(model, data, leaf, end):
    def _grow(tree, node, t, end):
        if t >= end:
            return

        for i in range(leaf):
            node.child[i] = tree.push(node.child[i].state)
            _grow(tree, node.child[i], t+1, end)

    tree = Tree(leaf, model)
    tree.root = tree.push(data)
    _grow(tree, tree.root, 0, end)

    return tree

def plot_tree(model, data, leaf, end, plot=True):
    """
    model(list) : list of lstm network
    data(ndarray) : init state, (1, 60, 21)
    leaf(int) : number of prong, lower than length of model
    end(int) : prediction time
    """
    tree = grow(model, data, leaf, end)
    l = tree.sort_by_level()

    if plot:
        for j in range(l.shape[0]):
            plt.scatter(j, dnn_model(l[j].reshape(1, 21)))

        plt.show()
    return l

if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    
    path = "C:\\code\\activ"
    lstm = tf.keras.models.load_model(path+'/model/power_lstm')
    scaler = MinMaxScaler()

    data = scaler.fit_transform(pd.read_excel(path+"/documents/nov_nine_var.xlsx").iloc[:,1:22].to_numpy())[-61:-1]
    plot_tree([lstm], data.reshape(1,60,21), 1, 10)