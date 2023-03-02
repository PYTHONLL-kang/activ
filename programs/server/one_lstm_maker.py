# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# %%
import os
os.chdir('../../')

# %%
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

tf.config.run_functions_eagerly(True)

# %%
def shift_data(origin, d):
    shift_d = np.concatenate((origin[0][1::], d), axis=0).reshape(1, 12, 1)
    return shift_d

# %%
df = pd.read_excel('./documents/nov_nine_var.xlsx').iloc[:,1::].to_numpy()

scaler = MinMaxScaler()
scale_df = scaler.fit_transform(df)

data = scale_df[:,0:21][-1-12:-1].reshape(1, 12, 21)

# %%
def make_dataset(d, length=12, test_size=0.2):
    X_data, y_data = [], []

    for i in range(0, len(d)-length-1):
        X_data.append(d[i:i+length])
        y_data.append(d[i+length+1])

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    da = train_test_split(X_data, y_data, test_size=test_size, shuffle=False)
    da[0] = da[0].reshape(len(da[0]), 12, 1)
    da[1] = da[1].reshape(len(da[1]), 12, 1)
    da[2] = da[2].reshape(len(da[2]), 1)
    da[3] = da[3].reshape(len(da[3]), 1)
    
    return da

# %%
class Model_(tf.keras.Model):
    def __init__(self):
        super(Model_, self).__init__()
        self.d0 = tf.keras.layers.LSTM(16, activation='tanh', return_sequences=False, dropout=0.2)
        self.d2 = tf.keras.layers.Dense(8, activation='relu')
        self.d3 = tf.keras.layers.Dense(units=1, activation='linear')

    def call(self, inputs):
        x = self.d0(inputs)
        x = self.d2(x)
        x = self.d3(x)

        return x

# %%
model_list = [[Model_() for i in range(21)] for j in range(5)] # model_list.shape = 5, 21, model
for j in range(5):
    for i in range(21):
        d = make_dataset(scale_df[:,i])
        model = model_list[j][i]
        model.build(input_shape=(1, 12, 1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(d[0], d[2], epochs=1, batch_size=128, validation_data=(d[1], d[3]), verbose=0)
        model.save('./model/one_lstm/one_lstm_{0}/{1}_model'.format(j, i))

# %%
real_axis = [i for i in range(scale_df.shape[0])]
pred_axis = [i + scale_df.shape[0] for i in range(500)]

for i in range(21):
    plt.subplot(3, 7, i+1)
    plt.plot(real_axis, scale_df[:, i])

    a = np.zeros((500, 1))
    pred = scale_df[:, i][-13:-1].reshape(1, 12, 1)
    for j in range(500):
        a[j] = pred[0][-1]
        pred = shift_data(pred, model_list[i](pred))
    
    plt.plot(pred_axis, a[:,0], c='r')

plt.show()

# %%
a = 0
for i in range(21):
    da = make_dataset(scale_df[:,-1])
    a += model_list[i].evaluate(da[1], da[3])
print(a)

# %%
a/21

# %%



