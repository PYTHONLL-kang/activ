import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_excel('aug_nine_var.xlsx')

X = df.iloc[:,1:22]
y = df.iloc[:,22:23].to_numpy()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_dim=21, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

model.summary()

hist = model.fit(X_train, y_train, epochs=5000, batch_size=8, validation_split=0.2, verbose=1,
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000, verbose=1, restore_best_weights=True
))

pred = model.predict(X)

model.save('dnn.h5')
np.save('dnn_out.npy', pred)