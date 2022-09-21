from glob import glob
from tabnanny import verbose
import numpy as np
# numpy e형태로 안나오도록 하는 코드
np.set_printoptions(precision=6, suppress=True)
#matplotlib 이라는 패키지로, 그래프를 그리거나 이미지를 편집하는 데 사용된다.
import matplotlib.pyplot as plt
#텐서플로우 패키지의 import문
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
#keras import하는 것
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
# tensorflow gpu error 무시
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#pandas 판다스
import pandas as pd
#randint 선언때에 사용된다.
from random import *
#MinMaxScaler 사용을 통해 0~1의 데이터로 만들고, 그걸 다시 복구 하는 방식이다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
#데이터를 train과 test를 나누는 패키지
from sklearn.model_selection import train_test_split
# 엑셀 데이터 받기
import random
from keras import Input
from keras.layers import Dense
df = pd.read_excel('aug_nine_var.xlsx')
df = df.dropna()
# x와 y데이터 일부 나누기
X = df.iloc[:,1:22]
y = df.iloc[:,22]
# minmaxsclaer 생성 후 fit_transform 시키기(0~1형태로 만드는 형태)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
numpY = y.to_numpy()
#X_test와 y_test의 경우 마지막 데이터가 우리 데이터의 최신값이다. dqn에서의 reward값을 받기 위해 X_test와 y_test를 전역변수로 만들었다.
global X_test
global y_test
# 처음 나누는 것은 test의 경우, 마지막, 최근 데이터를 얻기 위해 나누는 것이고, 두번째의 경우 학습용 데이터로 shuffle을 해놓음.
_,X_test,_,y_test=train_test_split(X, numpY, test_size=0.05, random_state=42,shuffle=False)
X_train,_,y_train,_=train_test_split(X, numpY, test_size=0.05, random_state=42,shuffle=True)
X = X_train
Y = y_train
#model을 생성한다. 원 dnn모델로, lstm모델과 합쳐 정확도를 높일 계획이다.
global model
model = tf.keras.Sequential()
model.add(layers.Dense(1024, input_shape=(21,), activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# 학습 시키기
hist = model.fit(X_train, y_train, epochs=120, batch_size=16, validation_split=0.2)

G_input = Input(shape = (15,))
G_model = Sequential()
G_model.add(G_input)

G_model.add(Dense(10, input_dim=15,activation='relu'))
G_model.add(Dense(20, activation='relu'))
G_model.add(Dense(20, activation='relu'))
G_model.add(Dense(20, activation='relu'))
G_out_x_data = Dense(21)
G_out_lab = Dense(1)
G_model = Model(G_input,[G_out_x_data(G_model(G_input)),G_out_lab(G_model(G_input))])
G_model.compile(loss='mse', optimizer='adam')
G_model.summary()

F_model = model
def real(real,fake):
    global K_loss
    c = tf.square(real-fake)
    c = c / c * K_loss
    return c
ll = []
ll_v = []
#######################################################################################################################
def train_step(F_model,G_model,epochs,batch,X,Y):
    r = np.array([])
    global X_test
    global y_test
    mk = []
    for j in range(epochs):
        for i in range(batch):
            noise = np.random.normal(size=15).reshape((1,15))
            train_data_G = G_model.predict(noise,verbose=0)[0]
            train_lab_G = G_model.predict(noise,verbose=0)[1]
            print("try: %d - %d" % (j,i))
            print(train_data_G[0],train_lab_G[0][0],F_model.predict([train_data_G],verbose=0)[0][0],"\n")
            global K_loss
            b = np.sqrt(np.sum(np.square(train_data_G-X[-1])))
            pred_1 = Y[-1]
            a = (np.square(y_test[-1]-pred_1))
            K_loss = pred_1+(np.abs(b*10-a))+np.square(F_model.predict([train_data_G],verbose=0) - train_lab_G)
            train_data_G = train_data_G.tolist()[0]
            train_lab_G = train_lab_G.tolist()[0]
            G_loss = F_model.evaluate(X_test,y_test,verbose=0)
            X_thing = np.array(X[i]).reshape((1,21,))
            Y_thing = np.array(Y[i]).reshape((1,1,))
            m = []
            m.append(X_thing)
            m.append(Y_thing)
            d = G_model.train_on_batch(noise,m)
            mk.append(K_loss)
            ll.append(G_loss)
        to_excel = np.array([train_data_G])
        to_excel = scaler.inverse_transform(to_excel)
        # to_excel = np.concatenate([to_excel,[train_lab_G]],axis=1)
        to_excel = np.concatenate([to_excel,F_model.predict([train_data_G],verbose=0)],axis=1)
        r = np.append(r,to_excel).reshape((-1,22))
        print(r)
        pl = pd.DataFrame(r)
        pl.to_excel("result.xlsx")
        train_data_G = np.array(train_data_G)
        print("g loss: %.4f" % (sum(mk)/batch))
    plt.plot(range(len(ll)),ll)

G_model.compile(loss=real,optimizer='sgd')
F_model.compile(loss='mse',optimizer='adam')
epochs = 2000
batch_size = 16
train_step(F_model,G_model,epochs,batch_size,X,Y)



