#matplotlib 이라는 패키지로, 그래프를 그리거나 이미지를 편집하는 데 사용된다.
import matplotlib.pyplot as plt
#텐서플로우 패키지의 import문
import tensorflow as tf
from math import * 
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#pandas 판다스
import pandas as pd
pd.options.display.float_format = '{:.5f}'.format

# numpy 임포트
import numpy as np
np.set_printoptions(precision=6, suppress=True)

#shuffle에 사용된다.
from random import *
#MinMaxScaler 사용을 통해 0~1의 데이터로 만들고, 그걸 다시 복구 하는 방식이다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
#데이터를 train과 test를 나누는 패키지
from sklearn.model_selection import train_test_split
def evaluate_loss(y_pred,y_test):
    global model
    global X_train
    global y_train
    global scaler
    y_pred = y_pred
    y_pred = np.array(y_pred).reshape((1,21,))
    last_data = X_train[-1]
    pred_1 = model.predict(y_pred,verbose=0)
    # pred_2 = model.predict(y_test,verbose=0)
    b = np.sqrt(np.sum(np.square(y_pred-last_data)))
    a = (np.square(y_train[-1]-pred_1))
    return a+(np.abs(b*10-a))
#matplotlib 이라는 패키지로, 그래프를 그리거나 이미지를 편집하는 데 사용된다.
import matplotlib.pyplot as plt
#텐서플로우 패키지의 import문
import tensorflow as tf
#keras import하는 것
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
# tensorflow gpu error 무시
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#randint 선언때에 사용된다.
from random import *
#MinMaxScaler 사용을 통해 0~1의 데이터로 만들고, 그걸 다시 복구 하는 방식이다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
#데이터를 train과 test를 나누는 패키지
from sklearn.model_selection import train_test_split
# 엑셀 데이터 받기
import random
df = pd.read_excel('aug_nine_var.xlsx')
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
global X_train
global y_train
_,X_test,_,y_test=train_test_split(X, numpY, test_size=0.05, random_state=42,shuffle=False)
X_train,_,y_train,_=train_test_split(X, numpY, test_size=0.05, random_state=42,shuffle=True)
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
model.save("test.h5")
# 학습 시키기
hist = model.fit(X_train, y_train, epochs=120, batch_size=16, validation_split=0.2, verbose=1)
# model = keras.models.load_model("test.h5")
epochs = 1000
latent_dim = 21 
model.trainable = False
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
noise_shape = keras.Input(shape=(latent_dim,))
my_model = Sequential([
    keras.Input((latent_dim,)),
    layers.Reshape((latent_dim,)),
    layers.Activation('relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(256,activation='selu'),
    layers.Dense(512,activation='relu'),
    layers.Dense(1024,activation='relu'),
    layers.Dense(21,activation='relu'),
    layers.Activation('relu')
    ])

generator = Model(noise_shape,my_model(noise_shape))
noise_shape = keras.Input(shape=(latent_dim,))
gen_thing_to_real_model = Model(noise_shape,model(my_model(noise_shape)))
g_optimizer=keras.optimizers.Adam(learning_rate=0.0001)
generator.compile(loss=evaluate_loss,optimizer=g_optimizer)
gen_thing_to_real_model.compile(loss='mse',optimizer=g_optimizer)

def train_step(real_model,generator,epochs,model):
    df = pd.read_excel('dataframe.xlsx')
    global X_train
    global y_train
    batch_size = 1
    y_train = list(y_train)
    x = []
    y = np.array([])
    result_ = np.array([])
    to_excel_list = []
    average = sum(y_train) / len(y_train)
    for j in range(epochs):
        noise = tf.random.normal([1,21])
        real_model.train_on_batch(noise,np.zeros((1,21)))
        y_pred = generator.predict(noise,verbose=0)[0]
        s = evaluate_loss(y_pred,_)
        average = model.predict(y_pred.reshape((1,21)),verbose=0)
        print("loss_value: %.4f average: %.4f try: %d" % (s,average,j))
        to_excel_list = generator.predict(noise,verbose=0)[0]
        to_excel_list = np.array(to_excel_list).reshape((1,21))
        c = to_excel_list
        to_excel_list = scaler.inverse_transform(to_excel_list)
        to_excel_list = np.append(to_excel_list,model.predict(c,verbose=0))
        result_ = np.append(result_,to_excel_list).reshape((-1,22))
        df = pd.DataFrame(result_)    
        df.to_excel("basic_formula.xlsx")
    
train_step(gen_thing_to_real_model,generator,epochs,model)