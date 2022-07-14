import numpy as np
import pandas as pd
from pandas import ExcelFile
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

df = pd.read_excel('yap.xlsx')

print(df.columns)

X = df.iloc[:,1:7]
y = df.iloc[:,-1]

scaler = MinMaxScaler()   
X_norm = scaler.fit_transform(X)

numpY = np.empty((278,1)) 
for i in range(278):
    numpY[i]=y[i]

# training set과 test set으로 나누기
X_train, y_train = X_norm[0:180], numpY[0:180]
X_test,  y_test  = X_norm[180::], numpY[180::]

# 모델 구조 정의하기
model = tf.keras.Sequential()  

#입력 8개로부터 전달받는 12개 노드의 layer 생성
model.add(layers.Dense(12, input_shape=(6,)))  
model.add(layers.Activation('relu'))  

model.add(layers.Dense(12))         
model.add(layers.Activation('relu'))

model.add(layers.Dense(12))         
model.add(layers.Activation('relu'))

#회귀모형(regression) 구축을 위해서 linear 활성함수 사용
model.add(layers.Dense(1))
model.add(layers.Activation('linear')) 

# 모델 구축하기
model.compile(
        loss='mse',         # mean_squared_error(평균제곱오차)의 alias
        optimizer='adam',   # 최적화 기법 중 하나
        metrics=['mae'])    # 실험 후 관찰하고 싶은 metric 들을 나열함. 

hist = model.fit(
    X_train, y_train,
    batch_size=10,    
    epochs=100,       
    validation_split=0.2,  
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],  
    verbose=2) 

# 테스트 데이터 입력
scores = model.evaluate(X_test, y_test)
print('test_loss: ', scores[0])
print('test_mae: ', scores[1])

# 모델 저장
model.save("dnn_estate.h5")

# 관찰된 metric 값들을 확인함
for i in range(len(scores)):
    print("%s: %.2f" % (model.metrics_names[i], scores[i]))

fig, loss_ax = plt.subplots(figsize=(15, 5))

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')   # 훈련데이터의 loss (즉, mse)
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss') # 검증데이터의 loss (즉, mse)

acc_ax.plot(hist.history['mae'], 'b', label='train mae')   # 훈련데이터의 mae
acc_ax.plot(hist.history['val_mae'], 'g', label='val mae') # 검증데이터의 mae

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('mean_absolute_error')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 모델 불러오기
loaded_model = load_model("dnn_estate.h5")

model.summary()

score = model.evaluate(X_test, y_test)
print('test_loss: ', score[0])
print('test_mse: ', score[1])