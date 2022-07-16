import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

df = pd.read_excel('dataframe.xlsx')
#메인 변수 5개 - 소비자물가, 금리, 국내총생산, 신생아 출산인구, 합계 출산율
#Y값(레이블) - 수도권 인구 과밀화율
X = df.iloc[:,0:5] #0~5까지 열, 즉 메인변수들
y = df.iloc[:,5] #마지막 열, 즉 레이블
scaler = StandardScaler() #표준화. 평균이 0이고 분산이 1인 정규분포로 만드는 것
X_norm = scaler.fit_transform(X) #메인 변수들끼리 단위가 모두 다르기 때문에 표준화 시켜야 함.

numpY = np.empty((278,1)) #pandas라서 numpy형으로 바꿈
for i in range(278):
    numpY[i]=y[i]

#training set과 test set으로 나누기 #랜덤으로 스플릿
X_train,X_test,y_train,y_test=train_test_split(X_norm, numpY, test_size=0.2, random_state=42,shuffle=True)

# 모델 구조 정의하기
model = tf.keras.Sequential()  

#입력 8개로부터 전달받는 12개 노드의 layer 생성
model.add(layers.Dense(128, input_shape=(5,),activation='sigmoid')) #메인변수 5개라서 input shape = 5. 활성함수 sigmoid로 
model.add(layers.Dense(64,activation='relu')) #활성함수 relu
model.add(layers.Dense(16,activation='relu')) #활성함수 relu
#회귀모형(regression) 구축을 위해서 linear 활성함수 사용
model.add(layers.Dense(1,activation='linear')) 
model.summary()

# 모델 구축하기
model.compile(
        loss='mse',         # mean_squared_error(평균제곱오차)의 alias
        optimizer='adam',   # 최적화 기법 중 하나
        metrics=['mae'])    # 실험 후 관찰하고 싶은 metric 들을 나열함. 

hist = model.fit(
    X_train, y_train,
    batch_size=10,    
    epochs=10000,       
    validation_split=0.2,  
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mae', patience=1)], #과적합 방지용. loss가 100 epoch 동안 개선되지 않으면 학습 중단 
    verbose=2) #학습 중 출력 문구 설정. 0이면 출력 X, 1이면 훈련 진행 막대, 2이면 미니배치마다 loss

# 모델 저장
model.save("test_dnn.h5") #test_dnn.h5라는 이름으로 모델 저장

scores = model.evaluate(X_test, y_test) #분리해둔 테스트 데이터로 모델 평가 후 scores 변수에 저장

# 관찰된 metric 값들을 확인함
for i in range(len(scores)):
    print("%s: %.2f" % (model.metrics_names[i], scores[i]))

#모델 손실 그래프 준비
fig, loss_ax = plt.subplots(figsize=(15, 5))

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train_mse')   # 훈련데이터의 loss (즉, mse)
loss_ax.plot(hist.history['val_loss'], 'r', label='test_mse') # 검증데이터의 loss (즉, mse)

acc_ax.plot(hist.history['mae'], 'b', label='train_mae')   # 훈련데이터의 mae
acc_ax.plot(hist.history['val_mae'], 'g', label='val_mae') # 검증데이터의 mae

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('mean_absolute_error')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 모델 불러오기
loaded_model = load_model("test_dnn.h5")

score = model.evaluate(X_test, y_test)
print('test_loss: ', score[0])
print('test_mae: ', score[1])