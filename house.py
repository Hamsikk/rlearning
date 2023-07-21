# 라이브러리 임포트
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam



# 데이터셋 임포트
dataset = pd.read_csv('C:/Users/user/Desktop/S/reinforcement learning/kc_house_data.csv')

# 특징과 타깃 구분
X = dataset.iloc[ : , 3 : ].values
X = X[: , np.r_[0:13, 14:18]]
y = dataset.iloc[:, 2].values

# 데이터셋을 훈련 집합과 테스트 집합으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 특징 스케일링
xscaler = MinMaxScaler(feature_range=(0,1))
X_train = xscaler.fit_transform(X_train)
X_test = xscaler.transform(X_test)

# 타깃 스케일링
yscaler = MinMaxScaler(feature_range=(0,1))
y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
y_test = yscaler.transform(y_test.reshape(-1, 1))

# 인공신경망 구축
model = Sequential()
model.add(Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim=17))
model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='relu'))
model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mean_absolute_error'])

# 인공 신경망 훈련
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# 테스트 집합에서 예측 생성해서 스케일링을 원복시킴
y_test = yscaler.inverse_transform(y_test)
prediction = yscaler.inverse_transform(model.predict(X_test))

# 오차율 계산
error = abs(prediction - y_test) / y_test
print(np.mean(error)*100)