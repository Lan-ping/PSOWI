import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D,Flatten
import os

import psowi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from psowi import PSOWI


data = np.genfromtxt('data2.csv', delimiter=',', dtype=np.float32)
# 20列输入值，1列输出
# 数据归一化
scaler = MinMaxScaler((0,1))
dataset = scaler.fit_transform(data)

x = np.expand_dims(dataset[:,:-1].astype(float),axis=2)
y = dataset[:,[-1]]
# print(y)

# 划分训练集、测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=0)

# 度量函数
def determin(y, y_pred):
    SS_res = K.sum(K.square(y - y_pred))
    SS_tot = K.sum(K.square(y - K.mean(y)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# 定义网络
model = Sequential()
model.add(Conv1D(32,3,input_shape=(20,1),activation='relu'))
model.add(Conv1D(64,3,activation='relu'))
model.add(MaxPooling1D(2))

model.add(Flatten())
model.add(Dense(60,activation='sigmoid'))
model.add(Dense(120,activation='sigmoid'))
model.add(Dense(240,activation='sigmoid'))
model.add(Dense(1,activation='linear'))
# plot_model(model,to_file='./model.png',show_shapes=True)
model.summary()
model.compile(optimizer='adam',loss='mse',metrics=[determin])

psowi = PSOWI(model,x_train,y_train,x_test,y_test)
model = psowi.get_optimal_model()

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=800,batch_size=30)

y_pred = model.predict(x)
dataset = pd.DataFrame(dataset)
dataset.iloc[:, 20] = y_pred

#反归一化
tran_data = scaler.inverse_transform(dataset)
y_pred = tran_data[:,[-1]]
# print(y_pred)
# print(y)

# 准确率
scores = model.evaluate(x_test, y_test, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

# 打印准确率和损失值
# loss,accuracy = model.evaluate(x_test, y_test, verbose=0)
# print(accuracy)
# print(loss)

# 打印权重列表
# weight = model.get_weights()
# print(weight)

