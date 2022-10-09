'''
基于zgpa_train.csv建立rnn模型，（单层rnn 输出有5个神经元，每次使用前8个预测第9个）预测股价
1.数据预处理转换成rnn的输入数据
2.对新数据zgpa_test.csv进行预测，可视化结果
3.储存结果，观察细节
'''
from lib2to3.pytree import type_repr
from platform import mac_ver
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
current_directory = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(current_directory+'/zgpa_train.csv')
# print(data.head())
price = data.loc[:, 'close']
print(price.head())
price_norm = price/max(price)
print(price_norm.head())

# visualize the prize
fig0 = plt.figure(figsize=(8, 5))
plt.plot(price_norm)
# plt.show()

# define method to extract X and y


def extract_data(data, step):
    X = []
    y = []
    for i in range(len(data)-step):
        X.append([a for a in data[i:i+step]])
        y.append(data[i+step])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


# X.shape(723,8,1)
X, y = extract_data(price, 8)
print(type(y))
y = np.array(y)
print('22', type(y))
model = Sequential()
# time_step=8 用前8个预测第9个
model.add(SimpleRNN(units=5, input_shape=(8, 1), activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
# TODO：这里又报错，有问题，先解决这个
# KILLED，rnn输入要求y为array，45行，将list转为array解决。
model.fit(X, y, batch_size=30, epochs=200)

# predict on training data
y_train_predict = model.predict(X)*max(price)
y_train = y*max(price)
print('y_train_predict', y_train_predict, 'y_train', y_train)
# visualize the predict result
fig1 = plt.figure(figsize=(10, 10))
plt.plot(y_train_predict, label='y_train_predict')
plt.plot(y_train, label='y_train')
plt.legend()
# 当图示效果不好的时候，看一下损失函数有没有改变，可能是深度学习给的初始状态不好。重新运行一次就好了
plt.show()

# load test data and preprocess the test data
data_test = pd.read_csv(current_directory+'/zgpa_test.csv')
price_test = data_test.loc[:, 'close']
# 将数据归一化的时候除以相同的price，保持不变。
price_test_norm = price_test/max(price)
# 步长为8，前八个为X，预测后一个为y
X_test_norm, y_test_norm = extract_data(price_test, 8)
print('X_test_norm', X_test_norm.shape, 'y_test_norm', y_test_norm)
y_test_predict = model.predict(X_test_norm)*max(price)
y_test = np.array(y_test_norm).reshape(-1, 1)*max(price)
print('y_test_predict', y_test_predict, 'y_test', y_test)
fig2 = plt.figure(figsize=(10, 10))
plt.plot(y_test_predict, label='y_test_predict')
plt.plot(y_test, label='y_test')
plt.legend()
# TODO：76行 extract_data(price_test, 8)的时候是拟合，extract_data(price_test_norm, 8)差一半
plt.show()

# download
# TODO：UNSKILLED function concatenate
print('4', y_test.shape, y_test_predict.shape)
result = np.concatenate((y_test, y_test_predict), axis=1)
result = pd.DataFrame(result, columns=['y_test', 'y_test_predict'])
result.to_csv(current_directory+'/test111.csv')
