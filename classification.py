'''
非线性二分类任务：

基于data.csv数据建立mlp模型，计算准确率，可视化预测结果
数据分离 test_size=0.33 ,random_state=10
一层隐藏层20个神经元
'''
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
current_directory = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(current_directory+'/'+'data.csv')
# print(data.head())
X = data.drop(['y'], axis=1)
y = data.loc[:, 'y']
# print('X', X, 'y', y)
# visualize the data
# fig0 = plt.figure(figsize=(3, 3))
passed = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
failed = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
plt.legend((passed, failed), ('passed', 'failed'))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('raw data')
# plt.show()

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=10)
# print(X_train.shape, X_test.shape, X.shape)
mlp = Sequential()
mlp.add(Dense(units=20, input_dim=2, activation='sigmoid'))
mlp.add(Dense(units=1, activation='sigmoid'))
mlp.summary()
mlp.compile(optimizer='adam', loss='binary_crossentropy')
mlp.fit(X_train, y_train, epochs=3000)
y_train_predict = mlp.predict_classes(X_train)
y_test_predict = mlp.predict_classes(X_test)
accuracy_train = accuracy_score(y_train, y_train_predict)
accuracy_test = accuracy_score(y_test, y_test_predict)
print('accuracy_train', accuracy_train, 'accuracy_test', accuracy_test)
# mlp出来之后是数组，没有索引，需要转换
# print(y_train_predict)
y_train_predict_form = pd.Series(i[0] for i in y_train_predict)
# y_train_predict_form 出来之后是带索引的列表，可以进行索引取值
# print(y_train_predict_form)

# TODO:UNSKILLED。
# generate new data to predict to plot
xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
x_range = np.c_[xx.ravel(), yy.ravel()]
y_range_predict = mlp.predict_classes(x_range)
y_range_predict_form = pd.Series(i[0] for i in y_range_predict)
fig1 = plt.figure(figsize=(5, 5))
predict_failed = plt.scatter(
    x_range[:, 0][y_range_predict_form == 0], x_range[:, 1][y_range_predict_form == 0])
predict_passed = plt.scatter(
    x_range[:, 0][y_range_predict_form == 1], x_range[:, 1][y_range_predict_form == 1])
failed = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
passed = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('mlp predict result')
# plt.show()
