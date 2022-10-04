'''
加载keras内置mnist数据集，可视化第一条数据
数据预处理：图像数据维度转化，归一化，输出格式转换
计算模型在预测数据集的准确率
模型结构 两层隐藏层，每层392个神经元
'''
# 多分类类别标签转化为卷积层概率数组
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
import imp
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train)

# visualize the data
fig1 = plt.figure(figsize=(5, 5))
plt.imshow(X_train[0])
plt.title(y_train[0])
# plt.show()

# TODO:UNSKILLED
# format the input data turn 60000*28*28 to 60000*784
feature_size = X_train[0].shape[0]*X_train[0].shape[1]
X_train_format = X_train.reshape(X_train.shape[0], feature_size)
X_test_format = X_test.reshape(X_test.shape[0], feature_size)
# print(X_train_format.shape)
# normalize the input data
X_train_normal = X_train_format/255
X_test_normal = X_test_format/255

# format the labels
# y_train:[5 0 4 ... 5 6 8],是识别之后的结果，要把它转换成卷积层形式，其他概率为0，识别分类为1的形式
# example y_train为5 要转换成[0,0,0,0,0,1,0,0,0,0]
# print(y_train)
y_train_format = to_categorical(y_train)
y_test_format = to_categorical(y_test)


# set up the model
mlp = Sequential()
mlp.add(Dense(units=392, activation='sigmoid', input_dim=feature_size))
mlp.add(Dense(units=392, activation='sigmoid'))
mlp.add(Dense(units=10, activation='softmax'))
mlp.summary()
# compile the model
mlp.compile(loss='categorical_crossentropy', optimizer='adam')
# fit
mlp.fit(X_train_normal, y_train_format, epochs=10)
y_train_predict = mlp.predict_classes(X_train_normal)
accuracy_train = accuracy_score(y_train, y_train_predict)
y_test_predict = mlp.predict_classes(X_test_normal)
accuracy_test = accuracy_score(y_test, y_test_predict)
print('accuracy_train', accuracy_train, 'accuracy_test', accuracy_test)
