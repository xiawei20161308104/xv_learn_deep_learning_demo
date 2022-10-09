'''
基于flare文本数据，建立LSTM模型（单层LSTM，输出20个神经元，使用前20个预测21个），预测序列文字
1.将文本数据预处理
2.查看预处理数据的数据结构，数据分离
3.预测
'''
from statistics import mode
from telnetlib import SE
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
current_director = os.path.dirname(os.path.abspath(__file__))

# load data and remove newline character
data = open(current_director+'/flare').read()
data = data.replace('\n', '').replace('\r', '')
# print(data)

# letter remove duplication
# set中的每一个元素不能出现多次，并且是无序存储的。
letters = list(set(data))
num_letters = len(letters)
print(letters)

'''
for:for … in，for … in range()，for … in enumerate()
----------
example:for … in
a = [1, 3, 4, 5]
for i in a:
	print(a)
1
3
4
5
----------
example:for … in range(a,b,c)
a,b左右区间阈值，左闭右开，c为步长，默认a，c为0，1，只设置b
for i in range(1, 5, 2)
    print(i)
1
3
example2:
a = [1, 3, 4, 5]
for i in range(len(a)):  #利用列表a的长度设置遍历次数
	print(a[i])
1
3
4
5
----------
example:for i,b in enumerate(a) 
同时对 i，b两个变量同时赋值，i 赋值为b当前元素索引，b为a当前的元素。
a = [1, 3, 4, 5]
for i, b in enumerate(a): 
	print("i = ", i)   #i 赋值为a当前元素的下标
	print("b = ", b)   #b赋值为a当前的元素
i = 0
b = 1
i = 1
b = 3
i = 2
b = 4
i = 3
b = 5
'''
# 根据文本内包含的字母建立字典，有字典之后才可以预处理。
int_to_char = {a: b for a, b in enumerate(letters)}
# {0: 'l', 1: 'r', 2: 'a', 3: 'H', 4: 'h', 5: 'm', 6: 'f', 7: 'A', 8: 't', 9: 's', 10: 'b', 11: '.',
# 12: 'd', 13: 'c', 14: 'u', 15: 'y', 16: 'S', 17: 'i', 18: 'e', 19: 'p', 20: ' ', 21: 'o', 22: 'n'}
# print(int_to_char)
char_to_int = {b: a for a, b in enumerate(letters)}
time_step = 20


# 批量字符预处理
# 滑动窗口提取数据
def extract_data(data, slide):
    x = []
    y = []
    for i in range(len(data) - slide):
        x.append([a for a in data[i:i+slide]])
        y.append(data[i+slide])
    return x, y

# 字符到数字的批量转化


def char_to_int_Data(x, y, char_to_int):
    x_to_int = []
    y_to_int = []
    for i in range(len(x)):
        x_to_int.append([char_to_int[char] for char in x[i]])
        y_to_int.append([char_to_int[char] for char in y[i]])
    return x_to_int, y_to_int

# 实现输入字符文章的批量处理，输入整个字符、滑动窗口大小、转化字典


def data_preprocessing(data, slide, num_letters, char_to_int):
    char_Data = extract_data(data, slide)
    int_Data = char_to_int_Data(char_Data[0], char_Data[1], char_to_int)
    Input = int_Data[0]
    Output = list(np.array(int_Data[1]).flatten())
    Input_RESHAPED = np.array(Input).reshape(len(Input), slide)
    new = np.random.randint(
        0, 10, size=[Input_RESHAPED.shape[0], Input_RESHAPED.shape[1], num_letters])
    for i in range(Input_RESHAPED.shape[0]):
        for j in range(Input_RESHAPED.shape[1]):
            new[i, j, :] = to_categorical(
                Input_RESHAPED[i, j], num_classes=num_letters)
    return new, Output


# extract X and y
X, y = data_preprocessing(data, time_step, num_letters, char_to_int)

# 处理之后X是这样的
# (56148,20,23) 56148个字母，20个预测数据，23行one—hot编码（取决于事先计算好的字典）
'''
flare is a teacher in ai industry. He obtained his phd in Australia.
['u', 'm', 'f', 'b', 'l', 't', 'i', 'S', 'n', 'e', 'A', 'a', 'p', 'h', 'y', 's', 'c', 'd', ' ', 'H', 'r', 'o', '.']
[[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] --------f
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] --------l
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0] ...
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
'''
# print(X[0, :, :])
# 56148
# print(len(y))
# 13 4 9
# print(y[1], y[2], y[3])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=10)

# 将y_train转换成one-hot格式
y_train_category = to_categorical(y_train, num_letters)
model = Sequential()
model.add(LSTM(units=20, input_shape=(
    X_train.shape[1], X_train.shape[2]), activation='relu'))
model.add(Dense(units=num_letters, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train_category, batch_size=1000, epochs=5)
y_train_predict = model.predict_classes(X_train)
# transform the int to letters
y_train_predict_char = [int_to_char[i] for i in y_train_predict]
# print(y_train_predict_char)
