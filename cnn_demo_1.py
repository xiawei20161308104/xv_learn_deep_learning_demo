'''
基于dataset，training_set数据 建立cnn结构，识别猫狗计算准确率
预测新图片准确率
'''
# 全连接神经网络模型(DNN),也称深度神经网络，与传统的感知机不同，每个结点和下一层所有结点都有运算关系，这就是名称中‘全连接’的含义，
# 全连接神经网络通常有多个隐藏层，增加隐藏层可以更好分离数据的特征，但过多的隐藏层也会增加训练时间以及产生过拟合。
# 如果用全连接神经网络处理大尺寸图像具有三个明显的缺点：
# 1）首先将图像展开为向量会丢失空间信息；（2）其次参数过多效率低下，训练困难；（3）同时大量的参数也很快会导致网络过拟合。

# 卷积神经网络的各层中的神经元是3维排列的:宽度、高度和深度。在卷积神经网络中的深度指的是 激活数据体 的第三个维度
# 卷积神经网络主要由这几类层构成：输入层、卷积层，(包含计算量)ReLU层、池化（Pooling）层和全连接层


import numpy as np
import tensorflow as tf
# 建立一个序列
from pyexpat import model
from tensorflow.keras.models import Sequential
# 建立各种层，分别是：卷积层图像是2D，池化层，展开层，全连接层
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

print(tf.__version__)
# 一、load data
# 设置255归一化
train_datagen = ImageDataGenerator(rescale=1./255)
# class_mode: "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式,
#  "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签."sparse"返回1D的整数标签,如果为None则不返回任何标签
#  target_size是cnn典型模型 的输入形状。batch_size是每次处理2张图片。
training_set = train_datagen.flow_from_directory(
    './orl_faces', target_size=(50, 50), batch_size=2, class_mode='categorical')
print('load training_set over')
print('类别', training_set.class_indices)
# 二、create cnn model（基于VGG-16模型）
model = Sequential()
# 一层卷积一层池化，然后在重复一遍
model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
# 展开层，将数据展开为一列
model.add(Flatten())
# 全连接层 设置128个神经元 激活函数不变
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
print('create model over')

# 三、configure the model
# 配置优化器，损失函数：分类交叉熵函数，指标评估参数
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
print('configure model over')

# 四、train the model
model.fit_generator(training_set, epochs=25)
# model.fit(training_set, epochs=25)

# 五、the model accuracy
accurary_train = model.evaluate_generator(training_set)
print('accurary_train', accurary_train)

test_set = train_datagen.flow_from_directory(
    './test_orl_faces', target_size=(50, 50), batch_size=2, class_mode='categorical')
accurary_test = model.evaluate_generator(test_set)
print('accurary_test', accurary_test)

# 六、预测新图片
# TODO:不会画预测结果，用哪个函数预测class_mode='categorical'的分类结果啊
# new_pic = load_img('./orl_faces/s1/3.jpg', target_size=(50, 50))
# new_pic = load_img('./orl_faces/s2/3.jpg', target_size=(50, 50))
new_pic = load_img('./orl_faces/s3/3.jpg', target_size=(50, 50))
new_pic = img_to_array(new_pic)/255
new_pic = new_pic.reshape(1, 50, 50, 3)
result = model.predict(new_pic, verbose=1)
print(np.argmax(result, axis=1))
