'''
使用VGG16提取特征，根据提取的特征建立模型，实现猫狗识别，
mlp一个隐藏层 10个神经元
'''
# from tensorflow.keras.preprocessing import img_to_array
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot as plt
import numpy as np
import os
current_directory = os.path.dirname(os.path.abspath(__file__))

# load the pic
img_path = current_directory+'/1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
print('type(img)', type(img), 'img.shape', img.shape)
# visualize the pic
# fig0 = plt.figure(figsize=(5, 5))
# plt.imshow(img)
# plt.show()

model_vgg = VGG16(weights='imagenet', include_top=False)
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
print('type(x)', type(x), 'x.shape', x.shape)

# get feature from model_vgg
x_feature = model_vgg.predict(x)
# x_feature.shape (1, 7, 7, 512)
print('x_feature', x_feature, 'x_feature.shape', x_feature.shape)
# do flatten
x_feature = x_feature.reshape(1, 7*7*512)


# 批量处理 图片 TODO:UNSKILLED
# load image and preprocess it with vgg16 structure
# --by flare

model_vgg = VGG16(weights='imagenet', include_top=False)
# define a method to load and preprocess the image


def modelProcess(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    x_vgg = model.predict(x)
    x_vgg = x_vgg.reshape(1, 25088)
    return x_vgg


# list file names of the training datasets
folder = current_directory+"/dataset/data_vgg/cats"
dirs = os.listdir(folder)
# generate path for the images
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == ".jpg":
        img_path.append(i)
img_path = [folder+"//"+i for i in img_path]

# preprocess multiple images
features1 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    feature_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features1[i] = feature_i

folder = current_directory+"/dataset/data_vgg/dogs"
dirs = os.listdir(folder)
img_path = []
for i in dirs:
    if os.path.splitext(i)[1] == ".jpg":
        img_path.append(i)
img_path = [folder+"//"+i for i in img_path]
features2 = np.zeros([len(img_path), 25088])
for i in range(len(img_path)):
    feature_i = modelProcess(img_path[i], model_vgg)
    print('preprocessed:', img_path[i])
    features2[i] = feature_i

# label the results
print(features1.shape, features2.shape)
y1 = np.zeros(300)
y2 = np.ones(300)

# generate the training data
X = np.concatenate((features1, features2), axis=0)
y = np.concatenate((y1, y2), axis=0)
y = y.reshape(-1, 1)
print(X.shape, y.shape)


# split train_test_data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50)
print('1111', X_train.shape, X_test.shape, X.shape)
# set up mlp model
model = Sequential()
model.add(Dense(units=10, activation='relu', input_dim=25088))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

# configure the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
# train the model
# TODO:第一次epoch的时候报错不知道为什么,应该是数组张量个数不符合输入要求
# KILLED：瞎改居然改好了，106行input_dim 写错了，写的25800改成25088 居然是写错了，，，，，
model.fit(X_train, np.array(y_train), epochs=50)
