import tensorflow as tf
#import introtodeeplearning as mdl
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import os
import cv2
import ShowPic

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU
#assert len(tf.config.experimental.list_physical_devices('GPU')) > 0

train_set = []

for filename in os.listdir("Y-PA-train"):
    img1 = cv2.imread("Y-PA-train\\"+filename,cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        continue
    img2 = cv2.resize(img1,(200,200))
    data = np.asarray(img2)  # 转换为矩阵
    train_set.append(data[np.newaxis,:])

for filename in os.listdir("N-PA-train"):
    img1 = cv2.imread("N-PA-train\\"+filename,cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        continue
    img2 = cv2.resize(img1,(200,200))
    data = np.asarray(img2)  # 转换为矩阵
    train_set.append(data[np.newaxis,:])

train_images = np.concatenate(train_set,axis = 0)

train_labelsY = np.ones(150)
train_labelsN = np.zeros(100)
train_labels = np.append(train_labelsY,train_labelsN)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
'''
cl1 = LungCut.lungmask(train_images[0])
ShowPic.showpic1(cl1)
'''
for i in range(250):
    train_images[i] = clahe.apply(train_images[i])#Histogram equalization

'''TODO:Scrambled train set'''
for i in range(125):
    if i%2 == 0:
        train_images[i],train_images[249-i] = train_images[249-i],train_images[i]
        train_labels[i],train_labels[249-i] = train_labels[249-i],train_labels[i]

test_set=[]

for filename in os.listdir("Y-PA-test"):
    img1 = cv2.imread("Y-PA-test\\"+filename,cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        continue
    img2 = cv2.resize(img1,(200,200))
    data = np.asarray(img2)  # 转换为矩阵
    test_set.append(data[np.newaxis,:])

for filename in os.listdir("N-PA-test"):
    img1 = cv2.imread("N-PA-test\\"+filename,cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        continue
    img2 = cv2.resize(img1,(200,200))
    data = np.asarray(img2)  # 转换为矩阵
    test_set.append(data[np.newaxis,:])

test_images = np.concatenate(test_set,axis = 0)

test_labels_Y = np.ones(41)
test_labels_N = np.zeros(35)
test_labels = np.append(test_labels_Y,test_labels_N)

for i in range(76):
    #test_images[i] = cv2.equalizeHist(test_images[i])
    test_images[i] = clahe.apply(test_images[i])

train_images = (np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)
test_labels = (test_labels).astype(np.int64)


def build_cnn_model():
    cnn_model = tf.keras.Sequential([

        # TODO: Define the first convolutional layer
        tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu),

        # TODO: Define the first max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # TODO: Define the second convolutional layer
        tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),

        # TODO: Define the second max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),

        # TODO: Define the last Dense layer to output the classification
        # probabilities. Pay attention to the activation needed a probability
        # output
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
        # [TODO Dense layer to output classification probabilities]
    ])

    return cnn_model


#cnn_model = xrv.models.DenseNet(num_classes=2, in_channels=3).cuda()  # DenseNet 模型，二分类
cnn_model = build_cnn_model()
# Initialize the model by passing some data through
cnn_model.predict(train_images[[0]])
# Print the summary of the layers in the model.
print(cnn_model.summary())

'''TODO: Define the compile operation with your optimizer and learning rate of choice'''
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''TODO: Use model.fit to train the CNN model, with the same batch_size and number of epochs previously used.'''
cnn_model.fit(train_images, train_labels, batch_size=4, epochs=5, shuffle=True,validation_split=0.8)

'''TODO: Use the evaluate method to test the model!'''
test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)