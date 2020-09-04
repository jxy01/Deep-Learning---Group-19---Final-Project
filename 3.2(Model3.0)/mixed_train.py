# python mixed_training.py
# import the necessary packages
from pyimagesearch import data_process
from pyimagesearch import model_build
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.layers.merge import concatenate
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import argparse
import locale
import os

#define the path of the dataset
inputPath1= "E:\\mit_online\\homework\\final\\3.2\\code\\metadata2.csv"
#define the path of the images
inputPath2 = "E:\\mit_online\\homework\\final\\3.2\\code\\data_50"
df = data_process.load_patient_attributes(inputPath1)

# load the  images and then scale the pixel intensities to the range [0, 1]
print("[INFO] loading images...")
images = data_process.load_patient_images(df,inputPath2)

images = images /255.0
print(images.shape)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing

print("[INFO] processing data...")
df = df.drop(["patientid"],axis=1)
df = df.drop(["filename"],axis=1)
df=df[1:]
df['condition'] = pd.to_numeric(df['condition'])
# transform "condition"  into one hot code
conditionBinarizer = LabelBinarizer().fit(df["condition"])
df['condition'] = conditionBinarizer.transform(df['condition'])

#split the data into two parts
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

trainY = trainAttrX["condition"]
testY = testAttrX["condition"]


# process the  attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
(trainAttrX, testAttrX) = data_process.process_patient_attributes(df,
	trainAttrX, testAttrX)
'''
print(trainAttrX)
print(type(trainAttrX))
print(trainAttrX.shape)
print("*************************************")
'''

# create the MLP and CNN models
mlp = model_build.create_mlp(trainAttrX.shape[1], regress=False)
cnn = model_build.create_cnn(64, 64, 3, regress=False)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(3, activation="softmax")(x)


# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

#define the learning rate and the decay rate
opt = Adam(lr=1e-3, decay=1e-3 / 400)
model.compile(loss="sparse_categorical_crossentropy",metrics = ['accuracy'], optimizer=opt,)

# train the model
print("[INFO] training model...")
model.fit(
	[trainAttrX, trainImagesX], trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=50, batch_size=7)

# make predictions on the testing data
print("[INFO] predicting risks...")
test_loss, test_acc = model.evaluate([testAttrX, testImagesX], testY)
print("\n","the test loss is {}\n".format(test_loss),"the test accuracy is {}".format(test_acc))
# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
