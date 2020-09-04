# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_patient_attributes(inputPath):
	# initialize the list of column names in the CSV file and then初始化纵坐标
	# load it using Pandas
	cols = ["patientid", "offset", "age", "location","filename","condition"]
#	cols = ["age", "gender", "death", "icu", "location"]
# the csv file had already been through data-cleaning
	df = pd.read_csv(inputPath, header=None, names=cols)
	return df
	# return the data frame


def process_patient_attributes(df, train, test):
	# initialize the column names of the continuous data
	continuous = ["offset", "age"]

	# performing min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])

	# one-hot encode the location categorical data (by definition of
	# one-hot encoing, all output features are now in the range [0, 1])
	locationBinarizer = LabelBinarizer().fit(df["location"])
	trainCategorical = locationBinarizer.transform(train["location"])
	testCategorical = locationBinarizer.transform(test["location"])


	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	trainX = np.hstack([trainCategorical, trainContinuous])
	testX = np.hstack([testCategorical, testContinuous])

	# return the concatenated training and testing data
	return (trainX, testX)
def load_patient_images(df, inputPath):
	# initialize our images array (i.e., the house images themselves)
	images = []
	# loop over the indexes of the data
	for i in df.index.values:
		# find the four images for the house and sort the file paths,
		# ensuring the four are always in the *same order*
		if i ==0:
			continue
		basePath = os.path.sep.join([inputPath, "{}.jpg".format(i-1)])
		outputImage = cv2.imread(basePath)
		outputImage = cv2.resize(outputImage,(64,64))
 	#	print(type(outputImage))
		images.append(outputImage)
	# return our set of images
	return np.array(images)


#test
df =  load_patient_attributes("E:\\mit_online\\homework\\final\\3_2\\code\\metadata2.csv")
inputPath = "E:\\mit_online\\homework\\final\\3_2\\code\\data_50"
img = load_patient_images(df, inputPath)
print(type(img))
#img = np.array(img,dtype="uint8")
