import matplotlib.pyplot as plt
import numpy as np
import cv2

def showpic2(images,i):
    # translate into numpy array
    flatNumpyArray = np.array(images[0])
    # Convert the array to make a 200*200 grayscale image(灰度图像)
    grayImage = flatNumpyArray.reshape(200, 200)
    # show gray image
    cv2.imshow('GrayImage', grayImage)
    # print image's array
    print(grayImage)
    cv2.waitKey()

def showpic1(image):
    # translate into numpy array
    flatNumpyArray = np.array(image)
    # Convert the array to make a 200*200 grayscale image(灰度图像)
    grayImage = flatNumpyArray.reshape(200, 200)
    # show gray image
    cv2.imshow('GrayImage', grayImage)
    # print image's array
    print(grayImage)
    cv2.waitKey()