# Readme

## 1 Introduction

This project builds a CNN model based on TensorFlow. It can predict whether the owner has COVID-19 based on X-rays. The accuracy rate can reach 0.6. The model is trained by COVID-19 Chest X-Ray dataset from Cohen J.P., Morrison, P., and Dao L.

## 2 Run with Virtual Environment

You'll need to find Project Interpreter in Preferences of PyCharm to create a new Virtual Environment.

The code can run normally under the environment of python3.7

> This is a brand new virtual environment and will not be affected by other environments in your computer.

### Install all the packages

```shell
pip install -r requirements.txt
```

#### Run

Please run the file "model1.0.py". 

After running this file, the computer will start training the model and then make predictions on the test set. In addition, the computer will print out the performance of training process and predicting process, including loss and accuracy.

