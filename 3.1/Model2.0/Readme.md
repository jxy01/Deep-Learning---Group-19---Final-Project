# Readme

## 1 Introduction

The project can predict whether the owner has the COVID-19 based on X-ray image. Considering the small dataset and equipment limitations, we believe that using pre-trained models can help us get better results.

The project is based on Pytorch and adopts DenseNet121model in torchxrayvision  which can predict the X-ray image of 18 diseases. We used the COVID-19 Chest X-Ray dataset to further train the model, and finally got an accuracy of 0.8.

## 2 Run with Virtual Environment

You'll need to find Project Interpreter in Preferences of PyCharm to create a new Virtual Environment.

The code can run normally under the environment of python3.7

> This is a brand new virtual environment and will not be affected by other environments in your computer.

### Install all the packages

```shell
pip install -r requirements.txt
```

#### Run

The pre-trained model is stored in backup/DenseNet.pt.

You can run the "predict.py" file directly, and the computer will make predictions on the test set and print out the specific performance of the model (including accuracy and loss)

In addition, you can also retrain the network by running the "train.py" file. If you want to retrain the network with a new data set, you need to organize your training set according to the file format in the data directory

