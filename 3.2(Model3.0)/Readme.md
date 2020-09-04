#Readme

##1 Introduction

This project builds a complex input model(including numeric, catigorical data and images) based on TensorFlow keras.By using cnn and mlp, it can predict  the risk and the severity of a COVID-19 patient based on his X-rays offset days age and location. The accuracy rate reached 0.69

------------


The model is trained by COVID-19 Chest X-Ray dataset from Cohen J.P., Morrison, P., and Dao L.

##2 Run with Virtual Environment

You'll need to find Project Interpreter in Preferences of PyCharm to create a new Virtual Environment.

The code can run normally under the environment of python3.7

This is a brand new virtual environment and will not be affected by other environments in your computer.

###Install all the packages

`pip install -r requirements.txt`

###Run the program

run python file named **"mixed_train.py"**

After running this file, the computer will start training the model and then make predictions on the test set. In addition, the computer will print out the performance of training process and predicting process, including loss and accuracy.

#3 Other files

- metadata2.csv : the dataset to be trained, including the path of the images , some numeric and catigorical data.

- data_50 : the images from the metadata2.csv which had already been renamed by number.

- pyimagesearch : containing two important py files **data_process.py** and **model_build.py**, which will be imported by the main function named **"mixed_train.py"**


