import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tf.keras.models import Sequential
from tf.keras.optimizers import Adam
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout,Dense,ReLU,Softmax

# data
data = pd.read_csv('D:/Final Year Project/Express-U/data/preprocessed/canny_labels.csv')

# splitting data

X_Train,X_Test = train_test_split(data,train_size=0.8,random_state=42,shuffle=True)

print(X_Train.shape)
print(X_Test.shape)

# Image Data Generator

Train_image_generator = ImageDataGenerator(
    rescale = 1/255,
    zoom_range= 0.5,
    brightness_range= [0.6,1.0],
    rotation_range= 20,
    vertical_flip=False,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.1
)
Test_image_generator = ImageDataGenerator(rescale=1./255)

Train_Set = Train_image_generator.flow_from_dataframe(dataframe=X_Train,
                                                   x_col="file_paths",
                                                   y_col="labels",
                                                   batch_size=32,
                                                   class_mode="categorical",
                                                   color_mode="grayscale",
                                                   subset="training")

Test_Set = Test_image_generator.flow_from_dataframe(dataframe=X_Test,
                                                   x_col="file_paths",
                                                   y_col="labels",
                                                   batch_size=32,
                                                   class_mode="categorical",
                                                   color_mode="grayscale",
                                                   shuffle=False)

# Model 

inputshape = (256,256,1)
outputshape = 35

model = Sequential()
model.add(Conv2D(input_shape=inputshape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(outputshape,activation="softmax"))

model.summary()

# model compile

lr=0.001
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])


