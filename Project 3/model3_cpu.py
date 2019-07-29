import os

import datetime
now = datetime.datetime.now()
unique = str(now.hour)+ str(now.minute)
import pickle
import pandas as pd
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from subprocess import check_output
import keras
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout, BatchNormalization, Activation, Dropout, MaxPool2D, Input
from keras.models import Model, Input, Sequential, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from math import ceil

def Unit(x, filters, pool=False):
    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2, 2))(x)
        res = Conv2D(filters=filters, kernel_size=[3, 3], strides=(2,
                     2), padding='same')(res)
    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1],
                 padding='same')(out)

    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1],
                 padding='same')(out)

    out = keras.layers.add([res, out])

    return out

def MiniModel(input_shape):
   images = Input(input_shape)
   net = Conv2D(filters=32, kernel_size=[5, 5], strides=[2, 2], padding="same")(images)
   net = Unit(images, 32)
   net = Unit(net, 32)
   net = Unit(net, 32)
   net = Unit(net, 32)
   net = Unit(net, 32)

   net = Unit(net, 64, pool=True)
   net = Unit(net, 64)
   net = Unit(net, 64)
   net = Unit(net, 64)
   net = Unit(net, 64)

   net = Unit(net, 128, pool=True)
   net = Unit(net, 128)
   net = Unit(net, 128)
   net = Unit(net, 128)
   net = Unit(net, 128)

   net = Unit(net, 256, pool=True)
   net = Unit(net, 256)
   net = Unit(net, 256)
   net = Unit(net, 256)
   net = Unit(net, 256)

   net = BatchNormalization()(net)
   net = Activation('relu')(net)
   net = Dropout(0.25)(net)

   net = AveragePooling2D(pool_size=(4, 4))(net)
   net = Flatten()(net)
   #net = Dense(units=128, activation='relu')(net)
   #net = Dense(units=64, activation='relu')(net)
   net = Dense(units=10, activation='softmax')(net)

   model = Model(inputs=images, outputs=net)

   return model

def threshold(dataSet):
    #for image in dataSet:
    #    image = (image > 230) * image
    return dataSet

with open('./train_images.pkl', 'rb') as f:
    data = pickle.load(f)

data=threshold(data)
num_classes=10
df = pd.read_csv('./train_labels.csv', header=0)
y = df['Category'].tolist()

(X_train, X_test, y_train, y_test) = train_test_split(data, y, test_size=0.10, random_state=1, shuffle=True, stratify=y)

input_shape = (X_train.shape[1], X_train.shape[2], 1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

X_train /= 255
X_test /= 255

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],X_train.shape[2], 1)

train_x = X_train / X_train.std(axis=0)
test_x = X_test / X_train.std(axis=0)

datagen = ImageDataGenerator(  # set input mean to 0 over the dataset
                               # set each sample mean to 0
                               # divide inputs by std of the dataset
                               # divide each input by its std
                               # apply ZCA whitening
                               # randomly rotate images in the range (degrees, 0 to 180)
                               # Randomly zoom image
                               # randomly shift images horizontally (fraction of total width)
                               # randomly shift images vertically (fraction of total height)
                               # randomly flip images
                               # randomly flip images
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=40,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
    )

datagen.fit(train_x)

bSize= 128
model = MiniModel(input_shape)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=250)
mc = ModelCheckpoint('Model'+str(unique)+'.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.summary()


model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit_generator(datagen.flow(train_x, y_train, batch_size=bSize), validation_data=[test_x, y_test],epochs=1000,steps_per_epoch=330, verbose=1, callbacks=[es,mc])

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('ResNN accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(unique+'_Acc.png')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ResNN loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(unique+'_Loss.png')

with open('./test_images.pkl', 'rb') as f:
    dataTest = pickle.load(f)
dataTest=threshold(dataTest)
dataTest /= 255
testSet = dataTest.reshape(dataTest.shape[0], dataTest.shape[1], dataTest.shape[2], 1)
testSet = testSet / X_train.std(axis=0)
#datagen.fit(testSet)

model = load_model('Model'+str(unique)+'.h5')
pred = model.predict(testSet)
pred = np.argmax(pred, axis = 1)
output = open("predictions_"+unique+".csv", "w")
output.write("Id,Category" + "\n")
for index,item in enumerate(pred):
    output.write(str(index) + "," + str(item) + "\n")
output.close()
print("Thanks for chosing us. Have a good day :)")

