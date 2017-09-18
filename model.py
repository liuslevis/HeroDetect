#!/usr/bin/python3

import keras
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, MaxPooling2D, Dense, Activation, Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

def cnn_6_layer(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_labels))
    model.add(Activation('sigmoid'))
    return model

def cnn_10_layer(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(n_labels))
    model.add(Activation('sigmoid'))
    return model

def cnn_13_layer(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(n_labels))
    model.add(Activation('sigmoid'))
    return model

def cnn_13_layer(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(256))
    model.add(Activation('relu'))
    
    model.add(Dense(n_labels))
    model.add(Activation('sigmoid'))
    return model

def cnn_13_layer_dropout(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(n_labels))
    model.add(Activation('sigmoid'))
    return model

def cnn_15_layer(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dense(n_labels))
    model.add(Activation('sigmoid'))
    return model

def cnn_vgg(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dense(n_labels))
    model.add(Activation('sigmoid'))
    return model

def cnn_vgg_dropout(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(4096))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(n_labels))
    model.add(Activation('sigmoid'))
    return model