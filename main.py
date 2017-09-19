#!/usr/bin/python3
import os
import random
import numpy as np
import cv2

import keras
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, MaxPooling2D, Dense, Activation, Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

import util
import model

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class HeroDetect(object):
    """docstring for HeroDetect"""
    def __init__(self, input_shape):
        super(HeroDetect, self).__init__()
        self.output_dir = './data/output'

        self.input_shape = (self.image_width, self.image_height, self.image_channels) = input_shape
        self.image_size = (self.image_width, self.image_height)
        
    def load_model(self, model_path, label_path):
        self.model_path = model_path
        self.label_path = label_path
        self.labels = self.read_labels(label_path)
        self.model = keras.models.load_model(model_path)

    def read_labels(self, label_path):
        with open(label_path) as f:
            return list(map(lambda x:x.rstrip(), f.readlines()))
    
    def create_labels(self, train_dir, label_path):
        labels = [i for i in os.listdir(train_dir) if os.path.isdir('{}/{}'.format(train_dir, i))]
        print('label =', labels)
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        return labels

    def read_image(self, path):
        image = cv2.imread(path)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
        return image

    def read_image_paths(self, directory):
        paths = []
        for base, folders, files in os.walk(directory):
            for file in files:
                if 'jpg' in file or 'png' in file:
                    paths.append('{}/{}'.format(base, file))
        return paths

    def prep_y(self, paths, labels):
        n = len(paths)
        y = np.ndarray((n, 1))
        for i, path in enumerate(paths):
            hero = path.split('/')[-2]
            y[i] = labels.index(hero)
        return to_categorical(y)

    def prep_X(self, paths, need_crop):
        n = len(paths)
        X = np.ndarray((n, *self.input_shape), dtype=np.uint8)
        for i, path in enumerate(paths):
            image = self.read_image(path)
            if need_crop:
                image = util.crop_skill_1(image, self.image_size)
            X[i] = image if image.shape == self.input_shape else image.T
            if i % 1000 == 0: print('Loading image {} of {}'.format(i, n))
        return X

    def train(self, ver, train_dir, model_init, epochs, batch_size):
        self.ver = ver
        self.model_path = '{}/{}.model.h5'.format(self.output_dir, self.ver)
        self.model_json_path = '{}/{}.model.json'.format(self.output_dir, self.ver)
        self.log_path = '{}/{}.detail.txt'.format(self.output_dir, self.ver)
        self.plot_path = '{}/{}.plot.png'.format(self.output_dir, self.ver)
        self.label_path = '{}/{}.label.txt'.format(self.output_dir, self.ver)
        # self.checkpoint_path = '{}/weights.epoch{epoch:02d}.val_loss{val_loss:.2f}.hdf5'.format(self.output_dir)

        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = keras.optimizers.Adadelta(lr=1e-1) # RMSprop()
        self.loss = keras.losses.categorical_crossentropy

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        labels = self.create_labels(train_dir, self.label_path)
        n_labels = len(labels)
        paths = self.read_image_paths(train_dir)
        random.shuffle(paths)
        print(train_dir, len(paths))

        split_index = int(len(paths) * 0.7)
        train_paths = paths[:split_index]
        valid_paths = paths[split_index:]

        X_train = self.prep_X(train_paths, need_crop=False)
        X_valid = self.prep_X(valid_paths, need_crop=False)
        y_train = self.prep_y(train_paths, labels)
        y_valid = self.prep_y(valid_paths, labels)

        print("Train X.shape:{} y.shape:{}".format(X_train.shape, y_train.shape))
        print("Valid X.shape:{} y.shape:{}".format(X_valid.shape, y_valid.shape))

        self.model = model_init(self.input_shape, n_labels)

        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

        self.model.summary()

        history = self.model.fit(X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(X_valid, y_valid),
            callbacks=[
                EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3, verbose=0, mode='auto'),
                # ModelCheckpoint(self.checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1),
                LossHistory(),
            ])

        self.model.save(self.model_path)

        util.plot_keras_history(history, self.plot_path, self.log_path, self.model_json_path, self.model) 
        # score = model.evaluate(X_valid, y_valid, verbose=0)
        # print('valid loss:', score[0])
        # print('valid accuracy:', score[1])

    def predict(self, X):
        return self.model.predict(X)

    def print_result(self, X, y, directory):
        paths = self.read_image_paths(directory)
        assert len(paths) == len(X) == len(y)
        n = len(paths)
        for i in range(n):
            topN = sorted(zip(self.labels, y[i]), key=lambda x:x[1])[:3]
            if topN[0][1] > 1e-4:
                print(paths[i], topN)

input_size = (50, 50)
input_shape = (*input_size, 3)
test_dir = './data/input/test_small'
train_dir = './data/input/train_small'
epochs = 100
batch_size = 250

def train():
    for model_init in [\
        model.cnn_6_layer,
        model.cnn_10_layer, 
        # model.cnn_13_layer, 
        # model.cnn_13_layer_dropout, 
        # model.cnn_15_layer,
        # model.cnn_vgg,
        # model.cnn_vgg_dropout,
        ]:
        for i in range(1):
            heroDetect = HeroDetect(input_shape=input_shape)
            heroDetect.train(
                ver='v2.iter{}.{}'.format(i, model_init.__name__), 
                train_dir=train_dir,
                model_init=model_init, 
                epochs=epochs, 
                batch_size=batch_size)

def test():
    heroDetect = HeroDetect(input_shape=input_shape)
    heroDetect.load_model(
        model_path='./data/output/v2.iter0.cnn_6_layer.model.h5', 
        label_path='./data/output/v2.iter0.cnn_6_layer.label.txt')

    paths = heroDetect.read_image_paths(test_dir)
    X = heroDetect.prep_X(paths, need_crop=False)
    y = heroDetect.predict(X)
    heroDetect.print_result(X, y, test_dir)

if __name__ == '__main__':
    # train()
    test()
