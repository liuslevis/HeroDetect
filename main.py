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
            for label in labels:
                f.write(label)
                f.write('\n')
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

    def prep_X(self, paths):
        n = len(paths)
        X = np.ndarray((n, *self.input_shape), dtype=np.uint8)
        for i, path in enumerate(paths):
            image = self.read_image(path)
            # image = util.crop_skill_1(image, self.image_size)
            X[i] = image #if image.shape == self.input_shape else image.T
            if i % 1000 == 0: print('Loading image {} of {}'.format(i, n))
        return X

    def train(self, ver, train_dir, valid_dir, model_init, epochs, batch_size, optimizer, data_arg):
        self.ver = ver
        self.model_path = '{}/{}.model.h5'.format(self.output_dir, self.ver)
        self.model_json_path = '{}/{}.model.json'.format(self.output_dir, self.ver)
        self.log_path = '{}/{}.detail.txt'.format(self.output_dir, self.ver)
        self.plot_path = '{}/{}.plot.png'.format(self.output_dir, self.ver)
        self.label_path = '{}/{}.label.txt'.format(self.output_dir, self.ver)
        # self.checkpoint_path = '{}/weights.epoch{epoch:02d}.val_loss{val_loss:.2f}.hdf5'.format(self.output_dir)

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.labels = self.create_labels(train_dir, self.label_path)
        n_labels = len(self.labels)
        paths = self.read_image_paths(train_dir)
        random.shuffle(paths)
        print(train_dir, len(paths))

        split_index = int(len(paths) * 0.7)
        train_paths = paths[:split_index]
        valid_paths = paths[split_index:]

        X_train = self.prep_X(train_paths)
        X_valid = self.prep_X(valid_paths)
        y_train = self.prep_y(train_paths, self.labels)
        y_valid = self.prep_y(valid_paths, self.labels)

        # print("Train X.shape:{} y.shape:{}".format(X_train.shape, y_train.shape))
        # print("Valid X.shape:{} y.shape:{}".format(X_valid.shape, y_valid.shape))

        self.model = model_init(self.input_shape, n_labels)

        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=optimizer,
            metrics=['accuracy'])

        self.model.summary()

        history = None
        if data_arg:
            # valid_datagen = ImageDataGenerator(rescale=1.)
            # valid_generator = valid_datagen.flow_from_directory(
            #     valid_dir,
            #     target_size=self.image_size,
            #     batch_size=batch_size,
            #     class_mode='categorical')
            train_datagen = ImageDataGenerator(
                rescale=1.,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=False)
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.image_size,
                batch_size=batch_size,
                class_mode='categorical')

            history = self.model.fit_generator(
                train_generator,
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                # validation_data=valid_generator,
                # validation_steps=len(X_valid) // batch_size,
                callbacks=[
                    EarlyStopping(monitor='loss', min_delta=0.1, patience=3, verbose=0, mode='auto'),
                    LossHistory(),
                ],
                )
        else:
            history = self.model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_valid, y_valid),
                callbacks=[
                    EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3, verbose=0, mode='auto'),
                    # ModelCheckpoint(self.checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1),
                    LossHistory(),
                ])

        self.model.save(self.model_path)

        #TODO
        if not data_arg:
            util.plot_keras_history(history, self.plot_path, self.log_path, self.model_json_path, self.model) 

    def predict(self, X):
        return self.model.predict(X)

    def print_test_result(self, test_dir, verbose=False):
        paths = self.read_image_paths(test_dir)
        X = self.prep_X(paths)
        y = self.predict(X)
        n = len(paths)
        hit = 0
        assert n > 0
        assert len(paths) == len(X) == len(y)
        for i in range(n):
            topN = sorted(zip(self.labels, y[i]), reverse=True, key=lambda x:x[1])[:1]
            for label, prob in topN:
                if label in paths[i]:
                    hit += 1
            if verbose:
                print('/'.join(paths[i].split('/')[-2:]), 'predict', topN)

        acc = hit / n
        print('acc: {} @ {}'.format(acc, test_dir))

input_size = (50, 50)
input_shape = (*input_size, 3)
test_dir = './data/input/test_tiny'
train_dir = './data/input/train_tiny'
valid_dir='./data/input/valid_tiny'

epochs = 2
batch_size = 50

def train():
    for model_init in [\
        model.cnn_6_layer,
        # model.cnn_10_layer, 
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
                valid_dir=train_dir,
                model_init=model_init, 
                epochs=epochs, 
                batch_size=batch_size,
                optimizer=keras.optimizers.Adadelta(lr=1e-1),
                data_arg=True,
                )

def test():
    heroDetect = HeroDetect(input_shape=input_shape)
    heroDetect.load_model(
        model_path='./data/output/v1.cnn_vgg_dropout.iter0.model.h5', 
        label_path='./data/output/v1.cnn_vgg_dropout.iter0.label.txt')

    heroDetect.print_test_result(test_dir, verbose=False)

if __name__ == '__main__':
    train()
    test()
