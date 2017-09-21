#!/usr/bin/python3
import os
import random
import numpy as np
import cv2
from sys import platform

import keras
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, MaxPooling2D, Dense, Activation, Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

        
class Util(object):
    def plot_keras_history(history, plot_path, log_path, model_json_path, model):
        log_path = log_path.replace('detail.txt', 'acc{0:.2f}.txt'.format(history.history['acc'][-1]))
        with open(log_path, 'w') as f:
            for key in ['val_acc', 'acc', 'val_loss', 'loss']:
                try:
                    f.write('\n{}='.format(key))
                    f.write(str(history.history[key]))
                except Exception as e:
                    pass
        if platform == "linux":
            return
        import matplotlib.pyplot as plt
        # TODO make it avalable on linux
        # summarize history for accuracy
        fig = plt.figure()
        fig.add_subplot(2,2,1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        # summarize history for loss
        fig.add_subplot(2,2,2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(plot_path)

    def plot_image(image):
        if platform == "linux":
            return
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.show()

    def rect_skill_1(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 385 / 480)
        y2 = int(w * 455 / 480)
        x1 = int(h * 600 / 848)
        x2 = int(h * 670 / 848)
        return x1, y1, x2, y2

    def crop_skill_1(image, size):
        x1, y1, x2, y2 = Util.rect_skill_1(image)
        image = image[y1:y2, x1:x2]
        return cv2.resize(image, size)


class Model(object):
    def cnn_5_layer(input_shape, n_labels):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, activation='relu', border_mode='same', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, kernel_size=3, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, kernel_size=3, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(n_labels))
        model.add(Activation('sigmoid'))
        return model

    def cnn_7_layer(input_shape, n_labels):
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
            X[i] = image
            if i % 1000 == 0: print('Loading image {} of {}'.format(i, n))
        return X

    def train(self, ver, train_dir, model_init, epochs, batch_size, optimizer):
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

        Util.plot_keras_history(history, self.plot_path, self.log_path, self.model_json_path, self.model) 
        print('Train success:{}'.format(ver))

    def predict(self, X):
        return self.model.predict(X)

    def predict_image(self, image):
        y = self.model.predict(np.array([image]))[0]
        return sorted(zip(self.labels, y), reverse=True, key=lambda x:x[1])[0]

    def predict_frame(self, frame):
        return self.predict_image(Util.crop_skill_1(frame, self.image_size))

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_sec = int(frame_count / fps)
        count = 0
        key = None
        predicts = []
        while True and key != 120:
            ret, frame = cap.read()
            if not ret:
                break
            # predict ervery 60 second
            count += 1
            if count % 60 * fps == 0:
                predict = self.predict_frame(frame)
                predicts.append(predict)
        cap.release()
        def reduce_predicts(predicts):
            d = {} # {label:prob}
            prob_sum = 0
            for label, prob in predicts:
                if prob > 0.1:
                    prob = 1
                    d.setdefault(label, 0)
                    d[label] += prob
                    prob_sum += prob
            return [(k, d[k] / prob_sum) for k in sorted(d, key=d.get, reverse=True)]
        return reduce_predicts(predicts) # first label
        
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

def train():
    for model_init in [\
        Model.cnn_5_layer,
        Model.cnn_7_layer,
        Model.cnn_10_layer, 
        # Model.cnn_13_layer, 
        Model.cnn_13_layer_dropout, 
        Model.cnn_15_layer,
        # Model.cnn_vgg,
        Model.cnn_vgg_dropout,
        ]:
        for i in range(1):
            heroDetect = HeroDetect(input_shape=(50, 50, 3))
            heroDetect.train(
                ver='v2.{}.iter{}'.format(model_init.__name__, i), 
                train_dir='./data/input/train',
                model_init=model_init, 
                epochs=100, 
                batch_size=250,
                optimizer=keras.optimizers.Adadelta(lr=1e-1),
                )

def test():
    heroDetect = HeroDetect(input_shape=(50, 50, 3))
    heroDetect.load_model(
        model_path='./data/output/v1.cnn_vgg.iter0.model.h5', 
        label_path='./data/output/v1.cnn_vgg.iter0.label.txt')
    # test_dir = './data/input/test_small'
    # heroDetect.print_test_result(test_dir, verbose=True)

if __name__ == '__main__':
    train()
    # test()
