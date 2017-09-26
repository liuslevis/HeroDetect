#!/usr/local/bin/python3

from detector import HeroDetect
import cv2

if __name__ == '__main__':
    heroDetect = HeroDetect(input_shape=(50, 50, 3))
    heroDetect.load_model(
        model_path='./model/cnn_vgg.model.h5', 
        label_path='./model/cnn_vgg.label.txt')

    frame = cv2.imread('./data/raw_test/bailishouyue/frame.png')
    print(heroDetect.predict_frame(frame))