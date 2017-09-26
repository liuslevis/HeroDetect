#!/usr/local/bin/python3

from detector import HeroDetect
import os

if __name__ == '__main__':                
    heroDetect = HeroDetect(input_shape=(50, 50, 3))
    heroDetect.load_model(
        model_path='./model/cnn_vgg.model.h5', 
        label_path='./model/cnn_vgg.label.txt')

    video_path = './data/raw_test/bailishouyue/q0532r8l8bq.p712.1.mp4'
    print(heroDetect.predict_video(video_path))
