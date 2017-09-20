#!/usr/local/bin/python3

from detector import HeroDetect
import os

if __name__ == '__main__':                
    heroDetect = HeroDetect(input_shape=(50, 50, 3))
    heroDetect.load_model(
        model_path='./data/output/v1.cnn_vgg.iter0.model.h5', 
        label_path='./data/output/v1.cnn_vgg.iter0.label.txt')

    for base, folders, files in os.walk('data/raw_test'):
        for file in files:
            if 'mp4' in file:
                video_path = '{}/{}'.format(base, file)
                print(video_path, heroDetect.predict_video(video_path))
