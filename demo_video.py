#!/usr/local/bin/python3

from detector import HeroDetect

if __name__ == '__main__':
    heroDetect = HeroDetect(input_shape=(50, 50, 3))
    heroDetect.load_model(
        model_path='./data/output/v1.cnn_vgg.iter0.model.h5', 
        label_path='./data/output/v1.cnn_vgg.iter0.label.txt')

    video_path = 'data/raw_test/huamulan/y0382qw3lsj.p712.1.mp4' # './data/raw_test/ake/t0530k7xavm.p712.1.mp4'
    print(heroDetect.predict_video(video_path))
