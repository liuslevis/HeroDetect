#!/usr/bin/python3

import os
import cv2
from detector import Util
# from skvideo.io import VideoCapture
from cv2 import VideoCapture

BEGIN_SEC = 60
END_SEC = 60

def gen_train(video_path, train_dir, crop_func):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    video_name = video_path.split('/')[-1]

    cap = VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = int(frame_count / fps)

    print('fps:{} size:{}*{}'.format(fps, frame_width, frame_height))
    count = 0
    while True:
        count +=1
        ret, frame = cap.read()
        if frame is None:
            break

        crop_image = crop_func(frame)
        if count % fps == 0:
            n_sec = int(count / fps)
            if BEGIN_SEC < n_sec < total_sec - END_SEC:
                train_path = train_dir + '/{}_{}_sec.jpg'.format(video_name, n_sec)
                cv2.imwrite(train_path, crop_image)
                print('    ', train_path)

        # if DEBUG and count > 1000:
            # cv2.imshow('frame', crop_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # break

def main():
    input_dir = './data/raw_new'
    output_dir = './data/input/train_skill_1/'
    crop_func = Util.crop_skill_1
    # crop_func = Util.crop_middle_hero
    for root_dir, sub_dirs, files in os.walk(input_dir):
        for file in files:
            if '.mp4' in file:
                video_path = root_dir + '/' + file
                train_dir = output_dir + os.path.basename(root_dir)
                print('processing', video_path, '->', train_dir)
                gen_train(video_path, train_dir, crop_func)

if __name__ == '__main__':
    main()
    