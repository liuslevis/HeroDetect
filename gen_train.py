#!/usr/bin/python3

import os
import cv2
import util

TRAIN_IMG_SIZE = (50, 50)
BEGIN_SEC = 60
END_SEC = 60

def gen_train(video_path, train_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    video_name = video_path.split('/')[-1]

    cap = cv2.VideoCapture(video_path)
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

        skill_1_image = util.crop_skill_1(frame, TRAIN_IMG_SIZE)
        if count % fps == 0:
            n_sec = int(count / fps)
            if BEGIN_SEC < n_sec < total_sec - END_SEC:
                train_path = train_dir + '/{}_{}_sec.jpg'.format(video_name, n_sec)
                cv2.imwrite(train_path, skill_1_image)
                print('    ', train_path)

        # if DEBUG and count > 1000:
            # cv2.imshow('frame', skill_1_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # break

def main():
    for root_dir, sub_dirs, files in os.walk('./data/raw/'):
        for file in files:
            if '.mp4' in file:
                video_path = root_dir + '/' + file
                train_dir = './data/input/train/' + os.path.basename(root_dir)
                print('processing', video_path, '->', train_dir)
                gen_train(video_path, train_dir)

if __name__ == '__main__':
    main()
    