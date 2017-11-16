#!/usr/local/bin/python3

from detector import HeroDetect
from detector import Util
from collections import deque, Counter
import cv2
import itertools
# from skvideo.io import VideoCapture
from cv2 import VideoCapture

# if __name__ == '__main__':
heroDetect = HeroDetect(input_shape=Util.shape_hero())
heroDetect.load_model(
    model_path='./model/v4.hero.cnn_vgg_dropout.iter0.model.h5', 
    label_path='./model/v4.hero.cnn_vgg_dropout.iter0.label.txt')

# video_path = './data/raw_test/bailishouyue/q0532r8l8bq.p712.1.mp4'
video_path = './data/raw_test/liubang/k0391sd2c3j.p712.1.mp4'
# video_path = './data/raw_test/huamulan/y0382qw3lsj.p712.1.mp4'
# video_path = './data/raw_test/random/d055299tzgr.p712.1.mp4'
# video_path = './data/raw_test/1.mp4'

cap = VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
count = 0
sample_sec = 10
votes = {} # {(i,j):deque([label], maxlen=10)}
color = (0, 255, 0)

key = None
while key != 120: # press x to stop
    ret, frame = cap.read()
    if not ret:
        break
    # display key frame ervery n second
    count += 1
    if count % int(sample_sec * fps) == 0:
        windows = [(i, -1) for i in range(-3,5)] + [(i, 0) for i in range(-4,5)] + [(i, 1) for i in range(-4,4)]       
        # windows = itertools.poduct(range(-3,4), range(-1,2))
        for i, j in windows:
            text_coord = '%d,%d' % (i,j)
            x1, y1, x2, y2 = Util.rect_grid_hero(frame, i, j)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img=frame, 
                text=text_coord, 
                org=(x1, y1 + 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=.5,
                color=color, 
                thickness=1)

            # prediction
            image = Util.crop_grid_hero(frame, i, j)
            label, prob = heroDetect.predict_image(image)
            votes.setdefault((i,j), deque([], maxlen=1))
            votes[(i, j)].append(label if prob > 0.1 else 'NA')

            most_common = Counter(votes[(i, j)]).most_common(n=1)[0] # ('hero', 0.2)
            text_vote = '%s %d' % (most_common[0], most_common[1]) 

            cv2.putText(img=frame, 
                text=text_vote, 
                org=(x1, y1 + 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=.5,
                color=color, 
                thickness=1)

            if i == j == 0:
                print('debug label, prob:', label, prob)

        print('votes:', votes)

        cv2.imshow(' Press "x" to close window. Press anykey to next frame.', frame)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
cap.release()