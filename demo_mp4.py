#!/usr/local/bin/python3

from detector import HeroDetect
from detector import Util
from collections import deque, Counter
import cv2
import skvideo.io

DEBUG = False

# if __name__ == '__main__':
heroDetect = HeroDetect(input_shape=(50, 50, 3))
heroDetect.load_model(
    model_path='./model/cnn_vgg.model.h5', 
    label_path='./model/cnn_vgg.label.txt')

# video_path = './data/raw_test/bailishouyue/q0532r8l8bq.p712.1.mp4'
# video_path = './data/raw_test/liubang/k0391sd2c3j.p712.1.mp4'
# video_path = './data/raw_test/huamulan/y0382qw3lsj.p712.1.mp4'
video_path = './data/raw_test/random/d055299tzgr.p712.1.mp4'
cap = skvideo.io.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
i_frame = 0
debug_info_per_sec = 1
detect_per_frames = 5

key = None
votes = deque([], maxlen=10)
color = (0, 255, 0)
text_hero = ''

def need_cut(i_sec):
    li = [(10, 20),  #10
        (40, 55),#15
        (64, 75),#10
        (90, 105), #15
        (120, 140),#20
        # (207, 227),#20
        ]
    for from_sec, to_sec in li:
        if from_sec <= i_sec <= to_sec:
            return True
    return False

def kill_info(i_sec):
    info = [(18, 2),
        (27, 3),
        (44, 2),
        (48, 3),
        (54, 4),
        (66, 2),
        (71, 3),
        (75, 4),
        (79, 5),
        (91, 2),
        (96, 3),
        (99, 4),
        (106, 5),
        (123, 2),
        (132, 3),
        (137, 4),
        (160, 2),
        (166, 3),
        (180, 6),
        (185, 2),
        (189, 3),
        (194, 4),
        (197, 5),
        (209, 2),
        (213, 2),
        (128, 3),
        (223, 4),
        (229, 5),]
    for sec, kill in info:
        if sec <= i_sec <= sec + 2:
            return '{} kill'.format(kill)
    return ''

out = cv2.VideoWriter('./data/output/demo.mp4', cv2.VideoWriter_fourcc(*'MPEG'), fps=25.0, frameSize=(848, 480))

while key != 120: # press x to stop
    ret, frame = cap.read()
    if not ret:
        break

    i_frame += 1
    i_sec = i_frame / fps 

    if i_frame % detect_per_frames == 0:
        # Hero Text
        label, prob = heroDetect.predict_frame(frame)
        if prob > 0.1:
            votes.append(label)

        top_vote = Counter(votes).most_common(n=1)
        text_hero = str(top_vote[0][0] if top_vote else '') # 'hero'

    # draw label
    x1, y1, x2, y2 = Util.rect_skill_1(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img=frame, 
        text=text_hero, 
        org=(x1, y1-10), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=.8,
        color=color, 
        thickness=2)

    x1, y1, x2, y2 = Util.rect_kill_info(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img=frame,
        text=kill_info(i_sec),
        org=(x1, y1-10), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=.8,
        color=color, 
        thickness=2
        )

    if DEBUG:
        if i_sec % debug_info_per_sec == 0:
            cv2.imshow('{} @ {} sec. Press "x" to close window'.format(text_hero, i_sec), frame)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif need_cut(i_sec):
        out.write(frame)

out.release()
cap.release()