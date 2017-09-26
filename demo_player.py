#!/usr/local/bin/python3

from detector import HeroDetect
from detector import Util
from collections import deque, Counter
import cv2
import skvideo.io

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
count = 0
sample_sec = 1
key = None
votes = deque([], maxlen=10)
color = (0, 255, 0)

while key != 120: # press x to stop
    ret, frame = cap.read()
    if not ret:
        break
    # display key frame ervery n second
    count += 1
    if count % int(sample_sec * fps) == 0:
        label, prob = heroDetect.predict_frame(frame)
        if prob > 0.1:
            votes.append(label)

        top_vote = Counter(votes).most_common(n=1)
        # text = str(top_vote[0][0] if top_vote else '') # 'hero'
        text = str(Counter(votes).most_common(n=1)) # ('hero', 0.2)
        # text = str(label, prob) # ('hero', 0.002)

        x1, y1, x2, y2 = Util.rect_skill_1(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img=frame, 
            text=text, 
            org=(x1, y1-10), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=.8,
            color=color, 
            thickness=2)

        cv2.imshow(text + ' Press "x" to close window', frame)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
cap.release()