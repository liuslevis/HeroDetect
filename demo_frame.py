#!/usr/local/bin/python3

from detector import HeroDetect
import cv2
from detector import Util

if __name__ == '__main__':
    heroDetect = HeroDetect(input_shape=(50, 50, 3))
    heroDetect.load_model(
        model_path='./data/output/v1.cnn_vgg.iter0.model.h5', 
        label_path='./data/output/v1.cnn_vgg.iter0.label.txt')

    video_path = './data/raw_test/bailishouyue/q0532r8l8bq.p712.3.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    key = None
    while True and key != 120:
        ret, frame = cap.read()
        if not ret:
            break
        # display key frame ervery 100 second
        count += 1
        if count % 100 * fps == 0:
            predict = heroDetect.predict_frame(frame)
            x1, y1, x2, y2 = Util.rect_skill_1(frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(str(predict) + ' Press "x" to close window', frame)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
    cap.release()