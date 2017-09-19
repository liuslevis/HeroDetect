#!/usr/local/bin/python3

import cv2
import util
from detector import HeroDetect
input_size = (50, 50)
input_shape = (*input_size, 3)

heroDetect = HeroDetect(input_shape=input_shape)
heroDetect.load_model(
    model_path='./data/output/v1.cnn_vgg.iter0.model.h5', 
    label_path='./data/output/v1.cnn_vgg.iter0.label.txt')

cap = cv2.VideoCapture('./data/raw_test/ake/t0530k7xavm.p712.1.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_sec = int(frame_count / fps)
count = 0
key = None
while True and key != 120:
    ret, frame = cap.read()
    if not ret:
        break
    # display key frame ervery 100 second
    count += 1
    if count % 100 * fps == 0:
        # image = util.crop_skill_1(frame, input_size)
        # predict = heroDetect.predict_image(image)
        predict = heroDetect.predict_frame(frame)
        x1, y1, x2, y2 = util.rect_skill_1(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(str(predict) + ' Press "x" to close window', frame)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
cap.release()
