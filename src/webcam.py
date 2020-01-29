# -*- coding: utf-8 -*-

# python imports
from __future__ import print_function
import cv2

# project imports
from predict import Predictor
from utils import draw_face_info, draw_landmarks
from features import get_all_landmarks


def run():
    predictor = Predictor()
    predictor.load_model()

    cv2.namedWindow("Webcam")
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        return

    while True:
        rval, frame = capture.read()
        if not rval:
            break

        image = frame
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for face_info in predictor.predict(gray_image):
            draw_face_info(image, face_info)
        draw_landmarks(image, get_all_landmarks(gray_image), draw_dots=True, visualize=False)

        cv2.imshow("Webcam", image)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'): # exit on ESC or Q
            break

    cv2.destroyWindow("Webcam")
    capture.release()



if __name__ == '__main__':
    run()
