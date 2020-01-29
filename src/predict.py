# -*- coding: utf-8 -*-

# python imports
from __future__ import print_function
from pprint import pprint
import sys
import cv2
import numpy as np

# project imports
from saving import load_model
from face_detector import FaceDetector
from features import get_features
from config import FER_MODEL_PATH


class Predictor:

    def __init__(self, threadsafe=False):
        self.threadsafe = threadsafe
        self.face_detector = FaceDetector(threadsafe)
        self.model = None


    def load_model(self):
        self.model = load_model(FER_MODEL_PATH)
        if self.threadsafe:
            self.model._make_predict_function()
        list(self.predict(np.zeros((1000, 1000)).astype(np.uint8))) # warmup


    def predict(self, gray_image):
        face_rects = self.face_detector.find(gray_image)

        for face_rect in face_rects:
            x, y, w, h = face_rect
            face = gray_image[y:y+h, x:x+w]

            features = get_features(face)
            if features is None:
                continue
            predicted_emotions = self._model_predict(features)[0]
            best_emotion = self.model.labels[np.argmax(predicted_emotions)]

            # Create a json serializable result
            yield dict(
                border = dict(
                    x = float(x),
                    y = float(y),
                    width = float(w),
                    height = float(h),
                ),
                prediction = {self.model.labels[i]: float(predicted_emotions[i]) for i in range(len(predicted_emotions))},
                emotion = best_emotion
            )


    def _model_predict(self, x):
        if len(x.shape) == 1:
            return self.model.predict(np.array([x]))
        return self.model.predict(x)



if __name__ == '__main__':
    from utils import load_image, draw_face_info, draw_landmarks, show_image
    from features import get_all_landmarks

    image, gray_image = load_image(sys.argv[1])
    p = Predictor()
    p.load_model()

    for face_info in p.predict(gray_image):
        pprint(face_info, indent=2)
        draw_face_info(image, face_info)
    draw_landmarks(image, get_all_landmarks(gray_image), draw_dots=True)

    show_image(image)
