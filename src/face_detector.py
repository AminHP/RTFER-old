# -*- coding: utf-8 -*-

# python imports
import numpy as np
import cv2
import dlib

# project imports
from config import PREDICTOR_FACE_CASCADE_PATH


class FaceDetector:

    def __init__(self, threadsafe=False):
        self.threadsafe = threadsafe
        if not self.threadsafe:
            self.cv_face_cascade = self._load_cv_face_cascade()
        self.dlib_face_detector = self._load_dlib_face_detector()


    def find(self, image, backend='dlib', surely_has_face=False):
        if not backend in ['dlib', 'opencv']:
            raise ValueError("Unknown backend '%s'. Supported backends are 'dlib' and 'opencv'" % backend)

        if backend == 'dlib':
            result = self._dlib_find(image)
        elif backend == 'opencv':
            result = self._cv_find(image)

        if surely_has_face:
            if len(result) == 0:
                result = [(0, 0, image.shape[1] - 1, image.shape[0] - 1)]
        return result


    def _cv_find(self, image):
        if self.threadsafe:
            face_cascade = self._load_cv_face_cascade()
        else:
            face_cascade = self.cv_face_cascade

        face_rects = face_cascade.detectMultiScale(
            image,
            scaleFactor = 1.1,
            minNeighbors = 22
        )
        return face_rects


    def _dlib_find(self, image):
        rects = self.dlib_face_detector(image, 0)
        result = []
        for rect in rects:
            result.append((rect.left(), rect.top(), rect.width(), rect.height()))
        return result


    def _load_cv_face_cascade(self):
        return cv2.CascadeClassifier(PREDICTOR_FACE_CASCADE_PATH)


    def _load_dlib_face_detector(self):
        return dlib.get_frontal_face_detector()
