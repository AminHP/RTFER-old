# -*- coding: utf-8 -*-

# python imports
import io
import numpy as np
import cv2
from PIL import Image
from imutils import face_utils
from matplotlib import pyplot as plt


def create_image_from_blob(blob):
    image = Image.open(io.BytesIO(blob))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image, gray_image


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image, gray_image


def put_text(image, rect, text):
    x, y, w, h = rect

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = h / 30.0
    font_thickness = int(round(font_scale * 1.5))
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    center_text_x = x + (w // 2)
    center_text_y = y + (h // 2)
    text_w, text_h = text_size

    lower_left_text_x = center_text_x - (text_w // 2)
    lower_left_text_y = center_text_y + (text_h // 2)

    cv2.putText(
        image, text,
        (lower_left_text_x, lower_left_text_y),
        font, font_scale, (0, 255, 0), font_thickness
    )


def draw_face_info(image, face_info):
    x = int(face_info['border']['x'])
    y = int(face_info['border']['y'])
    w = int(face_info['border']['width'])
    h = int(face_info['border']['height'])
    emotion = face_info['emotion']

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    put_text(image, (x, y, w, h // 5), emotion)


def draw_landmarks(image, landmarks, visualize=True, draw_dots=False):
    if not isinstance(landmarks, list):
        landmarks = [landmarks]
    for lm in landmarks:
        if visualize:
            image[:] = face_utils.visualize_facial_landmarks(image, lm)
        if draw_dots:
            for x, y in lm:
                cv2.circle(image, (x, y), 1, (255, 0, 0), 2)


def show_image(image, title='Result'):
    plt.subplot(111), plt.imshow(image), plt.title(title)
    plt.show()
