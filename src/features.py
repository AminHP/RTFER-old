# -*- coding: utf-8 -*-

# python imports
from __future__ import print_function
import numpy as np
import cv2
import dlib
from math import atan2, degrees
from imutils import face_utils

# project imports
from config import PREDICTOR_LANDMARKS_PATH


def get_roll_rotated_landmarks(landmarks):
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    origin = (left_eye + right_eye) / 2.
    rotation_angle = -np.arctan2(*((right_eye - left_eye)[::-1]))
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])

    normal_landmarks = landmarks - origin
    rotated_landmarks = np.dot(R, normal_landmarks.T).T
    result = rotated_landmarks + origin
    return result


def get_yaw_rotated_landmarks(landmarks):
    min_point = np.amin(landmarks, axis=0)
    normal_landmarks = landmarks - min_point

    max_x = np.amax(normal_landmarks, axis=0)[0]
    flipped = (np.array([max_x, 0]) - normal_landmarks) * np.array([1, -1])

    alpha = .5

    average = np.copy(normal_landmarks)
    average = average * alpha
    flipped = flipped * (1 - alpha)

    average[00:17] += flipped[00:17][::-1]
    average[17:27] += flipped[17:27][::-1]
    average[27:31] += flipped[27:31]
    average[31:36] += flipped[31:36][::-1]
    average[36:40] += flipped[42:46][::-1]
    average[40:42] += flipped[46:48][::-1]
    average[42:46] += flipped[36:40][::-1]
    average[46:48] += flipped[40:42][::-1]
    average[48:55] += flipped[48:55][::-1]
    average[55:60] += flipped[55:60][::-1]
    average[60:65] += flipped[60:65][::-1]
    average[65:68] += flipped[65:68][::-1]

    result = average + min_point
    return result


def get_wrinkle_density(image, points):
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, (255, 255, 255))
    roi = cv2.bitwise_and(image, mask)
    not_roi = cv2.bitwise_and(cv2.bitwise_not(image), mask)
    if not (roi.any() and not_roi.any()):
        return 0
    return np.divide(np.sum(roi), np.sum(roi + not_roi))


def calculate_angle_between_points_in_region(landmarks):
    angles = []
    for i in range(len(landmarks) - 1):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        x_diff = p2[0] - p1[0]
        y_diff = -(p2[1] - p1[1])
        angles.append(degrees(atan2(y_diff, x_diff)))
    return angles


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_LANDMARKS_PATH)

def get_landmarks(image):
    rects = detector(image, 0)
    if len(rects) == 0:
        return
    rect = rects[0]
    landmarks = predictor(image, rect)
    landmarks = face_utils.shape_to_np(landmarks)
    return landmarks


def get_all_landmarks(image):
    rects = detector(image, 0)
    result = []
    for rect in rects:
        landmarks = predictor(image, rect)
        landmarks = face_utils.shape_to_np(landmarks)
        result.append(landmarks)
    return result


def get_features(image, landmarks=None):
    if landmarks is None:
        landmarks = get_landmarks(image)
    if landmarks is None:
        return

    # laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # laplacian = np.absolute(laplacian)
    # laplacian = (((laplacian - np.min(laplacian)) / (np.max(laplacian) - np.min(laplacian))) * 255).astype('uint8')

    # sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)
    # sobel_x = np.absolute(sobel_x)
    # sobel_x = (((sobel_x - np.min(sobel_x)) / (np.max(sobel_x) - np.min(sobel_x))) * 255).astype('uint8')

    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
    sobel_y = (((sobel_y - np.min(sobel_y)) / (np.max(sobel_y) - np.min(sobel_y))) * 255).astype('uint8')

    edge = sobel_y
    orig_landmarks = landmarks
    landmarks = get_roll_rotated_landmarks(landmarks)
    landmarks = get_yaw_rotated_landmarks(landmarks)

    face_regions = {
        "upper_lip_top": landmarks[48:55],
        "upper_lip_bottom": landmarks[60:65],
        "lower_lip_top": np.flip(np.concatenate([landmarks[64:67], [landmarks[67]], [landmarks[60]]]), axis=0),
        "lower_lip_bottom": np.flip(np.concatenate([landmarks[54:60], [landmarks[48]]]), axis=0),

        "left_eyelid": np.concatenate([landmarks[17:22], np.flip(landmarks[36:40], axis=0)]),

        "between_eyes_wrinkle": np.concatenate([
            [orig_landmarks[21]], [orig_landmarks[22]], [orig_landmarks[42]],
            [orig_landmarks[28]], [orig_landmarks[39]]
        ]),
        "left_eye_wrinkle": np.concatenate([
            [orig_landmarks[17]],
            [[orig_landmarks[36][0], orig_landmarks[17][1]]],
            [[orig_landmarks[36][0], orig_landmarks[1][1]]],
            [[orig_landmarks[17][0], orig_landmarks[1][1]]]
        ]),
        "right_eye_wrinkle": np.concatenate([
            [orig_landmarks[26]],
            [[orig_landmarks[26][0], orig_landmarks[15][1]]],
            [[orig_landmarks[45][0], orig_landmarks[15][1]]],
            [[orig_landmarks[45][0], orig_landmarks[26][1]]]
        ])
    }

    f1 = calculate_angle_between_points_in_region(face_regions["upper_lip_top"])
    f2 = calculate_angle_between_points_in_region(face_regions["upper_lip_bottom"])
    f3 = calculate_angle_between_points_in_region(face_regions["lower_lip_top"])
    f4 = calculate_angle_between_points_in_region(face_regions["lower_lip_bottom"])

    f5 = calculate_angle_between_points_in_region(face_regions["left_eyelid"])

    f6 = get_wrinkle_density(edge, face_regions["between_eyes_wrinkle"])
    f7 = get_wrinkle_density(edge, face_regions["left_eye_wrinkle"])
    f8 = get_wrinkle_density(edge, face_regions["right_eye_wrinkle"])

    return f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
