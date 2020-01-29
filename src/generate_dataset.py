# -*- coding: utf-8 -*-

# python imports
from __future__ import print_function
import os
import numpy as np
import zipfile
import pickle

# project imports
from config import (DATA_DIR, FER_CSV_DATASET_PATH, FER_COMPRESSED_DATASET_PATH,
                    FER_DATASET_PATH, FER_USAGES, FER_SOURCES)


def extract():
    zip_ref = zipfile.ZipFile(FER_COMPRESSED_DATASET_PATH, 'r')
    zip_ref.extractall(DATA_DIR)
    zip_ref.close()


def create_pickle_data():
    with open(FER_CSV_DATASET_PATH) as file:
        file.readline()

        all_images = []
        all_landmarks = []
        all_details = []

        for i, line in enumerate(file):
            line = line.replace('\n', '').split(',')
            usage = FER_USAGES.index(line[0])
            source = FER_SOURCES.index(line[1])
            file_index = i
            width, height = int(line[3]), int(line[4])
            pixels = [int(p) for p in line[5].split(' ')]
            image = np.array(pixels).reshape((height, width)).astype('uint8')
            landmarks = [int(l) for l in line[6].split(' ')]
            landmarks = np.array(landmarks).reshape((68, 2))
            best_emotion = int(line[7])
            emotion_values = [int(v) for v in line[8:]]

            all_images.append(image)
            all_landmarks.append(landmarks)
            all_details.append([usage, source, file_index, best_emotion] + emotion_values)

        all_landmarks = np.array(all_landmarks)
        all_details = np.array(all_details)
        pickle.dump((all_images, all_landmarks, all_details), open(FER_DATASET_PATH, "wb"))

        # Test
        i, l, d = pickle.load(open(FER_DATASET_PATH, "rb"))
        assert len(all_images) == len(i) and (all_landmarks == l).all() and (all_details == d).all()
        assert all([(all_images[ind] == i[ind]).all() for ind in range(len(all_images))])



if __name__ == '__main__':
    extract()
    create_pickle_data()
