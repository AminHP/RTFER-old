# -*- coding: utf-8 -*-

import os

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# flask configs
FLASK_TEMPLATE_DIR = os.path.join(SRC_DIR, 'templates')
FLASK_STATIC_DIR = os.path.join(SRC_DIR, 'static')
FLASK_PORT = 8080
FLASK_DEBUG = True

# online video configs
VIDEO_DEFAULT_WIDTH = 640
VIDEO_DEFAULT_HEIGHT = 480
VIDEO_DEFAULT_FPS = 2
VIDEO_DEFAULT_QUALITY = 0.5  # float in range [0, 1]

# trainer configs
TRAINER_BATCH_SIZE = 64
TRAINER_EPOCHS = 1000

# predictor configs
PREDICTOR_FACE_CASCADE_PATH = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml')
PREDICTOR_LANDMARKS_PATH = os.path.join(DATA_DIR, 'shape_predictor_68_face_landmarks.dat')

# fer configs
FER_CPU_ONLY = True
FER_COMPRESSED_DATASET_PATH = os.path.join(DATA_DIR, 'ck_plus.zip')
FER_CSV_DATASET_PATH = os.path.join(DATA_DIR, 'ck_plus.csv')
FER_DATASET_PATH = os.path.join(DATA_DIR, 'ck_plus.pickle')
FER_MODEL_NAME = 'model'
FER_MODEL_PATH = os.path.join(MODEL_DIR, '%s.hdf5' % FER_MODEL_NAME)
FER_USAGES = ['train', 'test']
FER_SOURCES = ['ck+', 'fer+', 'lfw+']
FER_ALL_EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
FER_SELECTED_EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
FER_NEUTRAL_EFFECT = 1.  # float in range [0, 1]


####################################################

if FER_CPU_ONLY:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
