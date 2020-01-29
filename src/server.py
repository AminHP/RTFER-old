# -*- coding: utf-8 -*-

# python imports
from __future__ import print_function
from pprint import pprint

# flask imports
from flask import Flask, request, render_template, Response
from flask_socketio import SocketIO, emit

# project imports
from utils import create_image_from_blob
from predict import Predictor
from config import (FLASK_TEMPLATE_DIR, FLASK_STATIC_DIR, FLASK_PORT, FLASK_DEBUG,
                    VIDEO_DEFAULT_WIDTH, VIDEO_DEFAULT_HEIGHT, VIDEO_DEFAULT_FPS, VIDEO_DEFAULT_QUALITY)


# Create app
app = Flask(__name__, template_folder=FLASK_TEMPLATE_DIR, static_folder=FLASK_STATIC_DIR)
socketio = SocketIO(app)

# Create predictor
predictor = Predictor(threadsafe=True)
predictor.load_model()



@app.route('/')
def index():
    width = request.args.get('width', VIDEO_DEFAULT_WIDTH)
    height = request.args.get('height', VIDEO_DEFAULT_HEIGHT)
    fps = request.args.get('fps', VIDEO_DEFAULT_FPS)
    quality = request.args.get('quality', VIDEO_DEFAULT_QUALITY)

    return render_template(
        'index.html',
        width=width, height=height, fps=fps, quality=quality
    )


@socketio.on('connect', namespace='/fer')
def on_connect():
    print('connected!')


@socketio.on('predict', namespace='/fer')
def on_predict(message):
    image_data = message['image']
    _image, gray_image = create_image_from_blob(image_data)

    result = []
    for face_info in predictor.predict(gray_image):
        result.append(face_info)
    pprint(result)
    emit('predicted', result)


def run():
    socketio.run(app, host='0.0.0.0', port=FLASK_PORT, debug=FLASK_DEBUG)


if __name__ == '__main__':
    run()
