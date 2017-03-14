# Lightly modified drive.py
# Added command line parameter for throttle, import of preprocess() from shared.py

import argparse
import base64

import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import model_from_json

from shared import preprocess

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    #print('Telemetry call...')
    #print(data)
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    # do image pre-processing here, as per anything done in training
    img_processed = preprocess(transformed_image_array)

    steering_angle = float(model.predict(img_processed, batch_size=1))
    # The driving model takes a throttle setting as an argument.
    throttle = fixed_throttle
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    #print('Send control...')
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)
    #print('Sent data...')

if __name__ == '__main__':
    print('Starting auto driving...')
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('throttle', type=float,
                        help='Throttle setting, try 0.1, 0.2, up to 1.0 maximum.')
    args = parser.parse_args()
    sFile = args.model
    fixed_throttle = args.throttle
    print('Using file        : ', sFile)                # command line param
    print('Throttle fixed at : ', fixed_throttle)       # command line param

    with open(sFile, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")                # using Adam optimiser and MSE loss function
    print('Compiled Model')
    weights_file = sFile.replace('json', 'h5')
    model.load_weights(weights_file)
    print('Loaded model')

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
