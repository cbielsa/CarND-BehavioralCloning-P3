import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import cv2

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


errSum = 0.


# functions to pre-process images before they are feed to the CNN
# pre-processing is identical in training and inference

def crop_top_bottom_rows(img, numRowsBottomCropped=20, numRowsTopCropped=55):
    imgMod = np.empty((img.shape[0]-numRowsBottomCropped-numRowsTopCropped, img.shape[1], img.shape[2]), dtype=img.dtype)
    imgMod[:] = img[numRowsTopCropped:img.shape[0]-numRowsBottomCropped]
    return imgMod

def preprocessImage(img, numRowsBottomCropped=20, numRowsTopCropped=55, newNumRows=32, newNumCols=120):
    img = crop_top_bottom_rows(img, numRowsBottomCropped, numRowsTopCropped)
    return cv2.resize(img, (newNumCols, newNumRows)).astype(np.float32)  # retype and resize



# at each control cycle:
#  - steering angle is calculated by inference of the CNN defined by model.json and model.h5,
#    with the CNN having been trained with augmented data from a drive of circuit1 in the simulator
#  - throttle is calulated with a PID controller with target speed 20

@sio.on('telemetry')
def telemetry(sid, data):
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

    # image pre-processing
    image_array = preprocessImage(image_array)
    transformed_image_array = image_array[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    
    # PID speed controller target and gains
    speedTarget = 20.
    kP = 0.02; kI = 0.00012; kD = 0.15;

    err = float(speed) - speedTarget
    telemetry.errSum += err
    throttle = -kP*err -kI*telemetry.errSum - kD*(err-telemetry.err)
    
    telemetry.err = err  # store this cycle error to calculate error diff in next control cycle

    print("speed, steer, throttle:", float(speed), steering_angle, throttle)
    send_control(steering_angle, throttle)

# initialize sum of errors, for integral component of speed PID controller
telemetry.errSum = 0.

# initialize error in last control cycle, for derivative component of speed PID controller
telemetry.err = 0.



@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)
    print("\nJust connected!\n")


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

