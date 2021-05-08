from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

import keras
from tensorflow.keras.models import Sequential
import numpy as np


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask_gtts import gtts
from gevent.pywsgi import WSGIServer
from flask import Response
from yolo_detection_images import runModel
from walk import VideoCamera

app = Flask(__name__)

gtts(app)





#model = Sequential()

# Model saved with Keras model.save()
MODEL_PATH = 'mdl_wts_xc.hdf5'

MODEL_PATH_CURRENCY = 'mdl_wts_xc.hdf5'

MODEL_PATH_W = 'mdl_wts_xc.hdf5'

MODEL_PATH_B = 'BrailleNet.h5'

class_labels = {0:'No DR', 1:'Mild',2:'Moderate',3:'Severe',4:'Proliferative DR'}

class_labels_new = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


class_labels_curr = {0: '10', 1: '100', 2: '20', 3: '200', 4: '2000', 5: '50', 6: '500', 7: 'Background'}


class_labels_w = {0: 'Cloudy', 1: 'Sunny', 2: 'Rainy', 3: 'Snowy', 4: 'Foggy'}

model = load_model(MODEL_PATH)
model_c = load_model(MODEL_PATH_CURRENCY)
model_w = load_model(MODEL_PATH_W)
model_b = load_model(MODEL_PATH_B)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(100, 100))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    preds = model.predict(x)
    return preds


def model_predict_c(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    preds = model_c.predict(x)
    return preds



def model_predict_w(img_path, model):
    img = image.load_img(img_path, target_size=(100, 100))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    preds = model_w.predict(x)
    return preds



def model_predict_b(img_path, model):
    img = image.load_img(img_path, target_size=(28, 28))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    preds = model_b.predict(x)
    return preds

############################################## THE REAL DEAL ###############################################
@app.route('/detectObject' , methods=['POST'])
def mask_image():
	# print(request.files , file=sys.stderr)
	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	######### Do preprocessing here ################
	# img[img > 150] = 0
	## any random stuff do here
	################################################

	img = runModel(img)

	img = Image.fromarray(img.astype("uint8"))
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	img_base64 = base64.b64encode(rawBytes.read())
	return jsonify({'status':str(img_base64)})



@app.route('/test' , methods=['GET','POST'])
def test():
	print("log: got at test" , file=sys.stderr)
	return jsonify({'status':'succces'})

@app.route('/')
def home():
	return render_template('home.html')


#--------


#---------


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        
        preds = model_predict(file_path, model) 
        
        print(preds)

        pred_prob = np.max(preds)
        
        pred_class = preds.argmax(axis=-1) 
        
        result = pred_class[0]

        res = (class_labels[result])

 
        return res
    return None


@app.route('/weather', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        
        preds = model_predict_w(file_path, model) 
        
        print(preds)

        pred_prob = np.max(preds)
        
        pred_class = preds.argmax(axis=-1) 
        
        result = pred_class[0]

        res = (class_labels_w[result])

 
        return render_template('weather.html',pred_text_w = res) 
    return None

#----

@app.route('/braile', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        
        preds = model_predict_b(file_path, model) 
        
        print(preds)

        pred_prob = np.max(preds)
        
        pred_class = preds.argmax(axis=-1) 
        
        result = pred_class[0]

        res = (class_labels_new[result])

 
        return render_template('braile.html',pred_text_b = res) 
    return None


#---

@app.route('/currency', methods=['GET', 'POST'])
def upload3():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        
        preds = model_predict_c(file_path, model) 
        
        print(preds)

        pred_prob = np.max(preds)
        
        pred_class = preds.argmax(axis=-1) 
        
        result = pred_class[0]

        res = (class_labels_c[result])

 
        return render_template('currency.html',pred_text_c = res) 
    return None

#---open cv-----

def gen(walk):
    
    while True:
        #get camera frame
        frame = walk.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#-----


@app.route('/index')
def index():
	return render_template('yolo.html')


@app.route('/dis')
def dis():
	return render_template('index.html')


@app.route('/brailePG')
def brailePG():
	return render_template('braile.html')


@app.route('/weatherPG')
def weatherPG():
	return render_template('weather.html')


@app.route('/currencyPG')
def currencyPG():
	return render_template('currency.html')
	
    
@app.route('/audio')
def audio():
	return render_template('audio.html')

@app.route('/song')
def song():
	return render_template('songs.html')

@app.route('/walk')
def walk():
	return render_template('walk.html')

@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(debug = False)
