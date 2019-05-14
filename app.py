from __future__ import division, print_function

# coding=utf-8
import os

import cv2 as cv
import tensorflow as tf
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from models.medicine_model import MedicineNet

# Define a flask app
app = Flask(__name__)

# tf session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# Load trained model
model = MedicineNet((64, 64, 3))
saver = tf.train.Saver()
saver.restore(sess, "./models/tmp/model.ckpt")
print('Model loaded. Start serving...')


def model_predict(img_path, model):
    describe = []
    img = cv.resize(cv.imread(img_path) / 255.0, (64, 64)).reshape(-1, 64, 64, 3)
    img_dict = {model.inputs: img}
    or_out = sess.run(model.or_out, feed_dict=img_dict)
    sick = or_out[0][0]
    normal = or_out[0][1]
    describe.append('normal %.2f%%' % (100 - normal * 100))
    describe.append('sick %.2f%%' % (100 - sick * 100))
    return describe


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


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
        result = str(preds[0] + '\n' + preds[1])
        return result
    return None


if __name__ == '__main__':
    app.run(port=5088, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('192.168.2.76', 5088), app)
    #http_server.serve_forever()

