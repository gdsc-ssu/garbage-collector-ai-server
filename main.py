import numpy as np
from keras.models import load_model
import keras.applications.mobilenet_v2 as mobilenetv2
import tensorflow as tf

import io
from PIL import Image

from flask import Flask
from flask import request

model = load_model('trash_model.h5')

categories = {0: 'battery', 1: 'biological', 2: 'glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass',
              6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash',
              11: 'white-glass'}


def predict_garbage_type(garbage_image):
    # img = garbage_image.read()
    # img = Image.open(io.BytesIO(img))
    img = Image.open(garbage_image)
    img = np.asarray(img)
    img = tf.image.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3))

    predict = model.predict(img)

    prob = max(predict[0]) * 100
    prob = round(prob, 2)

    garbage_category = categories[np.argmax(predict[0])]

    return prob, garbage_category

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def test():
    return {"result": "Hello World!"}


@app.route("/garbage", methods=["GET"])
def get_garbage_type():
    garbage_image = request.files['image']
    prob, category = predict_garbage_type(garbage_image)

    return {"predicted_type": category, "probability": prob}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)