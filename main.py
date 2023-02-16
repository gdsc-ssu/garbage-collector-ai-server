import numpy as np
from keras.models import load_model
import keras.applications.mobilenet_v2 as mobilenetv2
import tensorflow as tf

import io
from PIL import Image
from fastapi import FastAPI, File

model = load_model('trash_model.h5')

categories = {0: 'battery', 1: 'biological', 2: 'glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass',
              6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash',
              11: 'white-glass'}


def mobilenetv2_preprocessing(img):
    return mobilenetv2.preprocess_input(img)


def predict_garbage_type(garbage_image: bytes = File()):
    img = await garbage_image.read()
    img = Image.open(io.BytesIO(img))
    # img = Image.open(garbage_image)
    img = np.asarray(img)
    img = tf.image.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3))

    predict = model.predict(img)

    prob = max(predict[0]) * 100
    prob = round(prob, 2)

    garbage_category = categories[np.argmax(predict[0])]

    return prob, garbage_category


app = FastAPI()


@app.get("/test")
async def test():
    return {"result": "Hello World!"}


@app.get("/garbage")
async def get_garbage_type(garbage_image: bytes = File()):
    prob, category = predict_garbage_type(garbage_image)

    return {"predicted_type": category, "probability": prob}
