import numpy as np

from keras.preprocessing import image
from keras.models import load_model
import keras.applications.mobilenet_v2 as mobilenetv2
import cv2
from PIL import Image
import tensorflow as tf

categories = {0: 'battery', 1: 'biological', 2: 'glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass',
              6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash',
              11: 'white-glass'}

def mobilenetv2_preprocessing(img):
    return mobilenetv2.preprocess_input(img)


model = load_model('trash_model.h5')

trash_img = Image.open('test-dataset/battery1.jpg')

trash_img = np.asarray(trash_img)

trash_img = tf.image.resize(trash_img, (224, 224))

trash_img = np.reshape(trash_img, (1, 224, 224, 3))

# print(trash_img.shape)

predict = model.predict(trash_img)

# print(predict)
# print(predict[0][1])
# print(np.sum(predict[0]))

max = max(predict[0]) * 100
# max_index = predict[0].where(predict[0] == max)

print(f"이 쓰레기는 {round(max, 2)}% 확률로 {categories[np.argmax(predict[0])]}입니다.")

#
# x = image.img_to_array(trash_img)
# x = np.expand_dims(x, axis=0)
#
# images = np.vstack([x])
# classes = model.predict_classes(images, batch_size=10)
# print classes
