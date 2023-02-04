import numpy as np

from keras.preprocessing import image
from keras.models import load_model

model = load_model('model12.h5')

trash_img = image.load_img('test-dataset/battery1.jpg')

print(model.predict(trash_img))

#
# x = image.img_to_array(trash_img)
# x = np.expand_dims(x, axis=0)
#
# images = np.vstack([x])
# classes = model.predict_classes(images, batch_size=10)
# print classes
