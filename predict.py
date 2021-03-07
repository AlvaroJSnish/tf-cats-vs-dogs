import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# import uuid

test_dir = 'data/test/'
model_dir = 'model'

# for file in os.listdir(test_dir):
#     path = test_dir + file
#     os.rename(path, test_dir + str(uuid.uuid4()) + '.jpg')

model = tf.keras.models.load_model(model_dir, compile=True)

for file in os.listdir(test_dir):
    path = test_dir + file
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    if classes[0] > 0.5:
        print('{} es un perro'.format(file))
    else:
        print('{} es un gato'.format(file))
