import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from use_gpu import use_gpu

test_dir = 'data/test/'
model_dir = 'model'

use_gpu()

# num = 0
# for file in os.listdir(test_dir):
#     path = test_dir + file
#     os.rename(path, test_dir + str(num) + '.jpg')
#     num += 1

model = tf.keras.models.load_model(model_dir, compile=True)

for file in os.listdir(test_dir):
    path = test_dir + file
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    print(classes)

    if classes[0] > 0.5:
        print('{} es un perro'.format(file))
    else:
        print('{} es un gato'.format(file))
