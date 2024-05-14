import math

from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'


def bone_image(path):
    img = load_img(path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = load_model('C:\\Users\\venka\\PycharmProjects\\pythonProject2\\bone\\Model.h5')
    preds = model.predict(x)

    print(preds)
    return math.floor(preds[0][0])
