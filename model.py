import tensorflow as tf
import numpy as np
from pathlib import Path

# Class based on https://github.com/phoenixfin/general-cnn/blob/master/cnn.py
# optimized for Plant CNN deployment


class PlantCNNModel(object):

    categories = [
        'Apple : Scab',
        'Apple : Black rot',
        'Apple : Cedar rust',
        'Apple : Healty',
        'Blueberry : Healthy',
        'Cherry : Powdery mildew',
        'Cherry : Healthy',
        'Corn : Cercospora / gray leaf spot',
        'Corn : Common rust',
        'Corn : Northern leaf blight',
        'Corn : Healthy',
        'Grape : Black rot',
        'Grape : Esca (black measles)',
        'Grape : Leaf blight',
        'Grape : Healthy',
        'Orange : Citrus greening',
        'Peach : Bacterial spot',
        'Peach : Healthy',
        'Bell Pepper : Bacterial spot',
        'Bell Pepper : Healthy',
        'Potato : Early blight',
        'Potato : Late blight',
        'Potato : Healthy',
        'Raspberry : Healthy',
        'Soybean : Healthy',
        'Squash : Powdery mildew',
        'Strawberry : Leaf scorch',
        'Strawberry : Healthy',
        'Tomato : Bacterial spot',
        'Tomato : Early blight',
        'Tomato : Late blight',
        'Tomato : Leaf mold',
        'Tomato : Septoria leaf spot',
        'Tomato : Spider mites',
        'Tomato : Target spot',
        'Tomato : Yellow leaf curl virus',
        'Tomato : Mosaic virus',
        'Tomato : Healty'
    ]

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    # Set categories
    def load_categories(self, categories):
        self.categories = categories

    # Predict one input image file
    def predict(self, img_path):
        IM = tf.keras.preprocessing.image
        size = 300

        img = IM.load_img(img_path, target_size=(size, size))
        img_array = IM.img_to_array(img)
        normalized = np.expand_dims(img, axis=0)/255
        res = self.model.predict(normalized)
        return self.categories[np.argmax(res[0])]
