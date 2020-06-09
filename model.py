import tensorflow as tf
import numpy as np
import requests
import json

from pathlib import Path

# Class based on https://github.com/phoenixfin/general-cnn/blob/master/cnn.py
# optimized for Plant CNN deployment


class PlantCNNProxy(object):

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

    def __init__(self, model_uri):
        self.model_uri = model_uri

    # Set categories
    def load_categories(self, categories):
        self.categories = categories

    # Predict one input image file
    def predict(self, img_path):
        IM = tf.keras.preprocessing.image
        img = IM.img_to_array(IM.load_img(
            img_path, target_size=(300, 300))) / 255.

        payload = {
            "instances": [{'input_1': img.tolist()}]
        }

        response = requests.post(self.model_uri, json=payload)

        result = json.loads(response.content.decode('utf-8'))
        return self.categories[np.argmax(result['predictions'][0])]
