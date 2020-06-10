import tensorflow as tf
import numpy as np
import requests
import json

from pathlib import Path

# Class based on https://github.com/phoenixfin/general-cnn/blob/master/cnn.py
# optimized for Plant CNN deployment


class PlantCNNProxy(object):

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

    # Plant categories
    categories = [
        {
            'plant': 'apple',
            'status': 'scab',
            'plant_info': 'https://en.wikipedia.org/wiki/Apple',
            'disease_info': 'https://en.wikipedia.org/wiki/Apple_scab'
        },
        {
            'plant': 'apple',
            'status': 'black rot',
            'plant_info': 'https://en.wikipedia.org/wiki/Apple',
            'disease_info': 'https://extension.umn.edu/plant-diseases/black-rot-apple'
        },
        {
            'plant': 'apple',
            'status': 'cedar rust',
            'plant_info': 'https://en.wikipedia.org/wiki/Apple',
            'disease_info': 'https://en.wikipedia.org/wiki/Gymnosporangium_juniperi-virginianae'
        },
        {
            'plant': 'apple',
            'status': 'healty',
            'plant_info': 'https://en.wikipedia.org/wiki/Apple',
            'disease_info': None
        },
        {
            'plant': 'blueberry',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Blueberry',
            'disease_info': None
        },
        {
            'plant': 'cherry',
            'status': 'powdery mildew',
            'plant_info': 'https://en.wikipedia.org/wiki/Cherry',
            'disease_info': 'http://treefruit.wsu.edu/article/cherry-powdery-mildew-questions-and-answers-from-2017/'
        },
        {
            'plant': 'cherry',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Cherry',
            'disease_info': None
        },
        {
            'plant': 'corn',
            'status': 'Cercospora / gray leaf spot',
            'plant_info': 'https://en.wikipedia.org/wiki/Corn',
            'disease_info': 'https://www.pioneer.com/us/agronomy/gray_leaf_spot_cropfocus.html'
        },
        {
            'plant': 'corn',
            'status': 'common rust',
            'plant_info': 'https://en.wikipedia.org/wiki/Corn',
            'disease_info': 'https://cropprotectionnetwork.org/resources/articles/diseases/common-rust-of-corn'
        },
        {
            'plant': 'corn',
            'status': 'Northern leaf blight',
            'plant_info': 'https://en.wikipedia.org/wiki/Corn',
            'disease_info': 'https://www.extension.purdue.edu/extmedia/BP/BP-84-W.pdf'
        },
        {
            'plant': 'corn',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Corn',
            'disease_info': None
        },
        {
            'plant': 'grape',
            'status': 'black rot',
            'plant_info': 'https://en.wikipedia.org/wiki/Grape',
            'disease_info': 'https://extension.psu.edu/black-rot-on-grapes-in-home-gardens'
        },
        {
            'plant': 'grape',
            'status': 'esca (black measles)',
            'plant_info': 'https://en.wikipedia.org/wiki/Grape',
            'disease_info': 'https://grapes.extension.org/grapevine-measles/'
        },
        {
            'plant': 'grape',
            'status': 'leaf blight (Isariopsis leaf spot)',
            'plant_info': 'https://en.wikipedia.org/wiki/Grape',
            'disease_info': 'https://ohioline.osu.edu/factsheet/plpath-fru-47'
        },
        {
            'plant': 'grape',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Grape',
            'disease_info': None
        },
        {
            'plant': 'orange',
            'status': 'citrus greening',
            'plant_info': 'https://en.wikipedia.org/wiki/Orange_(fruit)',
            'disease_info': 'https://en.wikipedia.org/wiki/Citrus_greening_disease'
        },
        {
            'plant': 'peach',
            'status': 'bacterial spot',
            'plant_info': 'https://en.wikipedia.org/wiki/Peach',
            'disease_info': 'https://www.canr.msu.edu/news/management_of_bacterial_spot_on_peaches_and_nectarines'
        },
        {
            'plant': 'peach',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Peach',
            'disease_info': None
        },
        {
            'plant': 'bell pepper',
            'status': 'bacterial spot',
            'plant_info': 'https://en.wikipedia.org/wiki/Bell_pepper',
            'disease_info': 'https://content.ces.ncsu.edu/bacterial-spot-of-pepper-and-tomato'
        },
        {
            'plant': 'bell pepper',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Bell_pepper',
            'disease_info': None
        },
        {
            'plant': 'potato',
            'status': 'early blight',
            'plant_info': 'https://en.wikipedia.org/wiki/Potato',
            'disease_info': 'https://www.ag.ndsu.edu/publications/crops/early-blight-in-potato'
        },
        {
            'plant': 'potato',
            'status': 'late blight',
            'plant_info': 'https://en.wikipedia.org/wiki/Potato',
            'disease_info': 'https://cropwatch.unl.edu/potato/late_blights_description'
        },
        {
            'plant': 'potato',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Potato',
            'disease_info': None
        },
        {
            'plant': 'raspberry',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Raspberry',
            'disease_info': None
        },
        {
            'plant': 'soybean',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Soybean',
            'disease_info': None
        },
        {
            'plant': 'squash',
            'status': 'powdery mildew',
            'plant_info': 'https://en.wikipedia.org/wiki/Cucurbita',
            'disease_info': 'https://content.ces.ncsu.edu/cucurbit-powdery-mildew'
        },
        {
            'plant': 'strawberry',
            'status': 'leaf scorch',
            'plant_info': 'https://en.wikipedia.org/wiki/Strawberry',
            'disease_info': 'https://content.ces.ncsu.edu/leaf-scorch-of-strawberry'
        },
        {
            'plant': 'strawberry',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Strawberry',
            'disease_info': None
        },
        {
            'plant': 'tomato',
            'status': 'bacterial spot',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': 'https://content.ces.ncsu.edu/bacterial-spot-of-pepper-and-tomato'
        },
        {
            'plant': 'tomato',
            'status': 'early blight',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': 'https://content.ces.ncsu.edu/early-blight-of-tomato'
        },
        {
            'plant': 'tomato',
            'status': 'late blight',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': 'https://extension.umn.edu/diseases/late-blight'
        },
        {
            'plant': 'tomato',
            'status': 'leaf mold',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': 'https://extension.umn.edu/diseases/leaf-mold-tomato'
        },
        {
            'plant': 'tomato',
            'status': 'septoria leaf spot',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': 'https://www.missouribotanicalgarden.org/gardens-gardening/your-garden/help-for-the-home-gardener/advice-tips-resources/pests-and-problems/diseases/fungal-spots/septoria-leaf-spot-of-tomato.aspx'
        },
        {
            'plant': 'tomato',
            'status': 'spider mites',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': 'https://entomology.k-state.edu/doc/misc.-extension-document/spider-mites-on-tomatoes.pdf'
        },
        {
            'plant': 'tomato',
            'status': 'target spot',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': 'http://203.64.245.61/web_crops/tomato/target.pdf'
        },
        {
            'plant': 'tomato',
            'status': 'yellow leaf curl virus',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': 'https://ipm.ifas.ufl.edu/Agricultural_IPM/tylcv_home_mgmt.shtml'
        },
        {
            'plant': 'tomato',
            'status': 'mosaic virus',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': 'https://extension.umn.edu/diseases/tomato-mosaic-virus-and-tobacco-mosaic-virus'
        },
        {
            'plant': 'tomato',
            'status': 'healthy',
            'plant_info': 'https://en.wikipedia.org/wiki/Tomato',
            'disease_info': None
        },
    ]
