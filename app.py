import json
import os
import requests

from dotenv import load_dotenv
from flask import Flask, render_template, url_for, request
from model import PlantCNNProxy

load_dotenv('.env')

# Set default model path, if not exist in .env
MODEL_PATH = os.getenv('MODEL_URI') if os.getenv(
    'MODEL_URI') != "" else "saved"


def create_app():
    app = Flask(__name__)

    model = PlantCNNProxy(os.getenv('MODEL_URI'))

    # Front page
    @app.route('/', methods=['GET', 'POST'])
    def index():

        # Define examples dict
        examples = {
            'Apple Cedar Rust': 'example_img/apple_cedar_rust.jpg',
            'Apple Scab': 'example_img/apple_scab.jpg',
            'Corn Common Rust': 'example_img/corn_common_rust.jpg',
            'Potato Early Blight': 'example_img/potato_early_blight.jpg',
            'Tomato Early Blight': 'example_img/tomato_early_blight.jpg',
            'Tomato Yellow Curl': 'example_img/tomato_yellow_curl_virus.jpg',
        }

        result = None

        # If previous image exists, delete image
        IMG_PATH = os.path.join('static/temp/input.jpg')
        img = None

        if os.path.isfile(IMG_PATH):
            os.remove(IMG_PATH)
            img = None

        if request.method not in ['POST', 'GET']:
            return '404 Not found'

        # Form submitted, do prediction
        elif request.method == 'POST':

            if 'img' not in request.files:
                return {"status": "error", "message": "No file uploaded "}

            img = request.files.get('img', None)

            try:
                img.save(IMG_PATH)
            except FileNotFoundError:
                return {"status": "error", "message": "File upload failed"}

            result = model.predict(IMG_PATH)

            result = {
                "status": "success",
                "message": result
            }

        return render_template('index.html', examples=examples, result=result, img_path=IMG_PATH)

    return app


if __name__ == '__main__':

    app = create_app()
    app.run(debug=True)
