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
    @app.route('/')
    def index():
        examples = {
            'Apple Cedar Rust': 'example_img/apple_cedar_rust.jpg',
            'Apple Scab': 'example_img/apple_scab.jpg',
            'Corn Common Rust': 'example_img/corn_common_rust.jpg',
            'Potato Early Blight': 'example_img/potato_early_blight.jpg',
            'Tomato Early Blight': 'example_img/tomato_early_blight.jpg',
            'Tomato Yellow Curl': 'example_img/tomato_yellow_curl_virus.jpg',
        }

        return render_template('index.html', examples=examples)

    # Return prediction
    @app.route('/', methods=['POST'])
    def predict():

        if 'img'not in request.files:
            return {"status": "error", "message": "No file uploaded "}

        img = request.files.get('img', None)
        img_path = os.path.join('./temp/', img.filename)

        try:
            img.save(img_path)
        except FileNotFoundError:
            return {"status": "error", "message": "File upload failed"}

        result = model.predict(img_path)

        # Clean up - delete image
        os.remove(img_path)
        return {
            "status": "success",
            "message": result
        }

    return app


if __name__ == '__main__':

    app = create_app()
    app.run(debug=True)
