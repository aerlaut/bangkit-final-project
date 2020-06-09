import json
import os

from dotenv import load_dotenv
from flask import Flask, render_template, url_for, request
from model import PlantCNNProxy

load_dotenv('.env')

# Set default model path, if not exist in .env
MODEL_PATH = os.getenv('MODEL_PATH') if os.getenv(
    'MODEL_PATH') != "" else "saved"


def create_app():
    app = Flask(__name__)

    model = new PlatnCNNProxy(os.getenv('MODEL_PATH'))

    # Front page
    @app.route('/')
    def index():
        return render_template('index.html')

    # Return prediction

    @app.route('/', methods=['POST'])
    def predict():

        img = request.files.get('img', None)
        if img is None:
            # TODO: return error
            return None

        img_path = os.path.join('./temp/', img.filename)

        try:
            img_path = os.path.join('./temp/', img.filename)
            img.save(img_path)

            result = model.predict(img_path)

            # Clean up - delete image
            os.remove(img_path)

        except FileNotFoundError:
            pass

        return json.dumps(result)

    return app


if __name__ == '__main__':

    app = create_app()
    app.run(debug=True)
