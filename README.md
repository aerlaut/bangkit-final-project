# BANGKIT FINAL PROJECT - BDG4-C

This is a project to fulfill the [Bangkit](https://events.withgoogle.com/bangkit/) program. For this project, we trained and implemented a machine learning model to differentiate several leaf diseases from 6 different types of agricultural plants using the [New Plant Disease Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset) from Kaggle.

The application can be seen at [bangkit.aerlaut.com](https://bangkit.aerlaut.com/)

### Authors

- [Aditya Firman Ihsan](https://github.com/phoenixfin)
- [Anugerah Erlaut](https://github.com/aerlaut)
- [Nova Lailatul Rizkiyah](https://github.com/novarizkiyah)

## Overview

The app is a Flask application passing uploaded images TensorFlow Model Server REST API. The prediction results are translated in this application. As such, to run this application, you have to install the application and the TensorFlow Model Server.

The training application is hosted at [https://github.com/phoenixfin/general-cnn](https://github.com/phoenixfin/general-cnn)

## Getting started

To deploy the trained model

1. Download the application files hosted at [this link](https://drive.google.com/drive/folders/11rtpBg7or2BfYXnYidRFYAGdxFCE4VE8?usp=sharing)
2. Crate a new directory and place the files following this structure
   ```
   /
       /1
           saved_model.pb
           /variables
               variables.index
               variables.data-00000-of-00002
               variables.data-00001-of-00002
       /assets
   ```
3. [Install tensorflow_model_server](https://www.tensorflow.org/tfx/serving/setup)
4. Run the model exposing a port
   ```
   tensorflow_model_server --model_base_path=<path_to_directory> --rest_api_port=5000 --model_name=PlantCNN
   ```

To deploy the application

1. Clone this repo

   ```
   git clone http://github.com/aerlaut/bangkit-final-project
   ```

2. Create virtual environment and activate

   OSX / Ubuntu :

   ```
   python3 -m venv venv
   source ./venv/bin/activate
   ```

   Windows :

   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies

   ```
   pip3 install -r requirements.txt
   ```

4. Create `.env` file at the root of the directory, with the following env variable

   ```
   MODEL_URI="http://localhost:5000/v1/models/PlantCNN:predict"
   ```

Change the port and model name according to the values used when setting up the model.
