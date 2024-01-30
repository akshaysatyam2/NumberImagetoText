import os

import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

path = (os.getcwd()).replace('\\', '/')
project_index = path.find('WebApp')

# Directories of project working
working_directory = path[:project_index + 6] + '/'
temp_directory = working_directory + 'temp/'


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == "POST":
        file = request.files['checkImage']
        file_name = secure_filename(file.filename)
        os.chdir(temp_directory)
        file.save(file_name)
        os.chdir(working_directory)
        result = getPredictions(temp_directory + file_name)
        response = {'Filename': file_name, 'Number': result}
        return jsonify(response)


@app.route('/', methods=['POST'])
def result():
    if request.method == "POST":
        file = request.files['checkImage']
        file_name = secure_filename(file.filename)
        os.chdir(temp_directory)
        file.save(file_name)
        os.chdir(working_directory)
        result = getPredictions(temp_directory + file_name)
        return render_template('result.html', result=result)


def getPredictions(file_url):
    model = tf.keras.models.load_model("num.model")

    img = cv2.imread(file_url)[:, :, 0]
    img = cv2.resize(img, (28, 28))

    img = np.invert(np.array([img]))

    img = img.reshape(1, 28, 28)

    img = img / 255.0

    prediction = model.predict(img)

    result = np.argmax(prediction)

    if result:
        return result
    else:
        return 'error'


if __name__ == "__main__":
    app.run(debug=True)

# To test this REST API, we can use a tool like curl or a web client(ex- Postman) to send POST requests to your endpoint at http://127.0.0.1:5000/api/predict.
# We should send a JSON payload with a 'new_review' field to get the prediction result.

# Command example curl -X POST -H "Content-Type: application/json" -d @payload.json http://127.0.0.1:5000/api/predict
# Output= {'Filename': 'file_name.png', 'Number': 6}
