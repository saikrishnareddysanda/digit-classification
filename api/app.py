from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

def load_model(model_type):
    api_folder = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(api_folder, "../models")
    for i in os.listdir(model_folder):
        if i.startswith(model_type):
            model_path = os.path.join(model_folder, i)

    loaded_model = joblib.load(model_path)
    return loaded_model


@app.route('/predict/<string:model_type>')
def predict(model_type):
    model = load_model(model_type)
    return jsonify({"message": f"Model ({model_type}) loaded successfully"})