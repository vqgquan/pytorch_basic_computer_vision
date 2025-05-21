from flask import Flask, request
import torch

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Preprocess input & run model inference here
    return {"prediction": result}

app.run(port=5000)