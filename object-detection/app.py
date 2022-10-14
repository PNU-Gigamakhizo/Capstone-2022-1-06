import time
from importlib.resources import path
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import base64
import io
import os
import aspose.words as aw
from backend.tf_inference import load_model, inference
import yolov5.detect
from torchvision import models
import torch
import base64
import json
from differnet.mkGrad import getGrad
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = Flask(__name__)

@app.route('/api/', methods=["POST"])
def main_interface():
    response = request.get_json()

    model = response['model']
    data_str = response['image']

    point = data_str.find(',')
    base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"

    image = base64.b64decode(base64_str)       
    img = Image.open(io.BytesIO(image))
    img.save("./myImage/input/myTest.jpg")
    img.save("./differnet/dataset/apple/test/anomaly/myTest.jpg")

    ####fruits detection
    yolov5.detect.run(root='./runs/fruits.pt', file_path = "./myImage/result/myFruits.jpg", save_dir = 'myImage/result2/')

    ####defect detection
    yolov5.detect.run()

    ###gradient map
    getGrad()

    results = []
    return jsonify(results)

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
