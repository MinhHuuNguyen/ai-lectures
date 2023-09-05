from flask import Flask, request
from PIL import Image
import torch

from io import BytesIO


app = Flask(__name__)


model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom


def predict(image):
    results = model(image).crop()
    for pred in results:
        pred['box'] = [float(pred_) for pred_ in pred['box']]
        pred['conf'] = float(pred['conf'])
        pred['cls'] = int(pred['cls'])
        pred.pop('im')
    return results


@app.route('/predict_path', methods=['POST'])
def route_predict_path():
    request_data = request.json
    results = predict(request_data['image_path'])
    return results


@app.route('/predict_file', methods=['POST'])
def route_predict_file():
    request_data = request.data
    image = Image.open(BytesIO(request_data))

    results = predict(image)
    return results


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1504)
