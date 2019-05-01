import time
from flask import Flask, request, jsonify, Response
import pickle
import numpy as np
import base64
from PIL import Image
from io import BytesIO

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto

app = Flask(__name__)

host = '35.244.20.1'
port = 8500
channel = implementations.insecure_channel(host, port)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

def sendRequest(im):
    req = predict_pb2.PredictRequest()
    req.model_spec.name = 'resnet'
    req.model_spec.signature_name = 'predict'
    req.inputs['input_tensor'].CopyFrom(make_tensor_proto(ids, shape=[1, 224, 224, 3], dtype=tf.float32))
    result = stub.Predict(req, 60.0)

    predictions = result.outputs['probabilities'].float32_val
    return predictions


@app.route('/ping')
def ping():
    print(request.get_json())
    return "pong"

'''
from PIL import Image
from io import BytesIO
import base64

data['img'] = 'R0lGODlhDwAPAKECAAAAzMzM/////wAAACwAAAAADwAPAAACIISPeQHsrZ5ModrLl
N48CXF8m2iQ3YmmKqVlRtW4MLwWACH+H09wdGltaXplZCBieSBVbGVhZCBTbWFydFNhdmVyIQAAOw=='

im = Image.open(BytesIO(base64.b64decode(data)))
# https://stackoverflow.com/questions/26070547/decoding-base64-from-post-to-use-in-pil
'''


@app.route('/image', methods=['POST'])
def api():
    img = Image.open(request.files.get("image"))
    numpy_img = np.array(img)
    nis = numpy_img.shape
    if nis[0] != nis[1]:
        if nis[0] > nis[1]:
            numpy_img = np.pad(numpy_img, [(0, 0), ((nis[0]-nis[1])//2, (nis[0]-nis[1]+1)//2), (0, 0)], 'constant')
        if nis[1] > nis[0]:
            numpy_img = np.pad(numpy_img, [((nis[1]-nis[0])//2, (nis[1]-nis[0]+1)//2), (0, 0), (0, 0)], 'constant')
    img = Image.fromarray(numpy_img)
    img = img.resize((224, 224), Image.NEAREST)
    reshaped_im = np.array(img).reshape([1, 224, 224, 3])
    preds = sendRequest(reshaped_im)
    # print({"shape": list(numpy_im.shape)})
    result = "".join(str(preds))
    resp = Response(result)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=443)
