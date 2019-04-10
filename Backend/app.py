import time
from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import image
from io import BytesIO

# Communication to TensorFlow server via gRPC
"""from grpc.beta import implementations
import tensorflow as tf

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto
"""

app = Flask(__name__)

"""host = '0.0.0.0'
port = 8500
channel = implementations.insecure_channel(host, port)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

def sendRequest(im):
    req = predict_pb2.PredictRequest()
    req.model_spec.name = 'resnet_exported'
    req.model_spec.signature_name = 'predict'
    req.inputs['x'].CopyFrom(make_tensor_proto(ids, shape=[1, 224, 224, 3], dtype=tf.float32))
    result = stub.Predict(req, 60.0)

    predictions = result.outputs['preds'].int64_val
    return predictions
"""

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
    im = Image.open(BytesIO(base64.b64decode(request.form['image'])))
    numpy_im = np.array(im)
    return jsonify({"shape": numpy_im.shape})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=443)