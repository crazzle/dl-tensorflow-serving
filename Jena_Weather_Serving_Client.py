from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
os.chdir("/Users/markkeinhorster/git/tensorflow_serving_weather_service")

#%%
from numpy import genfromtxt
float_data = genfromtxt('datasets/jena_climate_2009_2016.csv', 
	delimiter=',', 
	skip_header=True, 
	usecols=(range(1,15)))
print float_data.shape
print len(float_data)

#%%
mean = float_data[:200000].mean(axis=0)
normalized_data = float_data - mean
std = normalized_data[:200000].std(axis=0)
centered_data = normalized_data / std
print len(centered_data)

#%%
from numpy import genfromtxt
test_data = genfromtxt('datasets/jena_climate_2018.csv', 
    delimiter=',', 
    skip_header=True, 
    usecols=(range(1,15)))
print test_data.shape
print len(test_data)

#%%
test_normalized = test_data - mean
test_centered = test_normalized / std

#%%
step = 6 # = 1 Datenpunkt pro Stunde
lookback = 120 # = 5 Tage = 5 * 24 Stunden
delay = 24 # = 1 Tag in die Zukunft vorhersagen

#%%
subsampled = test_centered[::step, :]
day = test_centered[18617-(6*144)-2:18617-1]
subsampled = day[::step, :].astype(np.float32)
test_x = np.expand_dims(subsampled[:lookback], axis=0)
test_y = subsampled[lookback+delay][1]
print test_y

#%%
import grpc
from tensorflow_serving.apis import prediction_service_pb2

host = "localhost"
port = "9000"
channel = grpc.insecure_channel("localhost:9000")
stub = prediction_service_pb2.PredictionServiceStub(channel)

#%%
from tensorflow_serving.apis import get_model_metadata_pb2

status = get_model_metadata_pb2.GetModelMetadataRequest()
status.model_spec.name = 'weather'
status.model_spec.version.value = 4
status.metadata_field.append("signature_def")
response = stub.GetModelMetadata(status, 10)
print response

#%%
from tensorflow_serving.apis import predict_pb2
from tensorflow.contrib.util import make_tensor_proto

request = predict_pb2.PredictRequest()
request.model_spec.name = 'weather'
request.model_spec.version.value = 2
proto = make_tensor_proto(test_x)
request.inputs['input'].CopyFrom(proto)
result = stub.Predict(request)  # 10 secs timeout
print "prediction: %s, label: %f" % (result, test_y)
print "pred=%f, target=%f, mean=%f, std=%f" % (result.outputs['prediction'].float_val*std[1]+mean[1], test_y*std[1]+mean[1], mean[1], std[1])