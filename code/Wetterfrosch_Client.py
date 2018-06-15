import numpy as np
from numpy import genfromtxt
import grpc
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import get_model_metadata_pb2

#%%
"""
Der Durchschnitt und die Standardabweichung wurden nicht in das exportierte
Modell integriert. Da die Testdaten vorher zentriert werden, müssen Durchschnitt
und Standardabweichung noch einmal berechnet werden.
"""
trainingsdaten = genfromtxt('datasets/jena_climate_2009_2016.csv',
                            delimiter=',',
                            skip_header=True,
                            usecols=(range(1, 15)))
testdaten = genfromtxt('datasets/jena_climate_2018.csv',
                       delimiter=',',
                       skip_header=True,
                       usecols=(range(1, 15)))

#%%
"""
Damit das neuronale Netz schneller konvergiert werden die Daten vorverarbeitet:
"""
mean = trainingsdaten[:200000].mean(axis=0)
rohdaten_normalisiert = trainingsdaten - mean
std = rohdaten_normalisiert[:200000].std(axis=0)
normalisiert = testdaten - mean
zentriert = normalisiert / std

#%%
"""
Für die Vorhersage wird der Zeitraum
04.05.2018 bis zum 09.05.2018
gewählt und auf stuendliche Werte hochgerechnet
"""
anfang = 18617 - (6 * 144) - 2
ende = 18617 - 1
zieltag = zentriert[anfang:ende]
schrittweite = 6
stuendlich = zieltag[::schrittweite, :].astype(np.float32)

#%%
"""
Das Modell wird mit den Daten vom 04.05.2018 bis zum 08.06.2018 gefüttert
Das erwartete Ergebnis ist die Temperatur vom 09.05.2018
"""
zeitraum_vergangenheit = 120
zeitraum_zukunft = 24
test_x = np.expand_dims(stuendlich[:zeitraum_vergangenheit], axis=0)
test_y = stuendlich[zeitraum_vergangenheit + zeitraum_zukunft][1]

#%%
"""
Die Verbindungsdaten zum TensorFlow-Serving ModelServer
"""
host = "localhost"
port = "9000"
channel = grpc.insecure_channel("localhost:9000")
stub = prediction_service_pb2.PredictionServiceStub(channel)

#%%
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