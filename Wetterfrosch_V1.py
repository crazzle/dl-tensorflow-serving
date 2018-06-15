import keras
from keras import utils
from tensorflow.saved_model.utils import build_tensor_info
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir("/Users/markkeinhorster/git/tensorflow_serving_weather_service")

#%%
"""
Die Wetterdaten werden mit Numpy geladen
"""
from numpy import genfromtxt
rohdaten = genfromtxt(fname='datasets/jena_climate_2009_2016.csv', 
                      delimiter=',', 
                      skip_header=True, 
                      usecols=(range(1,15)))

#%%
"""
Damit das neuronale Netz schneller konvergiert werden die Daten vorverarbeitet:
"""
# Die Daten werden normalisiert
mean = rohdaten[:200000].mean(axis=0)
normalisiert = rohdaten - mean

# Die Daten werden zentriert
std = normalisiert[:200000].std(axis=0)
zentriert = normalisiert / std

# Plotten der Daten
temp = zentriert[:, 1]  # temperature (in degrees Celsius)\n",
plt.plot(range(len(temp)), temp)
plt.show()


#%%
"""
Die Methode erstellt uns aus den zentrierten Daten 
Trainings- und Validierungsdatensatz
"""
def generiere(data, lookback, delay):
    max_index = len(data) - delay - 1
    x = np.zeros((max_index-lookback, lookback, 14)).astype(np.float64)
    y = np.zeros((max_index-lookback)).astype(np.float64)
    for i in range(0+lookback, max_index):
        x[i-lookback] = data[i-lookback:i]
        y[i-lookback] = data[i+delay][1]
    return x,y

#%%
"""
Die Parameter geben an wie die Daten aussehen
mit denen das Modell gefüttert wird
"""
# Die Daten werden auf einen Datenpunkt pro 
# Stunde hochgerechnet
schrittweite = 6 # = 1 Datenpunkt pro Stunde
stuendlich = zentriert[::schrittweite, :]

# Ein Datensatz besteht aus den letzten 5 Tagen der Vergangenheit 
# für eine Vorhersage für den nächsten Tag
zeitraum_vergangenheit = 120 # = 5 Tage = 5 * 24 Stunden
zeitraum_zukunft = 24 # = 1 Tag in die Zukunft vorhersagen

#%%
"""
Das Dataset wird in training, validation und test aufgesplittet
"""
# Die Trainingsdaten
training = stuendlich[:43800, :]
training_x, training_y = generiere(training, 
                                   zeitraum_vergangenheit, 
                                   zeitraum_zukunft)

# Die Validierungsdaten
validierung = stuendlich[43800:52560]
validierung_x, validierung_y = generiere(training, 
                                         zeitraum_vergangenheit, 
                                         zeitraum_zukunft)

# Die Testdaten
test = stuendlich[52560:, :]
test_x, test_y = generiere(test, 
                           zeitraum_vergangenheit, 
                           zeitraum_zukunft)

#%%
"""
Das Keras-Modell für den WetterfroschV1 wird aufgebaut
"""
wetterfrosch_1 = keras.Sequential()
#model.add(keras.layers.Conv1D(filters=32, 
#                         kernel_size=5, 
#                         activation='relu',
#                         input_shape=(lookback, centered_data.shape[-1]) 
#                         ))
# model.add(keras.layers.MaxPooling1D(pool_size=3))
# model.add(keras.layers.Conv1D(filters=32, 
#                         kernel_size=5, 
#                         activation='relu'
#                         ))
wetterfrosch_1.add(keras.layers.GRU(units=16,
                                    activation='relu',
                                    dropout=0.1,
                                    recurrent_dropout=0.4,
                                    return_sequences=False,
                                    input_shape=(zeitraum_vergangenheit, stuendlich.shape[-1])))
wetterfrosch_1.add(keras.layers.Dense(1))
wetterfrosch_1.compile(optimizer=keras.optimizers.RMSprop(), 
                       loss='mae')

utils.plot_model(a=wetterfrosch_1, 
                to_file="wetterfrosch_v1.png",
                show_shapes=True,
                show_layer_names=True)

history = wetterfrosch_1.fit(x=training_x,
                             y=training_y,
                             epochs=120,
                             batch_size=32,
                             validation_data=(validierung_x, validierung_y),
                             shuffle=False,
                             verbose=2)

#%%
"""
Das Modell wird mit den Testdaten evaluiert
"""
evaluierung = wetterfrosch_1.evaluate(x=test_x, y=test_y)
print evaluierung

#%%
"""
Das Modell wird nach dem Training im HDF5-Format gespeichert
"""
wetterfrosch_1.save('keras_models/wetterfrosch_v1.h5')

#%%
tensor_info_input =  tf.saved_model.utils.build_tensor_info(wetterfrosch_v1.input)
tensor_info_output = tf.saved_model.utils.build_tensor_info(wetterfrosch_v1.output)

from tensorflow.python.saved_model import signature_constants
prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input': tensor_info_input},
        outputs={'prediction': tensor_info_output},
        method_name=signature_constants.PREDICT_METHOD_NAME))

#%%
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import builder

model_version = 2
export_path = os.path.join("./models/weather", str(model_version))
tf_builder = builder.SavedModelBuilder(export_path)
with tf.keras.backend.get_session() as sess:
    tf_builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            }
    )
    tf_builder.save()
