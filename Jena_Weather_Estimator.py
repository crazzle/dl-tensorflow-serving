from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
os.chdir("/Users/markkeinhorster/git/tensorflow_serving_weather_service")

#%%
from numpy import genfromtxt
float_data = genfromtxt('datasets/jena_climate_2009_2016.csv', delimiter=',', skip_header=True, usecols=(range(1,15)))
print float_data.shape
print len(float_data)

#%%
mean = float_data[:200000].mean(axis=0)
normalized_data = float_data - mean
std = normalized_data[:200000].std(axis=0)
centered_data = normalized_data / std
print len(centered_data)
from matplotlib import pyplot as plt
temp = centered_data[:, 1]  # temperature (in degrees Celsius)\n",
plt.plot(range(len(temp)), temp)
plt.show()


#%%
def generate_data(data, lookback, delay):
    max_index = len(data) - delay - 1
    x = np.zeros((max_index-lookback, lookback, 14)).astype(np.float64)
    y = np.zeros((max_index-lookback)).astype(np.float64)
    for i in range(0+lookback, max_index):
        x[i-lookback] = data[i-lookback:i]
        y[i-lookback] = data[i+delay][1]
    return x,y

#%%
step = 6 # = 1 Datenpunkt pro Stunde
lookback = 120 # = 5 Tage = 5 * 24 Stunden
delay = 24 # = 1 Tag in die Zukunft vorhersagen
max_index = len(centered_data) - delay - 1
subsampled = centered_data[::step, :]

#%%
# Datenset: x_shape = [-1, 120, 14]
training_data = subsampled[:43800, :]
train_x, train_y = generate_data(training_data, lookback, delay)

val_data = subsampled[43800:52560]
val_x, val_y = generate_data(val_data, lookback, delay)

test_data = subsampled[52560:, :]
test_x, test_y = generate_data(test_data, lookback, delay)

 #%%
model = keras.Sequential()
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
model.add(keras.layers.GRU(units=16,
                     activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.4,
                     return_sequences=False,
                     input_shape=(lookback, centered_data.shape[-1])
                     ))
model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.RMSprop(), loss='mae')

#%%
from keras import utils
utils.plot_model(model, "wetterfrosch_conv.png")

#%%
history = model.fit(x=train_x,
                    y=train_y,
                    epochs=120,
                    batch_size=32,
                    validation_data=(val_x, val_y),
                    shuffle=False,
                    verbose=2)

#%%
from matplotlib import pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#%%
evals = model.evaluate(x=test_x, y=test_y)
print evals

#%%
pred = model.predict(test_x)
print pred.shape

#%%
index = 10110
pred = model.predict(test_x[index:index+1])
print "pred=%f, target=%f, mean=%f, std=%f" % (pred[0,0]*std[1]+mean[1], test_y[index]*std[1]+mean[1], mean[1], std[1])
print pred.shape

#%%
from keras.utils import plot_model

plot_model(model, )

#%%
epochs = range(len(pred))
plt.figure()
plt.plot(epochs, pred, 'r-', label='Prediction')
plt.plot(epochs, test_y, 'b-', label='Label')
plt.title('Prediction and Label')
plt.legend()

plt.show()

#%%
model.save('keras_models/jena_temp_pred_v1_empty.h5')
model = keras.models.load_model("keras_models/jena_temp_pred.h5")

#%%
tensor_info_input = tf.saved_model.utils.build_tensor_info(model.input)
tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output)

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
