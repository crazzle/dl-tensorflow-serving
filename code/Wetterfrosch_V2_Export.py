import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from keras.models import load_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import builder
import os

#%%
"""
Das Modell wird mit Keras geladen
"""
wetterfrosch_v1 = load_model('keras_models/wetterfrosch_v2.h5')

#%%
"""
Die Wetterfrosch V1 Signatur für TensorFlow-Serving wird zusammengebaut
"""
tensor_info_input = tf.saved_model.utils.build_tensor_info(wetterfrosch_v1.input)
tensor_info_output = tf.saved_model.utils.build_tensor_info(wetterfrosch_v1.output)
prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input': tensor_info_input},
        outputs={'prediction': tensor_info_output},
        method_name=signature_constants.PREDICT_METHOD_NAME)
)

#%%
"""
Das Keras-Modell wird für TensorFlow-Serving exportiert
"""
model_version = "2"
export_path = os.path.join("./models/weather", model_version)
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
