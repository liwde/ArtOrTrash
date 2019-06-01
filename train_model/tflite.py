# https://github.com/tensorflow/tensorflow/issues/26672#issuecomment-472519970
# In the master, you can already use TFLiteConverter.from_keras_model()

import tensorflow as tf
from constants import *

model = tf.keras.models.load_model(PATH_MODEL)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(PATH_TFLITE_FLOAT,"wb").write(tflite_model)

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model_quant = converter_quant.convert()
open(PATH_TFLITE_QUANT,"wb").write(tflite_model_quant)
