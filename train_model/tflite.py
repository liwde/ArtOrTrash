# https://github.com/tensorflow/tensorflow/issues/26672#issuecomment-472519970
# In the master, you can already use TFLiteConverter.from_keras_model()

import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model_float.tflite","wb").write(tflite_model)

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model_quant = converter_quant.convert()
open("model_quant.tflite","wb").write(tflite_model_quant)
