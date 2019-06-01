import tensorflow as tf
from constants import *

def load_image(filename, label=None):
    img_raw = tf.io.read_file(filename)
    img_tensor = tf.image.decode_image(img_raw, channels=3)
    img_final = tf.image.resize(img_tensor, (IMAGE_SIZE, IMAGE_SIZE))
    img_final = (img_final-IMAGE_MEAN)/IMAGE_STD
    if label is None:
        return img_final
    return img_final, label
